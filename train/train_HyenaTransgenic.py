#!/usr/bin/env python
import torch, os, wandb, gc, time, sys, math, json, argparse
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from safetensors.torch import save_model

# sys.path.insert(0, f"{os.getcwd()}/src/transgenic")
from transgenic.datasets.datasets import isoformData, isoformDataHyena, makeDataLoader, hyena_collate_fn
from transgenic.model.tokenization_transgenic import GFFTokenizer
from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig

os.environ['HF_HOME'] = './HFmodels'

def linear_decay(step, total_steps, start_value=0.5, end_value=0.0):
	if step >= total_steps:
		return end_value 
	
	decay_rate = (start_value - end_value) / total_steps
	return start_value - (decay_rate * step)

def get_attr(obj, names):
	if len(names) == 1:
		return getattr(obj, names[0])
	elif type(obj) == torch.nn.modules.container.ModuleList:
		return get_attr(obj[int(names[0])], names[1:])
	else:
		return get_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
	if len(names) == 1:
		setattr(obj, names[0], val)
	elif type(obj) == torch.nn.modules.container.ModuleList:
		return set_attr(obj[int(names[0])], names[1:], val)
	else:
		set_attr(getattr(obj, names[0]), names[1:], val)


def _repeat_list(val, n):
	return [val for _ in range(n)]


def _count_parameters(model: torch.nn.Module) -> int:
	return sum(p.numel() for p in model.parameters())


def _write_json(path: str, obj: dict):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, sort_keys=True)


def _read_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

def trainTransgenicFCGAccelerate(
	train_ds:isoformData, 
	eval_ds:isoformData, 
	lr, 
	num_epochs,  
	schedule_lr, 
	eval, 
	batch_size,
	max_grad_norm=1.0,
	checkpoint_path="checkpoints_FCG/",
	output_dir="saved_models_FCG/",
	accumulation_steps=32,
	notes="",
	encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
	resume_from_checkpoint: str | None = None,
	save_every_epoch: bool = True,
	unlink=False,
	log_wandb=True):

	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# start a new wandb run to track this script
	if log_wandb:
		wandb.init(
			# set the wandb project where this run will be logged
			entity="transgenic-paper",
			project="transgenic",
			# track hyperparameters and run metadata
			config={
			"learning_rate": lr,
			"schedule_lr": schedule_lr,
			"architecture": "Hyena",
			"dataset": "10G_static6144_addExtra200_addRCIsoOnly_clean",
			"epochs": num_epochs,
			"max_grad_norm": max_grad_norm,
			"accumulation_steps": accumulation_steps,
			"Optimizer": "AdamW",
			"Checkpoints":checkpoint_path,
			"Outputs":output_dir,
			"Notes":notes,
			}
		)

	print(f"Training transgenic with Hyena. {checkpoint_path=} {output_dir=}", file=sys.stderr)

	accelerator = Accelerator(mixed_precision="fp16")
	device = accelerator.device
	print(f"Using: {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(234)
	torch.cuda.manual_seed_all(234)
	train_dl = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
	eval_dl = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
	
	# Larger model size (roughly 3x params vs d_model=768, layers=12):
	# (1152/768)^2 * (16/12) ~= 3.0
	base_d_model = 1152
	layers = 16
	attn_heads = 8

	attentionWindow = _repeat_list(1024, layers)
	ffn_dim = base_d_model * 4

	config = HyenaTransgenicConfig(
		d_model=base_d_model,
		encoder_layers=layers,
		decoder_layers=layers,
		encoder_n_layer=layers,
		encoder_ffn_dim=ffn_dim,
		decoder_ffn_dim=ffn_dim,
		attention_window=attentionWindow,
		dropout=0.1,
		encoder_attention_heads=attn_heads,
		decoder_attention_heads=attn_heads,
		encoder_model=encoder_model,
		unlink=unlink,
	)
	model = transgenicForConditionalGeneration(config)

	print(f"Model params: {_count_parameters(model):,}", file=sys.stderr)

	model.gradient_checkpointing_enable()
	model.to(device)
	model.train()

	
	#for param in model.transgenic.encoder.parameters():
	#	param.requires_grad = False

	#pretrained_params = []
	#new_params = []
	#for name, param in model.named_parameters():
		# Example condition: assume new layers contain 'new_layer' in their name.
	#	if name in new_keys.missing_keys:
	#		new_params.append(param)
	#	else:
	#		pretrained_params.append(param)

	# Setup the optimizer
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
	#optimizer = optim.AdamW([
	#	{'params': pretrained_params, 'lr': lr / 2},
	#	{'params': new_params, 'lr': lr}
	#], weight_decay=0.01)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	steps_per_epoch = max(1, math.ceil(len(train_dl) / accumulation_steps))
	t_total = steps_per_epoch * num_epochs
	lr_scheduler = None
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,#t_total*0.05,
		num_training_steps=t_total
		)

	# Prepare with accelerator
	if lr_scheduler is None:
		model, optimizer, train_dl, eval_dl = accelerator.prepare(model, optimizer, train_dl, eval_dl)
	else:
		model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
			model, optimizer, train_dl, eval_dl, lr_scheduler
		)

	# Optionally resume full training state (model/optimizer/scheduler/rng)
	start_epoch = 0
	if resume_from_checkpoint is not None:
		accelerator.load_state(resume_from_checkpoint)
		meta_path = os.path.join(resume_from_checkpoint, "meta.json")
		if os.path.exists(meta_path):
			meta = _read_json(meta_path)
			start_epoch = int(meta.get("next_epoch", 0))
		else:
			# If meta isn't present, resume from epoch 0 and just continue training.
			start_epoch = 0
		print(f"Resumed from {resume_from_checkpoint}; {start_epoch=}", file=sys.stderr)

	def _save_training_state(epoch_next: int, best_score):
		# Save a full Accelerate checkpoint for resume
		ckpt_dir = os.path.join(checkpoint_path, f"accelerate_epoch_{epoch_next}")
		os.makedirs(ckpt_dir, exist_ok=True)
		accelerator.save_state(ckpt_dir)
		_write_json(
			os.path.join(ckpt_dir, "meta.json"),
			{
				"next_epoch": epoch_next,
				"best_eval_score": None if best_score is None else float(best_score),
			},
		)

	# Training loop
	best_eval_score = None
	try:
		for epoch in range(start_epoch, num_epochs):
			total_loss = 0
			for step, batch in enumerate(tqdm(train_dl, miniters=10, disable=False)):
				if 'Zm' in batch[3][0]: 
					continue

				ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
				#if ii.shape[1] > 49000:
				#	continue
				
				dii = None
				try:
					outputs = None
					outputs = model(input_ids=ii, attention_mask=am, decoder_input_ids=dii, labels=lab, return_dict=True)
					total_loss += outputs.loss.detach().float()
					outputs.loss = outputs.loss / accumulation_steps
					#outputs.loss.backward()
					accelerator.backward(outputs.loss)
					if (step+1) % accumulation_steps == 0:
						clip_grad_norm_(model.parameters(), max_grad_norm)
						optimizer.step()
						if lr_scheduler is not None:
							lr_scheduler.step()
						# log metrics to wandb
						if log_wandb:
							wandb_log = {
								"epoch": epoch,
								"step": step,
								"loss": outputs.loss.detach().float() * accumulation_steps,
								"mean_loss": (total_loss) / (step + 1),
							}
							if lr_scheduler is not None:
								wandb_log["lr"] = lr_scheduler.get_last_lr()[0]
							for name, param in model.named_parameters():
								if (param.grad != None) & (param.requires_grad):
									grad_norm = param.grad.norm().detach().item()
									wandb_log[f"{name}_grad_norm"] = grad_norm
							wandb.log(wandb_log)
						optimizer.zero_grad()
					
					if (step % 5000 == 0) & (step != 0):
						print(f"Epoch {epoch}, Step {step}, Loss {outputs.loss.detach().float()*accumulation_steps}", file=sys.stderr)
						# Lightweight model-only snapshot (not enough to resume optimizer)
						save_model(accelerator.unwrap_model(model), f"{checkpoint_path}/model.safetensors")
					del outputs
				except Exception as e:
					print(f"Error in batch: {batch[3]}, {e}")
					optimizer.zero_grad()
					model.zero_grad()
					del outputs
					torch.cuda.empty_cache()
					gc.collect()
					time.sleep(1)
					continue
			
			train_epoch_loss = total_loss / len(train_dl)
			train_ppl = torch.exp(train_epoch_loss)

			if eval:
				eval_loss = 0
				for batch in tqdm(eval_dl, miniters=10, disable=False):
					with torch.no_grad():
						outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[2].to(device), return_dict=True)
					eval_loss += outputs.loss.detach().float()

				eval_epoch_loss = eval_loss / len(eval_dl)
				eval_ppl = torch.exp(eval_epoch_loss)
				print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
				if log_wandb:
					wandb_log = {"epoch_train_ppl":train_ppl, "epoch_train_loss":train_epoch_loss, "epoch_eval_ppl":eval_ppl, "epoch_eval_loss":eval_epoch_loss}
					wandb.log(wandb_log)
			else:
				print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
			
			if eval:
				if best_eval_score is None or eval_epoch_loss < best_eval_score:
					best_eval_score = eval_epoch_loss
					if not os.path.exists("checkpoints"):
						os.makedirs("checkpoints", exist_ok=True)
					save_model(accelerator.unwrap_model(model), f"{checkpoint_path}/model.safetensors")
					print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
			else:
				if best_eval_score is None or train_epoch_loss < best_eval_score:
					best_eval_score = train_epoch_loss
					if not os.path.exists("checkpoints"):
						os.makedirs("checkpoints", exist_ok=True)
					save_model(accelerator.unwrap_model(model), f"{checkpoint_path}/model.safetensors")
					print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)

			# Always save a resumable checkpoint at epoch boundary
			if save_every_epoch:
				_save_training_state(epoch_next=epoch + 1, best_score=best_eval_score)
			
			torch.cuda.empty_cache()
			gc.collect()
			total_loss = 0
			train_epoch_loss = 0
			train_ppl = 0
	except KeyboardInterrupt:
		print("KeyboardInterrupt: saving resume checkpoint...", file=sys.stderr)
		_save_training_state(epoch_next=epoch + 1 if 'epoch' in locals() else 0, best_score=best_eval_score)
		raise
	if log_wandb:
		wandb.finish()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train HyenaTransgenic model")
	parser.add_argument("--db", type=str,
		default="/home/framazan/data/Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db",
		help="Path to DuckDB training database")
	parser.add_argument("--no-wandb", action="store_true",
		help="Disable Weights & Biases logging")
	args = parser.parse_args()

	torch.manual_seed(123)

	db = args.db
	ds = isoformDataHyena(db, mode="train", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
	total = len(ds)
	train_size = int(total * 0.75)
	eval_size = int(total * 0.10)
	test_size = total - train_size - eval_size
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [train_size, eval_size, test_size])

	trainTransgenicFCGAccelerate(
		train_data, 
		eval_data, 
		lr=5e-5, 
		num_epochs=10, 
		schedule_lr=True, 
		eval=True, 
		batch_size=16,
		accumulation_steps=16,
		checkpoint_path="checkpoints_HyenaWide/", 
		output_dir="saved_models_HyenaWide/",
		max_grad_norm=1,
		notes="HyenaTransgenic ~3x params (d_model=1152, layers=16, heads=8), training from scratch",
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
		unlink = False,
		log_wandb=not args.no_wandb
	)