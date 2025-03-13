#!/usr/bin/env python
import torch, os, wandb, gc
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_transgenic import *
from configuration_transgenic import TransgenicHyenaConfig
from safetensors.torch import save_model
from modeling_HeynaTransgenic import HyenaForMLM

os.environ['HF_HOME'] = './HFmodels'


def trainTransgenicFCGAccelerate(
	train_ds:isoformData, 
	eval_ds:isoformData, 
	lr=1e-4, 
	num_epochs=10,  
	schedule_lr=True, 
	eval=True, 
	batch_size=1,
	max_grad_norm=1.0,
	checkpoint_path="checkpoints_FCG/",
	safetensors_model=None,
	output_dir="saved_models_FCG/",
	accumulation_steps=32,
	notes="",
	log_wandb=True):

	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# start a new wandb run to track this script
	if log_wandb:
		wandb.init(
			# set the wandb project where this run will be logged
			project="transgenic",

			# track hyperparameters and run metadata
			config={
			"learning_rate": lr,
			"schedule_lr": schedule_lr,
			"architecture": "HyenaForMLM",
			"dataset": "9G_6144nt",
			"epochs": num_epochs,
			"max_grad_norm": max_grad_norm,
			"accumulation_steps": accumulation_steps,
			"Optimizer": "AdamW",
			"Checkpoints":checkpoint_path,
			"Outputs":output_dir,
			"Notes":notes
			}
		)

	print(f"Training transgenic with HyenaforMLM. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)

	device = torch.device("cuda")
	print(f"Using: {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(123)
	torch.cuda.manual_seed_all(123)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyenaMLM_collate_fn)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyenaMLM_collate_fn)
	
	#model = getModel(None, safetensors_model=safetensors_model, device="cpu", mode="train")
	#decoder_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_E30.safetensors"#"checkpoints_HyenaPosEmbed/model.safetensors"
	#segment_checkpoint = None

	config = TransgenicHyenaConfig(
	do_segment=False, 
	d_model=512,
	encoder_layers=12,
	decoder_layers=12,
	encoder_n_layer=12,
	attention_window = [
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024
		])
	model = HyenaForMLM(config)

	#tensors = {}
	#with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
	#	for k in f.keys():
	#		if ("segmentation" not in k) and ("transgenic.decoder" not in k) and ("lm_head" not in k): 
	#			tensors[k] = f.get_tensor(k)
	#tensors["transgenic.decoder_embed_tokens.weight"] = tensors["lm_head.weight"]
	#tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]

	# Add missing shared HyenaSin freq weights
	#freq_tensors = {}
	#for k in tensors.keys():
	#	if "freq" in k:
	#		freq_tensors[".".join(k.split(".")[0:9]) + ".3.freq"] = tensors[k]
	#		freq_tensors[".".join(k.split(".")[0:9]) + ".5.freq"] = tensors[k]

	#model.load_state_dict(tensors | freq_tensors, strict=False)

	model.to(device)
	model.train()

	
	#for param in model.transgenic.encoder.parameters():
	#	param.requires_grad = False

	# Setup the optimizer
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	t_total = ((len(train_ds)//batch_size) // accumulation_steps) * num_epochs
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=t_total*0.05,
		num_training_steps=t_total
		)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			if 'Zm' in batch[3][0]: 
				continue
			if (ii.shape[1] % 3) != 0:
				continue

			ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
			#try:
			outputs = model(ii, mask_index=am, labels=lab)
			total_loss += outputs[1].detach().float()
			outputs[1] = outputs[1] / accumulation_steps
			
			outputs[1].backward()
			#except:
			#	print(f"Error in batch: {batch[3]}")
			
			if (step+1) % accumulation_steps == 0:
				clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				if schedule_lr: lr_scheduler.step()
				# log metrics to wandb
				if log_wandb:
					wandb_log = {"epoch":epoch, "step":step, "loss": outputs[1].detach().float()*accumulation_steps, "mean_loss": (total_loss) / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
					for name, param in model.named_parameters():
						if (param.grad != None) & (param.requires_grad):
							grad_norm = param.grad.norm().detach().item()
							wandb_log[f"{name}_grad_norm"] = grad_norm
					wandb.log(wandb_log)
				optimizer.zero_grad()
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch}, Step {step}, Loss {outputs[1].detach().float()*accumulation_steps}", file=sys.stderr)
				save_model(model, f"{checkpoint_path}/model.safetensors")
			del outputs
			torch.cuda.empty_cache()
			

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(batch[0].to(device), mask_index=batch[1].to(device), labels=batch[2].to(device))
				eval_loss += outputs[1].detach().float()

			eval_epoch_loss = eval_loss / len(eval_ds)
			eval_ppl = torch.exp(eval_epoch_loss)
			print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
			if log_wandb:
				wandb_log = {"epoch_train_ppl":train_ppl, "epoch_train_loss":train_epoch_loss, "epoch_eval_ppl":eval_ppl, "epoch_eval_loss":eval_epoch_loss}
				wandb.log(wandb_log)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		
		if eval:
			if best_eval_score is None or eval_epoch_loss < best_eval_score:
				best_eval_score = eval_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				save_model(model, f"{checkpoint_path}/model.safetensors")
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				save_model(model, f"{checkpoint_path}/model.safetensors")
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
		
		torch.cuda.empty_cache()
		gc.collect()
		total_loss = 0
		train_epoch_loss = 0
		train_ppl = 0
	
	if log_wandb:
		wandb.finish()


if __name__ == '__main__':
	torch.manual_seed(123)

	db="Segmentation_9Genomes_preprocessed_scodons.db"
	dt = GFFTokenizer()
	ds = MLMDatasetHyena(db, encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf")
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [534331, 71244,106867])

	trainTransgenicFCGAccelerate(
		train_data, 
		eval_data, 
		lr=1e-4, 
		num_epochs=20, 
		schedule_lr=True, 
		eval=True, 
		batch_size=4, 
		accumulation_steps=16,
		checkpoint_path="checkpoints_HyenaMLM/", 
		safetensors_model=None,
		output_dir="saved_models_HyenaMLM/",
		max_grad_norm=1,
		notes="Training with HyenaMLM",
		log_wandb=True
	)
