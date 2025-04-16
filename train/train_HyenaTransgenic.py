#!/usr/bin/env python
import torch, os, wandb, gc, time, sys
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from safetensors.torch import save_model, safe_open

sys.path.insert(0, f'{os.getcwd()}/src')
from datasets.datasets import isoformData, isoformDataHyena, makeDataLoader, hyena_collate_fn
from models.tokenization_transgenic import GFFTokenizer
from models.modeling_HeynaTransgenic import transgenicForConditionalGeneration
from models.configuration_transgenic import HyenaTransgenicConfig

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
	safetensors_model=None,
	output_dir="saved_models_FCG/",
	accumulation_steps=32,
	notes="",
	encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
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
			project="transgenic",

			# track hyperparameters and run metadata
			config={
			"learning_rate": lr,
			"schedule_lr": schedule_lr,
			"architecture": "Hyena",
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

	print(f"Training transgenic with Hyena. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)

	device = torch.device("cuda")
	print(f"Using: {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(234)
	torch.cuda.manual_seed_all(234)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
	
	#model = getModel(None, safetensors_model=safetensors_model, device="cpu", mode="train")
	# "checkpoints/Hyena_Gen9G_6144nt_E30.safetensors"
	# "checkpoints/Hyena_Gen9G_6144nt_wide_E12.safetensors"
	# #"checkpoints_HyenaPosEmbed/model.safetensors"
	# "checkpoints_HyenaWide/model.safetensors"
	# "checkpoints/Hyena_Gen9G_6144nt_512L18_E2.safetensors"
	decoder_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_768L12_E15.safetensors"#"checkpoints/Hyena_Gen9G_6144nt_768L12_E5.safetensors"#"checkpoints/Hyena_Gen9G_6144nt_wide_E12.safetensors"
	segment_checkpoint = None
	layers = 12
	attentionWindow = [
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024,
			#1024,1024,1024,1024,1024,1024
		]

	config = HyenaTransgenicConfig(
	do_segment=False, 
	numSegClasses=9,
	d_model=768,
	encoder_layers=layers,
	decoder_layers=layers,
	encoder_n_layer=layers,
	attention_window = attentionWindow,
	dropout=0.1,
	encoder_attention_heads=6,
	decoder_attention_heads=6
	)
	model = transgenicForConditionalGeneration(config)

	tensors = {}
	with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			#if ("segmentation" not in k) and ("transgenic.decoder" not in k) and ("lm_head" not in k): 
			tensors[k.replace("_orig_mod.", "")] = f.get_tensor(k)
	tensors["transgenic.decoder_embed_tokens.weight"] = tensors["lm_head.weight"]
	tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]

	# Add missing shared HyenaSin freq weights
	freq_tensors = {}
	for k in tensors.keys():
		if "freq" in k:
			freq_tensors[".".join(k.split(".")[0:9]) + ".3.freq"] = tensors[k]
			freq_tensors[".".join(k.split(".")[0:9]) + ".5.freq"] = tensors[k]

	state_dict = tensors | freq_tensors

	#missing_keys = []
	#for key, dict_param in state_dict.items():
	#
	#	submod_names = key.split(".")
	#	try:
	#		curr_param = get_attr(model, submod_names)
	#	except:
	#		curr_param = None
	#	
	#	if curr_param != None:
	#		if curr_param.shape != dict_param.shape: 
	#			with torch.no_grad():
	#				curr_param[tuple(slice(0, dim) for dim in dict_param.shape)] = dict_param
	#			set_attr(model, submod_names, curr_param)
	#		else:
	#			# Or re-use it (as done in load_state_dict) but the sizes have to match!
	#			set_attr(model, submod_names, curr_param)
	#	else:
	#		missing_keys.append(key)
	
	model.load_state_dict(state_dict, strict=True)

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
	t_total = (len(train_ds) // accumulation_steps) * num_epochs
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,#t_total*0.05,
		num_training_steps=t_total
		)
	
	# Add accelerator
	accelerator = Accelerator()
	model, optimizer, train_ds, schedule_lr = accelerator.prepare(
		model, optimizer, train_ds, schedule_lr
	)

	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
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
					if schedule_lr: lr_scheduler.step()
					# log metrics to wandb
					if log_wandb:
						wandb_log = {"epoch":epoch, "step":step, "loss": outputs.loss.detach().float()*accumulation_steps, "mean_loss": (total_loss) / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
						for name, param in model.named_parameters():
							if (param.grad != None) & (param.requires_grad):
								grad_norm = param.grad.norm().detach().item()
								wandb_log[f"{name}_grad_norm"] = grad_norm
						wandb.log(wandb_log)
					optimizer.zero_grad()
				
				if (step % 5000 == 0) & (step != 0):
					print(f"Epoch {epoch}, Step {step}, Loss {outputs.loss.detach().float()*accumulation_steps}", file=sys.stderr)
					save_model(model, f"{checkpoint_path}/model.safetensors")
				del outputs
				torch.cuda.empty_cache()
			except Exception as e:
				print(f"Error in batch: {batch[3]}, {e}")
				optimizer.zero_grad()
				model.zero_grad()
				del outputs
				torch.cuda.empty_cache()
				gc.collect()
				time.sleep(1)
				continue
			
			
			

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[2].to(device), return_dict=True)
				eval_loss += outputs.loss.detach().float()

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

	db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
	dt = GFFTokenizer()
	ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])

	trainTransgenicFCGAccelerate(
		train_data, 
		eval_data, 
		lr=5e-5, 
		num_epochs=10, 
		schedule_lr=True, 
		eval=True, 
		batch_size=1, 
		accumulation_steps=128,
		checkpoint_path="checkpoints_HyenaWide/", 
		safetensors_model=None,
		output_dir="saved_models_HyenaWide/",
		max_grad_norm=1,
		notes="d_model=768, 12 layers, 8 attn_heads, training from original 512 pretrained, restart from epoch5",
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
		unlink = False,
		log_wandb=True
	)