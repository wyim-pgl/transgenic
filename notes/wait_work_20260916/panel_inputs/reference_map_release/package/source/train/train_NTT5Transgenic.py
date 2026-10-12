#!/usr/bin/env python
import torch, os, wandb, gc, sys
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from safetensors.torch import save_model

from ..datasets.datasets import isoformData, makeDataLoader, target_collate_fn
from ..models.tokenization_transgenic import GFFTokenizer
from ..models.modeling_HeynaTransgenic import transgenicForConditionalGenerationT5
from ..models.configuration_transgenic import NTTransgenicT5Config

os.environ['HF_HOME'] = './HFmodels'


def trainT5Transgenic(
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
			
			project="transgenic", # set the wandb project where this run will be logged
			config={              # track hyperparameters and run metadata
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

	print(f"Training transgenic with T5. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)

	device = torch.device("cuda")
	print(f"Using: {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(123)
	torch.cuda.manual_seed_all(123)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=target_collate_fn)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=target_collate_fn)
	
	#model = getModel(None, safetensors_model=safetensors_model, device="cpu", mode="train")
	#decoder_checkpoint = "checkpoints_Hyena/model.safetensors"
	#segment_checkpoint = None

	config = NTTransgenicT5Config()
	model = transgenicForConditionalGenerationT5(config)

	model.to(device)
	model.train()

	for param in model.transgenic.encoder.esm.parameters():
		param.requires_grad = False

	# Setup the optimizer
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	t_total = len(train_ds) // num_epochs * accumulation_steps
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,#t_total*0.05,
		num_training_steps=t_total
		)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			if 'Zm' in batch[4][0]: 
				continue

			ii, am, lab = batch[0].to(device), batch[1].to(device), batch[3].to(device)
			dii = None
			try:
				outputs = model(input_ids=ii, attention_mask=am, decoder_input_ids=dii, labels=lab, return_dict=True)
				total_loss += outputs.loss.detach().float()
				outputs.loss = outputs.loss / accumulation_steps
			
				outputs.loss.backward()
			except:
				print(f"Error in batch: {batch[4]}")
			
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
			

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				if 'Zm' in batch[4][0]: 
					continue
				with torch.no_grad():
					outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[3].to(device), return_dict=True)
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
	ds = isoformData(db, dt, mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False)
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])

	trainT5Transgenic(
		train_data, 
		eval_data, 
		lr=8e-5, 
		num_epochs=15, 
		schedule_lr=True, 
		eval=True, 
		batch_size=1, 
		accumulation_steps=32,
		checkpoint_path="checkpoints_T5/", 
		safetensors_model=None,
		output_dir="saved_models_T5/",
		max_grad_norm=1,
		notes="Training with LongT5-base",
		encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b",
		unlink = False,
		log_wandb=True
	)