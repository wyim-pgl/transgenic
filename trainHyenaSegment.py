#!/usr/bin/env python
import torch, os, wandb, gc
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall
from utils_transgenic import *
from configuration_transgenic import TransgenicHyenaConfig
from safetensors.torch import save_model
from modeling_HeynaTransgenic import HyenaEncoder

os.environ['HF_HOME'] = './HFmodels'

def linear_decay(step, total_steps, start_value=0.5, end_value=0.0):
	if step >= total_steps:
		return end_value 
	
	decay_rate = (start_value - end_value) / total_steps
	return start_value - (decay_rate * step)

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
	torch.manual_seed(123)
	torch.cuda.manual_seed_all(123)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_segment_collate_fn)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_segment_collate_fn)
	
	#total_bp = 0
	#class_bp = [0,0,0,0,0,0,0,0,0]
	#for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
	#	total_bp += 6144
	#	for i in range(9):
	#		class_bp[i] += batch[2][:, :, i].sum().item()


	segment_checkpoint = "checkpoints/Hyena_SegementSkipConnect_E0-9.safetensors"
	config = TransgenicHyenaConfig(do_segment=True, numSegClasses=9)
	model = HyenaEncoder(config)

	tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			tensors[k] = f.get_tensor(k)
	tensors = {key.replace("transgenic.encoder.", ""):tensors[key] for key in tensors}
	new_tensors = {}
	for key in tensors:
		if "transgenic.decoder." not in key:
			if "segmentation_head" not in key:
				new_tensors[key] = tensors[key]

	model.load_state_dict(new_tensors, strict=False)

	model.to(device)
	model.train()
	for param in model.parameters():
		param.requires_grad = True
	for param in model.segmentation_head.parameters():
		param.requires_grad = True

	# Setup the optimizer
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	t_total = (len(train_ds) // accumulation_steps) * num_epochs
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0.05*t_total,
		num_training_steps=t_total
		)
	
	features = ["Gene", "Start_Codon", "Exon", "Intron", "SDonor", "SAcceptor", "UTR5", "UTR3", "Stop_Codon"]
	total_mlp = [
		{"total_mlp_Gene":0}, 
		{"total_mlp_Start_Codon":0}, 
		{"total_mlp_Exon":0}, 
		{"total_mlp_Intron":0}, 
		{"total_mlp_SDonor":0}, 
		{"total_mlp_SAcceptor":0}, 
		{"total_mlp_UTR5":0}, 
		{"total_mlp_UTR3":0}, 
		{"total_mlp_Stop_Codon":0}
	]
	total_mlr = [
		{"total_mlr_Gene":0}, 
		{"total_mlr_Start_Codon":0}, 
		{"total_mlr_Exon":0}, 
		{"total_mlr_Intron":0}, 
		{"total_mlr_SDonor":0}, 
		{"total_mlr_SAcceptor":0}, 
		{"total_mlr_UTR5":0}, 
		{"total_mlr_UTR3":0}, 
		{"total_mlr_Stop_Codon":0}
	]
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		total_mlp = [
			{"total_mlp_Gene":0}, 
			{"total_mlp_Start_Codon":0}, 
			{"total_mlp_Exon":0}, 
			{"total_mlp_Intron":0}, 
			{"total_mlp_SDonor":0}, 
			{"total_mlp_SAcceptor":0}, 
			{"total_mlp_UTR5":0}, 
			{"total_mlp_UTR3":0}, 
			{"total_mlp_Stop_Codon":0}
		]
		total_mlr = [
			{"total_mlr_Gene":0}, 
			{"total_mlr_Start_Codon":0}, 
			{"total_mlr_Exon":0}, 
			{"total_mlr_Intron":0}, 
			{"total_mlr_SDonor":0}, 
			{"total_mlr_SAcceptor":0}, 
			{"total_mlr_UTR5":0}, 
			{"total_mlr_UTR3":0}, 
			{"total_mlr_Stop_Codon":0}
		]
		genic_steps = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			if 'Zm' in batch[3][0]: 
				continue

			ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
			lab = lab[:, :, 0:9]
			dii = None
			#try:
			outputs = model(ii, segLabels=lab)
			
			if torch.sum(lab[:, :, 0]) > 0:
				genic_steps += 1
				predictions = torch.sigmoid(outputs.segmentation_logits).squeeze().cpu()
				labels = batch[2][:, :, 0:9].detach().cpu().squeeze().int()
				mlp = MultilabelPrecision(num_labels=9, average=None)(predictions, labels).tolist() # False positive rate
				for i,value in enumerate(mlp):
					total_mlp[i][f"total_mlp_{features[i]}"]+= value
				mlr = MultilabelRecall(num_labels=9, average=None)(predictions, labels).tolist() # False negative rate
				for i,value in enumerate(mlr):
					total_mlr[i][f"total_mlr_{features[i]}"]+= value
			
			#torch.sum(torch.sigmoid(outputs[1][:, :, 0]) > 0.5)
			#torch.sum(lab[:, :, 0])
			total_loss += outputs.segmentation_loss.detach().float()
			outputs.segmentation_loss /= accumulation_steps
			
			outputs.segmentation_loss.backward()
			#except:
			#	print(f"Error in batch: {batch[3]}")
			
			if (step+1) % accumulation_steps == 0:
				clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				if schedule_lr: lr_scheduler.step()
				# log metrics to wandb
				if log_wandb:
					wandb_log = {"epoch":epoch, "step":step, "loss": outputs[2].detach().float()*accumulation_steps, "mean_loss": (total_loss) / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
					for i, value in enumerate(total_mlp):
						wandb_log[list(value.keys())[0]] = value[list(value.keys())[0]]/(genic_steps+1)
					for i, value in enumerate(total_mlr):
						wandb_log[list(value.keys())[0]] = value[list(value.keys())[0]]/(genic_steps+1)
					for name, param in model.named_parameters():
						if (param.grad != None) & (param.requires_grad):
							grad_norm = param.grad.norm().detach().item()
							wandb_log[f"{name}_grad_norm"] = grad_norm
					wandb.log(wandb_log)
				optimizer.zero_grad()
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch}, Step {step}, Loss {outputs[2].detach().float()*accumulation_steps}", file=sys.stderr)
				save_model(model, f"{checkpoint_path}/model.safetensors")
			del outputs
			torch.cuda.empty_cache()
			

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(batch[0].to(device), segLabels=batch[2][:, :, 0:9].to(device))
				eval_loss += outputs.segmentation_loss.detach().float()
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

	dt = GFFTokenizer()
	ds  = preprocessedSegmentationDatasetHyena("Segmentation_9Genomes_preprocessed_scodons.db")
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [534331, 71244,106867])

	trainTransgenicFCGAccelerate(
		train_data, 
		eval_data, 
		lr=5e-5, 
		num_epochs=15, 
		schedule_lr=True, 
		eval=True, 
		batch_size=1, 
		accumulation_steps=256,
		checkpoint_path="checkpoints_HyenaSegment/", 
		safetensors_model=None,
		output_dir="saved_models_Hyena/",
		max_grad_norm=1,
		notes="Training with Hyena",
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
		unlink = False,
		log_wandb=True
	)