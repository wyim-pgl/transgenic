#!/usr/bin/env python
import torch, os, wandb, gc
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from safetensors.torch import save_model
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from utils_transgenic import *

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_HOME'] = './HFmodels'
#os.environ["NCCL_DEBUG"] = "INFO"

def trainSegmentNTAccelerate(
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
	encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
	unlink=False):

	# start a new wandb run to track this script
	wandb.init(
		# set the wandb project where this run will be logged
		project="AgroSegmentNT",

		# track hyperparameters and run metadata
		config={
		"learning_rate": lr,
		"schedule_lr": schedule_lr,
		"architecture": "ESM + SegmentNT",
		"dataset": "Segmentation_7Genomes.db",
		"epochs": num_epochs,
		"max_grad_norm": max_grad_norm,
		"accumulation_steps": accumulation_steps,
		"Optimizer": "AdamW",
		"Checkpoints":checkpoint_path,
		"Outputs":output_dir,
		"Notes":notes
		}
	)

	print(f"Training SegmentNT. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)
	
	# Set up accelerator
	#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	#accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # gradient_accumulation_steps=32
	#device = accelerator.device
	device = torch.device("cuda")
	print(f"Training SegmentNT with Accelerate on {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(123)
	torch.cuda.manual_seed_all(123)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=segment_collate_fn)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=segment_collate_fn)
	
	# initialize the model
	model = segmented_sequence_embeddings(encoder_model, segmentation_model, 14, do_segment=True)
	model.train()
	for param in model.esm.parameters():
		param.requires_grad = False
	for param in model.hidden_mapping.parameters():
		param.requires_grad = False
	for param in model.hidden_mapping_layernorm.parameters():
		param.requires_grad = False
	#for param in model.film.parameters():
	#	param.requires_grad = False
	
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)

	model.to(device)
	
	# Setup the optimizer
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_ds) * num_epochs),
		)
	
	# Prep objects for use with accelerator
	#model, optimizer, train_ds, eval_ds, lr_scheduler = accelerator.prepare(
	#	model, optimizer, train_ds, eval_ds, lr_scheduler
	#)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			outputs = model(batch[0].to(device), attention_mask=batch[1].to(device), segLabels=batch[2].to(device))
			loss = outputs[3] / accumulation_steps
			#accelerator.backward(loss)
			loss.backward()
			total_loss += outputs[3].detach().float()
			if (step+1) % accumulation_steps == 0:
				clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				if schedule_lr: lr_scheduler.step()
				# log metrics to wandb
				wandb_log = {"epoch":epoch, "step":step, "loss": outputs[3].detach().float(), "mean_loss": total_loss / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
				for name, param in model.named_parameters():
					if (param.grad != None) & (param.requires_grad):
						grad_norm = param.grad.norm().item()
						wandb_log[f"{name}_grad_norm"] = grad_norm
				wandb.log(wandb_log)
				optimizer.zero_grad()
			
			if (step % 300 == 0):
				print(f"Epoch {epoch=}, Step {step=}, Loss {loss=}", file=sys.stderr)
				#accelerator.save_state(output_dir=checkpoint_path)
				save_model(model, f"{checkpoint_path}/model.safetensors")
				

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(batch[0], attention_mask=batch[1], segLabels=batch[2])
				loss = outputs[3]
				eval_loss += loss.detach().float()

			eval_epoch_loss = eval_loss / len(eval_ds)
			eval_ppl = torch.exp(eval_epoch_loss)
			print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
			wandb_log = {"epoch_train_ppl":train_ppl, "epoch_train_loss":train_epoch_loss, "epoch_eval_ppl":eval_ppl, "epoch_eval_loss":eval_epoch_loss}
			wandb.log(wandb_log)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		
		if eval:
			if best_eval_score is None or eval_epoch_loss < best_eval_score:
				best_eval_score = eval_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				#accelerator.save_state(output_dir=checkpoint_path)
				save_model(model, f"{checkpoint_path}/model.safetensors")
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				#accelerator.save_state(output_dir=checkpoint_path)
				save_model(model, f"{checkpoint_path}/model.safetensors")
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
		
		torch.cuda.empty_cache()
		gc.collect()
		total_loss = 0
		loss = None
		train_epoch_loss = 0
		train_ppl = 0

	
	#accelerator.wait_for_everyone()
	#accelerator.save_model(model, output_dir)
	wandb.finish()


if __name__ == '__main__':
	torch.manual_seed(123)
	
	encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b"
	segmentation_model = "InstaDeepAI/segment_nt_multi_species"

	ds  = segmentationDataset(6144, 6000, "/home/jlomas/Segmentation_10Genomes.db")
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [800419, 106722,160085])

	trainSegmentNTAccelerate(
		train_data, 
		eval_data, 
		lr=1e-4, 
		num_epochs=5, 
		schedule_lr=True, 
		eval=True, 
		batch_size=16, 
		accumulation_steps=16,
		checkpoint_path="checkpoints_SegmentNT", 
		safetensors_model=None, #"checkpoints_SegmentNT/model.safetensors",
		output_dir="saved_models_SegmentNT/",
		max_grad_norm=2,
		notes="Short length, direct fine tuning of agro SegmentNT",
		encoder_model=encoder_model,
		unlink = False
	)

	#db = "Segmentation_7Genomes.db"
	#files = {
	#	"training_data/Athaliana_167_TAIR10.gene.exon.splice.gff3":["training_data/Athaliana_167_TAIR10.fa","ath"],
	#	"training_data/Bdistachyon_314_v3.1.gene_exons.exon.splice.gff3":["training_data/Bdistachyon_314_v3.0.fa","bdi"],
	#	"training_data/Sbicolor_730_v5.1.gene_exons.exon.splice.gff3":["training_data/Sbicolor_730_v5.0.fa","sbi"],
	#	"training_data/Sitalica_312_v2.2.gene_exons.exon.splice.gff3":["training_data/Sitalica_312_v2.fa","sit"],
	#	"training_data/Ptrichocarpa_533_v4.1.gene_exons.exon.splice.gff3":["training_data/Ptrichocarpa_533_v4.0.fa","ptr"],
	#	"training_data/Gmax_880_Wm82.a6.v1.gene_exons.exon.splice.gff3":[ "training_data/Gmax_880_v6.0.fa","gma"],
	#	"training_data/Ppatens_318_v3.3.gene_exons.exon.splice.gff3":["training_data/Ppatens_318_v3.fa","ppa"]
	#}
	#for file in files:
	#	genome2SegmentationSet(
	#	files[file][0], 
	#	file,
	#	files[file][1],
	#	db)
	#sys.exit()
