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
from modeling_HeynaTransgenic import HyenaEncoder, BiLSTMRegressionHead, UNet1DRegressionHead, DilatedCNNRegressionWithAttention

os.environ['HF_HOME'] = './HFmodels'

def trainTransgenicFCGAccelerate(
	train_ds:isoformData, 
	eval_ds:isoformData, 
	lr, 
	num_epochs,  
	schedule_lr, 
	eval, 
	batch_size,
	max_grad_norm=1.0,
	checkpoint_path="checkpoints_BiLSTM/",
	safetensors_model=None,
	output_dir="saved_models_BiLSTM/",
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
	torch.manual_seed(345)
	torch.cuda.manual_seed_all(345)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
	
	decoder_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_E30.safetensors"

	config = TransgenicHyenaConfig(do_segment=False)
	encoder_model = HyenaEncoder(config)

	tensors = {}
	with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			if ("segmentation" not in k) and ("transgenic.decoder" not in k) and ("lm_head" not in k): 
				tensors[k.replace("transgenic.encoder.", "")] = f.get_tensor(k)
	#tensors["transgenic.decoder_embed_tokens.weight"] = tensors["lm_head.weight"]
	#tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]

	encoder_model.load_state_dict(tensors, strict=False)

	encoder_model.to(device)
	encoder_model.eval()

	# Set up regression heads
	biLSTM_Start = DilatedCNNRegressionWithAttention()
	biLSTM_Start.load_state_dict(torch.load(f"{checkpoint_path}/biLSTM_Start.pth"))
	biLSTM_Start.to(device)
	biLSTM_Stop = DilatedCNNRegressionWithAttention()
	biLSTM_Stop.load_state_dict(torch.load(f"{checkpoint_path}/biLSTM_Stop.pth"))
	biLSTM_Stop.to(device)
	biLSTM_SD = DilatedCNNRegressionWithAttention()
	biLSTM_SD.load_state_dict(torch.load(f"{checkpoint_path}/biLSTM_SD.pth"))
	biLSTM_SD.to(device)
	biLSTM_SA = DilatedCNNRegressionWithAttention()
	biLSTM_SA.load_state_dict(torch.load(f"{checkpoint_path}/biLSTM_SA.pth"))
	biLSTM_SA.to(device)


	# Setup the optimizers
	start_optimizer = optim.AdamW(biLSTM_Start.parameters(), lr=lr)
	start_optimizer.zero_grad()
	stop_optimizer = optim.AdamW(biLSTM_Stop.parameters(), lr=lr)
	stop_optimizer.zero_grad()
	sd_optimizer = optim.AdamW(biLSTM_SD.parameters(), lr=lr)
	sd_optimizer.zero_grad()
	sa_optimizer = optim.AdamW(biLSTM_SA.parameters(), lr=lr)
	sa_optimizer.zero_grad()
	
	# Create the learning rate scheduler
	t_total = (len(train_ds) // accumulation_steps) * num_epochs
	if schedule_lr:
		start_lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=start_optimizer,
		num_warmup_steps=t_total*0.05,
		num_training_steps=t_total
		)
		stop_lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=stop_optimizer,
		num_warmup_steps=t_total*0.05,
		num_training_steps=t_total
		)
		sd_lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=sd_optimizer,
		num_warmup_steps=t_total*0.05,
		num_training_steps=t_total
		)
		sa_lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=sa_optimizer,
		num_warmup_steps=t_total*0.05,
		num_training_steps=t_total
		)
	
	loss_fn = nn.MSELoss()

	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss_start = 0
		total_loss_stop = 0
		total_loss_sd = 0
		total_loss_sa = 0
		step_start = 0
		step_stop = 0
		step_sd = 0
		step_sa = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			if 'Zm' in batch[3][0]: 
				continue

			ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
			
			try:
			
				outputs = encoder_model(ii)
				outputs.last_hidden_state = outputs.last_hidden_state.detach()
			except:
				print(f"Error in batch: {batch[3]}")
				continue
			
			true = dt.batch_decode(lab.detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
			sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace(" ", "")
			pp = PredictionProcessor(true, sequence, None)

			# Process start codons
			for cds in pp.start_cds:
				start = int(pp.features[pp.feature_index_dict[cds]][0])
				if pp.strand == "-":
					start = len(sequence)-start
				randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
				if start-randomNumber < 0:
					randomNumber = torch.tensor([start])
				if start+(200-randomNumber) > len(sequence):
					randomNumber = torch.tensor([start-len(sequence)+200])
				prediction = biLSTM_Start(outputs.last_hidden_state[:, start-randomNumber:start+(200-randomNumber)])
				loss = torch.log1p(loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device)))
				total_loss_start += loss.detach().cpu()
				loss /= accumulation_steps
				loss.backward()

				if (step_start+1) % accumulation_steps == 0:
					clip_grad_norm_(biLSTM_Start.parameters(), max_grad_norm)
					start_optimizer.step()
					if schedule_lr: start_lr_scheduler.step()
					# log metrics to wandb
					if log_wandb:
						wandb_log = {"step_start":step_start, "start_loss": loss.detach().float()*accumulation_steps, "start_mean_loss": (total_loss_start) / (step_start+1), "start_lr": start_lr_scheduler.get_last_lr()[0]}
						for name, param in biLSTM_Start.named_parameters():
							if (param.grad != None) & (param.requires_grad):
								grad_norm = param.grad.norm().detach().item()
								wandb_log[f"start_{name}_grad_norm"] = grad_norm
						wandb.log(wandb_log)
					start_optimizer.zero_grad()
			
				if (step_start % 5000 == 0) & (step_start != 0):
					torch.save(biLSTM_Start.state_dict(), f"{checkpoint_path}/biLSTM_Start.pth")
				
				step_start += 1
			
			# Process stop codons
			for cds in pp.end_cds:
				stop = int(pp.features[pp.feature_index_dict[cds]][2])
				if pp.strand == "-":
					stop = len(sequence)-stop
				randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
				if stop-randomNumber < 0:
					randomNumber = torch.tensor([stop])
				if stop+(200-randomNumber) > len(sequence):
					randomNumber = torch.tensor([stop-len(sequence)+200])
				prediction = biLSTM_Stop(outputs.last_hidden_state[:, stop-randomNumber:stop+(200-randomNumber)])
				loss = torch.log1p(loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device)))
				total_loss_stop += loss.detach().cpu()
				loss /= accumulation_steps
				loss.backward()

				if (step_stop+1) % accumulation_steps == 0:
					clip_grad_norm_(biLSTM_Stop.parameters(), max_grad_norm)
					stop_optimizer.step()
					if schedule_lr: stop_lr_scheduler.step()
					# log metrics to wandb
					if log_wandb:
						wandb_log = {"stop_step":step_stop, "stop_loss": loss.detach().float()*accumulation_steps, "stop_mean_loss": (total_loss_stop) / (step_stop+1), "stop_lr": stop_lr_scheduler.get_last_lr()[0]}
						for name, param in biLSTM_Stop.named_parameters():
							if (param.grad != None) & (param.requires_grad):
								grad_norm = param.grad.norm().detach().item()
								wandb_log[f"stop_{name}_grad_norm"] = grad_norm
						wandb.log(wandb_log)
					stop_optimizer.zero_grad()
			
				if (step_stop % 5000 == 0) & (step_stop != 0):
					torch.save(biLSTM_Stop.state_dict(), f"{checkpoint_path}/biLSTM_Stop.pth")
				
				step_stop += 1
			
			# Process Splice donors
			for cds in pp.cds_pairs:
				sd = int(pp.features[pp.feature_index_dict[cds[0]]][2])
				if pp.strand == "-":
					sd = len(sequence)-sd
				randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
				if sd-randomNumber < 0:
					randomNumber = torch.tensor([sd])
				if sd+(200-randomNumber) > len(sequence):
					randomNumber = torch.tensor([sd-len(sequence)+200])
				prediction = biLSTM_SD(outputs.last_hidden_state[:, sd-randomNumber:sd+(200-randomNumber)])
				loss = torch.log1p(loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device)))
				total_loss_sd += loss.detach().cpu()
				loss /= accumulation_steps
				loss.backward()

				if (step_sd+1) % accumulation_steps == 0:
					clip_grad_norm_(biLSTM_SD.parameters(), max_grad_norm)
					sd_optimizer.step()
					if schedule_lr: sd_lr_scheduler.step()
					# log metrics to wandb
					if log_wandb:
						wandb_log = {"sd_step":step_sd, "sd_loss": loss.detach().float()*accumulation_steps, "sd_mean_loss": (total_loss_sd) / (step_sd+1), "sd_lr": sd_lr_scheduler.get_last_lr()[0]}
						for name, param in biLSTM_SD.named_parameters():
							if (param.grad != None) & (param.requires_grad):
								grad_norm = param.grad.norm().detach().item()
								wandb_log[f"sd_{name}_grad_norm"] = grad_norm
						wandb.log(wandb_log)
					sd_optimizer.zero_grad()
			
				if (step_sd % 5000 == 0) & (step_sd != 0):
					torch.save(biLSTM_SD.state_dict(), f"{checkpoint_path}/biLSTM_SD.pth")
				
				step_sd += 1

			# Process Splice Acceptors
			for cds in pp.cds_pairs:
				sa = int(pp.features[pp.feature_index_dict[cds[1]]][0])
				if pp.strand == "-":
					sa = len(sequence)-sa
				randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
				if sa-randomNumber < 0:
					randomNumber = torch.tensor([sa])
				if sa+(200-randomNumber) > len(sequence):
					randomNumber = torch.tensor([sa-len(sequence)+200])
				prediction = biLSTM_SA(outputs.last_hidden_state[:, sa-randomNumber:sa+(200-randomNumber)])
				loss = torch.log1p(loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device)))
				total_loss_sa += loss.detach().cpu()
				loss /= accumulation_steps
				loss.backward()

				if (step_sa+1) % accumulation_steps == 0:
					clip_grad_norm_(biLSTM_SA.parameters(), max_grad_norm)
					sa_optimizer.step()
					if schedule_lr: sa_lr_scheduler.step()
					# log metrics to wandb
					if log_wandb:
						wandb_log = {"sa_step":step_sa, "sa_loss": loss.detach().float()*accumulation_steps, "sa_mean_loss": (total_loss_sa) / (step_sa+1), "sa_lr": sa_lr_scheduler.get_last_lr()[0]}
						for name, param in biLSTM_SA.named_parameters():
							if (param.grad != None) & (param.requires_grad):
								grad_norm = param.grad.norm().detach().item()
								wandb_log[f"sa_{name}_grad_norm"] = grad_norm
						wandb.log(wandb_log)
					sa_optimizer.zero_grad()
			
				if (step_sa % 5000 == 0) & (step_sa != 0):
					torch.save(biLSTM_SA.state_dict(), f"{checkpoint_path}/biLSTM_SA.pth")
				
				step_sa += 1
			del outputs
			torch.cuda.empty_cache()

		start_train_epoch_loss = total_loss_start / step_start
		start_train_ppl = torch.exp(start_train_epoch_loss)
		stop_train_epoch_loss = total_loss_stop / step_stop
		stop_train_ppl = torch.exp(stop_train_epoch_loss)
		sd_train_epoch_loss = total_loss_sd / step_sd
		sd_train_ppl = torch.exp(sd_train_epoch_loss)
		sa_train_epoch_loss = total_loss_sa / step_sa
		sa_train_ppl = torch.exp(sa_train_epoch_loss)

		if eval:
			start_eval_loss = 0
			step_start = 0
			stop_eval_loss = 0
			step_stop = 0
			sd_eval_loss = 0
			step_sd = 0
			sa_eval_loss = 0
			step_sa = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				if 'Zm' in batch[3][0]: 
					continue
				ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
				with torch.no_grad():
					outputs = encoder_model(input_ids=batch[0].to(device))
				
				true = dt.batch_decode(lab.detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
				sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace(" ", "")
				pp = PredictionProcessor(true, sequence, None)
				
				for cds in pp.start_cds:
					start = int(pp.features[pp.feature_index_dict[cds]][0])
					randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
					if pp.strand == "-":
						start = len(sequence)-start
					if start-randomNumber < 0:
						randomNumber = torch.tensor([start])
					if start+(200-randomNumber) > len(sequence):
						randomNumber = torch.tensor([start-len(sequence)+200])
					with torch.no_grad():
						prediction = biLSTM_Start(outputs.last_hidden_state[:, start-randomNumber:start+(200-randomNumber)])
					start_eval_loss += loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device))
					step_start += 1
				
				for cds in pp.end_cds:
					stop = int(pp.features[pp.feature_index_dict[cds]][2])
					randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
					if pp.strand == "-":
						stop = len(sequence)-stop
					if start-randomNumber < 0:
						randomNumber = torch.tensor([start])
					if stop+(200-randomNumber) > len(sequence):
						randomNumber = torch.tensor([stop-len(sequence)+200])
					with torch.no_grad():
						prediction = biLSTM_Stop(outputs.last_hidden_state[:, start-randomNumber:start+(200-randomNumber)])
					stop_eval_loss += loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device))
					step_stop += 1
				
				# Splice donors
				for cds in pp.cds_pairs:
					sd = int(pp.features[pp.feature_index_dict[cds[0]]][2])
					randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
					if pp.strand == "-":
						sd = len(sequence)-sd
					if sd+(200-randomNumber) > len(sequence):
						randomNumber = torch.tensor([sd-len(sequence)+200])
					with torch.no_grad():
						prediction = biLSTM_SD(outputs.last_hidden_state[:, start-randomNumber:start+(200-randomNumber)])
					sd_eval_loss += loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device))
					step_sd += 1
				
				# Splice acceptors
				for cds in pp.cds_pairs:
					sa = int(pp.features[pp.feature_index_dict[cds[1]]][0])
					randomNumber = torch.clamp(torch.round(torch.normal(mean=torch.FloatTensor([100.0]), std=torch.FloatTensor([33.3]))), min=0, max=200).to(torch.int)
					if pp.strand == "-":
						sa = len(sequence)-sa
					if sa-randomNumber < 0:
						randomNumber = torch.tensor([start])
					if sa+(200-randomNumber) > len(sequence):
						randomNumber = torch.tensor([sa-len(sequence)+200])
					with torch.no_grad():
						prediction = biLSTM_SA(outputs.last_hidden_state[:, sa-randomNumber:sa+(200-randomNumber)])
					sa_eval_loss += loss_fn(prediction.squeeze(), randomNumber.squeeze().to(torch.float).to(device))
					step_sa += 1


			start_eval_epoch_loss = start_eval_loss / step_start
			start_eval_ppl = torch.exp(start_eval_epoch_loss)
			stop_eval_epoch_loss = stop_eval_loss / step_stop
			stop_eval_ppl = torch.exp(stop_eval_epoch_loss)
			sd_eval_epoch_loss = sd_eval_loss / step_sd
			sd_eval_ppl = torch.exp(sd_eval_epoch_loss)
			sa_eval_epoch_loss = sa_eval_loss / step_sa
			sa_eval_ppl = torch.exp(sa_eval_epoch_loss)
			#print(f"{epoch=}: {start_train_ppl=}, {start_train_epoch_loss=}, {start_eval_ppl=}, {start_eval_epoch_loss=}", file=sys.stderr)
			#if log_wandb:
			#	wandb_log = {
			#		"start_epoch_train_ppl":start_train_ppl, "start_epoch_train_loss":start_train_epoch_loss, "start_epoch_eval_ppl":start_eval_ppl, "start_epoch_eval_loss":start_eval_epoch_loss,
			#		"stop_epoch_train_ppl":stop_train_ppl, "stop_epoch_train_loss":stop_train_epoch_loss, "stop_epoch_eval_ppl":stop_eval_ppl, "stop_epoch_eval_loss":stop_eval_epoch_loss,
			#		"sd_epoch_train_ppl":sd_train_ppl, "sd_epoch_train_loss":sd_train_epoch_loss, "sd_epoch_eval_ppl":sd_eval_ppl, "sd_epoch_eval_loss":sd_eval_epoch_loss,
			#		"sa_epoch_train_ppl":sa_train_ppl, "sa_epoch_train_loss":sa_train_epoch_loss, "sa_epoch_eval_ppl":sa_eval_ppl, "sa_epoch_eval_loss":sa_eval_epoch_loss
			#		}
			#	wandb.log(wandb_log)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		

		save_model(biLSTM_Start, f"{checkpoint_path}/biLSTM_Start.safetensors")
		save_model(biLSTM_Stop, f"{checkpoint_path}/biLSTM_Stop.safetensors")
		save_model(biLSTM_SD, f"{checkpoint_path}/biLSTM_SD.safetensors")
		save_model(biLSTM_SA, f"{checkpoint_path}/biLSTM_SA.safetensors")

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
		num_epochs=15, 
		schedule_lr=True, 
		eval=True, 
		batch_size=1, 
		accumulation_steps=64,
		checkpoint_path="checkpoints_biLSTMRegressionHead/", 
		safetensors_model=None,
		output_dir="saved_models_biLSTMRegressionHead/",
		max_grad_norm=1,
		notes="Training with biLSTMRegressionHead",
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
		unlink = False,
		log_wandb=True
	)