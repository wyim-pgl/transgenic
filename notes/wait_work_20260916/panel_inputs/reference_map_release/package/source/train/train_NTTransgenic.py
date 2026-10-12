#!/usr/bin/env python
import torch, os, wandb, gc, sys, re
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from safetensors.torch import save_model, safe_open
from peft import IA3Config

from ..datasets.datasets import isoformData, makeDataLoader, target_collate_fn
from ..models.tokenization_transgenic import GFFTokenizer
from ..models.modeling_HeynaTransgenic import transgenicForConditionalGeneration
from ..models.configuration_transgenic import NTTransgenicConfig
from ..utils.huggingface_integration import get_peft_model

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_HOME'] = './HFmodels'
#os.environ["NCCL_DEBUG"] = "INFO"

def linear_decay(step, total_steps, start_value=0.5, end_value=0.0):
	if step >= total_steps:
		return end_value 
	
	decay_rate = (start_value - end_value) / total_steps
	return start_value - (decay_rate * step)

def split_tokens(predicted_tokens):
	split_ptokens = [predicted_tokens[0:predicted_tokens.tolist().index(17)].tolist(), predicted_tokens[predicted_tokens.tolist().index(17):].tolist()]
	split_list = []
	for seg in split_ptokens:
		seg_list = []
		semicolons = [i for i, x in enumerate(seg) if (x == 21) | (x==2)]
		prev = 0
		if semicolons:
			for i in semicolons:
				if i == semicolons[-1]:
					i += 1
				seg_list.append(seg[prev:i])
				prev = i
			split_list.append(seg_list)
		else:
			split_list.append([seg])
	return split_list

def listFlatten(l):
	flatList = []
	for item in l:
		if type(item[0]) == list:
			flatList += listFlatten(item)
		else:
			flatList += item
	return flatList


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
	encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
	unlink=False,
	generation_mixin=False,
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
			"architecture": "FCG",
			"dataset": "49kb_7genomes_200addExtra",
			"epochs": num_epochs,
			"max_grad_norm": max_grad_norm,
			"accumulation_steps": accumulation_steps,
			"Optimizer": "AdamW",
			"Checkpoints":checkpoint_path,
			"Outputs":output_dir,
			"Notes":notes
			}
		)

	print(f"Training transgenic with reinitialized decoder. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)
	
	# Set up accelerator
	#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	#accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # gradient_accumulation_steps=32
	#device = accelerator.device
	device = torch.device("cuda")
	print(f"Training transgenic with Accelerate on {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(123)
	torch.cuda.manual_seed_all(123)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	#model = getModel(None, safetensors_model=safetensors_model, device="cpu", mode="train")
	decoder_checkpoint = "checkpoints/transgenic_Gen10G_6144nt_E4.safetensors"
	segment_checkpoint = "checkpoints_SegmentNT/model.safetensors"

	config = NTTransgenicConfig(do_segment=True)
	model = transgenicForConditionalGeneration(config)

	# Load decoder checkpoint
	decoder_tensors = {}
	with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			decoder_tensors[k] = f.get_tensor(k)
	decoder_tensors["transgenic.decoder_embed_tokens.weight"] = decoder_tensors["lm_head.weight"]
	decoder_tensors["transgenic.decoder.embed_tokens.weight"] = decoder_tensors["transgenic.decoder_embed_tokens.weight"]
	newDecoder_tensors = {}
	for k in decoder_tensors:
		if "ia3" not in k:
			newDecoder_tensors[k] = decoder_tensors[k]
		#if "final_logits_bias" in k:
		#	del newDecoder_tensors[k]
		#if "transgenic.decoder_embed_tokens.weight" in k:
		#	del newDecoder_tensors[k]
		#if "transgenic.decoder.embed_tokens.weight" in k:
		#	del newDecoder_tensors[k]
		#if "lm_head.weight" in k:
		#	del newDecoder_tensors[k]

	# Load segmentation checkpoint
	segment_tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			segment_tensors[k] = f.get_tensor(k)
	segment_tensors = {k.replace("unet", "transgenic.encoder.unet").replace("uFC", "transgenic.encoder.uFC"):segment_tensors[k] for k in segment_tensors}
	newSegment_tensors = {}
	for k in segment_tensors:
		if ("esm" not in k) & ("hidden" not in k) & ("lm_head" not in k):
			newSegment_tensors[k] = segment_tensors[k]

	# Merge dictionaries and load
	tensors = {**newDecoder_tensors, **newSegment_tensors}
	model.load_state_dict(tensors, strict=False)

	################ PEFT ####################
	target_modules = [
		r".*esm.encoder.layer.*.attention.self.query",
		r".*esm.encoder.layer.*.attention.self.key",
		r".*esm.encoder.layer.*.attention.self.value",
		r".*esm.encoder.layer.*.attention.output.dense",
		r".*esm.encoder.layer.*.intermediate.dense",
		r".*esm.encoder.layer.*.output.dense"]

	# Recompute activations for the dense layers
	feedforward_modules = [
		r".*esm.encoder.layer.*.attention.output.dense",
		r".*esm.encoder.layer.*.intermediate.dense",
		r".*esm.encoder.layer.*.output.dense"]

	# Find the target and feedforward modules
	peft_targets = []
	peft_feedforward = []
	for module in model.named_modules():
		for pattern in target_modules:
			if re.match(pattern, module[0]):
				peft_targets.append(module[0])
		for pattern in feedforward_modules:
			if re.match(pattern, module[0]):
				peft_feedforward.append(module[0])

	# Load the IA3 adaptor
	peft_config = IA3Config(task_type="SEQ_2_SEQ_LM", target_modules = peft_targets, feedforward_modules = peft_feedforward, init_ia3_weights = True)
	model = get_peft_model(model, peft_config)
	####################################

	model.to(device)
	model.train()
	#for param in model.transgenic.encoder.esm.parameters():
	#	param.requires_grad = False
	for param in model.base_model.transgenic.decoder.parameters():
		param.requires_grad = True
	for param in model.base_model.transgenic.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.base_model.transgenic.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	for param in model.base_model.transgenic.encoder.uFC.parameters():
		param.requires_grad = False
	for param in model.base_model.transgenic.encoder.unet.parameters():
		param.requires_grad = False
	for param in model.base_model.transgenic.encoder.unet_mapping.parameters():
		param.requires_grad = False
	for param in model.base_model.transgenic.encoder.unet_mapping_layernorm.parameters():
		param.requires_grad = False

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
	#scaler = GradScaler("cuda")
	# Prep objects for use with accelerator
	#model, optimizer, train_ds, eval_ds, lr_scheduler = accelerator.prepare(
	#	model, optimizer, train_ds, eval_ds, lr_scheduler
	#)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			if 'Zm0' in batch[4][0]: 
				continue
			#if step == 14:
			#	pass
			try:
				ii, am, gam, lab = batch[0].to(device), batch[1].to(device), batch[2], batch[3].to(device)
				dii = None
				if generation_mixin:
					try:
						p = linear_decay(step + len(train_ds)*epoch, len(train_ds)*num_epochs, start_value=0.0, end_value=0.5)
						model.eval()
						with torch.no_grad():
							# Generate output sequentially
							predicted_tokens = model.generate(inputs=ii, attention_mask=am, num_return_sequences=1, max_length=2048, num_beams=4,do_sample=True)[0,1:]
							
						# Split the output and target tokens into coherent chunks
						split_ptokens = split_tokens(predicted_tokens)
						split_ltokens = split_tokens(lab[0])

						# Replace target chunks with prediction chunks with probability 'p' if they are equal lengths
						if len(split_ptokens[0]) < len(split_ltokens[0]):
							split_ptokens[0][-1] = split_ptokens[0][-1][:-1] #Remove extra '21' added from splitting
						elif len(split_ptokens[0]) > len(split_ltokens[0]):
							split_ptokens[0][len(split_ltokens[0])-1] = split_ptokens[0][-1] + [21]
						feature_len = min(len(split_ptokens[0]), len(split_ltokens[0]))
						feature_mask = torch.bernoulli(torch.full((feature_len,), p)).tolist()
						for i, replace in enumerate(feature_mask):
							if replace:
								if len(split_ltokens[0][i]) == len(split_ptokens[0][i]):
									split_ltokens[0][i] = split_ptokens[0][i]
								
						# For now only substituting features, this keeps the lengths identical for loss calculation...
						#transcript_len = min(len(split_ptokens[1]), len(split_ltokens[1]))
						#transcript_mask = torch.bernoulli(torch.full((transcript_len,), p)).tolist()
						#for i, replace in enumerate(transcript_mask):
						#	if replace:
						#		split_ltokens[1][i] = split_ptokens[1][i]

						# Stitch tokens back together for input to the training step
						dii = torch.tensor(listFlatten(split_ltokens)).unsqueeze(0).to(device)
						model.train()
						for param in model.transgenic.encoder.esm.parameters():
							param.requires_grad = False
						for param in model.transgenic.decoder.parameters():
							param.requires_grad = True
						for param in model.transgenic.encoder.hidden_mapping.parameters():
							param.requires_grad = True
						for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
							param.requires_grad = True
					except:
						print("Failed to align generation output and prediction.", file=sys.stdout)
						dii = None

				#with autocast("cuda", dtype=torch.float16):
				outputs = model(input_ids=ii, attention_mask=am, global_attention_mask=gam, decoder_input_ids=dii, labels=lab, return_dict=True)
				total_loss += outputs.loss.detach().float()
				outputs.loss = outputs.loss / accumulation_steps
				
				outputs.loss.backward()
				#scaler.scale(outputs.loss).backward()
				
				
				
				if (step+1) % accumulation_steps == 0:
					clip_grad_norm_(model.parameters(), max_grad_norm)
					optimizer.step()
					#scaler.step(optimizer)
					if schedule_lr: lr_scheduler.step()
					# log metrics to wandb
					if log_wandb:
						wandb_log = {"epoch":epoch, "step":step, "loss": outputs.loss.detach().float()*accumulation_steps, "mean_loss": (total_loss) / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
						for name, param in model.named_parameters():
							if (param.grad != None) & (param.requires_grad):
								grad_norm = param.grad.norm().detach().item()
								wandb_log[f"{name}_grad_norm"] = grad_norm
						wandb.log(wandb_log)
					#scaler.update()
					optimizer.zero_grad()
				
				if (step % 5000 == 0) & (step != 0):
					print(f"Epoch {epoch}, Step {step}, Loss {outputs.loss.detach().float()*accumulation_steps}", file=sys.stderr)
					save_model(model, f"{checkpoint_path}/model.safetensors")
				del outputs
				torch.cuda.empty_cache()
			except Exception as e:
				print(f"WARNING: Failed on {batch}")

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			#model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), global_attention_mask=batch[2], labels=batch[3].to(device), return_dict=True)
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
		#loss = None
		train_epoch_loss = 0
		train_ppl = 0
	
	if log_wandb:
		wandb.finish()


if __name__ == '__main__':
	torch.manual_seed(123)

	#InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
	#InstaDeepAI/agro-nucleotide-transformer-1b
	#encoder_model = sys.argv[1]
	#unlink = bool(sys.argv[2])
	#notes = sys.argv[3]

	#with open("Train-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
	#	train_data = pickle.load(f)
	#with open("Eval-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
	#	eval_data = pickle.load(f)
	#with open("Test-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
	#	test_data = pickle.load(f)

	db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
	dt = GFFTokenizer()
	ds = isoformData(db, dt, mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False, shuffle=False)
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])

	trainTransgenicFCGAccelerate(
		train_data, 
		eval_data, 
		lr=5e-5, 
		num_epochs=5, 
		schedule_lr=True, 
		eval=True, 
		batch_size=1, 
		accumulation_steps=128,
		checkpoint_path="checkpoints_Gen9G_Large/", 
		safetensors_model="checkpoints/transgenic_Gen10G_6144nt_E4.safetensors",
		output_dir="saved_models_Gen9G_Large/",
		max_grad_norm=1,
		notes="Training with peft and 1 digit tokenizer",
		encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b",
		unlink = False,
		generation_mixin=False,
		log_wandb=False
	)

#	Keeping the training, validation, and testing datasets consistent after updating with reverse complemented isoforms
#	createDatabase(db="Flagship_6Genomes_49k_extra200_addRCIsoOnly.db", mode="train", maxLen=49152, addExtra=200, addRC=True, addRCIsoOnly=True, clean=True)
#	db = "Flagship_Genomes_49k_extra200_clean.db"
#	ds = isoformData(db, dt=GFFTokenizer(), mode="training", encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b")
#	train_data, eval_data, test_data = random_split(ds, [171071, 24438, 48879])
#	train_data_gms = []
#	eval_data_gms = []
#	test_data_gms = []
#	with duckdb.connect(db, config={"access_mode": "READ_ONLY"}) as con:
#		for i in train_data.indices:
#			train_data_gms.append(con.sql(f"SELECT geneModel FROM geneList WHERE rn={i+1}").fetchall()[0][0])
#		for i in eval_data.indices:
#			eval_data_gms.append(con.sql(f"SELECT geneModel FROM geneList WHERE rn={i+1}").fetchall()[0][0])
#		for i in test_data.indices:
#			test_data_gms.append(con.sql(f"SELECT geneModel FROM geneList WHERE rn={i+1}").fetchall()[0][0])
#
#	train_data_gms = pd.DataFrame(train_data_gms, columns=["geneModel"])
#	eval_data_gms = pd.DataFrame(eval_data_gms, columns=["geneModel"])
#	test_data_gms = pd.DataFrame(test_data_gms, columns=["geneModel"])
#
#	db = "Flagship_6Genomes_49k_extra200_addRCIsoOnly.db"
#	ds = isoformData(db, dt=GFFTokenizer(), mode="training", encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b")
#	with duckdb.connect(db, config={"access_mode": "READ_ONLY"}) as con:
#		train_data_index = con.sql('SELECT rn FROM geneList INNER JOIN train_data_gms ON geneList.geneModel = train_data_gms.geneModel').fetchall()
#		eval_data_index = con.sql('SELECT rn FROM geneList INNER JOIN eval_data_gms ON geneList.geneModel = eval_data_gms.geneModel').fetchall()
#		test_data_index = con.sql('SELECT rn FROM geneList INNER JOIN test_data_gms ON geneList.geneModel = test_data_gms.geneModel').fetchall()
#	train_data_index = [i[0]-1 for i in train_data_index]
#	eval_data_index = [i[0]-1 for i in eval_data_index]
#	test_data_index = [i[0]-1 for i in test_data_index]
#
#	old_set = set(train_data_index + eval_data_index + test_data_index)
#	full_set = set(range(len(ds)))
#	added_set = list(full_set - old_set)
#
#	train_data = torch.utils.data.Subset(ds, train_data_index)
#	eval_data = torch.utils.data.Subset(ds, eval_data_index)
#	test_data = torch.utils.data.Subset(ds, test_data_index)
#	add_data = torch.utils.data.Subset(ds, added_set)
#	add_train, add_eval, add_test = random_split(add_data, [36333, 4844, 7267])
#
#	train_data.indices = train_data_index + add_train.indices
#	eval_data.indices = eval_data_index + add_eval.indices
#	test_data.indices = test_data_index + add_test.indices
#
#	with open("Train-Flagship_6Genomes_49k_extra200_addRCIsoOnly.pkl", 'wb') as out:
#		pickle.dump(train_data, out)
#	with open("Eval-Flagship_6Genomes_49k_extra200_addRCIsoOnly.pkl", 'wb') as out:
#		pickle.dump(eval_data, out)
#	with open("Test-Flagship_6Genomes_49k_extra200_addRCIsoOnly.pkl", 'wb') as out:
#		pickle.dump(test_data, out)
#	sys.exit()
