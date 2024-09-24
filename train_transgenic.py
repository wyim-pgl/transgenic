#!/usr/bin/env python
import torch, os, wandb, gc
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from utils_transgenic import *
from safetensors.torch import save_model

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_HOME'] = './HFmodels'
#os.environ["NCCL_DEBUG"] = "INFO"

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
	unlink=False):

	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# start a new wandb run to track this script
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
	torch.manual_seed(345)
	torch.cuda.manual_seed_all(345)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	model = getModel(None, safetensors_model=safetensors_model, device="cpu", mode="train")
	model.to(device)
	model.train()
	
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
			ii, am, gam, lab = batch[0].to(device), batch[1].to(device), batch[2], batch[3].to(device)
			outputs = model(input_ids=ii, attention_mask=am, global_attention_mask=gam, labels=lab, return_dict=True)
			total_loss += outputs.loss.detach().float()
			outputs.loss = outputs.loss / accumulation_steps
			outputs.loss.backward()
			#accelerator.backward(outputs.loss / accumulation_steps)
			if (step+1) % accumulation_steps == 0:
				clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				if schedule_lr: lr_scheduler.step()
				# log metrics to wandb
				wandb_log = {"epoch":epoch, "step":step, "loss": outputs.loss.detach().float()*accumulation_steps, "mean_loss": total_loss / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
				for name, param in model.named_parameters():
					if (param.grad != None) & (param.requires_grad):
						grad_norm = param.grad.norm().detach().item()
						wandb_log[f"{name}_grad_norm"] = grad_norm
				wandb.log(wandb_log)
				optimizer.zero_grad()
			
			if (step % 1000 == 0) & (step != 0):
				print(f"Epoch {epoch}, Step {step}, Loss {outputs.loss.detach().float()*accumulation_steps}", file=sys.stderr)
				save_model(model, f"{checkpoint_path}/model.safetensors")
			del outputs
			torch.cuda.empty_cache()

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), global_attention_mask=batch[2], labels=batch[3].to(device), return_dict=True)
				eval_loss += outputs.loss.detach().float()

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
	ds = isoformData(db, dt, mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False)
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])

	trainTransgenicFCGAccelerate(
		train_data, 
		eval_data, 
		lr=1e-4, 
		num_epochs=10, 
		schedule_lr=True, 
		eval=True, 
		batch_size=1, 
		accumulation_steps=16,
		checkpoint_path="checkpoints_Gen10G/", 
		safetensors_model="checkpoints_Gen10G/model.safetensors",
		output_dir="saved_models_Gen10G/",
		max_grad_norm=1,
		notes="Continuing training with 3 more genomes and no PEFT",
		encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b",
		unlink = False
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
