import duckdb, sys, os, subprocess, re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import LEDForConditionalGeneration, LEDTokenizer, LEDConfig
from transformers import get_linear_schedule_with_warmup
from peft import IA3Config, get_peft_model
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import pickle

def print_gpu_allocation(s:str):
	#print(f"Allocated: {torch.cuda.memory_allocated()}, Cached: {torch.cuda.memory_reserved()}...{s}", file=sys.stderr)
	print(s, file=sys.stderr)
	result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
	print(result.stdout.decode('utf-8'), file=sys.stderr)

def loadGenome(genome):
	# Load genome into a dictionary
	# Return the dictionary
	genome_dict = {}
	with open(genome, 'r') as genomefile:
		lines = genomefile.readlines()
		header = None
		sequence_lines = []
		for line in lines:
			if line.startswith('>'):
				if header is not None:
					genome_dict[header] = ''.join(sequence_lines)
					sequence_lines = []
				header = line.strip().split(' ')[0][1:]
			else:
				sequence_lines.append(line.strip())
		if header is not None:
			genome_dict[header] = ''.join(sequence_lines)
	return genome_dict

def reverseComplement(sequence):
	# Reverse complement a sequence
	# Return the reverse complement
	sequence = sequence.upper()
	sequence = sequence.replace('A', 't')
	sequence = sequence.replace('T', 'a')
	sequence = sequence.replace('C', 'g')
	sequence = sequence.replace('G', 'c')
	sequence = sequence.upper()
	sequence = sequence[::-1]
	return sequence
	# Prepare raw genomic data for data loading 

def genome2GeneList(genome, gff3, db):
	# read in the genome and gff3 file
	# parse the gff3 file to get the gene models
	# for each gene model, create a list of isoforms and extract the gene region sequence (sense strand)
	# Store everything in a database
	# Call the function multiple times to append a new genome to the database

	# Load genome into a dictionary
	genome_dict = loadGenome(genome)

	# Connect to isoform database and create geneList table
	con = duckdb.connect(db)
	con.sql(
		"CREATE TABLE IF NOT EXISTS geneList ("
			"geneModel VARCHAR, "
			"start INT, "
			"fin INT, "
			"strand VARCHAR, "
			"chromosome VARCHAR, "
			"sequence VARCHAR, "
			"gff VARCHAR)")

	
	print(f"\nProcessing {gff3}...")
	num_lines = sum(1 for line in open(gff3, "r"))
	with open(gff3, 'r') as gff3file:
		region_start = None
		gff = ''
		skipGene = False
		
		for line in tqdm(gff3file, total=num_lines):
			if line.startswith('#') | (line == '\n'):
				continue
			else:
				line = line.strip().split('\t')
				chr, _, typ, start, fin, _, strand, phase, attributes = line
				if typ == 'gene':
					if (region_start != None) & (not skipGene):
						# Add previous gene model to the database
						con.sql(f"INSERT INTO geneList (geneModel, start, fin, strand, chromosome, sequence, gff) VALUES ('{geneModel}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence}', '{gff[:-1]}')")
						geneModel = None
						region_start = 0
						region_end = 0
						gff = ''
					
					# Construct current gene model
					skipGene = False
					geneModel = attributes.split(';')[0].split('=')[1]
					region_start = int(start)-1 # Gffs are 1-indexed
					region_end = int(fin)       # End not subtracted because python slicing is exclusive
					
					if region_end - region_start > 24552:
						print(f"Skipping {geneModel} because gene length > 24,552bp", file=sys.stderr)
						region_start = None
						region_end = None
						geneModel = None
						skipGene = True
						continue

					# Get sense strand sequence
					sequence = genome_dict[chr][region_start:region_end]
					if strand == '-':
						sequence = reverseComplement(sequence)
				
				elif skipGene:
					continue
				
				# Build gff string - start and end coordinates are relative to the gene model sense strand
				else: 
					start = (int(start) - 1) - region_start 
					end = int(fin) - region_start
					gff += f"{typ}|{start}|{end}|{strand}|{phase};"

	con.close()


def segmentSequence(seq, piece_size = 4092):
	# Segment the sequence into evenly sized chunks smaller than 6000bp (encoder max length of 1000 tokens)
	seqs = [seq[i:min(i+piece_size, len(seq))] for i in range(0, len(seq), piece_size)]
	return seqs


def setup(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group("nccl", rank=rank, world_size=world_size) # initialize the process group

def cleanup():
	if dist.is_initialized():
		dist.destroy_process_group()

def target_collate_fn(batch):
	# Unpack the batch items (each item in batch is a tuple of (sequence, attention_mask, target_sequence))
	sequences, attention_masks, labels = zip(*batch)

	# Stack the sequences and attention masks
	sequences = torch.stack(sequences)
	attention_masks = torch.stack(attention_masks)

	# Pad the labels to the same length and stack
	if labels:
		labels = [label.reshape([label.shape[1]]) for label in labels]
		labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

	return sequences, attention_masks, labels_padded

def makeDataLoader(dat, shuffle=True, batch_size=8, pin_memory=True, sampler=None):
	if sampler != None:
		shuffle = False
	
	return DataLoader(
		dat, 
		shuffle=shuffle, 
		collate_fn=target_collate_fn, 
		batch_size=batch_size, 
		pin_memory=pin_memory,
		sampler=sampler)

# geneList custom dataset class for use with DataLoader
class isoformData(Dataset):
	def __init__(self, db, mode="inference"):
		self.db = db
		self.mode = mode
		self.decoder_tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384", cache_dir="./HFmodels")
		self.encoder_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="./HFmodels", trust_remote_code=True)

	def __len__(self):
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM geneList").fetchall()[0][0]
	
	def __getitem__(self, idx):
		idx += 1
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			_,_,_,_,_,region_seq, gff = con.sql("WITH rn (geneModel,start, fin, strand, chromosome, sequence, gff, rnum) AS ("
								"SELECT *, row_number() OVER() FROM geneList) "
								f"SELECT * from rn where rnum={idx}").fetchall()[0][:-1]
		
		# Tokenize output targets
		if self.mode == 'inference':
			labels = self.decoder_tokenizer.batch_encode_plus( #TODO: will this work?
				"",
				return_tensors="pt",
				padding=True,
				truncation=True)["input_ids"]
		elif self.mode == "training":
			# Tokenize the labels
			labels = self.decoder_tokenizer.batch_encode_plus(
				[gff],
				return_tensors="pt",
				padding=True,
				truncation=True)["input_ids"]

		# Segment and tokenize the sequences
		seqs = segmentSequence(region_seq, piece_size=6000)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 682)["input_ids"]

		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)
	
		return (seqs, encoder_attention_mask, labels)

class gffTokenizer:
	def __init__(self):
		self.vocab = {"[PAD]": 0, "[UNK]": 1, "mRNA":2, "exon":3, "CDS":4, "five_prime_UTR":5, "three_prime_UTR":6,
				"1":7, "2":8, "3":9, "4":10, "5":11, "6":12, "7":13, "8":14, "9":15, "0":16, ".":17, "+":18, "-":19, 
				"|":20, ";":21}
		
	def encode(self, gff:str):
		tokens = []
		for model in gff.split(";"):
			for column in gff.split("|"):
				if re.search(r'^\d+$', column):
					for digit in column:
						tokens.append(self.vocab[digit])
				else:
					tokens.append(self.vocab[column])
		return torch.FloatTensor(tokens)
	
	def decode(self, tokens):
		gff = ""
		for token in tokens:
			for key, value in self.vocab.items():
				if token == value:
					gff += key
					break
		return gff
	
# Full generative model definition
class transgenic(LEDForConditionalGeneration):
	def __init__(self):
		self.cache_dir = "./HFmodels"
		self.decoder_model = "allenai/led-base-16384"
		self.encoder_model = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
		super().__init__(AutoConfig.from_pretrained(self.decoder_model))
		

		# Load the pre-trained encoder
		self.encoder = AutoModelForMaskedLM.from_pretrained(self.encoder_model, cache_dir=self.cache_dir, trust_remote_code=True)
		for param in self.encoder.parameters():
			param.requires_grad = False

		# Load the pre-trained decoder and freeze the parameters
		self.decoder = LEDForConditionalGeneration.from_pretrained(self.decoder_model, cache_dir=self.cache_dir)
		for param in self.decoder.parameters():
			param.requires_grad = False

		# Load the IA3 adaptor
		self.peft_config = IA3Config(
			task_type="SEQ_2_SEQ_LM", 
			target_modules = [
				"decoder.layers.*.self_attn.*",  # Targets all self-attention components
				"decoder.layers.*.fc1",  # Targets the first feedforward linear layer
				"decoder.layers.*.fc2",  # Targets the second feedforward linear layer
				],

			feedforward_modules = [
				"decoder.layers.*.fc1",  # Recompute activations for the first feedforward layer
				"decoder.layers.*.fc2",  # Recompute activations for the second feedforward layer
				])
		self.decoder = get_peft_model(self.decoder, self.peft_config)
		self.trainable_parameters = self.decoder.print_trainable_parameters()

		# TODO: Exlpore other options? (hidden states, BiLSTM, linear, attention, pooling, convolution)
		#plants -> 1500, multispecies -> 1024
		#T5 -> 512, longformer -> 768
		self.hidden_mapping = nn.Linear(1024, 768)

	def forward(self, seqs, encoder_attention_mask, labels, batch_size): # target_ids,
		# Compute the embeddings with nucleotide transformer encoder
		for i in range(batch_size):
			with torch.no_grad():
				embeddings = self.encoder(
					seqs[i, :, :],
					attention_mask=encoder_attention_mask[i,:,:],
					encoder_attention_mask=encoder_attention_mask[i,:,:],
					output_hidden_states=True
				)['hidden_states'][-1]
			if i == 0:
				batch_embeds = embeddings
			elif i == 1:
				batch_embeds = torch.stack((batch_embeds, embeddings), dim=0)
			else:
				batch_embeds = torch.cat((batch_embeds, embeddings.unsqueeze(0)), dim=0)
		
		# Combine last hidden state embeddings from multiple segments into a single sequence
		embeddings = batch_embeds.reshape(batch_size, 6*681, -1)					# Shape: (batch_size, seq_length, hidden_size)
		encoder_attention_mask = encoder_attention_mask.reshape(batch_size, 6*681)	# Shape: (batch_size, seq_length)
		
		# Use the last hidden state from the nucleotide encoder as input to the decoder
		# Transform the encoder hidden states to match the decoder input size
		decoder_inputs_embeds = self.hidden_mapping(embeddings)
		
		# Process the transformed encoder outputs through the decoder
		decoder_outputs = self.decoder(inputs_embeds=decoder_inputs_embeds, 
								 #decoder_input_ids=target_ids, 
								 attention_mask=encoder_attention_mask,
								 labels=labels
								 )
		
		return decoder_outputs

# Training loop
def trainTransgenicDDP(rank, 
		train_ds:isoformData, 
		eval_ds:isoformData, 
		lr, 
		num_epochs, 
		batch_size, 
		schedule_lr, 
		eval, 
		world_size,
		checkpoint_path="transgenic_checkpoint.pt"):
	# Set up GPU process group
	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
	print(f"Training transgenic on {device}, (world_size={world_size})", file=sys.stderr)
	setup(rank, world_size)

	# decoder_tokenizer = train_ds.decoder_tokenizer

	# Distribute data
	train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
	eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, sampler=eval_sampler)

	# Load the model and wrap in DDP
	model = transgenic().to(device)
	ddp_model = DDP(model, device_ids=[rank])
	
	# Define the loss function and optimizer
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_ds) * num_epochs),
		)

	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds)):
			batch = [item.to(device) for item in batch]
			outputs = ddp_model(batch[0], batch[1], batch[2], batch_size)
			loss = outputs.loss
			total_loss += loss.detach().float()
			loss.backward()
			optimizer.step()
			if schedule_lr: lr_scheduler.step()
			optimizer.zero_grad()

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for step, batch in enumerate(tqdm(eval_ds)):
				batch = [item.to(device) for item in batch]
				with torch.no_grad():
					outputs = ddp_model(batch[0], batch[1], batch[2], batch_size)
				loss = outputs.loss
				eval_loss += loss.detach().float()

			eval_epoch_loss = eval_loss / len(eval_ds)
			eval_ppl = torch.exp(eval_epoch_loss)
			print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		
		if rank == 0:
			checkpoint = {
						'epoch': epoch,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'lr_scheduler_state_dict': lr_scheduler.state_dict() if schedule_lr else None,
						'train_loss': train_epoch_loss,
						'eval_loss': eval_epoch_loss
					}
			if eval:
				if best_eval_score is None or eval_epoch_loss < best_eval_score:
					best_eval_score = eval_ppl
					if not os.path.exists("checkpoints"):
						os.makedirs("checkpoints", exist_ok=True)
					torch.save(checkpoint, f"checkpoints/{checkpoint_path}")
					print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
			else:
				if best_eval_score is None or train_epoch_loss < best_eval_score:
					best_eval_score = train_ppl
					if not os.path.exists("checkpoints"):
						os.makedirs("checkpoints", exist_ok=True)
					torch.save(checkpoint, f"checkpoints/{checkpoint_path}")
					print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)

def run_trainTransgenicDDP(train_ds, eval_ds=None, lr=8e-3, num_epochs=10, batch_size=8, schedule_lr=True, eval=False, checkpoint_path="transgenic_checkpoint.pt"):
	world_size = torch.cuda.device_count()
	mp.spawn(trainTransgenicDDP, 
		args=(train_ds, eval_ds, lr, num_epochs, batch_size, schedule_lr, eval, world_size), 
		nprocs=world_size, 
		join=True)
	cleanup()

# Prediction loop
def predictTransgenicDDP(rank, checkpoint:str, dataset:isoformData, outfile, split, batch_size, world_size):
	# Set up GPU process group
	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
	print(f"Running transgenic in prediction mode on {device}, (world_size={world_size})", file=sys.stderr)
	setup(rank, world_size)

	# configure map_location
	map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
	
	# Load the model and wrap in DDP
	
	model = transgenic().to(device)
	model.load_state_dict(torch.load(checkpoint, map_location=map_location)['model_state_dict'])
	ddp_model = DDP(model, device_ids=[rank])
	ddp_model.eval()
	
	# Create a DataLoader
	# Distribute data
	sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
	loader = makeDataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True, sampler=sampler)
	
	# Prediction loop
	predictions = []
	for step, batch in enumerate(tqdm(loader)):
		batch = [item.to(device) for item in batch]
		with torch.no_grad():
			outputs = ddp_model(batch[0], batch[1], batch[2], batch_size)
			ppl = torch.exp(outputs.loss).cpu().numpy()
			loss = outputs.loss.cpu().numpy()
			pred = dataset.dataset.decoder_tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
			true = dataset.dataset.decoder_tokenizer.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True)
		
		predictions.append([str(true[0]), str(pred[0]), str(ppl), str(loss)])
	
	for prediction in predictions:
		with open("predictions.pkl", 'wb') as out:
			pickle.dump(prediction, out)
		with open(outfile, 'a') as out:
			out.write("\t".join(prediction) + f"\t{split}\n")
	

def run_predictTransgenicDDP(checkpoint:str, dataset:isoformData, outfile, split, batch_size):
	world_size = torch.cuda.device_count()
	mp.spawn(predictTransgenicDDP, 
		args=(checkpoint, dataset, outfile, split, batch_size, world_size), 
		nprocs=world_size, 
		join=True)
	cleanup()

def k_fold_split(dataset:isoformData, k=10):
	fold_size = len(dataset) // k
	indices = torch.randperm(len(dataset)).tolist()

	for i in range(k):
		val_indices = indices[i*fold_size : (i+1)*fold_size]
		train_indices = indices[:i*fold_size] + indices[(i+1)*fold_size:]
		train_subset = Subset(dataset, train_indices)
		val_subset = Subset(dataset, val_indices)
		yield train_subset, val_subset

# 10-fold cross validation
def crossValidateTransgenic(dataset:isoformData, checkpoint_path, outfile, lr=8e-3, num_epochs=10, batch_size=1, schedule_lr=True):
	# Need to measure final loss, perplexity, and output the generated sequences for each of the 10 splits
	for fold, (train_subset, val_subset) in enumerate(k_fold_split(dataset, k=10)):
		train_subset.dataset.mode = 'training'
		run_trainTransgenicDDP(
			train_subset, 
			val_subset, 
			lr = 8e-3, 
			num_epochs=10, 
			batch_size=1, 
			schedule_lr=False,
			eval=False,
			checkpoint_path=checkpoint_path)

		# Generate predictions to evaluate trained model
		val_subset.dataset.mode = 'inference'
		run_predictTransgenicDDP(
			checkpoint_path, 
			val_subset, 
			outfile=outfile, 
			split=fold, 
			batch_size=1)


if __name__ == '__main__':
	fasta = "ATH_Chr4.fas"
	gff = "Athaliana_167_gene_Chr4.gff3"
	db = "AthChr4.db"
	
	#genome2GeneList(fasta, gff, db="AthChr4.db")
	
	# Create a training and evaluation DataLoaders
	ds = isoformData(db, mode="inference")
	#one = ds.__getitem__(0)
	
	batch_size = 1
	train_data, eval_data = random_split(ds, [4087, 40])

	model = transgenic()
	checkpoint = {'model_state_dict': model.state_dict()}
	torch.save(checkpoint, "transgenic.pt")
	#train_dataloader = makeDataLoader(train_data, shuffle=True, batch_size=batch_size, pin_memory=True)
	#eval_dataloader = makeDataLoader(eval_data, shuffle=True, batch_size=batch_size, pin_memory=True)

	#run_trainTransgenicDDP( 
	#	train_data, 
	#	eval_ds=eval_data, 
	#	lr=8e-3, 
	#	num_epochs=1, 
	#	batch_size=batch_size, 
	#	schedule_lr=True, 
	#	eval=True
	#)
	run_predictTransgenicDDP("transgenic.pt", eval_data, "predictions.txt", 0, batch_size)