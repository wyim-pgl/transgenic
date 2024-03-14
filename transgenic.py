import duckdb, sys, os, subprocess
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
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

def genome2Isoformlist(genome, gff3, db, split=False):
	# read in the genome and gff3 file
	# parse the gff3 file to get the gene models
	# for each gene model, create a list of isoforms and extract the gene region sequence (sense strand)
	# Store everything in a database
	# Call the function multiple times to append a new genome to the database

	# Load genome into a dictionary
	genome_dict = loadGenome(genome)

	# Connect to isoform database and create isoformList table
	con = duckdb.connect(db)
	con.sql(
		"CREATE TABLE IF NOT EXISTS isoformList ("
			"geneModel VARCHAR, "
			"isoforms VARCHAR, "
			"cds VARCHAR, "
			"start INT, "
			"fin INT, "
			"strand VARCHAR, "
			"chromosome VARCHAR, "
			"sequence VARCHAR, "
			"split INT)")

	
	print(f"\nProcessing {gff3}...")
	num_lines = sum(1 for line in open(gff3, "r"))
	with open(gff3, 'r') as gff3file:
		region_start = 0
		isoforms = ''
		cds = ''
		cds_list = []
		
		for line in tqdm(gff3file, total=num_lines):
			if line.startswith('#') | (line == '\n'):
				continue
			else:
				line = line.strip().split('\t')
				if line[2] == 'gene':
					if region_start != 0:
						# Add previous gene model to the database
						isoforms = ";".join(set(isoforms.split(";"))) # Locations exist where the isoform only differs in UTR sequence (ignoring UTR isoforms for now)
						con.sql(f"INSERT INTO isoformList (geneModel, isoforms, cds, start, fin, strand, chromosome, sequence) VALUES ('{geneModel}', '{isoforms}', '{cds}', {region_start}, {region_end}, '{strand}', '{chromosome}', '{sequence}')")
						geneModel = None
						region_start = 0
						region_end = 0
						strand = None
						chromosome = None
						isoforms = ''
						cds = ''
						cds_list = []
					
					# Construct current gene model
					geneModel = line[8].split(';')[0].split('=')[1]
					strand = line[6]
					chromosome = line[0]
					region_start = int(line[3])-1 # Gffs are 1-indexed
					region_end = int(line[4])     # End not subtracted because python slicing is exclusive
					
					# Get sense strand sequence
					sequence = genome_dict[chromosome][region_start:region_end]
					if strand == '-':
						sequence = reverseComplement(sequence)

					
					
				# Build isoform string - start and end coordinates are relative to the gene model sense strand
				elif line[2] == 'mRNA' and isoforms != "":
					isoforms += ";"
				elif line[2] == 'CDS':
					if strand == '+':
						start = (int(line[3]) - 1) - region_start 
						end = int(line[4]) - region_start
					else:
						start = region_end - int(line[4])
						end = region_end - (int(line[3]) - 1)
					
					if (isoforms != "") and not isoforms.endswith(';'):
						isoforms += f"|"
					isoforms += f"{start},{end}"

					#TODO need to expand overlapping CDS regions
					if f"{start}_{end}" not in cds_list:
						cds_list.append(f"{start}_{end}")
						if cds != "":
							cds += "|"
						cds += f"{start},{end}"
	if split:
		# Randomly assign a number 1-10 to the split column for cross validation  
		con.sql("UPDATE isoformList SET split = floor(random() * 10_ + 1)")
	con.close()

def setIsoformAccess(db, mode:str):
	# mode: AUTOMATIC, READ_ONLY, READ_WRITE
	with duckdb.connect(db) as con: 
		con.sql(f"SET access_mode='{mode}'")

def reMapIsoforms(isoforms, coords, keepIntrons=False):
	isoforms = isoforms.split(';')
	isoforms = [x.split('|') for x in isoforms]
	isoforms = [[list(map(int, y.split(','))) for y in x] for x in isoforms]

	# Remove the length corresponding to the 5' or 3' UTR
	removed_seq_length = min([min(x) for x in coords])

	out = ''
	for iso in isoforms:
		if out != "": out += ";"
		for exon in iso:
			# Find the start, end, and index of the CDS corresponding to the exon
			i = 0
			for cds_start, _ in coords:
				i += 1
				if cds_start >= exon[0]:
					break
			
			if not keepIntrons:
				# Remove the length of the cumulative introduced gaps
				removed_seq_length = sum([coords[j+1][0] - coords[j][1] for j in range(0, i-1)])

			if out != "" or not out.endswith(';'):
				out += "|"
			
			out += f"{exon[0]-removed_seq_length},"
			out += f"{exon[1]-removed_seq_length}"
	return out

def segmentSequence(seq):
	# Segment the sequence into 6 evenly sized chunks smaller than 4086bp
	# This allows a max sequence length of 24kb (24,516bp) and a encoder max length of 681 tokens
	# Longer sequences are truncated
	if len(seq) > (4086*6):
		print(f"WARNING: Sequence ({len(seq)}bp) was longer than 24,516bp  and was truncated")
		seq = seq[0:(4086*6)]
	
	piece_size = len(seq)//6 + len(seq)%6
	seqs = [seq[i:min(i+piece_size, len(seq))] for i in range(0, len(seq), piece_size)]
	
	# Split the sequence into 6 roughly equal segments
	base_size = len(seq) // 6 
	extras = len(seq) % 6 
	seqs = []
	for i in range(6):
		if i < extras:
			slice_ = seq[i*(base_size+1):(i+1)*(base_size+1)]
		else:
			slice_ = seq[extras*(base_size+1) + (i-extras)*base_size : extras*(base_size+1) + (i-extras+1)*base_size]
		seqs.append(slice_)

	return seqs

def encodeSeqs(seqs, out, models:list, mode='inference', device="cpu"):
	encoder_tokenizer, encoder, decoder_tokenizer = models
	
	# Tokenize output targets
	if mode == 'inference':
		out = decoder_tokenizer.pad_token_id # Start-of-sequence token for prediction
	elif mode == "training":
		out = decoder_tokenizer.batch_encode_plus(
			[out],
			return_tensors="pt",
			padding=True,
			truncation=True)["input_ids"]

	# Tokenize the sequences
	seqs = encoder_tokenizer.batch_encode_plus(
		seqs,
		return_tensors="pt",
		padding="max_length",
		truncation=True,
		max_length = 681)["input_ids"].to(device)

	# Compute the embeddings with nucleotide transformer encoder
	# Last hidden state: (batch_size, sequence_length, hidden_size) -> (1, 681, x)
	encoder_attention_mask = (seqs != encoder_tokenizer.pad_token_id).to(device)
	with torch.no_grad():
		embeddings = encoder(
				seqs,
				attention_mask=encoder_attention_mask,
				encoder_attention_mask=encoder_attention_mask,
				output_hidden_states=True
			)['hidden_states'][-1].detach().cpu()
	
	encoder_attention_mask.detach().cpu()

	# Combine last hidden state embeddings from multiple segments into a single sequence
	embeddings = embeddings.reshape(1, 6*681, -1)
	encoder_attention_mask = encoder_attention_mask.reshape(1, 6*681)
	
	# Clean up memory
	torch.cuda.empty_cache()

	return embeddings, encoder_attention_mask, out

# TODO - function to map predicted isoform coords back to the original gene model
def backMapIsoforms():
	# Need to retain idx somehow?
	pass

#def generateDatasetWorker(ds, method, encoder_tokenizer, encoder, decoder_tokenizer, mode, device):
#	# Move the model to a specific GPU
#	encoder = encoder.to(device)
#	
#	# Generate embeddings and isoform targets over the whole dataset
#	print(f"\n[{device}] - Generating encoded dataset...method: {method}, mode: {mode}", file=sys.stderr)
#	for idx, geneModel in enumerate(ds):
#		if method == "CDSOnly":
#			ds[idx] = getCDSOnly(geneModel, [encoder_tokenizer, encoder, decoder_tokenizer], mode=mode, device=device)
#		elif method == "CDSWithJunctions":
#			ds[idx] = getCDSWithJunction(geneModel, [encoder_tokenizer, encoder, decoder_tokenizer], mode=mode, device=device)
#		elif method == "IntronWithJunctions":
#			ds[idx] = getIntronWithJunction(geneModel, [encoder_tokenizer, encoder, decoder_tokenizer], mode=mode, device=device)

#	return ds

# Generate a dataset of embedded sequences and target isoforms from an isoformList database
def generateDataset(db, method="CDSOnly", outfile="CDSOnlyDataset.pt", mode="inference"):
	# Fix isoform database for parallel reading
	setIsoformAccess(db, "READ_ONLY")
	
	# Load models
	print("Loading models...", file=sys.stderr)
	encoder_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="./HFmodels",trust_remote_code=True)
	encoder = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="./HFmodels",trust_remote_code=True)
	decoder_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", cache_dir="./HFmodels")
	encoder.eval()
	encoder.half()
	
	# Load the isoforms from the pre-processed database
	print("Loading database...", file=sys.stderr)
	with duckdb.connect(db) as con:
		ds = con.sql("SELECT * FROM isoformList").fetchall()

	# Enable encoding on GPU if available
	if torch.cuda.is_available():
		device = torch.device('cuda')
		world_size = torch.cuda.device_count()
		print(f"Using device: {device} (world_size={world_size})", file=sys.stderr)
		
		# Split ds into 'world_size' number of chunks for parallel processing
		chunk_size = len(ds)//world_size
		ds = [ds[i:i + chunk_size] for i in range(0, len(ds), chunk_size)]
		if len(ds) > world_size:
			ds[-2].extend(ds[-1])
			ds = ds[:-1]

		# use multiprocessing to parallelize the encoding
		gpu_ids = [f"cuda:{i}" for i in list(range(world_size))]
		args = [(data_chunk, method, encoder_tokenizer, encoder, decoder_tokenizer, mode, gpu_id) for data_chunk, gpu_id in zip(ds, gpu_ids)]
		# Create a multiprocessing Pool with the number of available GPUs
		with mp.get_context('spawn').Pool(processes=len(gpu_ids)) as pool:
			# Map each data chunk, args, and GPU ID to the worker function
			#results = pool.starmap(generateDatasetWorker, args)
			pass
	else:
		sys.exit("No GPU available for parallel processing! Exiting...")

	results  = [gene for chunk in results for gene in chunk]
	torch.save(results, outfile)

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

# IsoformList custom dataset class for use with DataLoader
class isoformData(Dataset):
	def __init__(self, db, method="CDSOnly", mode="inference"):
		self.db = db
		self.method = method
		self.mode = mode
		self.decoder_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", cache_dir="./HFmodels")
		self.encoder_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="./HFmodels", trust_remote_code=True)

	# Get whole sequence without UTR and isoform coordinates for a gene model
	def getTranscript(self, geneModel):
		# Get gene model data
		geneModel, isoforms, cds,start,fin ,strand, chromosome ,region_seq, split = geneModel

		# Extract whole transcript  only sequence
		cds = cds.split('|')
		cds = [list(map(int, x.split(','))) for x in cds]
		seq = region_seq[min(cds):max(cds)]
		
		# Map isoform coordinates to transcript sequence coordinates
		out = reMapIsoforms(isoforms, cds, keepIntrons = True)
		
		# Segment new sequences
		seqs = segmentSequence(seq)

		return [seqs, out]
	# Get CDS only sequence and isoform coordinates for a gene model
	def getCDSOnly(self, geneModel):
		# Get gene model data
		geneModel, isoforms, cds,start,fin ,strand, chromosome ,region_seq, split = geneModel

		# Extract CDS only sequence
		cds = cds.split('|')
		cds = [list(map(int, x.split(','))) for x in cds]
		seq = ''
		for s, e in cds:
			seq += region_seq[s:e]
		
		# Map isoform coordinates to CDS only sequence coordinates
		out = reMapIsoforms(isoforms, cds)
		
		# Segment new sequences
		seqs = segmentSequence(seq)

		return [seqs, out]

	# Get CDS sequence including intron junctions. Map isoform to this sequence
	def getCDSWithJunction(self, geneModel):
		# Get gene model data
		geneModel, isoforms, cds,start,fin ,strand, chromosome ,region_seq, split = geneModel
		
		# Extract CDS with intron junctions (60 bp = 10 tokens)
		cds = cds.split('|')
		cds = [list(map(int, x.split(','))) for x in cds]
		seq = ''
		prev_intron_length = 0
		new_coords = []
		for i, coords in enumerate(cds):
			intron_length = cds[i+1][0] - coords[1]
			
			if prev_intron_length > 0:
				if prev_intron_length >= 120:
					start = coords[0] - 60
				else:
					start = coords[0] - prev_intron_length//2
			else:
				start = coords[0]
			
			if intron_length >= 120:
				end = coords[1] + 60
			else:
				end = coords[1] + intron_length//2
			
			new_coords.append((start, end))
			seq += region_seq[start:end]
			prev_intron_length = intron_length
		
		# Map isoform coordinates to CDS with intron junctions sequence coordinates
		out = reMapIsoforms(isoforms, new_coords)

		# Segment new sequences
		seqs = segmentSequence(seq)

		return [seqs, out]

	# Get intron sequences with CDS junctions and remap isoform coordinates
	# These sequences inclue the complete first and last CDS, complete introns and truncated internal CDS
	def getIntronWithJunction(self, geneModel):
		# Get gene model data
		geneModel, isoforms, cds,start,fin ,strand, chromosome ,region_seq, split = geneModel
		
		# Extract Introns with cds junctions (60 bp = 10 tokens)
		cds = cds.split('|')
		cds = [list(map(int, x.split(','))) for x in cds]
		seq = ''
		prev_cds_length = 0
		new_coords = []
		for i, coords in enumerate(cds):
			cds_length = coords[1]-coords[0]
			# Preserve the complete sequence of the first and last CDS
			if seq == "":
				start = coords[0]
				end = coords[1]
			elif i == len(cds)-1:
				if prev_cds_length >= 120:
					start = cds[i-1][1] - 60
				else:
					start = cds[i-1][1] - prev_cds_length//2
				end = coords[1]
			elif i == 1:
				start = cds[i-1][1]
				if cds_length >= 120:
					end = cds[i][0] + 60
				else:
					end = cds[i][0] + cds_length//2 # TODO: a base pair could be omitted for an odd sequence?
			else:
				if prev_cds_length >= 120:
					start = cds[i-1][1] - 60
				else:
					start = cds[i-1][1] - prev_cds_length//2
				if cds_length >= 120:
					end = cds[i][0] + 60
				else:
					end = cds[i][0] + cds_length//2
			
			new_coords.append((start, end))
			seq += region_seq[start:end]
			prev_cds_length = cds_length
		
			# Map isoform coordinates to CDS with intron junctions sequence coordinates
			out = reMapIsoforms(isoforms, new_coords)

			# Segment new sequences
			seqs = segmentSequence(seq)

			return [seqs, out]

	def __len__(self):
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM isoformList").fetchall()[0][0]
	
	def __getitem__(self, idx):
		idx += 1
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			con.sql(f"SET access_mode='READ_ONLY'")
			geneModel = con.sql("WITH rn (geneModel, isoforms, cds, start, fin, strand, chromosome, sequence, split, rnum) AS ("
								"SELECT *, row_number() OVER() FROM isoformList) "
								f"SELECT * from rn where rnum={idx}").fetchall()[0][:-1]
		print(geneModel[2])
		if self.method == "CDSOnly":
			seqs, out = self.getCDSOnly(geneModel)
		elif self.method == "CDSWithJunctions":
			seqs, out = self.getCDSWithJunction(geneModel)
		elif self.method == "IntronWithJunctions":
			seqs, out = self.getIntronWithJunction(geneModel)
		else:
			sys.exit("Invalid sequence chunking method! Exiting... (CDSOnly|CDSWithJunctions|IntronWithJunctions)")

		print(out)
		# Tokenize output targets
		if self.mode == 'inference':
			labels = None
		elif self.mode == "training":
			# Tokenize the labels
			labels = self.decoder_tokenizer.batch_encode_plus(
				[out],
				return_tensors="pt",
				padding=True,
				truncation=True)["input_ids"]

		# Tokenize the sequences
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 681)["input_ids"]

		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)
	
		return (seqs, encoder_attention_mask, labels)

# Full generative model definition
class transgenic(T5ForConditionalGeneration):
	def __init__(self):
		self.cache_dir = "./HFmodels"
		self.decoder_model = "google/flan-t5-small"
		self.encoder_model = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
		self.peft_config = IA3Config(task_type="SEQ_2_SEQ_LM")
		super().__init__(AutoConfig.from_pretrained(self.decoder_model))
		

		# Load the pre-trained encoder
		self.encoder = AutoModelForMaskedLM.from_pretrained(self.encoder_model, cache_dir=self.cache_dir, trust_remote_code=True)
		for param in self.encoder.parameters():
			param.requires_grad = False

		# Load the pre-trained decoder and freeze the parameters
		self.decoder = T5ForConditionalGeneration.from_pretrained(self.decoder_model, cache_dir=self.cache_dir)
		for param in self.decoder.parameters():
			param.requires_grad = False

		# Load the IA3 adaptor
		self.decoder = get_peft_model(self.decoder, self.peft_config)
		#self.trainable_parameters = self.decoder.print_trainable_parameters()

		# TODO: Exlpore other options? (hidden states, BiLSTM, linear, attention, pooling, convolution)
		#plants -> 1500, multispecies -> 1024
		self.hidden_mapping = nn.Linear(1024, 512)

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
	
	#genome2Isoformlist(fasta, gff, db="AthChr4.db")
	#with duckdb.connect("AthChr4.db") as con:
	#	gm = con.sql("SELECT * FROM isoformList where strand='-'").fetchall()[0]
	#	getCDSOnly(gm, [])
	#generateDataset("AthChr4.db", method="CDSOnly", outfile="CDSOnlyDataset.pt", mode="training")
	
	#model = transgenic()
	#checkpoint = {'model_state_dict': model.state_dict()}
	#torch.save(checkpoint, "transgenic.pt")
	#model.load_state_dict(torch.load("transgenic.pt")['model_state_dict'])
	# Create a training and evaluation DataLoaders
	ds = isoformData(db, method="CDSOnly", mode="training")
	one = ds.__getitem__(1)
	batch_size = 1
	train_data, eval_data = random_split(ds, [4087, 40])

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