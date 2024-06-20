import duckdb, sys, os, subprocess, re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import List, Optional, Tuple, Union
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, PreTrainedTokenizer
from transformers import LEDForConditionalGeneration, EsmForMaskedLM, LEDPreTrainedModel, LEDConfig
from transformers.modeling_outputs import ModelOutput
from transformers import get_linear_schedule_with_warmup
from peft import IA3Config, get_peft_model
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pickle
from dataclasses import dataclass
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from safetensors import safe_open

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_HOME'] = './HFmodels'
os.environ["NCCL_DEBUG"] = "INFO"

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
	con.sql("SET enable_progress_bar = false;")
	con.sql(
		"CREATE TABLE IF NOT EXISTS geneList ("
			"geneModel VARCHAR, "
			"start INT, "
			"fin INT, "
			"strand VARCHAR, "
			"chromosome VARCHAR, "
			"sequence VARCHAR, "
			"gff VARCHAR, "
			"rn INT PRIMARY KEY)")
	con.sql("CREATE SEQUENCE IF NOT EXISTS row_id START 1;")

	
	print(f"\nProcessing {gff3}...")
	num_lines = sum(1 for line in open(gff3, "r"))
	with open(gff3, 'r') as gff3file:
		region_start = None
		feature_list = ''
		mRNA_list = ''
		skipGene = False
		five_ps = {}
		three_ps = {}
		cds_num = {}
		
		for line in tqdm(gff3file, total=num_lines):
			if line.startswith('#') | (line == '\n'):
				continue
			else:
				line = line.strip().split('\t')
				chr, _, typ, start, fin, _, strand, phase, attributes = line
				if typ == 'gene':
					if (region_start != None) & (not skipGene):
						# Add previous gene model to the database
						mRNA_list = mRNA_list[1:-1]
						gff = f"{feature_list[:-1]}>{mRNA_list}"
						try:
							con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff) VALUES (nextval('row_id'), '{geneModel}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence}', '{gff}')")
						except Exception as e:
							print(f"{geneModel=}")
							print(f"{sequence=}")
							print(f"{gff=}")
							print(f"Error inserting {geneModel} into database: {e}", file=sys.stderr)
							con.close()
							sys.exit()

						geneModel = None
						region_start = 0
						region_end = 0
						feature_list = ''
						mRNA_list = ''
						five_ps = {}
						three_ps = {}
						cds_num = {}
					
					# Construct current gene model
					skipGene = False
					geneModel = attributes.split(';')[0].split('=')[1]
					region_start = int(start)-1 # Gffs are 1-indexed
					region_end = int(fin)       # End not subtracted because python slicing is exclusive
					
					# 49,104bp corresponds to 8,184 6-mer tokens (Max input)
	 				# 25,002 -> 4,167 6-mer tokens (Max output)
					if region_end - region_start > 25002:
						print(f"Skipping {geneModel} because gene length > 25,002bp", file=sys.stderr)
						region_start = None
						region_end = None
						geneModel = None
						skipGene = True
						continue

					# Get sense strand sequence
					sequence = genome_dict[chr][region_start:region_end]
					#if strand == '-':
					#	sequence = reverseComplement(sequence)
				
				elif skipGene:
					continue
				
				elif typ == 'mRNA':
					mRNA_list = mRNA_list[:-1] + ";"

				# Build gff string - start and end coordinates are relative to the gene model sense strand
				elif typ == 'CDS': 
					start = (int(start) - 1) - region_start 
					end = int(fin) - region_start
					if phase == '0':
						phase = 'A'
					elif phase == '1':
						phase = 'B'
					elif phase == '2':
						phase = 'C'
					else:
						phase = '.'

					try:
						num = cds_num[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(cds_num) + 1)
						cds_num[f"{start}-{end}-{strand}-{phase}"] = num
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"
					
					mRNA_list += f"{typ+num}|"
				
				elif typ == 'five_prime_UTR':
					start = (int(start) - 1) - region_start
					end = int(fin) - region_start
					try:
						num = five_ps[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(five_ps) + 1)
						five_ps[f"{start}-{end}-{strand}-{phase}"] = num
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"
					
					mRNA_list += f"{typ+num}|"
					
				elif typ == 'three_prime_UTR':
					start = (int(start) - 1) - region_start
					end = int(fin) - region_start
					try:
						num = three_ps[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(three_ps) + 1)
						three_ps[f"{start}-{end}-{strand}-{phase}"] = num
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"
					
					mRNA_list += f"{typ+num}|"

	con.close()


def segmentSequence(seq, piece_size = 4092):
	# Segment the sequence into evenly sized chunks smaller than 4092bp (encoder max length of 1024 tokens)
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
	sequences, attention_masks, labels, gm = zip(*batch)

	# Pad and stack the sequences
	max_segs = max([seq.shape[0] for seq in sequences])
	sequences = [seq.flatten() for seq in sequences]
	sequences = [F.pad(seq, (0, max_segs*1024 - seq.shape[0]), value=1) for seq in sequences]
	sequences = torch.stack(sequences)
	
	
	#Pad and stack the attention masks
	attention_masks = [mask.flatten() for mask in attention_masks]
	attention_masks = [F.pad(mask, (0, max_segs*1024 - mask.shape[0]), value=False) for mask in attention_masks]
	attention_masks = torch.stack(attention_masks)

	# Pad and stack the labels
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]
		labels_padded = torch.stack(labels_padded)

	if labels:
		return sequences, attention_masks, labels_padded, gm
	else:
		return sequences, attention_masks, gm

def makeDataLoader(dat, shuffle=True, batch_size=8, pin_memory=True, sampler=None, num_workers=0):
	if sampler != None:
		shuffle = False
	
	return DataLoader(
		dat, 
		shuffle=shuffle, 
		collate_fn=target_collate_fn, 
		batch_size=batch_size, 
		pin_memory=pin_memory,
		sampler=sampler,
		num_workers=num_workers)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# geneList custom dataset class for use with DataLoader
class isoformData(Dataset):
	def __init__(self, db, dt="gff", mode="inference"):
		self.db = db
		self.mode = mode
		self.dt = dt
		self.encoder_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", cache_dir="./HFmodels", trust_remote_code=True)
		if dt == "gff":
			self.decoder_tokenizer = GFFTokenizer()
			self.maxlength = 2048
		else:
			self.decoder_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384", cache_dir="./HFmodels", trust_remote_code=True)
			self.maxlength = 1024

	def __len__(self):
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM geneList").fetchall()[0][0]
	
	def __getitem__(self, idx):
		idx += 1
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			gm,_,_,_,_,region_seq, gff,_ = con.sql(f"SELECT * FROM geneList where rn={idx}").fetchall()[0]
		
		# Tokenize output targets
		if self.mode == "training":
			# Tokenize the labels
			labels = self.decoder_tokenizer.batch_encode_plus(
				[gff],
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True,
				max_length=self.maxlength)["input_ids"]
		
		if labels.shape[1] >= self.maxlength:
			labels = torch.cat((labels[:, 0:(self.maxlength-1)], torch.tensor([[self.decoder_tokenizer.vocab["</s>"]]])), dim=1)
			print(f"Warning {gm} label truncated to {self.maxlength} tokens", file=sys.stderr)
		#else:
		#	if self.dt == "gff":
		#		labels = torch.cat((labels, torch.tensor([[self.decoder_tokenizer.vocab["</s>"]]])), dim=1)

		# Segment and tokenize the sequences
		seqs = segmentSequence(region_seq, piece_size=6000)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 1024)["input_ids"]

		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)
		if self.mode == "training":
			return (seqs, encoder_attention_mask, labels, gm)
		else:
			return (seqs, encoder_attention_mask)
	
class GFFTokenizer(PreTrainedTokenizer):
	model_input_names = ["input_ids", "attention_mask"]

	def __init__(self, vocab=None, **kwargs):
		if vocab is None:
			self.vocab = {
				"<s>": 0, "<pad>": 1,"</s>":2, "<unk>":3, '00': 4, '01': 5, '02': 6, 
				'03': 7,'04': 8, '05': 9, '06': 10, '07': 11, '08': 12, 
				'09': 13, 'A': 14, 'B': 15, 'C': 16, ">":17, ".": 18, 
				"+": 19, "-": 20, ";":21
			}
			for i in range(0, 100):
				self.vocab[str(i)] =	i + 22
			for i in range(1, 151):
				self.vocab[f"CDS{i}"] = i + 121
			for i in range(1, 51):
				self.vocab[f"five_prime_UTR{i}"] = i + 271
				self.vocab[f"three_prime_UTR{i}"] = i + 321
		else:
			self.vocab = vocab

		self.ids_to_tokens = {id: token for token, id in self.vocab.items()}
		super().__init__(**kwargs)
		self.pad_token = "<pad>"
		self.unk_token = "<unk>"

	@property
	def vocab_size(self):
		return len(self.vocab)

	def get_vocab(self):
		return dict(self.vocab, **self.added_tokens_encoder)

	def _tokenize(self, text):
		tokens = ["<s>"]

		for features in text.split(">"):
			for feature in features.split(";"):
				for column in feature.split("|"):
					if re.search(r'^\d+$', column):
						pairs = [column[i:min(i+2, len(column))] for i in range(0, len(column), 2)]
						tokens.extend([pair for pair in pairs])
					else:
						tokens.append(column)
				tokens.append(";")
			tokens.append(">")
		return tokens[:-2] + ["</s>"]

	def _convert_token_to_id(self, token):
		return self.vocab.get(token, self.vocab.get(self.unk_token))

	def _convert_id_to_token(self, index):
		return self.ids_to_tokens.get(index, self.unk_token)

	def convert_tokens_to_string(self, tokens):
		toks = []
		for i,token in enumerate(tokens):
			if token.isnumeric() and i != 0:
				if tokens[i-1].isnumeric():
					toks[-1] = toks[-1] + token
					continue
			toks.append(token)
			
		toks = '|'.join([self._convert_id_to_token(token) if isinstance(token, int) else token for token in toks])
		toks = re.sub(r'\|;\|>\|', '>', toks)
		toks = re.sub(r';>', '>', toks)
		toks = re.sub(r'>\|', '>', toks)
		toks = re.sub(r'\|;\|', ';', toks)
		#toks = re.sub(r"(CDS\|\d+)", self.replace_pipe, toks)				# Condense CDS ids
		#toks = re.sub(r"(five_prime_UTR\|\d+)", self.replace_pipe, toks)	# Condense 5' UTR ids
		#toks = re.sub(r"(three_prime_UTR\|\d+)", self.replace_pipe, toks)	# Condense 3' UTR ids
		#toks = re.sub(r'\|(\d+\|)+', self.replace_pipe_in_digits, toks)		# Condense start and end numbers
		return toks

#Copied from transformers.models.led.modeling_led.py
class LEDLearnedPositionalEmbedding(nn.Embedding):
	"""
	This module learns positional embeddings up to a fixed maximum size.
	"""

	def __init__(self, num_embeddings: int, embedding_dim: int):
		super().__init__(num_embeddings, embedding_dim)

	def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
		"""`input_ids_shape` is expected to be [bsz x seqlen]."""
		bsz, seq_len = input_ids_shape[:2]
		positions = torch.arange(
			past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
		)
		return super().forward(positions)

#Copied from transformers.models.led.modeling_led.py
@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
	"""
	Base class for sequence-to-sequence language models outputs.

	Args:
		loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
			Language modeling loss.
		logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
		past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
			num_heads, sequence_length, embed_size_per_head)`).

			Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
			used (see `past_key_values` input) to speed up sequential decoding.
		decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
		decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
			weighted average in the cross-attention heads.
		encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder of the model.
		encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
		encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
			where `x` is the number of tokens with global attention mask.

			Global attentions weights after the attention softmax, used to compute the weighted average in the
			self-attention heads. Those are the attention weights from every token with global attention to every token
			in the sequence.
	"""

	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	past_key_values: Optional[List[torch.FloatTensor]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

#Copied from transformers.models.led.modeling_led.py
@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
	"""
	Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
	decoding.

	Args:
		last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
			Sequence of hidden-states at the output of the last layer of the decoder of the model.

			If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
			hidden_size)` is output.
		past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
			num_heads, sequence_length, embed_size_per_head)`).

			Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
			used (see `past_key_values` input) to speed up sequential decoding.
		decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
		decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
			weighted average in the cross-attention heads.
		encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder of the model.
		encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
		encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
			where `x` is the number of tokens with global attention mask.

			Global attentions weights after the attention softmax, used to compute the weighted average in the
			self-attention heads. Those are the attention weights from every token with global attention to every token
			in the sequence.
	"""

	last_hidden_state: torch.FloatTensor = None
	past_key_values: Optional[List[torch.FloatTensor]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class segmented_sequence_embeddings(EsmForMaskedLM):
	def __init__(self, model):
		self.cache_dir = "./HFmodels"
		#InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
		#InstaDeepAI/agro-nucleotide-transformer-1b
		self.encoder_model = model
		config = AutoConfig.from_pretrained(self.encoder_model, is_decoder=False, trust_remote_code=True)
		super().__init__(config)
		
		self.esm = AutoModelForMaskedLM.from_pretrained(self.encoder_model, cache_dir=self.cache_dir, trust_remote_code=True)
		for param in self.esm.parameters():
			param.requires_grad = False
		for param in self.esm.lm_head.parameters():
			param.requires_grad = False

		# TODO: Exlpore other options? (hidden states, BiLSTM, linear, attention, pooling, convolution)
		#plants -> 1500, multispecies -> 1024
		#T5 -> 512, longformer -> 768
		self.hidden_mapping = nn.Linear(config.hidden_size, 768)
		self.hidden_mapping_layernorm = nn.LayerNorm(768)
	
	def forward(self, input_ids, attention_mask=None, **kwargs):
		batch_size = input_ids.shape[0]
		num_segs = input_ids.shape[1] // 1024
		input_ids = input_ids.reshape(batch_size, int(num_segs), 1024)
		attention_mask = attention_mask.reshape(batch_size, int(num_segs), 1024)
		for i in range(batch_size):
			with torch.no_grad():
				embeddings = self.esm(
					input_ids[i, :, :],
					attention_mask=attention_mask[i,:,:],
					encoder_attention_mask=attention_mask[i,:,:],
					output_hidden_states=True
				)['hidden_states'][-1]
			
			if i == 0:
				batch_embeds = embeddings.reshape(1, num_segs*1024, -1)
				batch_mask = attention_mask[i,:,:].reshape(1, num_segs*1024)
			else:
				batch_embeds = torch.cat((batch_embeds, embeddings.reshape(1, num_segs*1024, -1)), dim=0)
				batch_mask = torch.cat((batch_mask, attention_mask[i,:,:].reshape(1, num_segs*1024)), dim=0)
		
		# Use the last hidden state from the nucleotide encoder as input to the decoder
		# Transform the encoder hidden states to match the decoder input size
		decoder_inputs_embeds = self.hidden_mapping(batch_embeds)
		decoder_inputs_embeds = self.hidden_mapping_layernorm(decoder_inputs_embeds)

		return ModelOutput(inputs_embeds=decoder_inputs_embeds, attention_mask=batch_mask)

# Full generative model definition
class transgenic(LEDForConditionalGeneration):
	def __init__(self):
		self.cache_dir = "./HFmodels"
		self.decoder_model = "allenai/led-base-16384"
		self.vocab_size = 372
		self.max_target_positions = 2048
		super().__init__(AutoConfig.from_pretrained(self.decoder_model, 
													vocab_size = self.vocab_size))
		
		# Add the pre-trained encoder
		self.encoder = segmented_sequence_embeddings()
		# Load the pre-trained decoder and freeze the parameters
		self.led = LEDForConditionalGeneration.from_pretrained(self.decoder_model, cache_dir=self.cache_dir)

		# Swap out the decoder embedding layers for the GffTokenizer vocabulary
		self.led.led.decoder.embed_tokens = nn.Embedding(self.vocab_size, 768, self.led.led.decoder.padding_idx)
		self.led.led.decoder.embed_positions = LEDLearnedPositionalEmbedding(self.max_target_positions, 768)
		self.led.led.decoder.layernorm_embedding = nn.LayerNorm(768)

		# Targets all self-attention components and feedforward linear layers for adaptors
		target_modules = [
			r"led.encoder.layers.*.self_attn.longformer_self_attn.query",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.key",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.value",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.query_global",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.key_global",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.value_global",
			r"led.encoder.layers.*.self_attn.output",
			r"led.encoder.layers.*.fc1",
			r"led.encoder.layers.*.fc2",
			r"led.decoder.layers.*.self_attn.k_proj",
			r"led.decoder.layers.*.self_attn.v_proj",
			r"led.decoder.layers.*.self_attn.q_proj",
			r"led.decoder.layers.*.self_attn.kout_proj",
			r"led.decoder.layers.*.fc1",
			r"led.decoder.layers.*.fc2",
			]
		# Recompute activations for the first and second feedforward layers
		feedforward_modules = [
			r"led.encoder.layers.*.fc1", 
			r"led.encoder.layers.*.fc2",
			r"led.decoder.layers.*.fc1",
			r"led.decoder.layers.*.fc2",
			]
		peft_targets = []
		peft_feedforward = []
		for module in self.led.named_modules():
			for pattern in target_modules:
				if re.match(pattern, module[0]):
					peft_targets.append(module[0])
			for pattern in feedforward_modules:
				if re.match(pattern, module[0]):
					peft_feedforward.append(module[0])

		# Load the IA3 adaptor
		#self.peft_config = IA3Config(
		#	task_type="SEQ_2_SEQ_LM",
		#	target_modules = peft_targets,
		#	feedforward_modules = peft_feedforward)
		#self.led = get_peft_model(self.led, self.peft_config)
		#print(self.led.print_trainable_parameters(), file=sys.stderr)
		
		# Custom head for GFF prediction
		self.lm_head = nn.Linear(768, 372)
	
	def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
		"""
		Shift input ids one token to the right.
		"""
		shifted_input_ids = input_ids.new_zeros(input_ids.shape)
		shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
		shifted_input_ids[:, 0] = decoder_start_token_id

		if pad_token_id is None:
			raise ValueError("config.pad_token_id has to be defined.")
		# replace possible -100 values in labels by `pad_token_id`
		shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

		return shifted_input_ids

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		decoder_attention_mask = None,
		decoder_head_mask = None,
		cross_attn_head_mask = None,
		past_key_values = None,
		output_attentions= None,
		output_hidden_states = None,
		return_dict = None,
		**kwargs
		) -> Tuple[torch.Tensor]:

		# Compute the embeddings with nucleotide transformer encoder
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids, 
										attention_mask=attention_mask, 
										return_dict=return_dict
										).to_tuple()
		else:
			encoder_outputs = (encoder_outputs["inputs_embeds"], encoder_outputs["attention_mask"])
		
		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = self.shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)

		# Process the transformed encoder outputs through the decoder
		decoder_outputs = self.led(
			inputs_embeds=encoder_outputs[0],
			attention_mask=encoder_outputs[1],
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=True,
			return_dict=return_dict
		)
	
		# Gff prediction head
		gff_logits = self.lm_head(decoder_outputs.decoder_hidden_states[-1])

		masked_lm_loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			masked_lm_loss = loss_fct(gff_logits.view(-1, self.vocab_size), labels.view(-1)).float()
		
		if not return_dict:
			output = (gff_logits,) + decoder_outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
		
		return LEDSeq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=gff_logits,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.decoder_hidden_states,
			decoder_attentions=decoder_outputs.decoder_attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
			encoder_hidden_states=decoder_outputs.encoder_hidden_states,
			encoder_attentions=decoder_outputs.encoder_attentions,
			encoder_global_attentions=decoder_outputs.encoder_global_attentions,
		)
	
	def get_encoder(self):
		return self.encoder
	
	def get_decoder(self):
		return self.led

class transgenicOriginalEmbed(LEDForConditionalGeneration):
	def __init__(self):
		self.cache_dir = "./HFmodels"
		self.decoder_model = "allenai/led-base-16384"
		self.max_target_positions = 1024
		super().__init__(AutoConfig.from_pretrained(self.decoder_model))
		
		# Add the pre-trained encoder
		self.encoder = segmented_sequence_embeddings()
		# Load the pre-trained decoder and freeze the parameters
		self.led = LEDForConditionalGeneration.from_pretrained(self.decoder_model, cache_dir=self.cache_dir)

		# Swap out the decoder embedding layers for the GffTokenizer vocabulary
		#self.led.led.decoder.embed_tokens = nn.Embedding(self.vocab_size, 768, self.led.led.decoder.padding_idx)
		#self.led.led.decoder.embed_positions = LEDLearnedPositionalEmbedding(self.max_target_positions, 768)
		#self.led.led.decoder.layernorm_embedding = nn.LayerNorm(768)

		# Targets all self-attention components and feedforward linear layers for adaptors
		target_modules = [
			r"led.encoder.layers.*.self_attn.longformer_self_attn.query",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.key",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.value",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.query_global",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.key_global",
			r"led.encoder.layers.*.self_attn.longformer_self_attn.value_global",
			r"led.encoder.layers.*.self_attn.output",
			r"led.encoder.layers.*.fc1",
			r"led.encoder.layers.*.fc2",
			r"led.decoder.layers.*.self_attn.k_proj",
			r"led.decoder.layers.*.self_attn.v_proj",
			r"led.decoder.layers.*.self_attn.q_proj",
			r"led.decoder.layers.*.self_attn.kout_proj",
			r"led.decoder.layers.*.fc1",
			r"led.decoder.layers.*.fc2",
			]
		# Recompute activations for the first and second feedforward layers
		feedforward_modules = [
			r"led.encoder.layers.*.fc1", 
			r"led.encoder.layers.*.fc2",
			r"led.decoder.layers.*.fc1",
			r"led.decoder.layers.*.fc2",
			]
		peft_targets = []
		peft_feedforward = []
		for module in self.led.named_modules():
			for pattern in target_modules:
				if re.match(pattern, module[0]):
					peft_targets.append(module[0])
			for pattern in feedforward_modules:
				if re.match(pattern, module[0]):
					peft_feedforward.append(module[0])

		# Load the IA3 adaptor
		self.peft_config = IA3Config(
			task_type="SEQ_2_SEQ_LM",
			target_modules = peft_targets,
			feedforward_modules = peft_feedforward)
		self.led = get_peft_model(self.led, self.peft_config)
		#print(self.led.print_trainable_parameters(), file=sys.stderr)
		
		# Custom head for GFF prediction
		#self.lm_head = nn.Linear(768, )

	def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
		"""
		Shift input ids one token to the right.
		"""
		shifted_input_ids = input_ids.new_zeros(input_ids.shape)
		shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
		shifted_input_ids[:, 0] = decoder_start_token_id

		if pad_token_id is None:
			raise ValueError("config.pad_token_id has to be defined.")
		# replace possible -100 values in labels by `pad_token_id`
		shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

		return shifted_input_ids

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		decoder_attention_mask = None,
		decoder_head_mask = None,
		cross_attn_head_mask = None,
		past_key_values = None,
		output_attentions= None,
		output_hidden_states = None,
		return_dict = None,
		**kwargs
		) -> Tuple[torch.Tensor]:

		# Compute the embeddings with nucleotide transformer encoder
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids, 
										attention_mask=attention_mask, 
										return_dict=return_dict
										).to_tuple()
		else:
			encoder_outputs = (encoder_outputs["inputs_embeds"], encoder_outputs["attention_mask"])
		
		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = self.shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)

		# Process the transformed encoder outputs through the decoder
		decoder_outputs = self.led(
			inputs_embeds=encoder_outputs[0],
			attention_mask=encoder_outputs[1],
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=True,
			return_dict=return_dict,
			labels=labels
		)
	
		# Gff prediction head
		#gff_logits = self.lm_head(decoder_outputs.decoder_hidden_states[-1])

		#masked_lm_loss = None
		#if labels is not None:
		#	loss_fct = nn.CrossEntropyLoss()
		#	masked_lm_loss = loss_fct(gff_logits.view(-1, self.vocab_size), labels.view(-1)).float()
		
		#if not return_dict:
		#	output = (gff_logits,) + decoder_outputs[1:]
		#	return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
		
		#return LEDSeq2SeqLMOutput(
		#	loss=masked_lm_loss,
		#	logits=gff_logits,
		#	past_key_values=decoder_outputs.past_key_values,
		#	decoder_hidden_states=decoder_outputs.decoder_hidden_states,
		#	decoder_attentions=decoder_outputs.decoder_attentions,
		#	cross_attentions=decoder_outputs.cross_attentions,
		#	encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
		#	encoder_hidden_states=decoder_outputs.encoder_hidden_states,
		#	encoder_attentions=decoder_outputs.encoder_attentions,
		#	encoder_global_attentions=decoder_outputs.encoder_global_attentions,
		#)
		return decoder_outputs
	
	def get_encoder(self):
		return self.encoder
	
	def get_decoder(self):
		return self.led
	
class transgenicModel(LEDPreTrainedModel):
	#_tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

	def __init__(self, config: LEDConfig, encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"):
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		self.decoder_embed_tokens = nn.Embedding(vocab_size, config.d_model, padding_idx)

		self.encoder = segmented_sequence_embeddings(encoder_model)
		self.decoder = LEDForConditionalGeneration(config).led.decoder

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.decoder_embed_tokens

	def set_input_embeddings(self, value):
		self.decoder_embed_tokens = value
		self.decoder.embed_tokens = self.decoder_embed_tokens

	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	#@add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
	#@add_code_sample_docstrings(
	#	checkpoint=_CHECKPOINT_FOR_DOC,
	#	output_type=Seq2SeqModelOutput,
	#	config_class=_CONFIG_FOR_DOC,
	#)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Using this like Bart, as LED is derived from it. So far
		# No checkpoint on the hub exists that uses that in practice.
		# https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)
		# Compute the embeddings with nucleotide transformer encoder
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids, 
										attention_mask=attention_mask, 
										return_dict=return_dict
										).to_tuple()
		else:
			encoder_outputs = (encoder_outputs["inputs_embeds"], encoder_outputs["attention_mask"])
		#if encoder_outputs is None:
		#	encoder_outputs = self.encoder(
		#		input_ids=input_ids,
		#		attention_mask=attention_mask,
		#		global_attention_mask=global_attention_mask,
		#		head_mask=head_mask,
		#		inputs_embeds=inputs_embeds,
		#		output_attentions=output_attentions,
		#		output_hidden_states=output_hidden_states,
		#		return_dict=return_dict,
		#	)

		# If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
		#elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
		#	encoder_outputs = LEDEncoderBaseModelOutput(
		#		last_hidden_state=encoder_outputs[0],
		#		hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
		#		attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
		#		global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
		#	)

		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_outputs[0],
			encoder_attention_mask=encoder_outputs[1].long(), #attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return LEDSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs[0],
			encoder_hidden_states=None,
			encoder_attentions=encoder_outputs[1],
			encoder_global_attentions=None,
		)

class transgenicForConditionalGeneration(LEDPreTrainedModel):
	base_model_prefix = "transgenic"
	_keys_to_ignore_on_load_missing = ["final_logits_bias"]
	_tied_weights_keys = ["transgenic.decoder.embed_tokens.weight", "lm_head.weight"]

	def __init__(self, config: LEDConfig, encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", unlink=False):
		if not unlink:
			_tied_weights_keys = []
		super().__init__(config)
		self.transgenic = transgenicModel(config, encoder_model=encoder_model)
		self.register_buffer("final_logits_bias", torch.zeros((1, self.transgenic.decoder_embed_tokens.num_embeddings)))
		self.lm_head = nn.Linear(config.d_model, self.transgenic.decoder_embed_tokens.num_embeddings, bias=False)

		# Initialize weights and apply final processing
		self.post_init()
		self.initialize_weights()

	def get_encoder(self):
		return self.transgenic.get_encoder()

	def get_decoder(self):
		return self.transgenic.get_decoder()

	def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
		new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
		self._resize_final_logits_bias(new_embeddings.weight.shape[0])
		return new_embeddings

	def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
		old_num_tokens = self.final_logits_bias.shape[-1]
		if new_num_tokens <= old_num_tokens:
			new_bias = self.final_logits_bias[:, :new_num_tokens]
		else:
			extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
			new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
		self.register_buffer("final_logits_bias", new_bias)

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings
	
	def initialize_weights(self):
		for m in self.transgenic.decoder.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		nn.init.xavier_uniform_(self.transgenic.encoder.hidden_mapping.weight)
		nn.init.constant_(self.transgenic.encoder.hidden_mapping.bias, 0)

	#@add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
	#@replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
	#@add_end_docstrings(LED_GENERATION_EXAMPLE)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
			config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
			(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:

		Conditional generation example:

		```python
		>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

		>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
		>>> TXT = "My friends are <mask> but they eat too many carbs."

		>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
		>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

		>>> prediction = model.generate(input_ids)[0]
		>>> print(tokenizer.decode(prediction, skip_special_tokens=True))
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)

		outputs = self.transgenic(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			encoder_outputs=encoder_outputs,
			global_attention_mask=global_attention_mask,
			head_mask=head_mask,
			decoder_head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			decoder_inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

		masked_lm_loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		return LEDSeq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
			encoder_global_attentions=outputs.encoder_global_attentions,
		)

	def prepare_inputs_for_generation(
		self,
		decoder_input_ids,
		past_key_values=None,
		attention_mask=None,
		global_attention_mask=None,
		head_mask=None,
		decoder_head_mask=None,
		cross_attn_head_mask=None,
		use_cache=None,
		encoder_outputs=None,
		**kwargs,
	):
		# cut decoder_input_ids if past is used
		if past_key_values is not None:
			decoder_input_ids = decoder_input_ids[:, -1:]

		return {
			"input_ids": None,  # encoder_outputs is defined. input_ids not needed
			"encoder_outputs": encoder_outputs,
			"past_key_values": past_key_values,
			"decoder_input_ids": decoder_input_ids,
			"attention_mask": attention_mask,
			"global_attention_mask": global_attention_mask,
			"head_mask": head_mask,
			"decoder_head_mask": decoder_head_mask,
			"cross_attn_head_mask": cross_attn_head_mask,
			"use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
		}

	def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
		return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			# cached cross_attention states don't have to be reordered -> they are always the same
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
				+ layer_past[2:],
			)
		return reordered_past

def plot_grad_flow(model, outprefix="gradient_flow"):
	'''Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.

	Usage: Plug this function in Trainer class after loss.backwards() as 
	"plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
	ave_grads = []
	max_grads= []
	layers = []
	for n, p in model.named_parameters():
		if(p.requires_grad) and ("bias" not in n):
			if p.grad != None:
				#print(f"{n=}, {p.grad=}")
				layers.append(n)
				ave_grads.append(p.grad.abs().mean().cpu())
				max_grads.append(p.grad.abs().max().cpu())
	#for n,p in model.lm_head.named_parameters():
	#	if p.grad != None:
	#		print(f"{n=}, {p.grad=}")
	#		layers.append("lm_head."+n)
	#		ave_grads.append(p.grad.abs().mean())
	#		max_grads.append(p.grad.abs().max())
	
	if len(ave_grads) == 0:
		print("No gradients to plot.", file=sys.stderr)
		return
	plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, lw=1, color="c")
	plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.8, lw=1, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(left=0, right=len(ave_grads))
	plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.legend([Line2D([0], [0], color="c", lw=4),
				Line2D([0], [0], color="b", lw=4),
				Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
	# save as pdf
	plt.savefig(f"{outprefix}.pdf")

# Training loop
def trainTransgenicDDP(rank, 
		train_ds:isoformData, 
		eval_ds:isoformData, 
		lr, 
		num_epochs,  
		schedule_lr, 
		eval, 
		world_size,
		batch_size,
		checkpoint_path="transgenic_checkpoint.pt"):
	# Set up GPU process group
	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
	print(f"Training transgenic on {device}, (world_size={world_size})", file=sys.stderr)
	setup(rank, world_size)
	
	# Distribute data
	train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
	eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, sampler=eval_sampler)
	
	# Load the model and wrap in DDP
	model = transgenic().to(device)
	ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
	ddp_model.train()
	
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

	dt = GFFTokenizer()
	# Training loop
	best_eval_score = None
	scaler = GradScaler()
	accumulation_steps = 4
	for epoch in range(num_epochs):
		train_sampler.set_epoch(epoch)
		total_loss = 0
		for step,batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			batch = [item.to(device) for item in batch[:-1]]
			with autocast(): # Mixed precision training to save memory
				outputs = ddp_model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
				loss = outputs.loss / accumulation_steps
			total_loss += loss.detach().float()
			scaler.scale(loss).backward()
			if (step + 1) % accumulation_steps == 0:
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
			if schedule_lr: lr_scheduler.step()
			

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				batch = [item.to(device) for item in batch]
				with torch.no_grad():
					outputs = ddp_model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
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

def run_trainTransgenicDDP(train_ds, eval_ds=None, lr=8e-3, num_epochs=10, schedule_lr=True, eval=False, batch_size=1,checkpoint_path="transgenic_checkpoint.pt"):
	world_size = torch.cuda.device_count()
	mp.spawn(trainTransgenicDDP, 
		args=(train_ds, eval_ds, lr, num_epochs, schedule_lr, eval, world_size, batch_size), 
		nprocs=world_size, 
		join=True)
	cleanup()

def trainTransgenicAccelerate(
	train_ds:isoformData, 
	eval_ds:isoformData, 
	lr, 
	num_epochs,  
	schedule_lr, 
	eval, 
	batch_size,
	checkpoint_path="checkpoints/",
	safetensors_model=None,
	output_dir="saved_transgenic_models/"):

	print(f"Training transgenic with custom embeddings. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)
	
	# Set up accelerator
	accelerator = Accelerator(mixed_precision="fp16")
	device = accelerator.device
	print(f"Training transgenic with Accelerate on {device}", file=sys.stderr)
	
	# Set up DataLoaders
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	model = transgenic()
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)
	model.to(device)

	for param in model.led.parameters():
		param.requires_grad = False
	for param in model.encoder.esm.parameters():
		param.requires_grad = False
	for param in model.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	for param in model.led.led.decoder.embed_tokens.parameters():
		param.requires_grad = True
	for param in model.led.led.decoder.embed_positions.parameters():
		param.requires_grad = True
	for param in model.led.led.decoder.layernorm_embedding.parameters():
		param.requires_grad = True
	for param in model.lm_head.parameters():
		param.requires_grad = True
	
	# Setup the optimizer
	optimizer = optim.AdamW(
		[{"params": model.encoder.hidden_mapping.parameters(), "lr": lr},
		{"params": model.led.led.decoder.embed_tokens.parameters(), "lr": lr},
		{"params": model.led.led.decoder.embed_positions.parameters(), "lr": lr},
		{"params": model.led.led.decoder.layernorm_embedding.parameters(), "lr": lr},
		#{"params": model.led.parameters(), "lr": lr},
		{"params": model.lm_head.parameters(), "lr": lr},
		{"params": model.encoder.hidden_mapping_layernorm.parameters(), "lr": lr}]
	)
	optimizer.zero_grad()
	
	# Create the learning rate scheduler
	if schedule_lr:
		lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=(len(train_ds) * num_epochs),
		)
	
	# Prep objects for use with accelerator
	model, optimizer, train_ds, eval_ds, lr_scheduler = accelerator.prepare(
		model, optimizer, train_ds, eval_ds, lr_scheduler
	)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			with accelerator.accumulate(model):
				#with autocast():
				outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
				loss = outputs.loss
				total_loss += loss.detach().float()
				accelerator.backward(loss)
				if schedule_lr: lr_scheduler.step()
				optimizer.step()
				optimizer.zero_grad()
				#for name, param in model.led.led.decoder.embed_tokens.named_parameters():
				#	print(f"{name=}, {param.grad=}", file=sys.stderr)
				#for name, param in model.led.led.decoder.embed_positions.named_parameters():
				#	print(f"{name=}, {param.grad=}", file=sys.stderr)
					
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch=}, Step {step=}, Loss {loss=}", file=sys.stderr)
				accelerator.save_state(output_dir=checkpoint_path)

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
				loss = outputs.loss
				eval_loss += loss.detach().float()

			eval_epoch_loss = eval_loss / len(eval_ds)
			eval_ppl = torch.exp(eval_epoch_loss)
			print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		
		if eval:
			if best_eval_score is None or eval_epoch_loss < best_eval_score:
				best_eval_score = eval_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
	
	accelerator.wait_for_everyone()
	accelerator.save_model(model, output_dir)

def trainTransgenicOriginalEmbedAccelerate(
	train_ds:isoformData, 
	eval_ds:isoformData, 
	lr, 
	num_epochs,  
	schedule_lr, 
	eval, 
	batch_size,
	checkpoint_path="checkpoints/",
	safetensors_model=None,
	output_dir="saved_transgenic_models/"):

	print(f"Training transgenic with original embeddings. {checkpoint_path=} {output_dir=} {safetensors_model=}", file=sys.stderr)
	
	# Set up accelerator
	accelerator = Accelerator(mixed_precision="fp16")
	device = accelerator.device
	print(f"Training transgenic with Accelerate on {device}", file=sys.stderr)
	
	# Set up DataLoaders
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	model = transgenicOriginalEmbed()
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)
	model.to(device)

	for param in model.encoder.esm.parameters():
		param.requires_grad = False
	for param in model.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	for param in model.led.led.decoder.embed_tokens.parameters():
		param.requires_grad = True
	for param in model.led.led.decoder.embed_positions.parameters():
		param.requires_grad = True
	for param in model.led.led.decoder.layernorm_embedding.parameters():
		param.requires_grad = True
	for param in model.led.lm_head.parameters():
		param.requires_grad = True
	
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
	model, optimizer, train_ds, eval_ds, lr_scheduler = accelerator.prepare(
		model, optimizer, train_ds, eval_ds, lr_scheduler
	)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			with accelerator.accumulate(model):
				#with autocast():
				outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
				loss = outputs.loss
				total_loss += loss.detach().float()
				accelerator.backward(loss)
				if schedule_lr: lr_scheduler.step()
				optimizer.step()
				optimizer.zero_grad()
				#for name, param in model.led.led.decoder.embed_tokens.named_parameters():
				#	print(f"{name=}, {param.grad=}", file=sys.stderr)
				#for name, param in model.led.led.decoder.embed_positions.named_parameters():
				#	print(f"{name=}, {param.grad=}", file=sys.stderr)
					
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch=}, Step {step=}, Loss {loss=}", file=sys.stderr)
				accelerator.save_state(output_dir=checkpoint_path)

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
				loss = outputs.loss
				eval_loss += loss.detach().float()

			eval_epoch_loss = eval_loss / len(eval_ds)
			eval_ppl = torch.exp(eval_epoch_loss)
			print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		
		if eval:
			if best_eval_score is None or eval_epoch_loss < best_eval_score:
				best_eval_score = eval_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
	
	accelerator.wait_for_everyone()
	accelerator.save_model(model, output_dir)

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

	# start a new wandb run to track this script
	wandb.init(
		# set the wandb project where this run will be logged
		project="transgenic",

		# track hyperparameters and run metadata
		config={
		"learning_rate": lr,
		"schedule_lr": schedule_lr,
		"architecture": "FCG",
		"dataset": "25kb-5genomes",
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
	ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.gradient_accumulation_steps > 1)
	accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # gradient_accumulation_steps=32
	device = accelerator.device
	print(f"Training transgenic with Accelerate on {device}", file=sys.stderr)
	
	# Set up DataLoaders
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	config = LEDConfig.from_pretrained("allenai/led-base-16384", 
									vocab_size=372, 
									max_decoder_position_embeddings=2048,
									decoder_layerdrop=0.2,
									dropout= 0.2)
	model = transgenicForConditionalGeneration(config, encoder_model=encoder_model, unlink=unlink)
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["lm_head.weight"] = tensors["transgenic.decoder.embed_tokens.weight"]
		model.load_state_dict(tensors)
	model.to(device)
	model.train()

	for param in model.transgenic.encoder.esm.parameters():
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
	
	# Prep objects for use with accelerator
	model, optimizer, train_ds, eval_ds, lr_scheduler = accelerator.prepare(
		model, optimizer, train_ds, eval_ds, lr_scheduler
	)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			#with accelerator.accumulate(model):
			outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
			loss = outputs.loss / accumulation_steps
			#loss.backward()
			accelerator.backward(loss)
			total_loss += outputs.loss.detach().float()
			if (step+1) % accumulation_steps == 0:
				clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				if schedule_lr: lr_scheduler.step()
				# log metrics to wandb
				wandb_log = {"epoch":epoch, "step":step, "loss": outputs.loss.detach().float(), "mean_loss": total_loss / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
				for name, param in model.named_parameters():
					if (param.grad != None) & (param.requires_grad):
						grad_norm = param.grad.norm().item()
						wandb_log[f"{name}_grad_norm"] = grad_norm
				wandb.log(wandb_log)
				#if accelerator.is_main_process: plot_grad_flow(model, outprefix=f"{checkpoint_path}/grad_flow")
				optimizer.zero_grad()
			
			
				
			#if schedule_lr: lr_scheduler.step()
			#optimizer.step()
			#optimizer.zero_grad()
					
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch=}, Step {step=}, Loss {loss=}", file=sys.stderr)
				accelerator.save_state(output_dir=checkpoint_path)

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
				loss = outputs.loss
				eval_loss += loss.detach().float()

			eval_epoch_loss = eval_loss / len(eval_ds)
			eval_ppl = torch.exp(eval_epoch_loss)
			print(f"{epoch=}: {train_ppl=}, {train_epoch_loss=}, {eval_ppl=}, {eval_epoch_loss=}", file=sys.stderr)
		else:
			print(f"Epoch {epoch=}: {train_ppl=}, {train_epoch_loss=}", file=sys.stderr)
		
		if eval:
			if best_eval_score is None or eval_epoch_loss < best_eval_score:
				best_eval_score = eval_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
	
	accelerator.wait_for_everyone()
	accelerator.save_model(model, output_dir)
	wandb.finish()

# Prediction loop
def predictTransgenicDDP(rank, checkpoint:str, dataset:isoformData, outfile, batch_size, world_size):
	# Set up GPU process group
	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
	print(f"Running transgenic in prediction mode on {device}, (world_size={world_size})", file=sys.stderr)
	setup(rank, world_size)

	# configure map_location
	map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
	
	# Load the model and wrap in DDP
	model = transgenic().to(device)
	model.load_state_dict(torch.load(checkpoint, map_location=map_location)['model_state_dict'])
	ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
	ddp_model.eval()
	
	# Create a DataLoader
	# Distribute data
	sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
	loader = makeDataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True, sampler=sampler)
	
	# Prediction loop
	dt = GFFTokenizer()
	predictions = []
	for step, batch in enumerate(tqdm(loader)):
		batch = [item.to(device) for item in batch]
		with torch.no_grad():
			outputs = model.generate(inputs=batch[0], attention_mask=batch[1], num_return_sequences=1, max_length=1024
									#temperature=0.1,  # Increase predictability of outputs by decreasing this
									#top_k=10,         # Limit next word sampling group to top-k groups
									#top_p=0.95,       # Top-p (nucleus) sampling
									#do_sample=True
									)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
		true = dt.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True)
		predictions.append([str(true[0]), str(pred[0])])
	
	with open(outfile, 'w') as out:
		for prediction in predictions:
			out.write("\t".join(prediction)+"\n")

	with open("predictions.pkl", 'wb') as out:
			pickle.dump(predictions, out)
	

def run_predictTransgenicDDP(checkpoint:str, dataset:isoformData, outfile, batch_size):
	world_size = torch.cuda.device_count()
	mp.spawn(predictTransgenicDDP, 
		args=(checkpoint, dataset, outfile, batch_size, world_size), 
		nprocs=world_size, 
		join=True)
	cleanup()

def predictTransgenic(model_path:str, dataset:isoformData, outfile="transgenic.out", batch_size=1):
	
	# Set up DataLoader
	dataset = makeDataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model
	model = transgenic()
	tensors = {}
	with safe_open("model.safetensors", framework="pt", device="cpu") as f:
		for k in f.keys():
			tensors[k] = f.get_tensor(k)
	model.load_state_dict(tensors)
	model.eval()

	# Prediction loop
	dt = GFFTokenizer()
	predictions = []
	for step, batch in enumerate(tqdm(dataset)):
		with torch.no_grad():
			outputs = model.generate(inputs=batch[0], attention_mask=batch[1], num_return_sequences=1, max_length=1024
									#temperature=0.1,  # Increase predictability of outputs by decreasing this
									#top_k=10,         # Limit next word sampling group to top-k groups
									#top_p=0.95,       # Top-p (nucleus) sampling
									#do_sample=True
									)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
		true = dt.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True)
		predictions.append([str(batch[3]), str(true[0]), str(pred[0])])
	
	with open(outfile, 'w') as out:
		for prediction in predictions:
			out.write("\t".join(prediction)+"\n")

	with open("predictions.pkl", 'wb') as out:
			pickle.dump(predictions, out)

def predictTransgenicAccelerate(model_path:str, dataset:isoformData, outfile="transgenic.out", batch_size=1):
	# Set up accelerator
	accelerator = Accelerator()
	device = accelerator.device
	print(f"Running transgenic in prediction mode on {device}", file=sys.stderr)
	
	# Set up DataLoader
	dataset = makeDataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model
	model = transgenic()
	model.to(device)
	tensors = {}
	with safe_open("saved_transgenic_models_accu32/model.safetensors", framework="pt", device="cpu") as f:
		for k in f.keys():
			tensors[k] = f.get_tensor(k)
	model.load_state_dict(tensors)
	model.eval()
	
	# Prep objects for use with accelerator
	model, dataset = accelerator.prepare(
		model, dataset
	)

	# Prediction loop
	dt = GFFTokenizer()
	predictions = []
	for step, batch in enumerate(tqdm(dataset)):
		with torch.no_grad():
			outputs = model.generate(inputs=batch[0], attention_mask=batch[1], num_return_sequences=1, max_length=1024
									#temperature=0.1,  # Increase predictability of outputs by decreasing this
									#top_k=10,         # Limit next word sampling group to top-k groups
									#top_p=0.95,       # Top-p (nucleus) sampling
									#do_sample=True
									)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
		true = dt.batch_decode(batch[2].reshape(batch_size,batch[2].size()[-1]).detach().cpu().numpy(), skip_special_tokens=True)
		predictions.append([str(batch[3]), str(true[0]), str(pred[0])])
	
	with open(f"{device}_{outfile}", 'w') as out:
		for prediction in predictions:
			out.write("\t".join(prediction)+"\n")

	with open(f"{device}_predictions.pkl", 'wb') as out:
			pickle.dump(predictions, out)

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
	torch.manual_seed(123)
	#fasta = "ATH_Chr4.fas"
	#gff = "Athaliana_167_gene_Chr4.gff3"
	#db = "AthChr4.db"
	
	db = "Flagship_Genomes_25k_stranded.db"
	#files = {
	#	"Athaliana_167_TAIR10.fa":"Athaliana_167_TAIR10.gene.clean.gff3",
	#	"Gmax_880_v6.0.fa":"Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
	#	"Ppatens_318_v3.fa":"Ppatens_318_v3.3.gene_exons.clean.gff3",
	#	"Ptrichocarpa_533_v4.0.fa":"Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
	#	"Sbicolor_730_v5.0.fa":"Sbicolor_730_v5.1.gene_exons.clean.gff3"
	#}
	#for fasta, gff in files.items():
	#	name = fasta.split("_")[0]
	#	print(f"Processing {name}...", file=sys.stderr)
	#	genome2GeneList("training_data/"+fasta, "training_data/"+gff, db=db)
	#	ds = isoformData(db, mode="training")
	#	length = len(ds)
	#	print(f"{name} {length=}")
	
	#model = transgenic()
	#tensors = {}
	#with safe_open("saved_transgenic_models_accu32/model.safetensors", framework="pt", device="cpu") as f:
	#	for k in f.keys():
	#		tensors[k] = f.get_tensor(k)

	#model.load_state_dict(tensors)
	#model.eval()
	#sys.exit()

	#config = AutoConfig.from_pretrained("allenai/led-base-16384", vocab_size=372, max_decoder_position_embeddings=2048)
	#model = transgenicForConditionalGeneration(config)
	#model = transgenicOriginalEmbed()
	#model.load_state_dict(torch.load("checkpoints_FCG/pytorch_model/mp_rank_00_model_states.pt")['module'])#, map_location=torch.device('cpu'))['module'])
	#model.to(torch.device('cuda:0'))
	#model.eval()
	# Create a training, evaluation, and testing DataLoaders (Dataset length: 175498)
	#ds = isoformData(db, dt="gff", mode="training")
	#train_data, eval_data, test_data, t_data = random_split(ds, [131470, 17399, 26167, 4])
	#batch_size = 1
	#for step, batch in enumerate(tqdm(t_data)):
	#	with torch.no_grad():
	#		out = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2], return_dict=True)
	#		prediction = out.logits.argmax(dim=-1)
	#		print(batch[3], file=sys.stderr)
	#		print(batch[2].to(torch.device('cpu')), file=sys.stderr)
	#		print(prediction.to(torch.device('cpu')), file=sys.stderr)
	#		print(out.loss, file=sys.stderr)
	#sys.exit()

	mode = sys.argv[1]
	encoder_model = sys.argv[2]
	unlink = bool(sys.argv[3])
	notes = sys.argv[4]
	print(f"Running in {mode} mode", file=sys.stderr)

	if mode == "test":
		config = AutoConfig.from_pretrained("allenai/led-base-16384", vocab_size=372, max_decoder_position_embeddings=2048)
		model = transgenicForConditionalGeneration(config)
		
		#model = transgenicOriginalEmbed()
		#model.load_state_dict(torch.load("checkpoints_FCG_accu32/pytorch_model/mp_rank_00_model_states.pt")['module'])#, map_location=torch.device('cpu'))['module'])
		
		tensors = {}
		with safe_open("checkpoints_FCG_accu32/model.safetensors", framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["lm_head.weight"] = tensors["transgenic.decoder.embed_tokens.weight"]
		model.load_state_dict(tensors)
		
		model.to(torch.device('cuda:0'))
		model.eval()
		# Create a training, evaluation, and testing DataLoaders (Dataset length: 175498)
		ds = isoformData(db, dt="gff", mode="training")
		train_data, eval_data, test_data, t_data = random_split(ds, [131470, 17399, 26167, 4])
		batch_size = 1
		for step, batch in enumerate(tqdm(t_data)):
			with torch.no_grad():
				out = model(input_ids=batch[0].to(torch.device('cuda:0')), attention_mask=batch[1].to(torch.device('cuda:0')), labels=batch[2].to(torch.device('cuda:0')), return_dict=True)
				prediction = out.logits.argmax(dim=-1)
				print(batch[3], file=sys.stderr)
				print(batch[2].to(torch.device('cpu')), file=sys.stderr)
				print(prediction.to(torch.device('cpu')), file=sys.stderr)
				print(out.loss, file=sys.stderr)
		sys.exit()
	elif mode == "train":
		ds = isoformData(db, dt="gff", mode="training")
		train_data, eval_data, test_data, t_data = random_split(ds, [131470, 17399, 26167, 4])
		trainTransgenicAccelerate(
			train_data, 
			eval_data, 
			lr=8e-2, 
			num_epochs=10, 
			schedule_lr=True, 
			eval=True, 
			batch_size=1, 
			checkpoint_path="checkpoints_accu32/", 
			safetensors_model=None, #"saved_transgenic_models_accu32/model.safetensors",
			output_dir="saved_transgenic_models_accu32/"
		)
	if mode == "FCG":
		ds = isoformData(db, dt="gff", mode="training")
		train_data, eval_data, test_data = random_split(ds, [131470, 17399, 26171])
		trainTransgenicFCGAccelerate(
			train_data, 
			eval_data, 
			lr=8e-3, 
			num_epochs=10, 
			schedule_lr=True, 
			eval=True, 
			batch_size=1, 
			checkpoint_path="checkpoints_FCG/", 
			safetensors_model=None, #"saved_transgenic_models_accu32/model.safetensors",
			output_dir="saved_models_FCG/",
			notes=notes,
			encoder_model=encoder_model,
			unlink = unlink
		)
	elif mode == "trainOriginalEmbed":
		ds = isoformData(db, dt="led", mode="training")
		train_data, eval_data, test_data, t_data = random_split(ds, [131470, 17399, 26167, 4])
		trainTransgenicOriginalEmbedAccelerate(
			train_data, 
			eval_data, 
			lr=8e-2, 
			num_epochs=10, 
			schedule_lr=True, 
			eval=True, 
			batch_size=1, 
			checkpoint_path="checkpoints_OriginalEmbed/", 
			safetensors_model=None, #"saved_transgenic_models_accu32/model.safetensors",
			output_dir="saved_transgenic_models_OriginalEmbed/"
		)
	elif mode == "predictAccelerate":
		predictTransgenicAccelerate(
			"saved_transgenic_models_accu32/model.safetensors", 
			test_data, 
			outfile="transgenic.out", 
			batch_size=1
		)
	elif mode == "predict":
		predictTransgenic(
			"saved_transgenic_models_accu32/model.safetensors", 
			test_data, 
			outfile="transgenic.out", 
			batch_size=1
			)