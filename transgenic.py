import duckdb, sys, os, subprocess, re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import List, Optional, Tuple, Union
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
import gc

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

def reverseComplement_gffString(gff, length):
	# Reverse complement a gff string for reverse complemented sequence
	feature_list, mRNA_list = gff.split('>')
	features = feature_list.split(';')


	order = []
	begin_cds = 0
	begin_utr5 = 0
	begin_utr3 = 0
	phase_lookup = {}
	for i, feature in enumerate(features):
		if "CDS" in feature:
			order.insert(begin_cds, i)
			begin_utr5 = i+1
			begin_utr3 = i+1
			phase_lookup[feature.split("|")[1]] = feature.split("|")[4]
		elif "five_prime_UTR" in feature:
			order.insert(begin_utr5, i)
			begin_cds = i+1
			begin_utr3 = i+1
		elif "three_prime_UTR" in feature:
			order.insert(begin_utr3, i)
			begin_cds = i+1
			begin_utr5 = i+1

	features = [feature.split('|') for feature in features]
	new_features = []
	feature_swap = {}
	for index, i in enumerate(order):
		old_start = int(features[i][0])
		old_end = int(features[i][2])
		new_start = str(int(length) - old_end)
		new_end = str(int(length) - old_start)
		new_type = features[index][1]
		feature_swap[features[i][1]] = new_type
		phase = features[i][4]

		if features[i][3] == "-":
			new_strand = "+"
		elif features[i][3] == "+":
			new_strand = "-"

		new_features.append([new_start, new_type, new_end, new_strand, phase])
	
	transcripts = [t.split("|") for t in mRNA_list.split(';')]
	new_transcripts = []
	for transcript in transcripts:
		new_utrs = []
		new_cds = []
		phase = ''
		for feature in transcript:
			if "UTR" in feature:
				new_utrs.append(feature_swap[feature])
			else:
				new_cds.insert(0, feature_swap[feature])
				phase += phase_lookup[feature]
		new_transcripts.append(new_cds + new_utrs)

	return f"{';'.join(['|'.join(feature) for feature in new_features])}>{';'.join(['|'.join(transcript) for transcript in new_transcripts])}"

def validateCDS(gff:str, seq:str, geneModel:str) -> Tuple[bool, str]:
	# Purpose:
	#      Validate that the CDS features in the gff produce a valid coding sequence
	# Inputs:
	#      gff: str, the gff string
	#      seq: str, the sequence
	#      geneModel: str, the gene model name
	# Outputs:
	#      (bool, str) - (valid, error message)

	# Break out the gene features
	features = [f.split("|") for f in gff.split(">")[0].split(";")]
	transcripts = [t.split("|") for t in gff.split(">")[1].split(";")]
	strand = features[0][3]
	feature_index_dict = {f[1]:i for i,f in enumerate(features)}

	# Check start/stop codons and frame
	stopCodons = ["TAA", "TAG", "TGA"]
	stopCodonsRC = ["TTA", "CTA", "TCA"]
	for transcript in transcripts:
		seqlen = 0
		for i,typ in enumerate(transcript):
			if "CDS" in typ:
				if i == 0:
					if strand == "+":
						if seq[int(features[feature_index_dict[typ]][0]):int(features[feature_index_dict[typ]][0])+3] != "ATG":
							valid = False
							return (valid, f"Start codon missing in {features[feature_index_dict[typ]][1]} of {geneModel}.")
					else:
						if seq[int(features[feature_index_dict[typ]][0]):int(features[feature_index_dict[typ]][0])+3] not in stopCodonsRC:
							valid = False
							return (valid, f"Stop codon missing in {features[feature_index_dict[typ]][1]} of {geneModel}.")
				lastCDS = typ
				seqlen += int(features[feature_index_dict[typ]][2]) - int(features[feature_index_dict[typ]][0])
		if seqlen % 3 != 0:
			valid = False
			return (valid, f"{geneModel} not a multiple of 3.")
		if strand == "+":
			if seq[int(features[feature_index_dict[lastCDS]][2])-3:int(features[feature_index_dict[lastCDS]][2])] not in stopCodons:
				valid = False
				return (valid, f"Stop codon missing in {features[feature_index_dict[lastCDS]][1]} of {geneModel}.")
		else:
			if seq[int(features[feature_index_dict[lastCDS]][2])-3:int(features[feature_index_dict[lastCDS]][2])] != "CAT":
				valid = False
				return (valid, f"Start codon missing in {features[feature_index_dict[lastCDS]][1]} of {geneModel}.")
	return (True, None)

def genome2GeneList(genome, gff3, db, maxLen=49152, addExtra=0, addRC=False, addRCIsoOnly=False, clean=False):
	# Purpose: 
	#   Read a genome assembly and gff3 annotation file into a 
	#   duckdb database for training or inference. For each gene 
	#   model, the corresponding nucleotide sequence is extracted
	#   from the genome and the target annotation string is created.
	#   The function may be called multiple times to append add 
	#   multiple genomes to the database
	# Inputs:
	#   genome: path to a fasta file containing the genome assembly
	#   gff3:     path to a gff3 file containing gene annotations
	#   db:       path to a duckdb database file (will be created if it does not exist)
	#   maxLen:   Maximum size of gene model sequence to include in the database (larger gene models are skipped)
	#   addExtra: Max size of random buffer to add to the gene model sequence (used to capture UTR start and end during training)
	#   addRC:    Add reverse complement of gene model sequence to the database (Used to augment training data)
	#   addRCIsoOnly: When adding RC seqs, only add gene models with alternative splicing
	#   clean: If true only sequences will be added that have start and stop codons and are a multiple of 3
	# Outputs:
	#   A duckdb database used to load data into the model
	# TODO:
	#   - Add support for bed files
	#   - Add inference mode with just target sequences for prediction

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
					# Add previous gene model to the database
					if (region_start != None) & (not skipGene):
						mRNA_list = mRNA_list[1:-1]
						gff = f"{feature_list[:-1]}>{mRNA_list}"
						valid = True
						if clean:
							valid, error = validateCDS(gff, sequence, geneModel)
							if not valid:
								print(f"{error}. Skipping", file=sys.stderr)
								region_start = 0
								region_end = 0
								geneModel = None
								feature_list = ''
								mRNA_list = ''
								five_ps = {}
								three_ps = {}
								cds_num = {}
					
						try:
							if valid:
								con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff) VALUES (nextval('row_id'), '{geneModel}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence}', '{gff}')")
						except Exception as e:
							print(f"{geneModel=}")
							print(f"{sequence=}")
							print(f"{gff=}")
							print(f"Error inserting {geneModel} into database: {e}", file=sys.stderr)
							con.close()
							sys.exit()
						
						if addRC:
							gff_rc = reverseComplement_gffString(gff, region_end - region_start)
							if addRCIsoOnly:
								if ';' in mRNA_list:
									try:
										if valid:
											con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff) VALUES (nextval('row_id'), '{geneModel + "-rc"}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence_rc}', '{gff_rc}')")
									except Exception as e:
										print(f"{geneModel=}-rc")
										print(f"{sequence_rc=}")
										print(f"{gff_rc=}")
										print(f"Error inserting {geneModel+"-rc"} into database: {e}", file=sys.stderr)
										con.close()
										sys.exit()
							else:
								try:
									if valid:
										con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff) VALUES (nextval('row_id'), '{geneModel + "-rc"}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence_rc}', '{gff_rc}')")
								except Exception as e:
									print(f"{geneModel=}-rc")
									print(f"{sequence_rc=}")
									print(f"{gff_rc=}")
									print(f"Error inserting {geneModel+"-rc"} into database: {e}", file=sys.stderr)
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
					five_prime_buffer = int(torch.randint(addExtra, (1,)))
					three_prime_buffer = int(torch.randint(addExtra, (1,)))
					skipGene = False
					geneModel = attributes.split(';')[0].split('=')[1]
					region_start = int(start) - five_prime_buffer - 1 # Gffs are 1-indexed
					if region_start < 0: region_start = 0
					region_end = int(fin) + three_prime_buffer        # End not subtracted because python slicing is exclusive
					
					# 49,152bp corresponds to 8,192 6-mer tokens (Max input)
	 				# 25,002 -> 4,167 6-mer tokens (Max output)
					if region_end - region_start > maxLen:
						print(f"Skipping {geneModel} because gene length > {maxLen}", file=sys.stderr)
						region_start = None
						region_end = None
						geneModel = None
						skipGene = True
						continue

					# Get forward strand sequence
					sequence = genome_dict[chr][region_start:region_end]
					if addRC:
						sequence_rc = reverseComplement(sequence)
				
				elif skipGene:
					continue
				
				elif typ == 'mRNA':
					mRNA_list = mRNA_list[:-1] + ";"

				# Build gff string - start and end coordinates are relative to the gene model sense strand (inclusive on start, exclusive on end)
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

def gffString2GFF3(gff:str, chr:str, region_start:int) -> List[str]:
	# Purpose: 
	#      Convert a GFF string (decoded model output) to GFF3 fromat.
	# Inputs:
	#      gff: str, the GFF string
	#      chr: str, the chromosome name
	#      region_start: int, start coords of sequence region in the chromosome
	#      region_end: int, end coords of sequence region in the chromosome
	# Outputs:
	#      An array of strings for each GFF3 lines
	# TODO: Error handling for strange model outputs
	
	# Break out the gene features
	phaseLookup = {"A":0,"B":1,"C":2,".":"."}
	gff = gff.replace ("<s>", "").replace("|</s>", "").replace("</s>", "")
	
	# Coordinates are 0-indexed and need to be 1 indexed
	region_start += 1
	
	try:
		features,transcripts = gff.split(">")
		features = [feat.split("|") for feat in features.split(";")]
		features = {feat[1]:feat for feat in features}
		transcripts = transcripts.split(";")
		geneStrand = features["CDS1"][3]
	except:
		print(f"Error parsing GFF string, skipping.\n{gff}", file=sys.stderr)
		return [""]
	
	# Get the gene start and end for each transcript
	transcriptBounds = []
	for transcript in transcripts:
		geneStart = None
		geneEnd = None
		for feat in transcript.split("|"):	
			if geneStart is None:
				geneStart = int(features[feat][0])
			elif int(features[feat][0]) < geneStart:
				geneStart = int(features[feat][0])
			
			if geneEnd is None:
				geneEnd = int(features[feat][2])
			elif int(features[feat][2]) > geneEnd:
				geneEnd = int(features[feat][2])
		transcriptBounds.append((geneStart+region_start, geneEnd+region_start))
	
	# Construct the GFF3 string
	import uuid
	id = str(uuid.uuid4())
	geneStart = min([x[0] for x in transcriptBounds])
	geneEnd = max([x[1] for x in transcriptBounds])
	geneModel = [f"{chr}\ttransgenic\tgene\t{geneStart}\t{geneEnd-1}\t.\t{geneStrand}\t.\tID={id}"]
	
	# Add mRNA models
	for i,transcript in enumerate(transcripts):
		geneModel.append(f"{chr}\ttransgenic\tmRNA\t{transcriptBounds[i][0]}\t{transcriptBounds[i][1]-1}\t.\t{geneStrand}\t.\tID={id}.t{i+1};Parent={id}")
		transcript = transcript.split("|")
		for featureID in transcript:
			if featureID == "":
				continue
			featureModel = features[featureID]
			featureType = re.sub(r'\d+', '', featureID)
			featureNum = re.sub(r'\D+', '', featureID)
			geneModel.append(f"{chr}\ttransgenic\t{featureType}\t{int(featureModel[0])+region_start}\t{int(featureModel[2])+region_start-1}\t.\t{geneStrand}\t{phaseLookup[featureModel[4]]}\tID={id}.t{i+1}.{featureType}{featureNum};Parent={id}.t{i+1}")
	
	return geneModel

def segmentSequence(seq, piece_size = 4092):
	# Segment the sequence into evenly sized chunks smaller than 4092bp (encoder max length of 1024 tokens)
	seqs = [seq[i:min(i+piece_size, len(seq))] for i in range(0, len(seq), piece_size)]
	return seqs

def setup(rank, world_size, pg="gloo"):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group(pg, rank=rank, world_size=world_size) # initialize the process group

def cleanup():
	if dist.is_initialized():
		dist.destroy_process_group()

def target_collate_fn(batch):
	# Unpack the batch items (each item in batch is a tuple of (sequence, attention_mask, target_sequence))
	sequences, attention_masks, global_attention_masks, labels, gm, chr, region_start, region_end= zip(*batch)

	# Pad and stack the sequences
	max_segs = max([seq.shape[0] for seq in sequences])
	sequences = [seq.flatten() for seq in sequences]
	sequences = [F.pad(seq, (0, max_segs*1024 - seq.shape[0]), value=1) for seq in sequences]
	sequences = torch.stack(sequences)
	
	
	#Pad and stack the attention masks
	attention_masks = [mask.flatten() for mask in attention_masks]
	attention_masks = [F.pad(mask, (0, max_segs*1024 - mask.shape[0]), value=False) for mask in attention_masks]
	attention_masks = torch.stack(attention_masks)

	if global_attention_masks[0] is not None:
		#Pad and stack the global attention masks
		global_attention_masks = [mask.flatten() for mask in global_attention_masks]
		global_attention_masks = [F.pad(mask, (0, max_segs*1024 - mask.shape[0]), value=False) for mask in global_attention_masks]
		global_attention_masks = torch.stack(global_attention_masks)

	# Pad and stack the labels
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]
		labels_padded = torch.cat(labels_padded)

	if labels:
		return sequences, attention_masks, global_attention_masks, labels_padded, gm, chr, region_start, region_end
	else:
		return sequences, attention_masks, global_attention_masks, None, gm, chr, region_start, region_end

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

def getSpliceBranchSeqs() -> List[str]:
	# Purpose:
	#      Generate a list of splice branch sequences for use in the model
	# Outputs:
	#      List[str], the splice branch sequences
	
	from itertools import product

	# Define the possible nucleotides for each position in the motif
	motif_positions = {
		'Y': ['C', 'T'],
		'N': ['A', 'C', 'G', 'T'],
		'R': ['A', 'G'],
		'A': ['A'],
		'U': ['T']}

	# Generate all possible motif sequences by creating a list of possible nucleotides for each position
	motif = 'YNYURAY'
	possible_nucleotides = [motif_positions[char] for char in motif]
	all_possible_sequences = [''.join(seq) for seq in product(*possible_nucleotides)]

	seqs = []
	for sequence in all_possible_sequences:
		seqs.append(sequence)
	rc = [reverseComplement(seq) for seq in seqs]
	return {"forward":seqs, "reverse":rc}

def scanGlobalAttentionTokens(vocab:dict, tokenized_sequence:List[int], seqlen:int) -> List[int]:
	# Purpose:
	#      Scan a tokenized sequence for sequence feartures which can be used for setting
	#      transient global attention in the model. Features include canonical and non-canonical 
	#      splice donor sites and spliceosome branch sequences. Splice donor sites are allowed to span two 
	#      tokens to account for split tokens and branch points must span two tokens. Both forward and 
	#      reverse strands are scanned and one direction is chosen based on the orientation with the most
	#      identified tokens.
	# Inputs:
	#      vocab: dict, the token to id dictionary
	#      tokenized_sequence: List[int], the tokenized sequence
	# Outputs:
	#      List[int], the positions of the global attention tokens

	# Donor splice sites in both directions
	spliceSites_fwd = ["AGGT", "AGGC","AGAT"]
	spliceSites_rev = 	["ACCT","GCCT","ATCT"]
	
	# Branch sequences in both directions
	spliceBranchSites = getSpliceBranchSeqs()
	spliceBranchSites_fwd = spliceBranchSites["forward"]
	spliceBranchSites_rev = spliceBranchSites["reverse"]

	# Swap vocab dictionary to id to allow for fast lookup
	id2vocab = {v:k for k,v in vocab.items()}

	# Initialize global attention masks
	global_attention_mask_fwd = [0 for i in range(len(tokenized_sequence))]
	global_attention_mask_rev = [0 for i in range(len(tokenized_sequence))]
	
	fwd_splice_count = 0
	rev_splice_count = 0
	fwd_branch_count = 0
	rev_branch_count = 0
	for i,token in enumerate(tokenized_sequence):
		if i == 0:
			continue
		
		double_token = f"{id2vocab[token]}{id2vocab[tokenized_sequence[i-1]]}"
		
		# Scan for forward donor splice sites
		for site in spliceSites_fwd:
			if site in id2vocab[token]:
				global_attention_mask_fwd[i] = 1 
				fwd_splice_count += 1
				break
			elif (site in double_token) and (site not in id2vocab[tokenized_sequence[i-1]]):
				global_attention_mask_fwd[i] = 1
				global_attention_mask_fwd[i-1] = 1
				fwd_splice_count += 1
				break
		# Scan for reverse donor splice sites
		for site in spliceSites_rev:
			if site in id2vocab[token]:
				global_attention_mask_rev[i] = 1 
				rev_splice_count += 1
				break
			elif (site in double_token) and (site not in id2vocab[tokenized_sequence[i-1]]):
				global_attention_mask_rev[i] = 1
				global_attention_mask_rev[i-1] = 1
				rev_splice_count += 1
				break

		# Scan for forward splice branch sequences
		for site in spliceBranchSites_fwd:
			if site in double_token:
				global_attention_mask_fwd[i] = 1
				global_attention_mask_fwd[i-1] = 1
				fwd_branch_count += 1
				break
		# Scan for reverse splice branch sequences
		for site in spliceBranchSites_rev:
			if site in double_token:
				global_attention_mask_rev[i] = 1
				global_attention_mask_rev[i-1] = 1
				rev_branch_count += 1
				break
	
	# TODO: is there a heurisitic we can use to determine the correct direction?
	picked = []
	for x in zip(global_attention_mask_fwd, global_attention_mask_rev):
		if sum(x) >= 1:
			picked.append(1)
		else:
			picked.append(0)

	return segmentSequence(picked, piece_size=1024)

# geneList custom dataset class for use with DataLoader
class isoformData(Dataset):
	def __init__(self, db, dt, mode="inference", encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", global_attention=False):
		self.db = db
		self.mode = mode
		self.dt = dt
		self.global_attention = global_attention
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)
		if dt != None:
			self.decoder_tokenizer = dt
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
			gm,region_start,region_end,strand,chr,region_seq, gff,_ = con.sql(f"SELECT * FROM geneList where rn={idx}").fetchall()[0]
		
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
		seqs = segmentSequence(region_seq, piece_size=4002)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 1024)["input_ids"]
		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		# Scan for global attention tokens
		if self.global_attention:
			global_attention_mask = scanGlobalAttentionTokens(self.encoder_tokenizer.get_vocab(), seqs.flatten().tolist(), int(region_end)-int(region_start))
			global_attention_mask = torch.LongTensor(global_attention_mask)
		else:
			global_attention_mask = None

		if self.mode == "training":
			return (seqs, encoder_attention_mask, global_attention_mask, labels, gm, chr, region_start, region_end)
		else:
			return (seqs, encoder_attention_mask, global_attention_mask, None, gm, chr, region_start, region_end)
	
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

class GFFTokenizer09(PreTrainedTokenizer):
	model_input_names = ["input_ids", "attention_mask"]

	def __init__(self, vocab=None, **kwargs):
		if vocab is None:
			self.vocab = {
				"<s>": 0, "<pad>": 1,"</s>":2, "<unk>":3, '0': 4, '1': 5, '2': 6, 
				'3': 7,'4': 8, '5': 9, '6': 10, '7': 11, '8': 12, 
				'9': 13, 'A': 14, 'B': 15, 'C': 16, ">":17, ".": 18, 
				"+": 19, "-": 20, ";":21
			}
			for i in range(1, 151):
				self.vocab[f"CDS{i}"] = i + 21
			for i in range(1, 51):
				self.vocab[f"five_prime_UTR{i}"] = i + 171
				self.vocab[f"three_prime_UTR{i}"] = i + 221
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
						tokens.extend([digit for digit in column])
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

class HybridSequenceLoss(nn.Module):
	# A custom loss function to combine CrossEntropy with MSE loss for numeric sequences.
	# The MSE is computed independently for each numeric sequence in the target.
	def __init__(self, num_weight=1.0):
		super(HybridSequenceLoss, self).__init__()
		self.tokenizer = GFFTokenizer09()
		self.num_token_ids = self.tokenizer.numeric_tokens
		self.num_weight = num_weight
		self.ce_loss = nn.CrossEntropyLoss(reduction='none')
		self.mse_loss = nn.MSELoss()

	def is_numeric(self, token_id):
		return token_id in self.num_token_ids

	def forward(self, outputs, targets):
		batch_size = outputs.size(0)
		seq_len = outputs.size(1)
		
		# Calculate CrossEntropy loss for each token
		ce_loss = self.ce_loss(outputs.view(-1, self.tokenizer.vocab_size), targets.view(-1))
		ce_loss = ce_loss.view(batch_size, seq_len)
		
		# Initialize masks and MSE loss
		total_mse_loss = 0.0
		num_numeric_sequences = 0

		# Iterate over each sequence in the batch
		for i in range(batch_size):
			output_seq = torch.argmax(outputs.view(-1, self.tokenizer.vocab_size), dim=1) #outputs[i]
			target_seq = targets[i]

			# Find all numeric sequences within target
			numeric_sequences_pred = []
			numeric_sequences_true = []
			current_seq_pred = []
			current_seq_true = []
			for j in range(seq_len):
				if self.is_numeric(target_seq[j].item()):
					current_seq_pred.append(self.tokenizer.decode(output_seq[j])) 
					current_seq_true.append(self.tokenizer.decode(target_seq[j]))
				else:
					if current_seq_pred:
						#TODO: handle cases when the sequence contains non-numeric tokens... '177' vs '17-'
						numeric_sequences_pred.append(int("".join(current_seq_pred)))
						current_seq_pred = []
					if current_seq_true:
						numeric_sequences_true.append(int("".join(current_seq_true)))
						current_seq_true = []
			if current_seq_pred:
				numeric_sequences_pred.append(current_seq_pred)
			if current_seq_true:
				numeric_sequences_true.append(current_seq_true)

			# Calculate MSE loss for each numeric sequence
			# TODO: these are too high?
			mse_loss = self.mse_loss(torch.tensor(numeric_sequences_pred, dtype=torch.float32),
									 torch.tensor(numeric_sequences_true, dtype=torch.float32))
			total_mse_loss += mse_loss
			num_numeric_sequences += 1

		if num_numeric_sequences > 0:
			total_mse_loss /= num_numeric_sequences

		# Combine losses
		total_loss = ce_loss.mean() + self.num_weight * total_mse_loss
		return total_loss

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

class NumberMaskEmbedTokens(nn.Embedding):
	def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
		super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
		self.num_feature_embed = nn.Embedding(2, embedding_dim)

	def forward(self, input_ids):
		# Create a mask for numerical tokens and embed it
		num_mask = self.create_numerical_mask(input_ids)
		num_mask = self.num_feature_embed(num_mask)
		
		# Generate embeddings for the input tokens
		num_embeddings = super().forward(input_ids)

		# Add the numerical mask to the embeddings
		num_embeddings = num_embeddings + num_mask
		
		return num_embeddings

	def create_numerical_mask(self, input_ids):
		# Create a binary mask where 1 indicates a numerical token and 0 otherwise
		num_mask = torch.zeros_like(input_ids, dtype=torch.long)
		for i, sentence in enumerate(input_ids):
			for j, token in enumerate(sentence):
				if (token.item() >= 4) & (token.item() <= 13):
					num_mask[i,j] = 1
		return num_mask

class segmented_sequence_embeddings(EsmForMaskedLM):
	def __init__(self, model):
		self.cache_dir = "./HFmodels"
		self.encoder_model = model
		config = AutoConfig.from_pretrained(self.encoder_model, is_decoder=False, trust_remote_code=True)
		super().__init__(config)
		
		self.esm = AutoModelForMaskedLM.from_pretrained(self.encoder_model, cache_dir=self.cache_dir, trust_remote_code=True)
		
		#print(self.led.print_trainable_parameters(), file=sys.stderr)
		#for param in self.esm.parameters():
		#	param.requires_grad = False
		#for param in self.esm.lm_head.parameters():
		#	param.requires_grad = False

		# TODO: Exlpore other options? (hidden states, BiLSTM, linear, attention, pooling, convolution)
		#plants -> 1500, multispecies -> 1024
		#T5 -> 512, longformer -> 768
		self.hidden_mapping = nn.Linear(config.hidden_size, 768)
		self.hidden_mapping_layernorm = nn.LayerNorm(768, eps=1e-5)
	
	def forward(self, input_ids, attention_mask=None, **kwargs):
		batch_size = input_ids.shape[0]
		num_segs = input_ids.shape[1] // 1024
		input_ids = input_ids.reshape(batch_size, int(num_segs), 1024)
		attention_mask = attention_mask.reshape(batch_size, int(num_segs), 1024)
		for i in range(batch_size):
			#with torch.no_grad():
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
	_tied_weights_keys = ["decoder_embed_tokens.weight", "decoder.embed_tokens.weight"]

	def __init__(self, config: LEDConfig, encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"):
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		#self.decoder_embed_tokens = NumberMaskEmbedTokens(vocab_size, config.d_model, padding_idx)
		self.decoder_embed_tokens = nn.Embedding(vocab_size, config.d_model, padding_idx)
		
		self.encoder = segmented_sequence_embeddings(encoder_model)
		self.decoder = LEDForConditionalGeneration(config).led.decoder
		self.decoder.embed_tokens = self.decoder_embed_tokens

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
			global_attention_mask=global_attention_mask,
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
	_tied_weights_keys = ["transgenic.decoder_embed_tokens.weight", "lm_head.weight"]

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

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		masked_lm_loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
			#loss_fct = HybridSequenceLoss()
			#masked_lm_loss = loss_fct(lm_logits, labels)

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
		checkpoint_path,
		safetensors_model,
		encoder_model,
		unlink,
		accumulation_steps,
		max_grad_norm):
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
	config = LEDConfig.from_pretrained("allenai/led-base-16384", 
									vocab_size=272, 
									max_decoder_position_embeddings=2048,
									decoder_layerdrop=0.1,
									dropout= 0.1)
	model = transgenicForConditionalGeneration(config, encoder_model=encoder_model, unlink=unlink)

	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
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

	# Add gradients back for the entire decoder and the hidden mapping layers
	for param in model.transgenic.decoder.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	model.print_trainable_parameters()

	# Load checkpoint if provided
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["base_model.model.lm_head.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
		tensors["base_model.model.transgenic.decoder.embed_tokens.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
		model.load_state_dict(tensors)
	model.to(device)
	ddp_model = DDP(model, find_unused_parameters=True)
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

	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			outputs = ddp_model(input_ids=batch[0], attention_mask=batch[1], global_attention_mask=batch[2], labels=batch[3], return_dict=True)
			loss = outputs.loss / accumulation_steps
			loss.backward()
			total_loss += outputs.loss.detach().float()
			if (step+1) % accumulation_steps == 0:
				clip_grad_norm_(ddp_model.model.parameters(), max_grad_norm)
				optimizer.step()
				if schedule_lr: lr_scheduler.step()
				# log metrics to wandb
				wandb_log = {"epoch":epoch, "step":step, "loss": outputs.loss.detach().float(), "mean_loss": total_loss / (step+1), "lr": lr_scheduler.get_last_lr()[0]}
				for name, param in ddp_model.model.named_parameters():
					if (param.grad != None) & (param.requires_grad):
						grad_norm = param.grad.norm().item()
						wandb_log[f"{name}_grad_norm"] = grad_norm
				wandb.log(wandb_log)
				optimizer.zero_grad()
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch=}, Step {step=}, Loss {loss=}", file=sys.stderr)
				if torch.distributed.get_rank() == 0:
					torch.save({'state_dict': ddp_model.state_dict()}, checkpoint_path)

		train_epoch_loss = total_loss / len(train_ds)
		train_ppl = torch.exp(train_epoch_loss)

		if eval:
			model.eval()
			eval_loss = 0
			for batch in tqdm(eval_ds, miniters=10, disable=False):
				with torch.no_grad():
					outputs = model(input_ids=batch[0], attention_mask=batch[1], global_attention_mask=batch[2], labels=batch[3], return_dict=True)
				loss = outputs.loss
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
				if torch.distributed.get_rank() == 0:
					torch.save({'state_dict': ddp_model.state_dict()}, checkpoint_path)
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				if torch.distributed.get_rank() == 0:
					torch.save({'state_dict': ddp_model.state_dict()}, checkpoint_path)
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
	
	if torch.distributed.get_rank() == 0:
		torch.save({'state_dict': ddp_model.state_dict()}, checkpoint_path)
	wandb.finish()

def run_trainTransgenicDDP(
		train_ds:isoformData, 
		eval_ds:isoformData, 
		lr=1e-4, 
		num_epochs=10,  
		schedule_lr=True, 
		eval=True, 
		world_size=1,
		batch_size=1,
		checkpoint_path="checkpoints/transgenic_checkpoint.pt",
		safetensors_model=None,
		encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
		unlink=False,
		accumulation_steps=16,
		max_grad_norm=1.0):
	ws = torch.cuda.device_count()
	if ws == 0:
		ws = world_size
	mp.spawn(trainTransgenicDDP, 
		args=(train_ds, 
		eval_ds, 
		lr, 
		num_epochs,  
		schedule_lr, 
		eval, 
		ws,
		batch_size,
		checkpoint_path,
		safetensors_model,
		encoder_model,
		unlink,
		accumulation_steps,
		max_grad_norm), 
		nprocs=ws, 
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
	ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.gradient_accumulation_steps > 1)
	accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # gradient_accumulation_steps=32
	device = accelerator.device
	print(f"Training transgenic with Accelerate on {device}", file=sys.stderr)
	
	# Set up DataLoaders
	torch.manual_seed(345)
	torch.cuda.manual_seed_all(345)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	config = LEDConfig.from_pretrained("allenai/led-base-16384", 
									vocab_size=272, 
									max_decoder_position_embeddings=2048,
									decoder_layerdrop=0.1,
									dropout= 0.1)
	model = transgenicForConditionalGeneration(config, 
											encoder_model=encoder_model, 
											unlink=unlink)
	
	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
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

	# Add gradients back for the entire decoder and the hidden mapping layers
	for param in model.transgenic.decoder.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	model.print_trainable_parameters()
	if f"{device}" != "cpu":
		try:
			model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
		except:
			print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# Load checkpoint if provided
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["base_model.model.lm_head.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
		tensors["base_model.model.transgenic.decoder.embed_tokens.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
		#tensors["base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight"] = model.base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight
		#tensors["base_model.model.transgenic.decoder.embed_tokens.num_feature_embed.weight"] = model.base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight
		if "base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight" in tensors.keys():
			del tensors["base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight"]
		model.load_state_dict(tensors)
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
	model, optimizer, train_ds, eval_ds, lr_scheduler = accelerator.prepare(
		model, optimizer, train_ds, eval_ds, lr_scheduler
	)
	
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			#with accelerator.accumulate(model):
			outputs = model(input_ids=batch[0], attention_mask=batch[1], global_attention_mask=batch[2], labels=batch[3], return_dict=True)
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
					outputs = model(input_ids=batch[0], attention_mask=batch[1], global_attention_mask=batch[2], labels=batch[3], return_dict=True)
				loss = outputs.loss
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
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
		
		torch.cuda.empty_cache()
		gc.collect()
		total_loss = 0
		loss = None
		train_epoch_loss = 0
		train_ppl = 0

	
	accelerator.wait_for_everyone()
	accelerator.save_model(model, output_dir)
	wandb.finish()


def trainTransgenicFCGCPU(
	train_ds:isoformData, 
	eval_ds:isoformData, 
	lr, 
	num_epochs,  
	schedule_lr, 
	eval, 
	batch_size,
	max_grad_norm=1.0,
	checkpoint_path="checkpoints_CPU/",
	safetensors_model=None,
	output_dir="saved_models_CPU/",
	accumulation_steps=32,
	notes="",
	encoder_model="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
	unlink=False,
	log_wandb=False):

	if log_wandb:
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
	

	print(f"Training transgenic on cpu", file=sys.stderr)
	
	# Set up DataLoaders
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	if eval_ds:
		eval_ds = makeDataLoader(eval_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	config = LEDConfig.from_pretrained("allenai/led-base-16384", 
									vocab_size=272, 
									max_decoder_position_embeddings=2048,
									decoder_layerdrop=0.1,
									dropout= 0.1)
	model = transgenicForConditionalGeneration(config, encoder_model=encoder_model, unlink=unlink)
	
	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
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

	# Add gradients back for the entire decoder and the hidden mapping layers
	for param in model.transgenic.decoder.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	model.print_trainable_parameters()

	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["base_model.model.lm_head.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
		tensors["base_model.model.transgenic.decoder.embed_tokens.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
		#tensors["base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight"] = model.base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight
		#tensors["base_model.model.transgenic.decoder.embed_tokens.num_feature_embed.weight"] = model.base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight
		model.load_state_dict(tensors)
	#model.to(device)
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
	from torch.profiler import profile, record_function, ProfilerActivity
	# Training loop
	best_eval_score = None
	for epoch in range(num_epochs):
		total_loss = 0
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
				with record_function("model_training"):
					outputs = model(input_ids=batch[0], attention_mask=batch[1], global_attention_mask=batch[2], labels=batch[3], return_dict=True)
					loss = outputs.loss / accumulation_steps
					loss.backward()
			
					total_loss += outputs.loss.detach().float()
					if (step+1) % accumulation_steps == 0:
						clip_grad_norm_(model.parameters(), max_grad_norm)
						optimizer.step()
						if schedule_lr: lr_scheduler.step()
						if log_wandb:
							wandb_log = {"epoch":epoch, "step":step, "loss": outputs.loss.detach().float(), "mean_loss": total_loss / (step+1)}#, "lr": lr_scheduler.get_last_lr()[0]}
							for name, param in model.named_parameters():
								if (param.grad != None) & (param.requires_grad):
									grad_norm = param.grad.norm().item()
									wandb_log[f"{name}_grad_norm"] = grad_norm
							for name, param in model.lm_head.named_parameters():
								if (param.grad != None) & (param.requires_grad):
									grad_norm = param.grad.norm().item()
									wandb_log[f"{name}_grad_norm"] = grad_norm
							wandb.log(wandb_log)
						#if accelerator.is_main_process: plot_grad_flow(model, outprefix=f"{checkpoint_path}/grad_flow")
						optimizer.zero_grad()
			#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
			#print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
			break
			
			
			if (step % 5000 == 0) & (step != 0):
				print(f"Epoch {epoch=}, Step {step=}, Loss {loss=}", file=sys.stderr)
				#accelerator.save_state(output_dir=checkpoint_path)

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
				#accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {eval_epoch_loss=}", file=sys.stderr)
		else:
			if best_eval_score is None or train_epoch_loss < best_eval_score:
				best_eval_score = train_ppl
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints", exist_ok=True)
				#accelerator.save_state(output_dir=checkpoint_path)
				print(f"New best model saved with {train_epoch_loss=}", file=sys.stderr)
	
	#accelerator.wait_for_everyone()
	#accelerator.save_model(model, output_dir)
	wandb.finish()
	return model

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
	
	# Load the model and add to device
	config = LEDConfig.from_pretrained("allenai/led-base-16384", 
									vocab_size=272, 
									max_decoder_position_embeddings=2048,
									decoder_layerdrop=0.1,
									dropout= 0.1)
	model = transgenicForConditionalGeneration(config, encoder_model=encoder_model, unlink=unlink)
	
	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
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

	# Load checkpoint
	tensors = {}
	with safe_open(model_path, framework="pt", device="cpu") as f:
		for k in f.keys():
			tensors[k] = f.get_tensor(k)
	tensors["base_model.model.lm_head.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
	tensors["base_model.model.transgenic.decoder.embed_tokens.weight"] = tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
	if "base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight" in tensors.keys():
		del tensors["base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight"]
	model.load_state_dict(tensors)
	model.to(device)
	model.eval()
	
	# Prep objects for use with accelerator
	model, dataset = accelerator.prepare(
		model, dataset
	)

	# Prediction loop
	dt = GFFTokenizer09()
	predictions = []
	for step, batch in enumerate(tqdm(dataset)):
		with torch.no_grad():
			outputs = model.module.generate(inputs=batch[0], 
											attention_mask=batch[1], 
											max_length=2048,
											#num_return_sequences=1,
											#num_beams=4,         # Using beam search
											#do_sample=True,      # Enable sampling
											#early_stopping=True  # Stop early if all beams finished
											#top_k=10,            # Top-k sampling
											#top_p=0.95,          # Top-p (nucleus) sampling
											#temperature=0.7,     # Temperature scaling
											#length_penalty=2.0  # Length penalty to avoid short sequences
											#penalty_alpha=0.6, top_k=4
											)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
		true = dt.batch_decode(batch[3].reshape(batch_size,batch[3].size()[-1]).detach().cpu().numpy(), skip_special_tokens=True)
		predictions.append([batch[3], outputs.detach().cpu(), true[0], pred[0], batch[4]])	
	
		with open(f"{device}_{outfile}.txt", 'w') as out:
			for prediction in predictions:
				out.write("\t".join(str(prediction))+"\n")

		with open(f"{device}_{outfile}.pkl", 'wb') as out:
			pickle.dump(predictions, out)

def testTransgenicAccelerate(
		safetensors_model:str, 
		test_ds:isoformData, 
		encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", 
		unlink=False, 
		batch_size=1,
		outfile="test_results.pkl"):
	print(f"Testing transgenic using {safetensors_model=}", file=sys.stderr)
	
	# Set up accelerator
	ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.gradient_accumulation_steps > 1)
	accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # gradient_accumulation_steps=32
	device = accelerator.device
	print(f"Accelerate active on {device}", file=sys.stderr)
	outfile = outfile.split(".")[0]+"_"+str(device)+".pkl"
	
	# Set up DataLoader
	test_ds = makeDataLoader(test_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	config = LEDConfig.from_pretrained("allenai/led-base-16384", 
									vocab_size=372, 
									max_decoder_position_embeddings=2048,
									decoder_layerdrop=0.1,
									dropout= 0.1)
	model = transgenicForConditionalGeneration(config, encoder_model=encoder_model, unlink=unlink)
	# Load checkpoint
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["lm_head.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]
		tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]
		model.load_state_dict(tensors)
	model.to(device)
	
	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
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
	peft_config = IA3Config(task_type="SEQ_2_SEQ_LM", target_modules = peft_targets, feedforward_modules = peft_feedforward)
	#model = get_peft_model(model, peft_config)
	model.eval()

	# Prep objects for use with accelerator
	model, test_ds = accelerator.prepare(
		model, test_ds
	)

	tokenizer = GFFTokenizer()
	import pandas as pd
	output_list = []
	
	# generation loop
	try:
		total_loss = 0
		for step, batch in enumerate(tqdm(test_ds, miniters=10, disable=False)):
			outputs = model(input_ids=batch[0], attention_mask=batch[1], global_attention_mask=batch[2], labels=batch[3], return_dict=True)
			loss = outputs.loss.detach().cpu().float()
			total_loss += loss
			generated_tokens = model.module.generate(input_ids=batch[0], attention_mask=batch[1], num_return_sequences=1, max_length=2048).cpu()
			generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
			output = outputs.logits.argmax(dim=-1).cpu()
			lab = tokenizer.batch_decode(batch[3].cpu(), skip_special_tokens=True)
			outp = tokenizer.batch_decode(output, skip_special_tokens=True)
			output_list.append({"loss":loss, "chromosome":batch[5], "region_start":batch[6], "region_end":batch[7], "geneModel":batch[4], "label_tokens":batch[3].cpu(), "output_tokens":output, "generated_tokens":generated_tokens,"label":lab, "output":outp, "generated":generated})
		avg_loss = total_loss / len(test_ds)
		print(f"Average loss: {avg_loss}", file=sys.stderr)
		with open(outfile, 'wb') as out:
			pickle.dump(output_list, out)
	except Exception as e:
		print(f"Error at step {step}", file=sys.stderr)
		print(e, file=sys.stderr)
		with open(outfile, 'wb') as out:
			pickle.dump(output_list, out)

def createDatabase(db="transgenic.db", mode="train", maxLen=49152, addExtra=0, addRC=False, addRCIsoOnly=False, clean=False):
	# Purpose: 
	#      Create or append to transgenic database. The database is a duckdb database 
	#      used for both either training or infrence. In 'train' mode, fasta sequences and 
	#      full GFF3 annotations are required. In 'infer' mode, fasta sequences
	#      and BED formatted annotations are required. 
	# Inputs:
	#      db: str, the name of the database. If exists, will append to it.
	#      mode: str, the mode of the database. Either "train" or "infer".
	#      maxLen: int, the maximum length of the input sequence.
	#	   addExtra: int, the number of extra bases to add to the ends of the input sequence.
	#      addRC: bool, whether to add the reverse complement of the input sequence.
	# TODO: 
	#      - Decide how to input filenames (add input argument)
	#      - Make BED-parsing DB creation function

	files = {
		"Athaliana_167_TAIR10.fa":"Athaliana_167_TAIR10.gene.clean.gff3",
		"Gmax_880_v6.0.fa":"Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
		#"Ppatens_318_v3.fa":"Ppatens_318_v3.3.gene_exons.clean.gff3",
		"Ptrichocarpa_533_v4.0.fa":"Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
		"Sbicolor_730_v5.0.fa":"Sbicolor_730_v5.1.gene_exons.clean.gff3",
		"Bdistachyon_314_v3.0.fa": "Bdistachyon_314_v3.1.gene_exons.clean.gff3",  
		"Sitalica_312_v2.fa": "Sitalica_312_v2.2.gene_exons.clean.gff3"
	}
	for fasta, gff in files.items():
		name = fasta.split("_")[0]
		print(f"Processing {name}...", file=sys.stderr)
		genome2GeneList("training_data/"+fasta, "training_data/"+gff, db=db, maxLen=maxLen, addExtra=addExtra, addRC=addRC, addRCIsoOnly=addRCIsoOnly, clean=clean)
		ds = isoformData(db, dt=GFFTokenizer09(), mode="training")
		length = len(ds)
		print(f"{name} {length=}")
	
def predictionToGFF3(device_type, world_size, suffix, outfile):
	for i in range(world_size):
		with open(f"{device_type}:{i}_{suffix}.pkl", 'rb') as f:
			predictions = pickle.load(f)
		with open(outfile, 'a') as out:
			for prediction in predictions:
				out.write(f"{prediction[0]}\t{prediction[1]}\t{prediction[2]}\t{prediction[3]}\t{prediction[4]}\t{prediction[5]}\t{prediction[6]}\t{prediction[7]}\n")

class PredictionProcessor():
	def __init__(self, gff:str, sequence:str):
		self.gff = gff
		self.sequence = sequence
		self.stopCodons = ["TAA", "TAG", "TGA"]
		self.spliceDonors = ["AGGT", "GGT","GT","AGGC", "AGAT","GGC","GAT"]
		#self.spliceAcceptors_highConfidence = ["AGG","AGG","ACG"]
		self.spliceAcceptors = ["AG","AC"]
		self.parsable = True

		# Remove nonsensical features from the prediction
		self.gff = self.tidyPrediction()

		if len(gff.split("-")) > len(gff.split("+")):
			self.strand = "-"
		else:
			self.strand = "+"
		
		if self.strand == '-':
			try:
				self.reverseComplement()
			except:
				print(f"Error: sequence could not be reverse complemented.", file=sys.stderr)
				self.parsable = False
				return
		
		# Get lists of features and transcripts
		try:
			feature_list, mRNA_list = self.gff.split('>')
			self.features = [f.split('|') for f in feature_list.split(';')]
			self.transcripts = [m.split('|') for m in mRNA_list.split(';')]
		except:
			print(f"Error: prediction could not be parsed.\n{self.gff}", file=sys.stderr)
			self.parsable = False
			return

		# Identify initial and final CDS for each transcript
		self.start_cds = [t[0] for t in self.transcripts]
		self.end_cds = []
		for transcript in self.transcripts:
			for i, typ in enumerate(transcript):
				if 'UTR' in typ:
					self.end_cds.append(transcript[i-1])
					break
				elif i == len(transcript)-1:
					self.end_cds.append(transcript[-1])

		# Get indexes of features in feature list for lookup
		self.feature_index_dict = {}
		for i, feature in enumerate(self.features):
			self.feature_index_dict[feature[1]] = i

		# Get pairs of start/stop codons and their corresponding 5'/3' UTRs
		# to update simultaneously
		self.start_pairs = []
		for i, cds in enumerate(self.start_cds):
			new_pair = (cds,None)
			for feature in self.transcripts[i]:
				if 'five_prime_UTR' in feature:
					new_pair = (cds, feature)
			self.start_pairs.append(new_pair)

		self.end_pairs = []
		for i, cds in enumerate(self.end_cds):
			new_pair = (cds,None)
			for feature in self.transcripts[i]:
				if 'three_prime_UTR' in feature:
					new_pair = (cds, feature)
					break
			self.end_pairs.append(new_pair)
	
		# Get cds pairs for splice junction validation
		self.cds_pairs = []
		for transcript in self.transcripts:
			for i, typ in enumerate(transcript):
				if i == 0:
					continue
				if 'CDS' in typ and (transcript[i-1], typ) not in self.cds_pairs:
					self.cds_pairs.append((transcript[i-1], typ))

	def tidyPrediction(self):
		types = ["CDS", "five_prime_UTR", "three_prime_UTR"]
		phases = ["A", "B", "C", "."]
		strands = ["+", "-"]

		try:
			feature_list, mRNA_list = self.gff.split('>')
			features = [f.split('|') for f in feature_list.split(';')]
			transcripts = [m.split('|') for m in mRNA_list.split(';')]
		except:
			print(f"Error: prediction could not be parsed.\n{self.gff}", file=sys.stderr)
			self.parsable = False
			return self.gff
		
		# Check features for valid types, start/end coordinates, phase, and strand
		try:
			validFeatures = []
			invalidFeatureIndexes = []
			for i, feature in enumerate(features):
				valid = True
				typeCount = [typ in feature[1] for typ in types]
				try:
					start = int(feature[0])
					end = int(feature[2])
				except:
					valid = False
					start = 0
					end = 0
				
				if start == end:
					valid = False
				elif feature[4] not in phases:
					valid = False
				elif feature[3] not in strands:
					valid = False
				elif sum(typeCount) < 1:
					valid = False
					
				if not valid:
					invalidFeatureIndexes.append(i)
				else:
					validFeatures.append(feature[1])
		except:
			print(f"Error: prediction could not be tidied.\n{self.gff}", file=sys.stderr)
			self.parsable = False
			return self.gff
		
		for i in sorted(invalidFeatureIndexes, reverse=True):
			del features[i]
		
		# Check transcripts for valid types
		for i, transcript in enumerate(transcripts):
			invalidFeatureIndexes = []
			for j,feature in enumerate(transcript):
				typCount = [typ in feature for typ in types]
				if sum(typCount) < 1:
					invalidFeatureIndexes.append(j)
				elif feature not in validFeatures:
					if j not in invalidFeatureIndexes:
						invalidFeatureIndexes.append(j)
				elif feature in transcript[0:j]:
					if j not in invalidFeatureIndexes:
						invalidFeatureIndexes.append(j)
			for entry in sorted(invalidFeatureIndexes, reverse=True):
				del transcripts[i][entry]
		
		# If transcript is empty, make one primary transcript
		for i,transcript in enumerate(transcripts):
			if len(transcript) == 0:
				transcripts[i] = [feature[1] for feature in features]

		
		return f"{';'.join(['|'.join(feature) for feature in features])}>{';'.join(['|'.join(transcript) for transcript in transcripts])}"

	def reverseComplement(self):
		self.sequence = reverseComplement(self.sequence)
		self.gff = reverseComplement_gffString(self.gff, len(self.sequence))
	
	# Check start_cds for start codons, if missing search sequence for new start codon
	# Use the start codon closest to the predicted coordinates
	def checkStartCodons(self, buff=50):
		for i, cds in enumerate(self.start_cds):
			start = int(self.features[self.feature_index_dict[cds]][0])
			end = int(self.features[self.feature_index_dict[cds]][2])
			if self.sequence[start:start+3] != "ATG":
				if end-start < 2*buff:
					buf = (end-start)//2
				else:
					buf = buff
				buffer = self.sequence[start-buf:start+buf]
				start_codons = [m.start() for m in re.finditer("ATG", buffer)]
				if start_codons:
					pick = int(np.argmin([abs(loc-buf) for loc in start_codons]))
					new_start = start + start_codons[pick]-buf
					self.features[self.feature_index_dict[cds]][0] = str(new_start)
			
		for cds, utr in self.start_pairs:
			if utr:
				self.features[self.feature_index_dict[utr]][2] = self.features[self.feature_index_dict[cds]][0]

	# Check end_cds for stop codons, if missing search sequence for new stop codon
	# Use the stop codon closest to the predicted coordinates
	def checkStopCodons(self, buff=50):
		for i, cds in enumerate(self.end_cds):
			start = int(self.features[self.feature_index_dict[cds]][0])
			end = int(self.features[self.feature_index_dict[cds]][2])
			if self.sequence[end-3:end] not in self.stopCodons:
				if end-start < 2*buff:
					buf = (end-start)//2
				else:
					buf = buff
				buffer = self.sequence[end-buf:end+buf]
				stop_codons = [m.end() for m in re.finditer("|".join(self.stopCodons), buffer)]
				if stop_codons:
					pick = int(np.argmin([abs(loc-buf) for loc in stop_codons]))
					new_end = end + stop_codons[pick]-buf
					self.features[self.feature_index_dict[cds]][2] = str(new_end)

		for cds, utr in self.end_pairs:
			if utr:
				self.features[self.feature_index_dict[utr]][0] = self.features[self.feature_index_dict[cds]][2]
	
	# Used to check if a subsequence from cds_pairs makes internal stop codons
	def containsStopCodon(self, p):
		if self.cds_pairs.index(p)+1 == len(self.cds_pairs):
			return False
		f = []
		for pair in self.cds_pairs[0:self.cds_pairs.index(p)+1]:
			for feature in pair:
				if feature not in f:
					f.append(feature)
		f = [self.features[self.feature_index_dict[feature]] for feature in f]
		seq = ''
		for feat in f:
			seq += self.sequence[int(feat[0]):int(feat[2])]
		seq = [seq[i:min(i+3, len(seq))] for i in range(0, len(seq), 3)]
		return "TAG" in seq or "TAA" in seq or "TGA" in seq

	def checkSpliceJunctions(self, buff=50):
		# Use known splice junctions to adjust feature cordinates.
		# Search for donor and acceptor splice sites
		# Find the highest quality donor site closest to prediction. Preference order (AGGT/GGT/AGGC/AGAT/GGC/GAT)
		# Select the matching acceptor site closest to the prediction which does not introduce in-frame stop codons
		# buff: search buffer for cds boundary markers
		# note: decisionTensors are (2, #donor sites,#acceptor sites). Dimension 0[0] is stop codon presence, Dimension 0[1] frame congruence
		
		# TODO: Need a way to mitigate the cumulative effect of incorrectly predicted splice junctions when using phase
		#       to update the prediction. I need to balance the phase correction approach with the fact that some phase
		#       calls will be wrong and I will not find the correct splice junction everytime.

		self.sjLog = {}
		for f in self.features:
			if "CDS" in f[1]:
				self.sjLog[f[1]] = {"donor":{
									"AGGT":[],
									"GGT":[],
									"GT":[],
									"AGGC":[],
									"AGAT":[],
									"GGC":[],
									"GAT":[],
									"chosen":None}
								  ,"acceptor":{
									"AG":[],
									"AC":[],
									"chosen":None}
							  }
		
		for donor, acceptor in self.cds_pairs:
			donor_start = int(self.features[self.feature_index_dict[donor]][0])
			donor_end = int(self.features[self.feature_index_dict[donor]][2])
			acceptor_start = int(self.features[self.feature_index_dict[acceptor]][0])
			acceptor_end = int(self.features[self.feature_index_dict[acceptor]][2])

			#Find closest donor sites
			if donor_end-donor_start < buff:
				buf = donor_end-donor_start
			else:
				buf = buff
			buffer = self.sequence[donor_end-buf:donor_end+buff]
			for site in self.spliceDonors:
				HC = [m.end() for m in re.finditer(site, buffer)]
				pick = [loc-buf-2 for loc in HC]
				pick = sorted(pick, key=abs)
				self.sjLog[donor]["donor"][site] = pick

			# Find closest acceptor sites
			if acceptor_end-acceptor_start < buff:
				buf = acceptor_end-acceptor_start
			else:
				buf = buff
			buffer = self.sequence[acceptor_start-buff:acceptor_start+buf]
			for site in self.spliceAcceptors:
				HC = [m.start() for m in re.finditer(site, buffer)]
				pick = [loc-buf+2 for loc in HC]
				pick = sorted(pick, key=abs)
				self.sjLog[acceptor]["acceptor"][site] = pick

			
			# Select the highest quality donor site with phase correction
			phase_lookup = {"A":0, "B":1,"C":2}
			moveOn = False
			if len(self.sjLog[donor]["donor"]["AGGT"]) > 0:
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "AGGT", "AG")
			if (len(self.sjLog[donor]["donor"]["GGT"]) > 0) & (not moveOn):
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "GGT", "AG")
			if (len(self.sjLog[donor]["donor"]["GT"]) > 0) & (not moveOn):
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "GT", "AG")
			if (len(self.sjLog[donor]["donor"]["AGGC"]) > 0) & (not moveOn):
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "AGGC", "AG")
			if (len(self.sjLog[donor]["donor"]["AGAT"]) > 0) & (not moveOn):
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "AGAT", "AC")
			if (len(self.sjLog[donor]["donor"]["GGC"]) > 0) & (not moveOn):
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "GGC", "AG")
			if (len(self.sjLog[donor]["donor"]["GAT"]) > 0) & (not moveOn):
				moveOn = self.selectDonorByPhase(donor, donor_end, acceptor, "GAT", "AC")
			
			# If no phase correction, select the closest matching donor site
			if not moveOn:
				if len(self.sjLog[donor]["donor"]["AGGT"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "AGGT"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AG"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["AGGT"][0]))
				elif len(self.sjLog[donor]["donor"]["GGT"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "GGT"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AG"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["GGT"][0]))
				elif len(self.sjLog[donor]["donor"]["GT"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "GT"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AG"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["GT"][0]))
				elif len(self.sjLog[donor]["donor"]["AGGC"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "AGGC"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AG"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["AGGC"][0]))
				elif len(self.sjLog[donor]["donor"]["AGAT"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "AGAT"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AC"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["AGAT"][0]))
				elif len(self.sjLog[donor]["donor"]["GGC"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "GGC"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AG"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["GGC"][0]))
				elif len(self.sjLog[donor]["donor"]["GAT"]) > 0:
					self.sjLog[donor]["donor"]["chosen"] = "GAT"
					self.sjLog[acceptor]["acceptor"]["chosen"] = "AC"
					self.features[self.feature_index_dict[donor]][2] = str(int(donor_end + self.sjLog[donor]["donor"]["GAT"][0]))

			# Select the closest matching acceptor site, then scan for those which do not introduce stop codons
			# If no acceptor has been found, don't update
			try:
				closestAcceptor = self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]].pop(0)
			except:
				closestAcceptor = None

			if closestAcceptor:
				self.features[self.feature_index_dict[acceptor]][0] = str(int(acceptor_start + closestAcceptor))
				if self.containsStopCodon((donor,acceptor)):
					if len(self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]]) > 1:
						while self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]]:
							self.features[self.feature_index_dict[acceptor]][0] = str(int(acceptor_start + self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]].pop(0)))
							if not self.containsStopCodon((donor,acceptor)):
								break

	def selectDonorByPhase(self, donor:str, donor_end:int, acceptor:str, dsite:str, asite:str) -> bool:
		# Select the donor site with the lowest distance to the predicted site which maintains phase in the first transcript
		phase_lookup = {"A":0, "B":2,"C":1, ".":0} # modulo of cumulative previous sequence 
		moveOn = False
		if len(self.sjLog[donor]["donor"][dsite]) > 0:
			chosen = None
			for loc in self.sjLog[donor]["donor"][dsite]:
				pick = str(int(donor_end + loc))
				transcript = self.transcripts[0]
				if (donor in transcript) & (acceptor in transcript):
					transcript_length = 0
					for typ in transcript:
						if typ == donor:
							break
						if 'CDS' in typ:
							transcript_length += int(self.features[self.feature_index_dict[typ]][2])-int(self.features[self.feature_index_dict[typ]][0])
					transcript_length += int(pick)-int(self.features[self.feature_index_dict[typ]][0])
					if (transcript_length % 3) == phase_lookup[self.features[self.feature_index_dict[acceptor]][4]]:
						chosen = dsite
						break

			if chosen:
				self.sjLog[donor]["donor"]["chosen"] = dsite
				self.sjLog[acceptor]["acceptor"]["chosen"] = asite
				self.features[self.feature_index_dict[donor]][2] = pick
				moveOn = True
				
			return moveOn

	def stitchGFF(self, originalStrand=True):
		updatedGFF = f"{';'.join(['|'.join(feature) for feature in self.features])}>{';'.join(['|'.join(transcript) for transcript in self.transcripts])}"
		if originalStrand and (self.strand == "-"):
			updatedGFF = reverseComplement_gffString(updatedGFF, len(self.sequence))
		return updatedGFF
	
	def postProcessPrediction(self, buff=50) -> str:
		# Use start/stop codon, splice junction, and valid protein translation to adjust feature cordinates and phases.
		# Enforce that each transcript is a valid protein-coding sequence.
		# Ensure that updates do not make overlapping exons
		# buff: search buffer for cds boundary markers

		if self.parsable:
			self.checkStartCodons(buff=buff)
			self.checkStopCodons(buff=buff)
			self.checkSpliceJunctions(buff=buff)
			return self.stitchGFF()
		else:
			return self.gff

def checkPhase(gff):
	features, transcripts = gff.split('>')
	features = [f.split('|') for f in features.split(';')]
	transcripts = [m.split('|') for m in transcripts.split(';')]
	
	feature_index_dict = {}
	for i, feature in enumerate(features):
		feature_index_dict[feature[1]] = i
	
	for transcript in transcripts:
		seqlen = 0
		for i, typ in enumerate(transcript):
			if 'CDS' in typ:
				seqlen += int(features[feature_index_dict[typ]][2])-int(features[feature_index_dict[typ]][0])
				print(f"{typ}, {seqlen%3}")

def processPredictions(files:list, db:str, buffer=50, outPrefix="transgenic_predictions"):
	# Processes output predctions and converts to gff3 format
	# files: list of files to process
	# db: Gene list database used for generation
	# buffer: buffer size for splice junction search
	# outPrefix: output files prefix. Files are made for standard GFF3 and transgenic input format
	with open(outPrefix+".txt", 'w') as out_txt:
		with open(outPrefix+".pred.gff3", 'w') as out_gff:
			with open(outPrefix+".true.gff3", 'w') as out_true:
				for file in files:
					with open(file, 'rb') as f:
						predictions = pickle.load(f)
					
					for prediction in predictions:
						with duckdb.connect(db, config={"access_mode": "READ_ONLY"}) as con:
							df = con.sql(f"select * from geneList where gff = '{prediction[2].replace("<s>","").replace("|</s>","").replace("</s>", "")}' limit 1").df()
						pp = PredictionProcessor(prediction[3].replace("|</s>", "").replace("</s>", "").replace("<s>", ""), df.loc[0, "sequence"])
						newGFF = pp.postProcessPrediction(buff=buffer)
						out_txt.write(newGFF+'\n')
						try:
							gff = gffString2GFF3(newGFF, df.loc[0, "chromosome"], df.loc[0, "start"])
							gff = [line+";geneModel="+df.loc[0, "geneModel"] for line in gff]
							out_gff.write('\n'.join(gff)+'\n')
						except:
							print(f"Error: prediction could not be converted to GFF3.\n{newGFF}", file=sys.stderr)

						true = gffString2GFF3(prediction[2].replace("<s>","").replace("|</s>","").replace("</s>", ""), df.loc[0, "chromosome"], df.loc[0, "start"])
						true = [line+";geneModel="+df.loc[0, "geneModel"] for line in true]
						out_true.write('\n'.join(true)+'\n')


if __name__ == '__main__':
	#createDatabase(db="Flagship_6Genomes_49k_extra200_addRCIsoOnly.db", mode="train", maxLen=49152, addExtra=200, addRC=True, addRCIsoOnly=True, clean=True)
	#sys.exit()
	files = []
	files = [
		"cpu:0_ContrastiveSearch.out.pkl",
		"cpu:1_ContrastiveSearch.out.pkl",
		"cpu:2_ContrastiveSearch.out.pkl",
		"cpu:3_ContrastiveSearch.out.pkl",
		"cpu:8_ContrastiveSearch.out.pkl",
		"cpu:10_ContrastiveSearch.out.pkl",
		"cpu:6_ContrastiveSearch.out.pkl",
		"cpu:9_ContrastiveSearch.out.pkl",
		"cpu:7_ContrastiveSearch.out.pkl",
		"cpu:4_ContrastiveSearch.out.pkl",
		"cpu:5_ContrastiveSearch.out.pkl",
		"cpu:11_ContrastiveSearch.out.pkl"
	]
	#processPredictions(files, "Flagship_Genomes_49k_extra200_clean.db", outPrefix="contrast_predictions", buffer=75)
	#sys.exit()

	torch.manual_seed(123)
	#with duckdb.connect("Flagship_Genomes_49k_extra200_clean.db", config={"access_mode": "READ_ONLY"}) as con:
	#	df = con.sql("select * from geneList where geneModel = 'Seita.9G262400.v2.2'").df()
	#	df = con.sql("select * from geneList limit 400").df()
	#checkPhase(reverseComplement_gffString(df.loc[4, "gff"], len(df.loc[4, "sequence"])))
	#pp = PredictionProcessor("309|CDS1|534|-|A;1375|CDS2|1612|-|A;1706|CDS3|1833|-|B;2547|CDS4|2784|-|B;2875|CDS5|2974|-|B;3080|CDS6|3196|-|A;3847|CDS7|4074|-|C;4323|CDS8|4436|-|B;4548|CDS9|4669|-|C;4759|CDS10|4974|-|B;5072|CDS11|5256|-|C;5364|CDS12|5617|-|A;5617|five_prime_UTR1|5662|-|.;5785|five_prime_UTR2|6924|-|.;83|three_prime_UTR1|309|-|.;5364|CDS13|5662|-|A;5785|CDS14|5886|-|C;7444|CDS15|7588|-|C;8037|CDS16|8089|-|A;8089|five_prime_UTR3|8385|-|.;5785|CDS17|5877|-|C>CDS1|CDS2|CDS3|CDS4|CDS5|CDS6|CDS7|CDS8|CDS9|CDS10|CDS11|CDS12|five_prime_UTR1|five_prime_UTR2|three_prime_UTR1;CDS1|CDS2|CDS3|CDS4|CDS5|CDS6|CDS7|CDS8|CDS9|CDS10|CDS11|CDS13|CDS14|CDS15|CDS16|five_prime_UTR3|three_prime_UTR1;CDS1|CDS2|CDS3|CDS4|CDS5|CDS6|CDS7|CDS8|CDS9|CDS10|CDS11|CDS13|CDS17|CDS15|CDS16|five_prime_UTR3|three_prime_UTR1", df["sequence"].item())
	#newGFF = pp.postProcessPrediction(buff=75)
	#print(newGFF)
	#model = transgenic()
	#tensors = {}
	#with safe_open("saved_transgenic_models_accu32/model.safetensors", framework="pt", device="cpu") as f:
	#	for k in f.keys():
	#		tensors[k] = f.get_tensor(k)

	#model.load_state_dict(tensors)
	#model.eval()
	#createDatabase(db="7Genomes_25k_extra200.db", mode="train", maxLen=25002, addExtra=200)
	#sys.exit()

	#config = AutoConfig.from_pretrained("allenai/led-base-16384", vocab_size=372, max_decoder_position_embeddings=2048)
	#model = transgenicForConditionalGeneration(config)
	#model = transgenicOriginalEmbed()
	#model.load_state_dict(torch.load("checkpoints_FCG/pytorch_model/mp_rank_00_model_states.pt")['module'])#, map_location=torch.device('cpu'))['module'])
	#model.to(torch.device('cuda:0'))
	#model.eval()
	# Create a training, evaluation, and testing DataLoaders (Dataset length: 175498)
	#ds = isoformData("Flagship_Genomes_49k_extra200.db", dt="gff", mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b")
	#train_data, eval_data, test_data = random_split(ds, [171071, 24438, 48879])
	#train_data = makeDataLoader(train_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=1)
	#train_data.dataset.__getitem__(train_data.dataset.indices.index(108613))
	#for step, batch in enumerate(tqdm(train_data)):
	#	print(batch[3], file=sys.stderr)
		#print(batch[2], file=sys.stderr)
		#print(batch[0], file=sys.stderr)
		#print(batch[1], file=sys.stderr)
	#sys.exit()
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

	# Keeping the training, validation, and testing datasets consistent after updating with reverse complemented isoforms
#	db = "Flagship_Genomes_49k_extra200_clean.db"
#	ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b")
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
	db = "Flagship_6Genomes_49k_extra200_addRCIsoOnly.db"
	ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b")
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
	with open("Train-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
		train_data = pickle.load(f)
	with open("Eval-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
		eval_data = pickle.load(f)
	with open("Test-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
		test_data = pickle.load(f)

	mode = sys.argv[1]
	#InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
	#InstaDeepAI/agro-nucleotide-transformer-1b
	encoder_model = sys.argv[2]
	unlink = bool(sys.argv[3])
	notes = sys.argv[4]
	print(f"Running in {mode} mode", file=sys.stderr)

	if mode == "test":
		# Create a training, evaluation, and testing DataLoaders (Dataset length: 175498)
		ds = isoformData("Flagship_Genomes_25k_stranded.db", dt="gff", mode="training", encoder_model = encoder_model)
		train_data, eval_data, test_data = random_split(ds, [131470, 17399, 26171])
		testTransgenicAccelerate(
			"saved_models_FCG/model.safetensors",
			test_data,
			encoder_model=encoder_model,
			unlink=unlink,
			batch_size=1,
			outfile="25kb_5genomes_test.pkl"
		)
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
		#ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model=encoder_model, global_attention=False)
		#train_data, eval_data, test_data = random_split(ds, [237345, 31646, 47470])#[171071, 24438, 48879])316461
		trainTransgenicFCGAccelerate(
			train_data, 
			eval_data, 
			lr=1e-4, 
			num_epochs=4, 
			schedule_lr=True, 
			eval=True, 
			batch_size=1, 
			accumulation_steps=16,
			checkpoint_path="checkpoints_ESMpeftReal_local09/", 
			safetensors_model="checkpoints_ESMpeftReal_local09/model.safetensors", #"saved_models_FCG/model.safetensors",
			output_dir="saved_models_ESMpeftReal_local09/",
			max_grad_norm=1,
			notes=notes,
			encoder_model=encoder_model,
			unlink = unlink
		)
	elif mode == "CPU":
		ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model=encoder_model)
		train_data, eval_data, test_data, t_data = random_split(ds, [171071, 24438, 48878, 1])
		outmod = trainTransgenicFCGCPU(
			t_data, 
			eval_data, 
			lr=1e-4, 
			num_epochs=40, 
			schedule_lr=False, 
			eval=False, 
			batch_size=1, 
			accumulation_steps=1,
			checkpoint_path="checkpoints_CPU/", 
			safetensors_model="checkpoints_ESMpeftReal_local09/model.safetensors", #"saved_models_FCG/model.safetensors",
			output_dir="saved_models_CPU/",
			max_grad_norm=1,
			notes=notes,
			encoder_model=encoder_model,
			unlink = unlink
		)
		print("Done")
	elif mode == "trainOriginalEmbed":
		ds = isoformData(db, dt="led", mode="training", encoder_model=encoder_model)
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
		db = "Flagship_Genomes_49k_extra200_clean.db"
		ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model=encoder_model)
		train_data, eval_data, test_data, t_data = random_split(ds, [171071, 24438, 48878, 1])
		predictTransgenicAccelerate(
			"checkpoints_ESMpeftReal_local09/model.safetensors", 
			eval_data, 
			outfile="GreedySearch.out", 
			batch_size=1	
		)
	elif mode == "predict":
		db = "Flagship_Genomes_49k_extra200_clean.db"
		ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model=encoder_model)
		train_data, eval_data, test_data, t_data = random_split(ds, [171071, 24438, 48878, 1])
		predictTransgenic(
			"saved_transgenic_models_accu32/model.safetensors", 
			test_data, 
			outfile="transgenic.out", 
			batch_size=1
			)
	elif mode == "DDP":
		ds = isoformData(db, dt=GFFTokenizer09(), mode="training", encoder_model=encoder_model)
		train_data, eval_data, test_data = random_split(ds, [171071, 24438, 48879])
		run_trainTransgenicDDP(
			train_data, 
			eval_data, 
			lr=1e-4, 
			num_epochs=10,  
			schedule_lr=True, 
			eval=True, 
			world_size=12,
			batch_size=1,
			checkpoint_path="checkpoints/transgenic_ddp_checkpoint.pt",
			safetensors_model="checkpoints_ESMpeft_local09/model.safetensors",
			encoder_model=encoder_model,
			unlink=False,
			accumulation_steps=16,
			max_grad_norm=1.0
		)