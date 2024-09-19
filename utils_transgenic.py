import subprocess, sys, os, duckdb, torch, re, pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch.distributed as dist
from torch.utils.data import  Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import IA3Config, get_peft_model
from safetensors import safe_open

from modeling_transgenic import transgenicForConditionalGeneration, segmented_sequence_embeddings
from tokenization_transgenic import GFFTokenizer
from configuration_transgenic import TransgenicConfig

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

# geneList custom dataset class for use with DataLoader
class isoformData(Dataset):
	def __init__(self, db, dt, mode="inference", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False):
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

# Create a database for loading genomic data to SegmentNT
def genome2SegmentationSet(genome_file, gff_file, organism, db):

	# load gff into database with organism name
	gff3 = pd.read_csv(gff_file, sep='\t', header=None, comment='#')
	gff3.columns = ['chromosome', 'source', 'feature', 'start', 'fin', 'score', 'strand', 'frame', 'attribute']
	gff3['organism'] = organism
	with duckdb.connect(db) as con:
		con.sql(
			'CREATE TABLE IF NOT EXISTS gff ('
			'chromosome VARCHAR, '
			'source VARCHAR, '
			'feature VARCHAR, '
			'start INT, '
			'fin INT, '
			'score VARCHAR, '
			'strand VARCHAR, '
			'frame VARCHAR, '
			'attribute VARCHAR, '
			'organism VARCHAR)')

		con.sql(
			'INSERT INTO gff '
			'SELECT * '
			'FROM gff3; '
		)

	# load chromosome sequences into database with organism name
	genome_dict = loadGenome(genome_file)
	genome_df = pd.DataFrame(genome_dict.items(), columns=['chromosome', 'sequence'])
	genome_df['organism'] = organism
	genome_df['length'] = genome_df['sequence'].apply(len)
	with duckdb.connect(db) as con:
		con.sql(
			"CREATE TABLE IF NOT EXISTS genome ("
			'chromosome VARCHAR, '
			'sequence VARCHAR, '
			'organism VARCHAR, '
			'length INT)')

		con.sql(
			'INSERT INTO genome '
			'SELECT * '
			'FROM genome_df; '
		)

class segmentationDataset(Dataset):
	def __init__(self, window_size, step_size, db, encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b"):
		self.window_size = window_size
		self.step_size = step_size
		self.db = db
		self.encoder_model =  encoder_model
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)


		self.classes = ['protein_coding_gene', 
				'lncRNA', 
				'exon', 
				'intron', 
				'splice_donor', 
				'splice_acceptor', 
				'5UTR', 
				'3UTR', 
				'CTCF-bound', 
				'polyA_signal', 
				'enhancer_Tissue_specific', 
				'enhancer_Tissue_invariant', 
				'promoter_Tissue_specific', 
				'promoter_Tissue_invariant']

		self.gffClassMap = {'gene': 'protein_coding_gene',  
					'exon': 'exon', 
					'intron': 'intron',
					'five_prime_cis_splice_site': 'splice_donor', 
					'three_prime_cis_splice_site': 'splice_acceptor', 
					'five_prime_UTR': '5UTR', 
					'three_prime_UTR': '3UTR'}

		# Partition the genomes into windows based on step_size and window_size
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			seqLengths = con.sql("SELECT organism, chromosome, length FROM genome").df()	
		
		window_list = []
		for i in range(len(seqLengths)):
			organism = seqLengths.loc[i, 'organism']
			chromosome = seqLengths.loc[i, 'chromosome']
			length = seqLengths.loc[i, 'length']
			windows = (length - self.window_size) // self.step_size + 1
			
			for j in range(windows):
				start = j * self.step_size
				end = start + self.window_size
				window_list.append([organism, chromosome, start, end])
		
		self.windows = pd.DataFrame(window_list, columns=['organism', 'chromosome', 'start', 'end'])

	
	def __len__(self):
		return len(self.windows)

	def __getitem__(self, idx):
		# Get the windowed sequence from the database
		window = self.windows.loc[idx]
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			sequence = con.sql(
				"SELECT sequence "
				"FROM genome "
				f"WHERE chromosome = '{window['chromosome']}' "
				f"AND organism = '{window['organism']}'").fetchall()[0][0]
		sequence = sequence[window['start']:window['end']]

		if "N" in sequence:
			return self.__getitem__(torch.randint(0, len(self.windows), (1,)).item())

		# Get the gff for the window
		# Use any feature that overlaps the window
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			annotations = con.sql(
				"SELECT feature, start, fin "
				"FROM gff "
				f"WHERE chromosome = '{window['chromosome']}' "
				f"AND organism = '{window['organism']}' "
				f"AND (start <= {window['end']} AND fin >= {window['start']})").df()

		# Adjust the start and end coordinates for the window
		annotations['start'] = annotations['start'].apply(lambda x: x - window['start'] -1)
		annotations['fin'] = annotations['fin'].apply(lambda x: x - window['start'])
		annotations['start'] = annotations['start'].apply(lambda x: max(x, 0))
		annotations['fin'] = annotations['fin'].apply(lambda x: min(x, self.window_size))

		# Create the class tensor and populate with the annotations
		class_tensor = torch.zeros((self.window_size, len(self.classes)), dtype=torch.float32)
		
		for i in range(len(annotations)):
			start = annotations.loc[i, 'start']
			end = annotations.loc[i, 'fin']
			feature = annotations.loc[i, 'feature']
			if feature in self.gffClassMap:
				class_idx = self.classes.index(self.gffClassMap[feature])
				class_tensor[start:end, class_idx] = 1
		
		# Segment and tokenize the sequences (piece size is 6144 nucleotides)
		seqs = segmentSequence(sequence, piece_size=6144)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 1024)["input_ids"]
		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, window['organism'], window['chromosome'], window['start'], window['end'])

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

def segmentSequence(seq, piece_size = 6144):
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

def segment_collate_fn(batch):
	# Unpack the batch items (each item in batch is a tuple of (sequence, attention_mask, target_sequence))
	sequences, attention_masks, labels, organism, chromosome, start, end = zip(*batch)

	# Remove invalid sequences from the batch
	#sequences = [item for item, keep in zip(sequences, valid) if keep]
	#attention_masks = [item for item, keep in zip(attention_masks, valid) if keep]
	#labels = [item for item, keep in zip(labels, valid) if keep]

	#if len(sequences) == 0:
	#	return None, None, None

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
		labels_padded = torch.cat(labels_padded)

	if labels:
		return sequences, attention_masks, labels_padded, organism, chromosome, start, end
	else:
		return sequences, attention_masks, None, organism, chromosome, start, end

def makeDataLoader(dat, shuffle=True, batch_size=8, pin_memory=True, sampler=None, num_workers=0, collate_fn=target_collate_fn):
	if sampler != None:
		shuffle = False
	
	return DataLoader(
		dat, 
		shuffle=shuffle, 
		collate_fn=collate_fn, 
		batch_size=batch_size, 
		pin_memory=pin_memory,
		sampler=sampler,
		num_workers=num_workers)

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
		ds = isoformData(db, dt=GFFTokenizer(), mode="training")
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
				if 'CDS' in typ and (transcript[i-1], typ) not in self.cds_pairs and 'CD' in transcript[i-1]:
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

		# Uniqueify the  transcripts
		transcripts = [list(x) for x in set(tuple(x) for x in transcripts)]

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
					try:
						with open(file, 'rb') as f:
							predictions = pickle.load(f)
					except:
						print(f"Error: could not load predictions from {file}.", file=sys.stderr)
						continue
					
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

def getModel(config, safetensors_model=None, device="cpu", mode="predict"):
	if not config:
		# Load the model and add to device
		config = TransgenicConfig()

	model = transgenicForConditionalGeneration(config)

	if mode == "train":
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
		if "base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight" in tensors.keys():
			del tensors["base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight"]
		
		newtensors = {k.replace("base_model.model.", "").replace(".base_layer", ""):tensors[k] for k in tensors}
		newnewtensors = {}
		for k in newtensors:
			if "ia3" not in k:
				newnewtensors[k] = newtensors[k]
		model.load_state_dict(newnewtensors)
	
	return model

def getPeftModel(encoder_model, config=None, unlink=False, safetensors_model=None, device="cpu", mode="predict"):
	if not config:
		# Load the model and add to device
		config = TransgenicConfig()

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

	if mode == "train":
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
		if "base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight" in tensors.keys():
			del tensors["base_model.model.transgenic.decoder_embed_tokens.num_feature_embed.weight"]
		model.load_state_dict(tensors)
	
	return model

def registerModel():
	from modeling_transgenic import transgenicForConditionalGeneration, transgenicModel
	from configuration_transgenic import TransgenicConfig

	TransgenicConfig.register_for_auto_class()
	transgenicModel.register_for_auto_class("AutoModel")
	transgenicForConditionalGeneration.register_for_auto_class("AutoModel")

	model = getModel(TransgenicConfig(), safetensors_model="checkpoints_ESMpeftReal_local09/model.safetensors", device="cpu", mode="predict")
	model.push_to_hub("jlomas/transgenic-agro-E9")

def mergeAndProcessPredictions(device, searchType, world_size, db, outprefix, buffer):
	files = [f"{device}:{i}_{searchType}Search.out.pkl" for i in range(world_size)]
	processPredictions(files, db, outPrefix=outprefix, buffer=buffer)
	#sys.exit()

def analyzePerGeneTranscriptPerformance(label_tokens, prediction_tokens):
	# Purpose: 
	#      Analyze the transcript performance of a prediction on a per-gene basis. 
	# Inputs:
	#      label_tokens: tensor containing tokenized labels
	#      prediction_tokens: tensor containeing tokenized predictions
	# Outputs:
	#      performance: dictionary of gene-level performance metrics

	performance = {
		"label_transcript_count": None,
		"pred_transcript_count": None
	}

	label_tokens = label_tokens.tolist()
	prediction_tokens = prediction_tokens.tolist()

	label_transcripts = label_tokens.split(17)[1].split(21)
	prediction_transcripts = prediction_tokens.split(17)[1].split(21)

	# Transcript count
	performance["label_transcript_count"] = len(label_transcripts)
	performance["pred_transcript_count"] = len(prediction_transcripts)

	# Double loop through label transcripts and prediction transcripts
	# Use pairs with highest F1 or MCC as matching then remove from consideration
	# record metrics (sensitivity, specificity, F1, MCC) for matching pairs
	for i, transcript in enumerate(label_transcripts):
		prev_MCC = 0
		for j, prediction in enumerate(prediction_transcripts):
			#TODO
			pass


if __name__ == '__main__':
	
	db = "Segmentation_10Genomes.db"
	files = {
		"training_data/Athaliana_167_TAIR10.gene.exon.splice.gff3":["training_data/Athaliana_167_TAIR10.fa","ath"],
		"training_data/Bdistachyon_314_v3.1.gene_exons.exon.splice.gff3":["training_data/Bdistachyon_314_v3.0.fa","bdi"],
		"training_data/Sbicolor_730_v5.1.gene_exons.exon.splice.gff3":["training_data/Sbicolor_730_v5.0.fa","sbi"],
		"training_data/Sitalica_312_v2.2.gene_exons.exon.splice.gff3":["training_data/Sitalica_312_v2.fa","sit"],
		"training_data/Ptrichocarpa_533_v4.1.gene_exons.exon.splice.gff3":["training_data/Ptrichocarpa_533_v4.0.fa","ptr"],
		"training_data/Gmax_880_Wm82.a6.v1.gene_exons.exon.splice.gff3":[ "training_data/Gmax_880_v6.0.fa","gma"],
		"training_data/Ppatens_318_v3.3.gene_exons.exon.splice.gff3":["training_data/Ppatens_318_v3.fa","ppa"],
		"training_data/Vvinifera_PN40024_5.1_on_T2T_ref.exon.splice.gff3" : ["training_data/Vvinifera_T2T_ref.fasta", "Vvi"],
		"training_data/Osativa_323_v7.0.gene_exons.exon.splice.gff3" : ["training_data/Osativa_323_v7.0.fa", "Osa"],
		"training_data/Zmays_493_RefGen_V4.gene_exons.exon.splice.gff3" : ["training_data/Zmays_493_APGv4.fa", "Zma"]
	}
	for file in files:
		genome2SegmentationSet(
		files[file][0], 
		file,
		files[file][1],
		db)
	

	ds  = segmentationDataset(6144, 6000, "Segmentation_10Genomes.db")

	print(len(ds))