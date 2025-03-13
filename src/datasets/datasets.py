import duckdb, random, torch, sys, zlib
import pandas as pd
import numpy as np
import torch.nn.functional as F 
from torch.utils.data import  Dataset, DataLoader
from transformers import AutoTokenizer

from ..utils.sequence import segmentSequence, scanGlobalAttentionTokens, mask_sequences

class isoformData(Dataset):
	def __init__(self, db, dt, mode="inference", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False, shuffle=False):
		self.db = db
		self.mode = mode
		self.dt = dt
		self.global_attention = global_attention
		self.shuffle = shuffle
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
			try:
				gm,region_start,region_end,strand,chr,region_seq, gff,sfpb, stpb, fpb, tpb,_ = con.sql(f"SELECT * FROM geneList where rn={idx}").fetchall()[0]
			except:
				newidx = torch.randint(self.__len__(), (1,)).item()
				print(f"Warning {idx=} produced an error... using {newidx}", file=sys.stderr)
				gm,region_start,region_end,strand,chr,region_seq, gff,sfpb, stpb, fpb, tpb,_ = con.sql(f"SELECT * FROM geneList where rn={newidx}").fetchall()[0]
		
		if self.shuffle:
			gff_shuffle = [g.split(";") for g in gff.split(">")]
			random.shuffle(gff_shuffle[0])
			random.shuffle(gff_shuffle[1])
			gff = ";".join(gff_shuffle[0]) + ">" + ";".join(gff_shuffle[1])

		# Tokenize labels
		if self.mode == "training":
			labels = self.decoder_tokenizer.batch_encode_plus(
				[gff],
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True,
				max_length=self.maxlength)["input_ids"]
		
		# Ensure labels are less than the maxlength
		if labels.shape[1] >= self.maxlength:
			labels = torch.cat((labels[:, 0:(self.maxlength-1)], torch.tensor([[self.decoder_tokenizer.vocab["</s>"]]])), dim=1)
			print(f"Warning {gm} label truncated to {self.maxlength} tokens", file=sys.stderr)

		# Segment and tokenize the sequences
		seqs = segmentSequence(region_seq, piece_size=6144)
		numSeqs = len(seqs)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 1024)["input_ids"]
		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		# Complete the attention mask based on buffers
		#fiveP_mask_index = (sfpb - fpb)//6
		#threeP_mask_index = (stpb - tpb)//6
		#encoder_attention_mask[0, 0:fiveP_mask_index] = False
		#encoder_attention_mask[numSeqs-1, 1024-threeP_mask_index:] = False


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

class isoformDataHyena(Dataset):
	def __init__(self, db, dt, mode="inference", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False):
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
			try:
				gm,region_start,region_end,strand,chr,region_seq, gff,sfpb, stpb, fpb, tpb,_ = con.sql(f"SELECT * FROM geneList where rn={idx}").fetchall()[0]
			except:
				newidx = torch.randint(self.__len__(), (1,)).item()
				print(f"Warning {idx=} produced an error... using {newidx}", file=sys.stderr)
				gm,region_start,region_end,strand,chr,region_seq, gff,sfpb, stpb, fpb, tpb,_ = con.sql(f"SELECT * FROM geneList where rn={newidx}").fetchall()[0]

		# Tokenize labels
		if self.mode == "training":
			labels = self.decoder_tokenizer.batch_encode_plus(
				[gff],
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True,
				max_length=self.maxlength)["input_ids"]
		
		# Ensure labels are less than the maxlength
		if labels.shape[1] >= self.maxlength:
			labels = torch.cat((labels[:, 0:(self.maxlength-1)], torch.tensor([[self.decoder_tokenizer.vocab["</s>"]]])), dim=1)
			#print(f"Warning {gm} label truncated to {self.maxlength} tokens", file=sys.stderr)

		# Tokenize the input sequences and remove the [SEP] token
		seqs = self.encoder_tokenizer.batch_encode_plus([region_seq], return_tensors="pt")
		seqs["input_ids"] = seqs["input_ids"][:, :-1]

		attention_mask = (seqs["input_ids"] != self.encoder_tokenizer.pad_token_id)

		if self.mode == "training":
			return (seqs["input_ids"], attention_mask, labels, gm, chr, region_start, region_end)
		else:
			return (seqs["inputs_ids"], attention_mask, None, gm, chr, region_start, region_end)

# Create a database for loading genomic data to SegmentNT
class segmentationDataset(Dataset):
	def __init__(self, table, window_size, step_size, db, encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", preprocess=False):
		self.window_size = window_size
		self.step_size = step_size
		self.db = db
		self.preprocess = preprocess
		self.encoder_model =  encoder_model
		self.table = table
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
			seqLengths = con.sql(f"SELECT organism, chromosome, length FROM {table}_genome").df()	
		
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
				f"FROM {self.table}_genome "
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
				f"FROM {self.table}_gff "
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
		
		if self.preprocess:
			class_tensor = zlib.compress(class_tensor.numpy().tobytes())
		
		# Segment and tokenize the sequences (piece size is 6144 nucleotides)
		if self.preprocess:
			seqs = sequence
			encoder_attention_mask = None
		else:
			seqs = segmentSequence(sequence, piece_size=6144)
			seqs = self.encoder_tokenizer.batch_encode_plus(
				seqs,
				return_tensors="pt",
				padding="max_length",
				truncation=True,
				max_length = 1024)["input_ids"]
			encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, window['organism'], window['chromosome'], window['start'], window['end'])

class preprocessedSegmentationDataset(Dataset):
	def __init__(self, db, encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b"):
		self.db = db
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)
	
	def __len__(self):
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM data").fetchall()[0][0]
	
	def __getitem__(self, idx):
		idx = idx + 1
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			_, sequence,label,organism,chromosome,start,fin,_ = con.sql(f"SELECT * FROM data where rn={idx}").fetchall()[0]
		
		if "N" in sequence:
			return self.__getitem__(torch.randint(0, self.__len__(), (1,)).item())

		# Retreive tensor from BLOB
		try:
			class_tensor = np.frombuffer(zlib.decompress(label), dtype=np.float32).reshape(6144, 14)
			class_tensor = torch.from_numpy(class_tensor)
		except:
			newidx = torch.randint(0, self.__len__(), (1,)).item()
			print(f"Warning labe with {idx=} could not be parsed... using {newidx}")
			return self.__getitem__(newidx)

		# Tokenize sequence
		seqs = segmentSequence(sequence, piece_size=6144)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length = 1024)["input_ids"]
		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, organism, chromosome, start, fin)

class preprocessedSegmentationDatasetHyena(Dataset):
	def __init__(self, db, encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf"):
		self.db = db
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)
	
	def __len__(self):
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM data").fetchall()[0][0]
	
	def __getitem__(self, idx):
		idx = idx + 1
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			_, sequence,label,organism,chromosome,start,fin,_ = con.sql(f"SELECT * FROM data where rn={idx}").fetchall()[0]
		
		if "N" in sequence:
			return self.__getitem__(torch.randint(0, self.__len__(), (1,)).item())

		# Retreive tensor from BLOB
		try:
			class_tensor = np.frombuffer(zlib.decompress(label), dtype=np.float32).reshape(6144, 14)
			class_tensor = torch.from_numpy(class_tensor)
		except:
			newidx = torch.randint(0, self.__len__(), (1,)).item()
			print(f"Warning labe with {idx=} could not be parsed... using {newidx}")
			return self.__getitem__(newidx)

		# Tokenize the input sequences and remove the [SEP] token
		seqs = self.encoder_tokenizer.batch_encode_plus([sequence], return_tensors="pt")
		seqs["input_ids"] = seqs["input_ids"][:, :-1]
		encoder_attention_mask = (seqs["input_ids"] != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, organism, chromosome, start, fin)

class MLMDatasetHyena(Dataset):
	def __init__(self, db, encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf"):
		self.db = db
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)
	
	def __len__(self):
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM data").fetchall()[0][0]
	
	def __getitem__(self, idx):
		idx = idx + 1
		with duckdb.connect(self.db, config = {"access_mode": "READ_ONLY"}) as con:
			_, sequence,label,organism,chromosome,start,fin,_ = con.sql(f"SELECT * FROM data where rn={idx}").fetchall()[0]

		# Tokenize the input sequences and remove the [SEP] token
		seqs = self.encoder_tokenizer.batch_encode_plus([sequence], return_tensors="pt")
		seqs["input_ids"] = seqs["input_ids"][:, :-1]
		encoder_attention_mask = (seqs["input_ids"] != self.encoder_tokenizer.pad_token_id)

		masked_seqs, mask_index = mask_sequences(seqs.input_ids)

		return (masked_seqs, mask_index, seqs.input_ids, organism, chromosome, start, fin)

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

def hyena_collate_fn(batch):
	# Unpack the batch items (each item in batch is a tuple of (sequence, attention_mask, target_sequence))
	sequences, attention_masks, labels, gm, chr, region_start, region_end= zip(*batch)

	# Pad and stack the sequences
	max_len = max([seq.shape[1] for seq in sequences])
	sequences = [F.pad(seq, (max_len - seq.shape[1], 0)) for seq in sequences]
	sequences = torch.cat(sequences)
	
	
	#Pad and stack the attention masks
	max_len = max([mask.shape[1] for mask in attention_masks])
	attention_masks = [F.pad(mask, (max_len - mask.shape[1], 0)) for mask in attention_masks]
	attention_masks = torch.cat(attention_masks)

	# Pad and stack the labels
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]
		labels_padded = torch.cat(labels_padded)

	if labels:
		return sequences, attention_masks, labels_padded, gm, chr, region_start, region_end
	else:
		return sequences, attention_masks, None, gm, chr, region_start, region_end

def hyenaMLM_collate_fn(batch):
	# Unpack the batch items (each item in batch is a tuple of (sequence, attention_mask, target_sequence))
	sequences, attention_masks, labels, gm, chr, region_start, region_end= zip(*batch)

	# Pad and stack the sequences
	max_len = max([seq.shape[1] for seq in sequences])
	sequences = [F.pad(seq, (max_len - seq.shape[1], 0)) for seq in sequences]
	sequences = torch.cat(sequences)
	
	#Pad and stack the attention masks
	max_len = max([mask.shape[1] for mask in attention_masks])
	attention_masks = [F.pad(mask, (max_len - mask.shape[1], 0)) for mask in attention_masks]
	attention_masks = torch.cat(attention_masks)

	# Pad and stack the labels
	max_len = max([label.shape[1] for label in labels])
	labels = [F.pad(label, (max_len - label.shape[1], 0)) for label in labels]
	labels = torch.cat(labels)

	return sequences, attention_masks, labels, gm, chr, region_start, region_end

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
		labels_padded = torch.stack(labels_padded)

	if labels:
		return sequences, attention_masks, labels_padded, organism, chromosome, start, end
	else:
		return sequences, attention_masks, None, organism, chromosome, start, end

def hyena_segment_collate_fn(batch):
	# Unpack the batch items (each item in batch is a tuple of (sequence, attention_mask, target_sequence))
	sequences, attention_masks, labels, organism, chromosome, start, end = zip(*batch)

	# Pad and stack the sequences
	max_len = max([seq["input_ids"].shape[1] for seq in sequences])
	sequences = [F.pad(seq["input_ids"], (max_len - seq["input_ids"].shape[1], 0)) for seq in sequences]
	sequences = torch.cat(sequences)
	
	#Pad and stack the attention masks
	max_len = max([mask.shape[1] for mask in attention_masks])
	attention_masks = [F.pad(mask, (max_len - mask.shape[1], 0)) for mask in attention_masks]
	attention_masks = torch.cat(attention_masks)

	# Pad and stack the labels
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]
		labels_padded = torch.stack(labels_padded)

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
