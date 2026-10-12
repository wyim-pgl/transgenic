"""
Dataset and DataLoader utilities for TransGenic training and inference.

This module provides PyTorch Dataset classes that load genomic data from DuckDB
databases and tokenize them for the TransGenic model. Each sample consists of:

  Input:  Raw DNA nucleotide sequence → tokenized by encoder tokenizer (HyenaDNA or NT)
  Target: GFF gene annotation string → tokenized by GFFTokenizer (decoder vocabulary)

Dataset classes:
  - isoformData:          Original NT-encoder variant (Agro Nucleotide Transformer)
  - isoformDataHyena:     HyenaDNA-encoder variant (primary, used in production training)
  - segmentationDataset:  Sliding-window segmentation dataset (SegmentNT-style)
  - preprocessedSegmentationDataset:     Pre-segmented data for NT encoder
  - preprocessedSegmentationDatasetHyena: Pre-segmented data for HyenaDNA encoder
  - MLMDatasetHyena:      Masked Language Modeling dataset for HyenaDNA pretraining

Collate functions:
  - target_collate_fn:         Pads multi-segment NT encoder batches
  - hyena_collate_fn:          Pads variable-length HyenaDNA batches (primary)
  - hyenaMLM_collate_fn:       Pads MLM batches for HyenaDNA pretraining
  - segment_collate_fn:        Pads segmentation batches (NT encoder)
  - hyena_segment_collate_fn:  Pads segmentation batches (HyenaDNA encoder)

DuckDB connection management:
  isoformDataHyena uses persistent per-worker connections with PID-based fork
  detection. When a DataLoader worker process is forked, the inherited connection
  is detected (PID mismatch) and replaced with a fresh one. This avoids the
  overhead of opening/closing connections on every __getitem__ call.

Data flow:
  DuckDB → raw DNA + GFF strings → tokenize → pad/collate → DataLoader → model
"""

import duckdb, random, torch, sys, zlib, os
import math
from numbers import Integral, Real
import pandas as pd
import numpy as np
import torch.nn.functional as F  # For F.pad (dynamic padding in collate functions)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  # HuggingFace tokenizer loading utility

# Internal imports for sequence processing and tokenization
from ..utils.sequence import segmentSequence, scanGlobalAttentionTokens, mask_sequences
from ..model.tokenization_transgenic import GFFTokenizer


class isoformData(Dataset):
	"""Dataset for Agro Nucleotide Transformer (NT) encoder variant.

	Loads genomic regions from a DuckDB database and tokenizes them
	using the NT encoder tokenizer. DNA sequences are segmented into
	6144-nt chunks (each encoded to 1024 tokens), then flattened for
	the multi-segment encoder. Labels are tokenized GFF strings.

	This class opens a new DuckDB connection per __getitem__ call,
	which is simpler but slower than isoformDataHyena's persistent
	connection approach.
	"""
	def __init__(self, db, dt, mode="inference", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False, shuffle=False):
		"""
		Args:
			db: Path to DuckDB database file containing geneList table.
			dt: Decoder tokenizer (GFFTokenizer instance) or None for default.
			mode: "training" for train+labels, "inference" for input-only.
			encoder_model: HuggingFace model ID for the encoder tokenizer.
			global_attention: Whether to compute global attention masks for Longformer.
			shuffle: Whether to randomly shuffle GFF feature order (data augmentation).
		"""
		self.db = db
		self.mode = mode
		self.dt = dt
		self.global_attention = global_attention
		self.shuffle = shuffle
		# Load the encoder tokenizer (NT uses 6-mer vocabulary)
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)
		if dt != None:
			self.decoder_tokenizer = dt
			self.maxlength = 2048  # GFF token sequence length limit
		else:
			# Default: use LED tokenizer as fallback decoder tokenizer
			self.decoder_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384", cache_dir="./HFmodels", trust_remote_code=True)
			self.maxlength = 1024

	def __len__(self):
		"""Return total number of gene samples in the database."""
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM geneList").fetchall()[0][0]

	def __getitem__(self, idx):
		"""Load and tokenize one gene sample by index.

		Reads the gene record from DuckDB, optionally shuffles GFF features
		(data augmentation), tokenizes the DNA sequence into 6144-nt segments,
		and tokenizes the GFF labels.

		Returns:
			Tuple of (input_ids, attention_mask, global_attention_mask, labels, gene_model, chr, start, end)
		"""
		idx += 1  # DuckDB row numbers are 1-indexed
		# Open a read-only connection to fetch the gene record
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			try:
				# Unpack all columns from the geneList table
				gm, region_start, region_end, strand, chr, region_seq, gff, sfpb, stpb, fpb, tpb, _ = con.sql(f"SELECT geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf, rn FROM geneList where rn={idx}").fetchall()[0]
			except:
				# Fallback: if this index fails, try a random different index
				newidx = torch.randint(self.__len__(), (1,)).item()
				print(f"Warning {idx=} produced an error... using {newidx}", file=sys.stderr)
				gm, region_start, region_end, strand, chr, region_seq, gff, sfpb, stpb, fpb, tpb, _ = con.sql(f"SELECT geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf, rn FROM geneList where rn={newidx}").fetchall()[0]

		# Data augmentation: randomly shuffle the order of GFF features and transcripts
		if self.shuffle:
			gff_shuffle = [g.split(";") for g in gff.split(">")]
			random.shuffle(gff_shuffle[0])   # Shuffle features
			random.shuffle(gff_shuffle[1])   # Shuffle transcripts
			gff = ";".join(gff_shuffle[0]) + ">" + ";".join(gff_shuffle[1])

		# Tokenize GFF labels using the decoder tokenizer
		if self.mode == "training":
			labels = self.decoder_tokenizer.batch_encode_plus(
				[gff],
				return_tensors="pt",
				padding=True,
				truncation=True,
				add_special_tokens=True,
				max_length=self.maxlength)["input_ids"]

		# Truncate labels that exceed max length, preserving the </s> end token
		if labels.shape[1] >= self.maxlength:
			labels = torch.cat((labels[:, 0:(self.maxlength-1)], torch.tensor([[self.decoder_tokenizer.vocab["</s>"]]])), dim=1)
			print(f"Warning {gm} label truncated to {self.maxlength} tokens", file=sys.stderr)

		# Segment the DNA sequence into 6144-nt chunks for NT encoder
		# Each chunk is tokenized to 1024 tokens (6144 nt / 6-mer = 1024 tokens)
		seqs = segmentSequence(region_seq, piece_size=6144)
		numSeqs = len(seqs)  # Number of segments
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length=1024)["input_ids"]
		# Create attention mask: 1 for real tokens, 0 for padding
		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		# Compute global attention mask for Longformer (identifies splice sites etc.)
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
	"""Primary dataset class for HyenaDNA encoder variant.

	Unlike isoformData, this class:
	  - Uses HyenaDNA tokenizer (single-nucleotide vocabulary, not 6-mer)
	  - Does NOT segment sequences (HyenaDNA handles full-length sequences)
	  - Uses GFFTokenizer directly instead of HF encoder for labels
	  - Maintains persistent DuckDB connections per worker process
	  - Supports exclude_prefix for cross-species evaluation (e.g., exclude 'Zm')

	DuckDB connection lifecycle:
	  - One connection is opened per DataLoader worker (or main process)
	  - Fork detection via PID comparison: if PID changes after fork,
	    the inherited connection is closed and a fresh one is opened
	  - Connections are closed in __del__ when the dataset is garbage-collected
	"""
	def __init__(self, db, mode="inference", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False, exclude_prefix=None, split=None, gff_vocab_version="v2", window_len=None, strict=False, *, species=None, evidence_db=None, evidence_k=8, aug_seed=None, channel_dropout_p=0.5, thin_fractions=(0.01, 0.1, 0.5, 1.0), eval_fraction=None, return_tile_ids=False):
		"""
		Args:
			db: Path to DuckDB database file.
			mode: "train" for training (returns labels), "inference" for input-only.
			encoder_model: HuggingFace model ID for the HyenaDNA encoder tokenizer.
			global_attention: Whether to use global attention (not used with HyenaDNA).
			exclude_prefix: Gene name prefix to exclude (e.g., "Zm" for maize).
			split: "train" | "valid" | "test" — select rows by the frozen split column of a B5 database.
			strict: Protocol A35 applied to the dataset. When True a row that cannot be read,
				that holds an empty sequence, or that tokenises to zero length raises instead of
				being replaced by a randomly chosen different row. A frozen-recipe run must set
				this: silently substituting a row removes one sample from a pre-registered study
				and duplicates another, and the only trace is a line in a SLURM log.
			species: Optional parameterized species_id filter in every selection branch.
			evidence_db: Separate frozen C0-lite DB; enables strict Athaliana K=8 tracks.
			aug_seed: Required integer; private CPU augmentation stream and molecule ranks.
			channel_dropout_p: Whole-source dropout probability.
			thin_fractions: Nonempty uniform choices of molecule retention fraction.
			eval_fraction: Fixed retention fraction, with no augmentation RNG draws.
			return_tile_ids: Append rn/has_evidence metadata (10-tuple instead of 9).
			gff_vocab_version: Vocabulary version for the decoder GFF tokenizer.
				Use "v1" (legacy 272 tokens) when pairing with the published
				jlomas/HyenaTransgenic-* checkpoints (vocab_size 272); "v2"
				(288 tokens) matches newly trained isoform-aware models.
		"""
		self.db = db
		self.mode = mode
		self.split = split
		self.evidence_db = evidence_db
		self.species = species
		self.return_tile_ids = return_tile_ids
		self._evidence_con = self._evidence_pid = None
		self._aug_generator = self._aug_key = None
		self._validated_tiles = set()
		if return_tile_ids and evidence_db is None:
			raise ValueError('return_tile_ids requires evidence_db')
		if evidence_db is not None:
			if species != 'Athaliana' or isinstance(evidence_k, bool) or not isinstance(evidence_k, Integral) or evidence_k != 8:
				raise ValueError('MVP evidence requires species=Athaliana and evidence_k=8')
			if isinstance(aug_seed, bool) or not isinstance(aug_seed, Integral):
				raise ValueError('evidence requires an explicit integer aug_seed')
			try:
				thin_fractions = tuple(thin_fractions)
			except TypeError as exc:
				raise ValueError('thin_fractions must be nonempty') from exc
			values = (channel_dropout_p,) + thin_fractions + (() if eval_fraction is None else (eval_fraction,))
			if not thin_fractions or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
				raise ValueError('dropout and fractions must be finite values in [0,1]')
		self.aug_seed = aug_seed
		self.channel_dropout_p = channel_dropout_p
		self.thin_fractions = thin_fractions
		self.eval_fraction = eval_fraction
		self.strict = strict or evidence_db is not None  # Never substitute evidence tile identities
		self.dt = GFFTokenizer(vocab_version=gff_vocab_version)  # Decoder tokenizer for GFF annotation strings
		self.gff_vocab_version = gff_vocab_version
		self.global_attention = global_attention
		# Load HyenaDNA tokenizer (single-nucleotide: A=0, C=1, G=2, T=3, etc.)
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)
		self.maxlength = 8192 if gff_vocab_version == "v3" else 2048  # Maximum GFF token sequence length (v3 windows hold many genes)
		self._worker_pid = None  # PID of the process that owns _con
		self._con = None         # Persistent DuckDB connection (lazily created)

		# Build an index map at init time: maps dataset index → database row number
		# This is done once in the main process and handles exclude_prefix filtering
		# Also compute isoform counts for sampling weight rebalancing
		self._sample_weights = None
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			if split:
				# B5: membership comes from the frozen split column (docs/gsf_spec_v1.md §7), never from random_split
				q = "SELECT rn, gff FROM geneList WHERE split = ? AND gff IS NOT NULL AND COALESCE(train_weight, 1.0) > 0"
				params = [split]
				if window_len:                      # #18: restrict to one tile tier (fin - start)
					q += " AND fin - start = ?"
					params.append(int(window_len))
				if species is not None:
					q += " AND species_id = ?"
					params.append(species)
				rows = con.sql(q + " ORDER BY rn", params=params).fetchall()
				if not rows:
					raise ValueError(f"no rows with split={split!r} in {db}; build it with scripts/build_b5_database.py")
			elif exclude_prefix:
				# SQL-level filtering: exclude genes whose name starts with the prefix
				q = "SELECT rn, gff FROM geneList WHERE geneModel NOT LIKE ?"
				params = [f"{exclude_prefix}%"]
				if species is not None:
					q += " AND species_id = ?"
					params.append(species)
				rows = con.sql(q + " ORDER BY rn", params=params).fetchall()
			else:
				q = "SELECT rn, gff FROM geneList"
				params = []
				if species is not None:
					q += " WHERE species_id = ?"
					params.append(species)
				rows = con.sql(q + " ORDER BY rn", params=params).fetchall()
			self._index_map = [row[0] for row in rows]  # List of valid row numbers

			# Compute moderate isoform count rebalancing weights for training.
			# Genes with multiple isoforms get a sqrt(isoform_count) weight boost.
			# This gently upsamples multi-isoform genes without being drastic.
			if mode == "train":
				weights = []
				for row in rows:
					gff_str = row[1] if row[1] else ""
					# Count isoforms: number of semicolons in transcript section + 1
					if ">" in gff_str:
						transcript_section = gff_str.split(">", 1)[1]
						iso_count = transcript_section.count(";") + 1
					else:
						iso_count = 1
					# Moderate boost: sqrt(isoform_count) so 4-isoform gene gets 2x weight
					weights.append(math.sqrt(max(iso_count, 1)))
				self._sample_weights = weights

		self._length = len(self._index_map)  # Cache length to avoid repeated DB queries

	def get_sample_weights(self):
		"""Return per-sample weights for isoform count rebalancing.

		Multi-isoform genes receive sqrt(isoform_count) weight boost.
		Returns None in inference mode or if weights were not computed.
		Use with torch.utils.data.WeightedRandomSampler for rebalanced training.
		"""
		return self._sample_weights

	def _get_connection(self):
		"""Return a persistent read-only DuckDB connection for this worker process.

		After fork(), child processes inherit the parent's connection object,
		but DuckDB connections are not fork-safe. We detect this by comparing
		the current PID with the PID that created the connection, and create
		a new connection if they differ.
		"""
		pid = os.getpid()
		if self._worker_pid != pid or self._con is None:
			# Close the inherited (stale) connection from parent process
			if self._con is not None:
				try:
					self._con.close()
				except Exception:
					pass  # Ignore errors from already-dead connections
			# Open a fresh read-only connection for this worker
			self._con = duckdb.connect(self.db, config={"access_mode": "READ_ONLY"})
			self._worker_pid = pid  # Record which PID owns this connection
		return self._con

	def _get_evidence_connection(self):
		pid = os.getpid()
		if self._evidence_pid != pid or self._evidence_con is None:
			if self._evidence_con is not None:
				try:
					self._evidence_con.close()
				except Exception:
					pass
			self._evidence_con = duckdb.connect(self.evidence_db, config={'access_mode': 'READ_ONLY'})
			self._evidence_pid = pid
			self._validated_tiles.clear()
		return self._evidence_con

	def _get_aug_generator(self):
		# Private CPU stream: fixed worker/order reproducibility, not exact resume.
		worker = torch.utils.data.get_worker_info()
		key = (os.getpid(), None if worker is None else (worker.id, worker.seed))
		if self._aug_generator is None or self._aug_key != key:
			seed = int(self.aug_seed) + (0 if worker is None else worker.seed)
			self._aug_generator = torch.Generator(device='cpu').manual_seed(seed % (2**64))
			self._aug_key = key
		return self._aug_generator

	def __getstate__(self):
		state = self.__dict__.copy()
		for name in ('_con', '_worker_pid', '_evidence_con', '_evidence_pid', '_aug_generator', '_aug_key'):
			state[name] = None
		state['_validated_tiles'] = set()
		return state

	def __del__(self):
		"""Also safe when construction failed before opening either connection."""
		for name in ('_con', '_evidence_con'):
			con = getattr(self, name, None)
			if con is not None:
				try:
					con.close()
				except Exception:
					pass
				setattr(self, name, None)

	def __len__(self):
		"""Return the number of samples (after exclude_prefix filtering)."""
		return self._length

	def __getitem__(self, idx):
		"""Load and tokenize one gene sample by dataset index.

		Maps the dataset index to a database row number (via _index_map),
		fetches the record using a persistent DuckDB connection, tokenizes
		the DNA sequence with HyenaDNA tokenizer and the GFF labels with
		GFFTokenizer.

		Args:
			idx: Dataset index (0-based).

		Returns:
			Tuple of (input_ids, attention_mask, labels, gene_model, chr, start, end)
			Labels are None in inference mode.
		"""
		rn = self._index_map[idx]  # Map dataset index to database row number
		con = self._get_connection()  # Get persistent connection for this worker
		try:
			# Fetch all columns from the geneList table by row number
			# Columns: geneModel, region_start, region_end, strand, chr, region_seq, gff, ...
			gm, region_start, region_end, strand, chr, region_seq, gff, sfpb, stpb, fpb, tpb, _ = con.sql("SELECT geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf, rn FROM geneList where rn=?", params=[rn]).fetchall()[0]
		except Exception as e:
			# A35 applied one level below the batch loop: under a frozen recipe a row that cannot be
			# read stops the run. The legacy path keeps the random substitute so published behaviour
			# is unchanged.
			if self.strict:
				raise RuntimeError(
					f"dataset row rn={rn} (index {idx}, split={self.split!r}) could not be read from {self.db}: "
					f"{type(e).__name__}: {e}. Substituting another row would drop this sample and duplicate "
					f"another one silently (protocol A35)."
				) from e
			# Fallback: try a random different sample if this one fails
			newidx = torch.randint(self._length, (1,)).item()
			rn_fallback = self._index_map[newidx]
			print(f"Warning rn={rn} produced {type(e).__name__}: {e}; falling back to rn={rn_fallback}", file=sys.stderr)
			gm, region_start, region_end, strand, chr, region_seq, gff, sfpb, stpb, fpb, tpb, _ = con.sql("SELECT geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf, rn FROM geneList where rn=?", params=[rn_fallback]).fetchall()[0]

		# Tokenize GFF labels using the custom GFFTokenizer
		tx_count = 1  # Default transcript count
		if self.mode == "train":
			# Count the number of isoforms/transcripts for regression head
			if gff and ">" in gff:
				transcript_section = gff.split(">", 1)[1]
				tx_count = min(transcript_section.count(";") + 1, 15)
			tokens = self.dt._tokenize(gff)  # Split GFF into token strings
			token_ids = [self.dt._convert_token_to_id(t) for t in tokens]  # Convert to integer IDs
			# Truncate if over max length, preserving </s> end token
			if len(token_ids) > self.maxlength:
				token_ids = token_ids[:self.maxlength-1] + [self.dt.vocab["</s>"]]
			labels = torch.tensor([token_ids])  # Shape: (1, seq_len)

		# Tokenize the full DNA sequence with HyenaDNA tokenizer
		# HyenaDNA uses single-nucleotide vocabulary (no segmentation needed)
		if not region_seq:
			if self.strict:
				raise RuntimeError(
					f"dataset row rn={rn} (index {idx}, split={self.split!r}, geneModel={gm!r}) holds an empty "
					f"sequence in {self.db}; refusing to substitute another row (protocol A35)."
				)
			fallback_idx = torch.randint(self._length, (1,)).item()
			print(
				f"Warning rn={rn} produced empty sequence region; falling back to rn={self._index_map[fallback_idx]}",
				file=sys.stderr,
			)
			return self.__getitem__(fallback_idx)

		seqs = self.encoder_tokenizer(region_seq, return_tensors="pt")
		if seqs["input_ids"].shape[1] > 1:
			seqs["input_ids"] = seqs["input_ids"][:, :-1]  # Remove the trailing [SEP] token
		if seqs["input_ids"].shape[1] == 0:
			if self.strict:
				raise RuntimeError(
					f"dataset row rn={rn} (index {idx}, split={self.split!r}, geneModel={gm!r}) tokenised to zero "
					f"length ({len(region_seq)} nt of sequence); refusing to substitute another row (protocol A35)."
				)
			fallback_idx = torch.randint(self._length, (1,)).item()
			print(
				f"Warning rn={rn} tokenized to zero length; falling back to rn={self._index_map[fallback_idx]}",
				file=sys.stderr,
			)
			return self.__getitem__(fallback_idx)

		# Create attention mask: True for real nucleotide tokens, False for padding
		attention_mask = (seqs["input_ids"] != self.encoder_tokenizer.pad_token_id)

		sample = (seqs['input_ids'], attention_mask, labels if self.mode == 'train' else None,
			gm, chr, region_start, region_end, tx_count if self.mode == 'train' else 0)
		if self.evidence_db is None:
			return sample

		from ..evidence.mvp_tracks import rasterise_est
		metadata = con.execute('SELECT split, is_rc FROM geneList WHERE rn=?', [rn]).fetchall()
		if len(metadata) != 1 or type(metadata[0][1]) is not bool:
			raise ValueError(f'rn={rn}: requires explicit non-null boolean is_rc')
		split, is_rc = metadata[0]
		if region_end - region_start != len(region_seq) or len(region_seq) != seqs['input_ids'].shape[1]:
			raise ValueError(f'rn={rn}: region, sequence and nucleotide token lengths must agree')
		evidence_con = self._get_evidence_connection()
		if rn not in self._validated_tiles:
			expected = (rn, chr, region_start, region_end, split, is_rc)
			frozen = evidence_con.execute('SELECT rn, chrom, start0, end0, split, is_rc FROM corpus_tile WHERE rn=?', [rn]).fetchall()
			if frozen != [expected] or type(frozen[0][5]) is not bool:
				raise ValueError(f'rn={rn}: evidence corpus_tile metadata mismatch')
			self._validated_tiles.add(rn)
		if self.eval_fraction is not None:
			fraction, dropped = self.eval_fraction, False
		else:
			u = torch.rand(2, generator=self._get_aug_generator())
			dropped = u[0].item() < self.channel_dropout_p
			fraction = self.thin_fractions[int(u[1].item() * len(self.thin_fractions))]
		tracks = rasterise_est(evidence_con, sequence=region_seq, chrom=chr,
			offset=region_start, L=region_end-region_start, is_rc=is_rc,
			keep_fraction=fraction, aug_seed=self.aug_seed, rn=rn,
			population='retained', source_supplied=not dropped)
		sample += (torch.from_numpy(np.ascontiguousarray(tracks, dtype=np.float16)),)
		if self.return_tile_ids:
			has_evidence = evidence_con.execute('SELECT EXISTS (SELECT 1 FROM tile_est WHERE rn=?)', [rn]).fetchone()[0]
			sample += ({'rn': rn, 'has_evidence': has_evidence},)
		return sample


class segmentationDataset(Dataset):
	"""Sliding-window segmentation dataset for SegmentNT-style training.

	Partitions entire chromosomes into fixed-size windows with overlap,
	then creates per-position multi-class segmentation labels from GFF
	annotations. Each sample has:
	  Input: DNA window → tokenized by NT encoder
	  Target: (window_size, num_classes) binary tensor

	Classes include: protein_coding_gene, lncRNA, exon, intron,
	splice_donor, splice_acceptor, 5UTR, 3UTR, CTCF-bound,
	polyA_signal, enhancer/promoter (tissue-specific/invariant).
	"""
	def __init__(self, table, window_size, step_size, db, encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", preprocess=False):
		"""
		Args:
			table: Database table prefix (e.g., "rice" → uses rice_genome, rice_gff tables).
			window_size: Size of each genomic window in base pairs.
			step_size: Stride between consecutive windows (overlap = window_size - step_size).
			db: Path to DuckDB database.
			encoder_model: HuggingFace model ID for the encoder tokenizer.
			preprocess: If True, compress labels to zlib BLOBs (for storage).
		"""
		self.window_size = window_size
		self.step_size = step_size
		self.db = db
		self.preprocess = preprocess
		self.encoder_model = encoder_model
		self.table = table
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)

		# Segmentation class definitions (14 classes for multi-label classification)
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

		# Map GFF feature types to segmentation class names
		self.gffClassMap = {'gene': 'protein_coding_gene',
					'exon': 'exon',
					'intron': 'intron',
					'five_prime_cis_splice_site': 'splice_donor',
					'three_prime_cis_splice_site': 'splice_acceptor',
					'five_prime_UTR': '5UTR',
					'three_prime_UTR': '3UTR'}

		# Pre-compute all genomic windows across all chromosomes
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			seqLengths = con.sql(f"SELECT organism, chromosome, length FROM {table}_genome").df()

		window_list = []
		for i in range(len(seqLengths)):
			organism = seqLengths.loc[i, 'organism']
			chromosome = seqLengths.loc[i, 'chromosome']
			length = seqLengths.loc[i, 'length']
			# Number of windows that fit in this chromosome
			windows = (length - self.window_size) // self.step_size + 1

			for j in range(windows):
				start = j * self.step_size
				end = start + self.window_size
				window_list.append([organism, chromosome, start, end])

		# Store all windows in a DataFrame for indexed access
		self.windows = pd.DataFrame(window_list, columns=['organism', 'chromosome', 'start', 'end'])

	def __len__(self):
		"""Return total number of genomic windows."""
		return len(self.windows)

	def __getitem__(self, idx):
		"""Load one genomic window and its segmentation labels.

		Fetches the DNA sequence for the window, constructs per-position
		multi-class labels from overlapping GFF annotations, and tokenizes
		the sequence for the encoder.

		Sequences containing 'N' (ambiguous nucleotides) are rejected and
		replaced with a random different window.
		"""
		window = self.windows.loc[idx]
		# Fetch the full chromosome sequence from the database
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			sequence = con.sql(
				"SELECT sequence "
				f"FROM {self.table}_genome "
				f"WHERE chromosome = '{window['chromosome']}' "
				f"AND organism = '{window['organism']}'").fetchall()[0][0]
		# Extract the window from the chromosome
		sequence = sequence[window['start']:window['end']]

		# Reject sequences with ambiguous nucleotides (N)
		if "N" in sequence:
			return self.__getitem__(torch.randint(0, len(self.windows), (1,)).item())

		# Fetch all GFF annotations that overlap this window
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			annotations = con.sql(
				"SELECT feature, start, fin "
				f"FROM {self.table}_gff "
				f"WHERE chromosome = '{window['chromosome']}' "
				f"AND organism = '{window['organism']}' "
				f"AND (start <= {window['end']} AND fin >= {window['start']})").df()

		# Adjust annotation coordinates to be relative to the window start
		annotations['start'] = annotations['start'].apply(lambda x: x - window['start'] - 1)
		annotations['fin'] = annotations['fin'].apply(lambda x: x - window['start'])
		# Clip to window boundaries
		annotations['start'] = annotations['start'].apply(lambda x: max(x, 0))
		annotations['fin'] = annotations['fin'].apply(lambda x: min(x, self.window_size))

		# Build per-position multi-class label tensor: (window_size, num_classes)
		class_tensor = torch.zeros((self.window_size, len(self.classes)), dtype=torch.float32)

		for i in range(len(annotations)):
			start = annotations.loc[i, 'start']
			end = annotations.loc[i, 'fin']
			feature = annotations.loc[i, 'feature']
			# Map GFF feature type to class index and set label to 1
			if feature in self.gffClassMap:
				class_idx = self.classes.index(self.gffClassMap[feature])
				class_tensor[start:end, class_idx] = 1

		# Optionally compress labels for storage (preprocessing mode)
		if self.preprocess:
			class_tensor = zlib.compress(class_tensor.numpy().tobytes())

		# Tokenize the DNA sequence for the encoder
		if self.preprocess:
			seqs = sequence  # Raw string for storage
			encoder_attention_mask = None
		else:
			# Segment into 6144-nt chunks and tokenize with NT encoder
			seqs = segmentSequence(sequence, piece_size=6144)
			seqs = self.encoder_tokenizer.batch_encode_plus(
				seqs,
				return_tensors="pt",
				padding="max_length",
				truncation=True,
				max_length=1024)["input_ids"]
			encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, window['organism'], window['chromosome'], window['start'], window['end'])


class preprocessedSegmentationDataset(Dataset):
	"""Pre-segmented dataset for NT encoder segmentation training.

	Reads pre-computed windows from a DuckDB table where labels are
	stored as zlib-compressed binary BLOBs. This avoids the overhead
	of fetching full chromosome sequences and computing windows at
	runtime.
	"""
	def __init__(self, db, encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b"):
		self.db = db
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)

	def __len__(self):
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM data").fetchall()[0][0]

	def __getitem__(self, idx):
		idx = idx + 1  # 1-indexed row numbers
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			_, sequence, label, organism, chromosome, start, fin, _ = con.sql(f"SELECT * FROM data where rn={idx}").fetchall()[0]

		# Reject ambiguous sequences
		if "N" in sequence:
			return self.__getitem__(torch.randint(0, self.__len__(), (1,)).item())

		# Decompress label tensor from zlib BLOB: (6144, 14) float32
		try:
			class_tensor = np.frombuffer(zlib.decompress(label), dtype=np.float32).reshape(6144, 14)
			class_tensor = torch.from_numpy(class_tensor)
		except:
			newidx = torch.randint(0, self.__len__(), (1,)).item()
			print(f"Warning labe with {idx=} could not be parsed... using {newidx}")
			return self.__getitem__(newidx)

		# Segment and tokenize for NT encoder (6144-nt chunks → 1024-token segments)
		seqs = segmentSequence(sequence, piece_size=6144)
		seqs = self.encoder_tokenizer.batch_encode_plus(
			seqs,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length=1024)["input_ids"]
		encoder_attention_mask = (seqs != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, organism, chromosome, start, fin)


class preprocessedSegmentationDatasetHyena(Dataset):
	"""Pre-segmented dataset for HyenaDNA encoder segmentation training.

	Same as preprocessedSegmentationDataset but uses HyenaDNA tokenizer
	(single-nucleotide vocabulary) instead of NT's 6-mer tokenizer.
	No sequence segmentation is needed since HyenaDNA handles full-length input.
	"""
	def __init__(self, db, encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf"):
		self.db = db
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)

	def __len__(self):
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM data").fetchall()[0][0]

	def __getitem__(self, idx):
		idx = idx + 1  # 1-indexed row numbers
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			_, sequence, label, organism, chromosome, start, fin, _ = con.sql(f"SELECT * FROM data where rn={idx}").fetchall()[0]

		# Reject ambiguous sequences
		if "N" in sequence:
			return self.__getitem__(torch.randint(0, self.__len__(), (1,)).item())

		# Decompress label tensor from zlib BLOB
		try:
			class_tensor = np.frombuffer(zlib.decompress(label), dtype=np.float32).reshape(6144, 14)
			class_tensor = torch.from_numpy(class_tensor)
		except:
			newidx = torch.randint(0, self.__len__(), (1,)).item()
			print(f"Warning labe with {idx=} could not be parsed... using {newidx}")
			return self.__getitem__(newidx)

		# Tokenize with HyenaDNA (single nucleotide per token, no segmentation)
		seqs = self.encoder_tokenizer.batch_encode_plus([sequence], return_tensors="pt")
		seqs["input_ids"] = seqs["input_ids"][:, :-1]  # Remove trailing [SEP] token
		encoder_attention_mask = (seqs["input_ids"] != self.encoder_tokenizer.pad_token_id)

		return (seqs, encoder_attention_mask, class_tensor, organism, chromosome, start, fin)


class MLMDatasetHyena(Dataset):
	"""Masked Language Modeling dataset for HyenaDNA pretraining.

	Masks random groups of 3 contiguous nucleotides (to preserve codon
	structure) and trains the model to predict the original nucleotides.
	Used for self-supervised pretraining of the HyenaDNA encoder.

	Default masking: 921 nucleotides per sequence (307 groups of 3),
	which is approximately 15% of a 6144-nt window.
	"""
	def __init__(self, db, encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf"):
		self.db = db
		self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir="./HFmodels", trust_remote_code=True)

	def __len__(self):
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			return con.sql("SELECT COUNT(*) FROM data").fetchall()[0][0]

	def __getitem__(self, idx):
		idx = idx + 1  # 1-indexed row numbers
		with duckdb.connect(self.db, config={"access_mode": "READ_ONLY"}) as con:
			_, sequence, label, organism, chromosome, start, fin, _ = con.sql(f"SELECT * FROM data where rn={idx}").fetchall()[0]

		# Tokenize the full DNA sequence with HyenaDNA tokenizer
		seqs = self.encoder_tokenizer.batch_encode_plus([sequence], return_tensors="pt")
		seqs["input_ids"] = seqs["input_ids"][:, :-1]  # Remove trailing [SEP] token
		encoder_attention_mask = (seqs["input_ids"] != self.encoder_tokenizer.pad_token_id)

		# Apply random masking: replace groups of 3 nucleotides with mask token
		# Returns (masked_sequences, boolean_mask_positions)
		masked_seqs, mask_index = mask_sequences(seqs.input_ids)

		# Return: masked input, mask positions, original tokens (labels), metadata
		return (masked_seqs, mask_index, seqs.input_ids, organism, chromosome, start, fin)


# ═══════════════════════════════════════════════════════════════════════════
# Collate Functions
# ═══════════════════════════════════════════════════════════════════════════
# Collate functions handle padding batches of variable-length sequences
# to a uniform size for batched GPU processing. Each variant matches
# a specific dataset class and encoder type.

def target_collate_fn(batch):
	"""Collate function for isoformData (NT encoder, multi-segment).

	Pads variable-segment-count inputs to the maximum number of segments
	in the batch, and pads labels to the maximum label length.
	"""
	# Unpack batch tuples
	sequences, attention_masks, global_attention_masks, labels, gm, chr, region_start, region_end = zip(*batch)

	# Pad sequences: flatten segments, pad to max total length, stack
	max_segs = max([seq.shape[0] for seq in sequences])  # Max segment count in batch
	sequences = [seq.flatten() for seq in sequences]      # Flatten (num_segs, 1024) → (num_segs*1024,)
	sequences = [F.pad(seq, (0, max_segs*1024 - seq.shape[0]), value=1) for seq in sequences]  # Pad with pad_token_id=1
	sequences = torch.stack(sequences)

	# Pad attention masks to match padded sequence length
	attention_masks = [mask.flatten() for mask in attention_masks]
	attention_masks = [F.pad(mask, (0, max_segs*1024 - mask.shape[0]), value=False) for mask in attention_masks]
	attention_masks = torch.stack(attention_masks)

	# Pad global attention masks if present
	if global_attention_masks[0] is not None:
		global_attention_masks = [mask.flatten() for mask in global_attention_masks]
		global_attention_masks = [F.pad(mask, (0, max_segs*1024 - mask.shape[0]), value=False) for mask in global_attention_masks]
		global_attention_masks = torch.stack(global_attention_masks)

	# Pad labels to max label length in batch
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]  # Right-pad with 0
		labels_padded = torch.cat(labels_padded)  # Concatenate along batch dim

	if labels:
		return sequences, attention_masks, global_attention_masks, labels_padded, gm, chr, region_start, region_end
	else:
		return sequences, attention_masks, global_attention_masks, None, gm, chr, region_start, region_end


def hyena_collate_fn(batch):
	"""Collate function for isoformDataHyena (HyenaDNA encoder, variable-length).

	Pads variable-length DNA sequences to the maximum length in the batch.
	LEFT-pads sequences (padding at the beginning) because HyenaDNA's
	causal convolutions are more stable with trailing real tokens.
	Labels are RIGHT-padded (padding at the end) as is standard.

	This is the PRIMARY collate function used during TransGenic training.
	"""
	arity = len(batch[0]) if batch else 0
	if arity not in (7, 8, 9, 10) or any(len(sample) != arity for sample in batch):
		raise ValueError('expected a nonempty batch with uniform tuple arity 7, 8, 9 or 10')
	if arity >= 8:
		sequences, attention_masks, labels, gm, chr, region_start, region_end, tx_counts = zip(*(sample[:8] for sample in batch))
		tx_counts = torch.tensor(tx_counts, dtype=torch.float32)
	else:
		# Backward compatibility with 7-element tuples
		sequences, attention_masks, labels, gm, chr, region_start, region_end = zip(*batch)
		tx_counts = None

	# LEFT-pad sequences to the longest sequence in the batch
	dna_max_len = max([seq.shape[1] for seq in sequences])
	if arity >= 9:
		for sample in batch:
			t = sample[8]
			if not isinstance(t, torch.Tensor) or t.dtype != torch.float16 or t.shape != (sample[0].shape[1], 8):
				raise ValueError('evidence tracks must be float16 [DNA token length, 8]')
		tracks = torch.stack([F.pad(sample[8], (0, 0, dna_max_len-sample[8].shape[0], 0)) for sample in batch])
	sequences = [F.pad(seq, (dna_max_len - seq.shape[1], 0)) for seq in sequences]  # Left-pad
	sequences = torch.cat(sequences)  # Stack into (batch_size, max_len)

	# LEFT-pad attention masks to match
	max_len = max([mask.shape[1] for mask in attention_masks])
	attention_masks = [F.pad(mask, (max_len - mask.shape[1], 0)) for mask in attention_masks]
	attention_masks = torch.cat(attention_masks)

	# RIGHT-pad labels to the longest label in the batch
	if None not in labels:
		label_max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, label_max_len - label.shape[1]), value=-100) for label in labels]  # Right-pad with ignore index
		labels_padded = torch.cat(labels_padded)

	if None not in labels:
		result = (sequences, attention_masks, labels_padded, gm, chr, region_start, region_end, tx_counts)
	else:
		result = (sequences, attention_masks, None, gm, chr, region_start, region_end, None)
	if arity >= 9:
		result += (tracks,)
	if arity == 10:
		result += ({'rn': torch.tensor([s[9]['rn'] for s in batch], dtype=torch.int64),
			'has_evidence': torch.tensor([s[9]['has_evidence'] for s in batch], dtype=torch.bool)},)
	return result


def hyenaMLM_collate_fn(batch):
	"""Collate function for MLMDatasetHyena (masked language modeling).

	Same padding strategy as hyena_collate_fn: left-pad sequences,
	left-pad masks, left-pad labels (original tokens for reconstruction).
	"""
	sequences, attention_masks, labels, gm, chr, region_start, region_end = zip(*batch)

	# LEFT-pad masked sequences
	max_len = max([seq.shape[1] for seq in sequences])
	sequences = [F.pad(seq, (max_len - seq.shape[1], 0)) for seq in sequences]
	sequences = torch.cat(sequences)

	# LEFT-pad attention masks
	max_len = max([mask.shape[1] for mask in attention_masks])
	attention_masks = [F.pad(mask, (max_len - mask.shape[1], 0)) for mask in attention_masks]
	attention_masks = torch.cat(attention_masks)

	# LEFT-pad original (unmasked) labels for reconstruction loss
	max_len = max([label.shape[1] for label in labels])
	labels = [F.pad(label, (max_len - label.shape[1], 0)) for label in labels]
	labels = torch.cat(labels)

	return sequences, attention_masks, labels, gm, chr, region_start, region_end


def segment_collate_fn(batch):
	"""Collate function for segmentationDataset (NT encoder, multi-segment).

	Pads multi-segment sequences and per-position class tensors for
	batched segmentation training.
	"""
	sequences, attention_masks, labels, organism, chromosome, start, end = zip(*batch)

	# Pad sequences: flatten segments, pad to max total length, stack
	max_segs = max([seq.shape[0] for seq in sequences])
	sequences = [seq.flatten() for seq in sequences]
	sequences = [F.pad(seq, (0, max_segs*1024 - seq.shape[0]), value=1) for seq in sequences]
	sequences = torch.stack(sequences)

	# Pad attention masks
	attention_masks = [mask.flatten() for mask in attention_masks]
	attention_masks = [F.pad(mask, (0, max_segs*1024 - mask.shape[0]), value=False) for mask in attention_masks]
	attention_masks = torch.stack(attention_masks)

	# Pad class label tensors (per-position multi-class labels)
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]
		labels_padded = torch.stack(labels_padded)

	if labels:
		return sequences, attention_masks, labels_padded, organism, chromosome, start, end
	else:
		return sequences, attention_masks, None, organism, chromosome, start, end


def hyena_segment_collate_fn(batch):
	"""Collate function for preprocessedSegmentationDatasetHyena.

	Pads HyenaDNA-tokenized sequences and per-position class tensors
	for batched segmentation training.
	"""
	sequences, attention_masks, labels, organism, chromosome, start, end = zip(*batch)

	# LEFT-pad input sequences (HyenaDNA dict format)
	max_len = max([seq["input_ids"].shape[1] for seq in sequences])
	sequences = [F.pad(seq["input_ids"], (max_len - seq["input_ids"].shape[1], 0)) for seq in sequences]
	sequences = torch.cat(sequences)

	# LEFT-pad attention masks
	max_len = max([mask.shape[1] for mask in attention_masks])
	attention_masks = [F.pad(mask, (max_len - mask.shape[1], 0)) for mask in attention_masks]
	attention_masks = torch.cat(attention_masks)

	# Pad class label tensors
	if labels:
		max_len = max([label.shape[1] for label in labels])
		labels_padded = [F.pad(label, (0, max_len - label.shape[1])) for label in labels]
		labels_padded = torch.stack(labels_padded)

	if labels:
		return sequences, attention_masks, labels_padded, organism, chromosome, start, end
	else:
		return sequences, attention_masks, None, organism, chromosome, start, end


def makeDataLoader(dat, shuffle=True, batch_size=8, pin_memory=True, prefetch_factor=2, sampler=None, num_workers=0, collate_fn=target_collate_fn, persistent_workers=False, generator=None):
	"""Create a PyTorch DataLoader with sensible defaults for TransGenic training.

	Args:
		dat: Dataset instance.
		shuffle: Whether to shuffle data (disabled when sampler is provided).
		batch_size: Number of samples per batch.
		pin_memory: Pin CPU tensors in page-locked memory for faster GPU transfer.
		            Set False on unified memory systems (GB10).
		sampler: Optional custom sampler (disables shuffle when set).
		num_workers: Number of DataLoader worker processes (0 = main process only).
		collate_fn: Function to collate/pad a list of samples into a batch.
		persistent_workers: Keep workers alive between epochs to avoid respawn cost.
		generator: Optional torch.Generator for deterministic shuffle order.
		           Used for reproducible mid-epoch resume.

	Returns:
		Configured DataLoader instance.
	"""
	if sampler != None:
		shuffle = False  # Sampler and shuffle are mutually exclusive in PyTorch

	return DataLoader(
		dat,
		shuffle=shuffle,
		collate_fn=collate_fn,
		batch_size=batch_size,
		pin_memory=pin_memory,
		sampler=sampler,
		num_workers=num_workers,
		prefetch_factor=prefetch_factor if num_workers > 0 else None,  # Pre-fetch 2 batches per worker
		persistent_workers=persistent_workers if num_workers > 0 else False,  # Only valid with workers
		generator=generator,)
