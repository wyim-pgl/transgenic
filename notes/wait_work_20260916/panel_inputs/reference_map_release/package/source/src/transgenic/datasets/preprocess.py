"""
DuckDB Database Builder for TransGenic Training and Inference.

Converts genome FASTA files and GFF3/BED annotation files into DuckDB databases
that serve as the data backend for TransGenic's PyTorch DataLoader pipeline.

The main function, genome2GSFDataset(), processes a sorted GFF3 file line-by-line,
extracting each gene region with flanking genomic context and converting its
annotation to the compact GSF (Gene Sentence Format) string representation.

Database Schema (geneList table):
    rn              INT PRIMARY KEY   -- Auto-incrementing row ID
    geneModel       VARCHAR           -- Gene ID from GFF3 ID= attribute
    start           INT               -- 0-indexed region start in chromosome
    fin             INT               -- Region end (exclusive, Python-style)
    strand          VARCHAR           -- Gene strand: "+" or "-"
    chromosome      VARCHAR           -- Chromosome/scaffold name
    sequence        VARCHAR           -- Extracted DNA sequence (gene + flanking)
    gff             VARCHAR           -- GSF annotation string (NULL in predict mode)
    static_fpb      INT               -- Static 5' flanking buffer size (bp)
    static_tpb      INT               -- Static 3' flanking buffer size (bp)
    five_prime_buf  INT               -- Random 5' offset for training augmentation
    three_prime_buf INT               -- Random 3' offset for training augmentation

Key design decisions:
    - Sequences are padded with flanking genomic context to the nearest multiple
      of staticSize (default 6144bp), matching the HyenaDNA encoder chunk size.
    - In training mode, random buffer offsets (addExtra) teach the model to locate
      UTR boundaries regardless of their position within the input window.
    - Reverse complement augmentation (addRC) doubles data for splice variant
      diversity; addRCIsoOnly restricts this to multi-isoform genes.
    - The clean flag filters genes with invalid CDS (missing start/stop codons
      or reading frame errors) to ensure training labels are biologically valid.
    - GFF3 phase values (0/1/2) are remapped to GSF phase tokens (A/B/C).
    - Feature deduplication ensures that CDS/UTR features shared across
      alternative transcripts appear only once in the feature list.

This module also includes two legacy/auxiliary functions for segmentation tasks:
    - genome2SegmentationDataset: Stores raw GFF3 + genome in DuckDB (slow)
    - genome2PreprocessedSegmentationDt: Creates windowed segmentation data with
      14-class binary labels for nucleotide-level feature prediction
"""

import os
import sys
import zlib
import duckdb
import torch
import pandas as pd
from tqdm import tqdm

from ..utils.sequence import loadGenome, reverseComplement, validateCDS
from ..utils.gsf import reverseComplement_gffString

def genome2GSFDataset(
		genome: str,
		gff3: str,
		db: str,
		anoType = 'gff',
		mode = 'predict',
		maxLen=49152,
		addExtra=0,
		staticSize=6144,
		addRC=False,
		addRCIsoOnly=False,
		clean=False,
		speciesPrefix: str = ''
	):
	"""
	Build a DuckDB database from a genome FASTA and GFF3/BED annotation file.

	Reads a sorted GFF3 (or BED) file line-by-line, extracts each gene region
	with flanking genomic context padded to the nearest multiple of staticSize,
	converts the annotation to GSF format, and inserts the result into a DuckDB
	database. Can be called multiple times to append multiple genomes.

	Args:
		genome (str):       Path to genome FASTA file.
		gff3 (str):         Path to sorted GFF3 or BED annotation file.
		db (str):           Path to DuckDB database file (created if not exists).
		anoType (str):      Input format - "gff" for GFF3, "bed" for BED. Default "gff".
		mode (str):         "train" includes GSF labels; "predict" stores only
		                    gene regions (skips sub-gene features). Default "predict".
		maxLen (int):       Skip genes whose padded region exceeds this length.
		                    Default 49152 (8 x 6144 = max HyenaDNA encoder input).
		addExtra (int):     Max random buffer (bp) added to each side for training
		                    augmentation. Helps model learn UTR boundaries. Default 0.
		staticSize (int):   Pad sequences to multiples of this size. Must match
		                    the encoder chunk size. Default 6144.
		addRC (bool):       Add reverse complement copies for data augmentation.
		addRCIsoOnly (bool): When addRC=True, only augment genes with alternative
		                    splicing (multiple mRNA transcripts).
		clean (bool):       Validate CDS integrity (start/stop codons, reading
		                    frame) before insertion; skip invalid genes.
		speciesPrefix (str): Prefix to prepend to chromosome names (e.g., "Zm"
		                    turns "Chr01" into "Zm_Chr01"). Useful for multi-species
		                    databases to avoid chromosome name collisions. Default "".
	"""
	# Since 2026-09-02 this function delegates to build_b5.build_species (docs/gsf_spec_v1.md):
	# the last gene of a file is flushed, exact-multiple windows get no extra chunk, records over the
	# caps are rejected instead of truncated, inserts are parameterised and the GSF text is canonical
	# (gsf-order-v1). Legacy columns are unchanged; split/provenance columns are NULL when no split
	# table is supplied. B5 builds must use scripts/build_b5_database.py with the frozen split table.
	from .build_b5 import build_species
	import duckdb as _duckdb
	if anoType != 'gff':
		raise NotImplementedError("BED input is no longer supported; convert to GFF3 first")
	if addRCIsoOnly and not addRC:
		addRC = True  # --add-rc-iso-only alone used to be a silent no-op
	rc = 'none' if not addRC else ('isoform-only' if addRCIsoOnly else 'all')
	species_id = speciesPrefix or os.path.splitext(os.path.basename(genome))[0]
	con = _duckdb.connect(db)
	try:
		result = build_species(con, species_id, genome, gff3, split_rows={}, split_sha="", rc=rc, add_extra=addExtra,
		                       max_len=maxLen, clean=clean, mode=mode, allow_missing_split=True)
	finally:
		con.close()
	for r in result["rejected"]:
		print(f"Skipping {r['gene_id']}: {r['reason']}", file=sys.stderr)
	if staticSize != 6144:
		print(f"Warning: staticSize={staticSize} ignored; the frozen window policy is sym6144-v1", file=sys.stderr)
	return result


def genome2SegmentationDataset(genome_file, gff_file, organism, db):
	"""
	Load raw GFF3 annotations and genome sequences into DuckDB tables.

	Creates two tables per organism:
	  - {organism}_gff:    Full GFF3 annotation rows
	  - {organism}_genome: Chromosome sequences with lengths

	This is a LEGACY implementation. On-the-fly feature extraction during
	dataloading is very slow. Use genome2PreprocessedSegmentationDt instead.

	Args:
		genome_file (str): Path to genome FASTA file.
		gff_file (str):    Path to GFF3 annotation file.
		organism (str):    Organism name (used as table name prefix).
		db (str):          Path to DuckDB database file.
	"""
	# Load GFF3 into a pandas DataFrame and store in DuckDB
	table = organism
	gff3 = pd.read_csv(gff_file, sep='\t', header=None, comment='#')
	gff3.columns = ['chromosome', 'source', 'feature', 'start', 'fin', 'score', 'strand', 'frame', 'attribute']
	gff3['organism'] = organism
	with duckdb.connect(db) as con:
		con.sql(
			f'CREATE TABLE IF NOT EXISTS {table}_gff ('
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
			f'INSERT INTO {table}_gff '
			'SELECT * '
			'FROM gff3; '
		)

	# Load chromosome sequences into DuckDB with length metadata
	genome_dict = loadGenome(genome_file)
	genome_df = pd.DataFrame(genome_dict.items(), columns=['chromosome', 'sequence'])
	genome_df['organism'] = organism
	genome_df['length'] = genome_df['sequence'].apply(len)
	with duckdb.connect(db) as con:
		con.sql(
			f"CREATE TABLE IF NOT EXISTS {table}_genome ("
			'chromosome VARCHAR, '
			'sequence VARCHAR, '
			'organism VARCHAR, '
			'length INT)')

		con.sql(
			f'INSERT INTO {table}_genome '
			'SELECT * '
			'FROM genome_df; '
		)


def genome2PreprocessedSegmentationDt(db, genome, gff, table, window_size, step_size):
	"""
	Create a pre-processed segmentation dataset with sliding windows.

	Converts genome + GFF3 into a TSV file where each row is a fixed-size
	genomic window with a 14-class binary label tensor (zlib-compressed).
	This pre-computes the nucleotide-level labels once instead of doing it
	on-the-fly during training, dramatically improving dataloading speed.

	The 14 segmentation classes are:
	    [0]  protein_coding_gene      [7]  3UTR
	    [1]  lncRNA                   [8]  CTCF-bound
	    [2]  exon                     [9]  polyA_signal
	    [3]  intron                   [10] enhancer_Tissue_specific
	    [4]  splice_donor             [11] enhancer_Tissue_invariant
	    [5]  splice_acceptor          [12] promoter_Tissue_specific
	    [6]  5UTR                     [13] promoter_Tissue_invariant

	Args:
		db (str):          Output TSV file path (appended to).
		genome (str):      Path to genome FASTA file.
		gff (str):         Path to GFF3 annotation file.
		table (str):       Organism/dataset label for the TSV rows.
		window_size (int): Size of each sliding window (bp).
		step_size (int):   Step between consecutive windows (bp).
	"""
	# 14 nucleotide-level segmentation classes for multi-label prediction
	classes = ['protein_coding_gene',
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

	# Map GFF3 feature types to segmentation class names
	gffClassMap = {'gene': 'protein_coding_gene',
					'exon': 'exon',
					'intron': 'intron',
					'five_prime_cis_splice_site': 'splice_donor',
					'three_prime_cis_splice_site': 'splice_acceptor',
					'five_prime_UTR': '5UTR',
					'three_prime_UTR': '3UTR'}

	# Read genome sequence and GFF3 annotation into memory
	genome = loadGenome(genome)
	gff = pd.read_csv(gff, sep='\t', header=None, comment='#')
	gff.columns = ['chromosome', 'source', 'feature', 'start', 'fin', 'score', 'strand', 'frame', 'attributes']


	for chr in genome:
		# Build a per-nucleotide binary class tensor for this chromosome
		sequence = genome[chr]
		chr_gff = gff[gff['chromosome'] == chr].reset_index(drop=True)
		# Shape: (chromosome_length, 14) -- one-hot per nucleotide per class
		class_tensor = torch.zeros((len(sequence), len(classes)), dtype=torch.float32)

		print(f"Processing chr {chr}...", file=sys.stderr)
		skip = False
		for i, row in tqdm(chr_gff.iterrows()):
			start = row['start']
			end = row['fin']
			feature = row['feature']
			# Look ahead to detect lncRNA genes (skip their children)
			nextfeature = chr_gff.loc[i+1, 'feature'] if i+1 < len(chr_gff) else None
			if feature == 'gene':
				skip = False
			if nextfeature == 'lncRNA':
				skip = True
			if not skip:
				if feature in gffClassMap:
					# Set the binary label for this class across the feature's span
					class_idx = classes.index(gffClassMap[feature])
					class_tensor[start:end, class_idx] = 1

		# Slide a fixed-size window across the chromosome and output each window
		print(f"Adding {chr} windows...", file=sys.stderr)
		for i in tqdm(range(0, len(sequence), step_size)):
			start = i
			end = i + window_size
			if end > len(sequence):
				end = len(sequence)
			window_seq = sequence[start:end]
			window_class = class_tensor[start:end]
			# Compress the label tensor with zlib to reduce file size
			window_class = zlib.compress(window_class.numpy().tobytes())
			# Append as tab-separated: sequence, compressed_labels, organism, chr, start, end
			with open(db, 'a') as f:
				f.write(f"{window_seq}\t{window_class}\t{table}\t{chr}\t{start}\t{end}\n")
