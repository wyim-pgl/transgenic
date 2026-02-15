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
		clean=False
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
	"""
	# Purpose:
	#   Read a genome assembly and gff3 annotation file into a
	#   duckdb database for training or inference. For each gene
	#   model, the corresponding nucleotide sequence is extracted
	#   from the genome and the target annotation string is created.
	#   The function may be called multiple times to append add
	#   multiple genomes to the database

	# Inputs:
	#   genome: 	path to a fasta file containing the genome assembly
	#   gff3:		path to a gff3 file containing gene annotations
	#   db:			path to a duckdb database file (will be created if it does not exist)
	#   anoType:	'gff' or 'bed' (default: 'gff'). The type of annotation file to use
	#   mode:		'train' or 'predict' (default: 'predict'). Train mode includes gff3 labels in the database
	#   maxLen:		Maximum size of gene model sequence to include in the database (larger gene models are skipped)
	#   addExtra:	Max size of random buffer to add to the gene model sequence (used to capture UTR start and end during training)
	#   staticSize:	Extracted sequences will be a multiple of this size (padded with random amounts of neighboring sequence)
	#   addRC:		Add reverse complement of gene model sequence to the database (Used to augment training data)
	#   addRCIsoOnly: When adding RC seqs, only add gene models with alternative splicing
	#   clean:		Only add sequences that have start and stop codons and are a multiple of 3

	# Outputs:
	#   A duckdb database used to load data into the model

	# ═══════════════════════════════════════════════════════════════
	# Load genome FASTA into a dict: {chromosome_name: sequence_string}
	# ═══════════════════════════════════════════════════════════════
	genome_dict = loadGenome(genome)

	# ═══════════════════════════════════════════════════════════════
	# Connect to DuckDB and initialize the geneList table + row counter
	# "CREATE TABLE IF NOT EXISTS" allows appending multiple genomes
	# "CREATE SEQUENCE" provides auto-incrementing primary key
	# ═══════════════════════════════════════════════════════════════
	con = duckdb.connect(db)
	con.sql("SET enable_progress_bar = false;")
	con.sql(
		"CREATE TABLE IF NOT EXISTS geneList ("
			"geneModel VARCHAR, "       # Gene ID (e.g., "AT1G01010")
			"start INT, "               # 0-indexed region start in chromosome
			"fin INT, "                 # Region end (exclusive)
			"strand VARCHAR, "          # "+" or "-"
			"chromosome VARCHAR, "      # Chromosome/scaffold name
			"sequence VARCHAR, "        # Extracted DNA sequence with flanking context
			"gff VARCHAR, "             # GSF annotation string (NULL in predict mode)
			"static_fpb INT, "          # Static 5' flanking buffer size
			"static_tpb INT, "          # Static 3' flanking buffer size
			"five_prime_buf INT, "      # Random 5' buffer offset (training augmentation)
			"three_prime_buf INT, "     # Random 3' buffer offset (training augmentation)
			"rn INT PRIMARY KEY)")      # Auto-incrementing primary key
	con.sql("CREATE SEQUENCE IF NOT EXISTS row_id START 1;")

	# ═══════════════════════════════════════════════════════════════
	# Parse the GFF3/BED file line-by-line, building GSF strings per gene
	# The GFF3 is expected to be sorted so that gene lines precede their
	# child features (mRNA, CDS, UTR), and all children of one gene
	# appear before the next gene line.
	# ═══════════════════════════════════════════════════════════════
	print(f"\nProcessing {gff3}...")
	num_lines = sum(1 for line in open(gff3, "r"))
	with open(gff3, 'r') as gff3file:
		region_start = None       # Genomic start of the current extraction window
		feature_list = ''         # Accumulates GSF feature definitions (semicolon-delimited)
		mRNA_list = ''            # Accumulates GSF transcript definitions
		skipGene = False          # Flag to skip oversized or non-coding genes
		five_ps = {}              # Dedup dict for five_prime_UTR features: "start-end-strand-phase" → num
		three_ps = {}             # Dedup dict for three_prime_UTR features
		cds_num = {}              # Dedup dict for CDS features

		for line in tqdm(gff3file, total=num_lines):
			# Skip comment lines and blank lines in the GFF3
			if line.startswith('#') | (line == '\n'):
				continue
			else:
				line = line.strip().split('\t')

				# Parse GFF3 or BED format into common variables
				if anoType == 'gff':
					# GFF3 columns: chr, source, type, start, end, score, strand, phase, attributes
					chr, _, typ, start, fin, _, strand, phase, attributes = line
				elif anoType == 'bed':
					# BED columns: chr, start, end, name, score, strand
					# Convert BED to gene-level entry (no sub-features in BED mode)
					chr, start, fin, attributes, _, strand = line
					attributes = f"ID={attributes};"
					typ = 'gene'
					phase = "."

				# In prediction mode, only process gene-level entries (skip CDS, UTR, mRNA)
				# since we only need the sequence region, not the annotation labels
				if (mode == 'predict') and (typ != "gene"):
					continue

				# ─────────────────────────────────────────────────────
				# GENE line: finalize the previous gene and start a new one
				# ─────────────────────────────────────────────────────
				if typ == 'gene':
					# ── Insert the PREVIOUS gene into the database ──
					if (region_start != None) & (not skipGene):
						# Strip leading/trailing pipe delimiters from mRNA_list
						mRNA_list = mRNA_list[1:-1]
						# Assemble full GSF string: "feature_list>transcript_list"
						if feature_list and mRNA_list:
							gff = f"{feature_list[:-1]}>{mRNA_list}"
						else:
							gff = None

						# Optionally validate CDS integrity (start/stop codons, reading frame)
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
								# Generate random 5'/3' buffer offsets for training augmentation
								# These shift the gene within its extraction window, teaching the
								# model to handle variable UTR positions
								if addExtra:
									five_prime = int(torch.randint(addExtra, (1,)))
									three_prime = int(torch.randint(addExtra, (1,)))
								else:
									five_prime = 0
									three_prime = 0
								# Insert the gene record with all metadata
								con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf) VALUES (nextval('row_id'), '{geneModel}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence}', '{gff}', '{five_prime_buffer}', '{three_prime_buffer}','{five_prime}', '{three_prime}')")
						except Exception as e:
							print(f"{geneModel=}")
							print(f"{sequence=}")
							print(f"{gff=}")
							print(f"Error inserting {geneModel} into database: {e}", file=sys.stderr)
							con.close()
							sys.exit()

						# ── Reverse complement augmentation ──
						# Doubles training data by adding RC versions of gene sequences,
						# which helps the model learn strand-agnostic features
						if addRC:
							# Transform the GSF annotation for the reverse complement strand
							gff_rc = reverseComplement_gffString(gff, region_end - region_start)
							if addRCIsoOnly:
								# Only augment genes with multiple transcripts (alternative splicing)
								# Semicolons in mRNA_list indicate multiple transcript isoforms
								if ';' in mRNA_list:
									try:
										if valid:
											con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf) VALUES (nextval('row_id'), '{geneModel + "-rc"}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence_rc}', '{gff_rc}', '{five_prime_buffer}', '{three_prime_buffer}', '{int(torch.randint(addExtra, (1,)))}', '{int(torch.randint(addExtra, (1,)))}')")
									except Exception as e:
										print(f"{geneModel=}-rc")
										print(f"{sequence_rc=}")
										print(f"{gff_rc=}")
										print(f"Error inserting {geneModel+"-rc"} into database: {e}", file=sys.stderr)
										con.close()
										sys.exit()
							else:
								# Add RC for ALL valid genes
								try:
									if valid:
										con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf) VALUES (nextval('row_id'), '{geneModel + "-rc"}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence_rc}', '{gff_rc}', '{five_prime_buffer}', '{three_prime_buffer}', {int(torch.randint(addExtra, (1,)))}', '{int(torch.randint(addExtra, (1,)))}')")
								except Exception as e:
									print(f"{geneModel=}-rc")
									print(f"{sequence_rc=}")
									print(f"{gff_rc=}")
									print(f"Error inserting {geneModel+"-rc"} into database: {e}", file=sys.stderr)
									con.close()
									sys.exit()

						# Reset accumulators for the next gene
						geneModel = None
						region_start = 0
						region_end = 0
						feature_list = ''
						mRNA_list = ''
						five_ps = {}
						three_ps = {}
						cds_num = {}

					# ── Calculate flanking buffer to pad gene to nearest multiple of staticSize ──
					# The encoder processes fixed-size chunks (e.g., 6144bp), so sequences
					# must be padded to multiples of staticSize with neighboring genomic context
					gene_length = int(fin) - int(start) + 1
					if gene_length <= staticSize:
						# Small gene: pad to exactly one chunk
						additional_sequence = staticSize - (gene_length % staticSize)
					else:
						# Large gene: pad to next full multiple of staticSize
						additional_sequence = ((gene_length // staticSize) + 1)*staticSize - gene_length
					# Split padding roughly evenly between 5' and 3' flanks
					three_prime_buffer = additional_sequence//2
					if not (additional_sequence % 2):
						five_prime_buffer = additional_sequence//2
					else:
						# Odd padding: give the extra base to the 5' side
						five_prime_buffer = additional_sequence//2 + 1


					skipGene = False
					# Extract gene ID from GFF3 attributes (e.g., "ID=AT1G01010;Name=...")
					geneModel = attributes.split(';')[0].split('=')[1]
					# Clamp 5' buffer if it would extend past chromosome start
					if (int(start) - five_prime_buffer - 1) < 0:
						five_prime_buffer = int(start) - 1
					# GFF3 is 1-indexed; convert to 0-indexed for Python slicing
					region_start = int(start) - five_prime_buffer - 1

					# Ensure the total region is at least staticSize when gene + buffers fall short
					if (five_prime_buffer + gene_length + three_prime_buffer) <= staticSize:
						three_prime_buffer = staticSize - (five_prime_buffer + gene_length)
					# End is not decremented because Python slicing is exclusive on the right
					region_end = int(fin) + three_prime_buffer

					# 49,152bp corresponds to 8,192 6-mer tokens (Max input)
 					# 25,002 -> 4,167 6-mer tokens (Max output)
					if (region_end - region_start > maxLen):
						print(f"Skipping {geneModel} because gene length > {maxLen}", file=sys.stderr)
						region_start = None
						region_end = None
						geneModel = None
						skipGene = True
						continue
					# Warn if padding math produced a non-multiple (should not happen)
					if ((region_end - region_start)% staticSize) != 0:
						print(f"Warning: {geneModel} not a multiple of {staticSize=}", file=sys.stderr)

					# Extract the forward-strand DNA sequence for this gene region
					sequence = genome_dict[chr][region_start:region_end]
					# Pre-compute reverse complement if RC augmentation is enabled
					if addRC:
						sequence_rc = reverseComplement(sequence)

				elif skipGene:
					# Skip all child features of an oversized or non-coding gene
					continue

				elif typ == 'lncRNA':
					# Long non-coding RNAs are excluded from the dataset entirely
					skipGene = True
					continue

				# ─────────────────────────────────────────────────────
				# mRNA line: start a new transcript in the current gene
				# ─────────────────────────────────────────────────────
				elif typ == 'mRNA':
					# Close the previous transcript's pipe-delimited feature list
					# and start a new one (semicolons separate transcripts in GSF)
					mRNA_list = mRNA_list[:-1] + ";"

				# ─────────────────────────────────────────────────────
				# CDS line: build feature + transcript entries for coding sequence
				# Coordinates are converted from GFF3 (1-indexed, inclusive)
				# to GSF (0-indexed relative to extracted region, end-exclusive)
				# ─────────────────────────────────────────────────────
				elif typ == 'CDS':
					# Convert to 0-indexed coordinates relative to the extraction region
					start = (int(start) - 1) - region_start   # GFF3 start is 1-indexed → subtract 1
					end = int(fin) - region_start              # GFF3 end is inclusive → no -1 needed (GSF is exclusive)
					# Remap GFF3 numeric phase (0/1/2) to GSF letter phase (A/B/C)
					# Phase = number of bases to skip at the start of the first codon
					if phase == '0':
						phase = 'A'    # Reading frame starts at position 0
					elif phase == '1':
						phase = 'B'    # Reading frame starts at position 1
					elif phase == '2':
						phase = 'C'    # Reading frame starts at position 2
					else:
						phase = '.'    # Phase not applicable

					# Deduplicate CDS features: multiple transcripts may share the same CDS
					# If this exact CDS (same coords, strand, phase) was already seen,
					# reuse its number; otherwise assign a new sequential number
					try:
						num = cds_num[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(cds_num) + 1)
						cds_num[f"{start}-{end}-{strand}-{phase}"] = num
						# Add to feature list only for new (unique) features
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"

					# Always add the CDS reference to the current transcript
					mRNA_list += f"{typ+num}|"

				# ─────────────────────────────────────────────────────
				# five_prime_UTR line: 5' untranslated region
				# Same coordinate conversion and deduplication as CDS
				# ─────────────────────────────────────────────────────
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

				# ─────────────────────────────────────────────────────
				# three_prime_UTR line: 3' untranslated region
				# ─────────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════
# Legacy segmentation dataset builders
# ═════════════════════════════════════════════════════════════════════

# Note: Legacy implementation (very slow during dataloading). Recommended to use 'genome2PreprocessedSegmentationDataset' instead
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
