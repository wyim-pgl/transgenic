"""
Sequence processing utilities for TransGenic.

This module provides functions for manipulating DNA sequences and preparing
them for model input:

  - loadGenome:               Parse a FASTA genome file into a dict of {chr: sequence}
  - reverseComplement:        Compute the reverse complement of a DNA sequence
  - validateCDS:              Verify CDS features produce valid coding sequences
  - segmentSequence:          Split long DNA sequences into fixed-size chunks
  - scanGlobalAttentionTokens: Identify splice-site tokens for Longformer global attention
  - getSpliceBranchSeqs:      Generate all possible spliceosome branch point motifs
  - mask_sequences:           Apply codon-aware random masking for MLM pretraining

These utilities are used by the dataset classes (datasets.py) during data
loading and preprocessing.
"""

import torch, random
from typing import List, Tuple


def loadGenome(genome):
	"""Parse a FASTA genome file into a dictionary.

	Reads a standard FASTA file where each entry starts with '>' followed by
	the chromosome/contig name, and subsequent lines contain the DNA sequence.

	Args:
		genome: Path to the FASTA genome file.

	Returns:
		Dict mapping chromosome names to their full DNA sequences.
		Example: {"chr1": "ATCG...", "chr2": "GCTA..."}
	"""
	genome_dict = {}
	with open(genome, 'r') as genomefile:
		lines = genomefile.readlines()
		header = None
		sequence_lines = []
		for line in lines:
			if line.startswith('>'):
				# Save the previous chromosome's sequence (if any)
				if header is not None:
					genome_dict[header] = ''.join(sequence_lines)
					sequence_lines = []
				# Parse new chromosome name (first word after '>')
				header = line.strip().split(' ')[0][1:]
			else:
				# Accumulate sequence lines for current chromosome
				sequence_lines.append(line.strip())
		# Save the last chromosome
		if header is not None:
			genome_dict[header] = ''.join(sequence_lines)
	return genome_dict


def reverseComplement(sequence):
	"""Compute the reverse complement of a DNA sequence.

	Applies Watson-Crick complementarity (A↔T, C↔G) and reverses
	the resulting string. Uses a lowercase intermediary to avoid
	double-substitution (e.g., A→t then t→A would fail without this trick).

	Args:
		sequence: DNA string containing A, T, C, G characters.

	Returns:
		Reverse-complemented DNA string.

	Example:
		reverseComplement("ATCG") → "CGAT"
	"""
	sequence = sequence.upper()
	# Use lowercase intermediary to avoid double-substitution
	sequence = sequence.replace('A', 't')  # A → t (temporary lowercase)
	sequence = sequence.replace('T', 'a')  # T → a (temporary lowercase)
	sequence = sequence.replace('C', 'g')  # C → g (temporary lowercase)
	sequence = sequence.replace('G', 'c')  # G → c (temporary lowercase)
	sequence = sequence.upper()             # Convert all back to uppercase
	sequence = sequence[::-1]               # Reverse the complemented sequence
	return sequence


def validateCDS(gff: str, seq: str, geneModel: str) -> Tuple[bool, str]:
	"""Validate that CDS features in a GFF annotation produce valid coding sequences.

	Checks each transcript in the GFF for:
	  1. Correct start codon (ATG for forward strand, reverse-complement stop for reverse)
	  2. Correct stop codon (TAA/TAG/TGA for forward strand)
	  3. Total CDS length is a multiple of 3 (maintains reading frame)

	Args:
		gff: GFF annotation string ("features>transcripts" format).
		seq: Full DNA sequence of the genomic region.
		geneModel: Gene model name (for error messages).

	Returns:
		Tuple of (is_valid, error_message).
		If valid: (True, None)
		If invalid: (False, "description of the problem")
	"""
	try:
		# Parse GFF into features and transcript structures
		features = [f.split("|") for f in gff.split(">")[0].split(";")]
		transcripts = [t.split("|") for t in gff.split(">")[1].split(";")]
		strand = features[0][3]  # Strand from the first feature (+ or -)
		# Build lookup: feature name → index in features list
		feature_index_dict = {f[1]: i for i, f in enumerate(features)}

		# Standard genetic code stop codons (forward and reverse complement)
		stopCodons = ["TAA", "TAG", "TGA"]
		stopCodonsRC = ["TTA", "CTA", "TCA"]  # Reverse complement of stop codons

		# Validate each transcript independently
		for transcript in transcripts:
			seqlen = 0  # Cumulative CDS length for reading frame check
			for i, typ in enumerate(transcript):
				if "CDS" in typ:
					if i == 0:
						# First CDS in transcript: check start/stop codon
						if strand == "+":
							# Forward strand: first CDS must start with ATG
							if seq[int(features[feature_index_dict[typ]][0]):int(features[feature_index_dict[typ]][0]) + 3] != "ATG":
								valid = False
								return (valid, f"Start codon missing in {features[feature_index_dict[typ]][1]} of {geneModel}.")
						else:
							# Reverse strand: first CDS (3' end) must have reverse-complement stop codon
							if seq[int(features[feature_index_dict[typ]][0]):int(features[feature_index_dict[typ]][0]) + 3] not in stopCodonsRC:
								valid = False
								return (valid, f"Stop codon missing in {features[feature_index_dict[typ]][1]} of {geneModel}.")
					lastCDS = typ  # Track the last CDS for stop codon check
					# Accumulate CDS length (end - start)
					seqlen += int(features[feature_index_dict[typ]][2]) - int(features[feature_index_dict[typ]][0])

			# Reading frame check: total CDS length must be a multiple of 3
			if seqlen % 3 != 0:
				valid = False
				return (valid, f"{geneModel} not a multiple of 3.")

			# Check stop codon at the end of the last CDS
			if strand == "+":
				if seq[int(features[feature_index_dict[lastCDS]][2]) - 3:int(features[feature_index_dict[lastCDS]][2])] not in stopCodons:
					valid = False
					return (valid, f"Stop codon missing in {features[feature_index_dict[lastCDS]][1]} of {geneModel}.")
			else:
				# Reverse strand: last CDS (5' end) must have ATG reverse complement = CAT
				if seq[int(features[feature_index_dict[lastCDS]][2]) - 3:int(features[feature_index_dict[lastCDS]][2])] != "CAT":
					valid = False
					return (valid, f"Start codon missing in {features[feature_index_dict[lastCDS]][1]} of {geneModel}.")
	except Exception as e:
		return (False, f"Error validating {geneModel}, skipping: {e}")
	return (True, None)


def segmentSequence(seq, piece_size=6144):
	"""Split a DNA sequence into fixed-size chunks.

	Used by the Nucleotide Transformer (NT) encoder variant which has a
	max input length of 1024 tokens (corresponding to ~6144 nucleotides
	with 6-mer tokenization). The last chunk may be shorter than piece_size.

	Not used by HyenaDNA encoder (which handles full-length sequences natively).

	Args:
		seq: DNA sequence string.
		piece_size: Maximum chunk size in nucleotides (default 6144).

	Returns:
		List of DNA string chunks.

	Example:
		segmentSequence("ATCGATCG", piece_size=4) → ["ATCG", "ATCG"]
	"""
	seqs = [seq[i:min(i + piece_size, len(seq))] for i in range(0, len(seq), piece_size)]
	return seqs


def scanGlobalAttentionTokens(vocab: dict, tokenized_sequence: List[int], seqlen: int) -> List[int]:
	"""Identify tokens that should receive global attention in the Longformer encoder.

	Scans a tokenized DNA sequence for biologically important motifs that
	benefit from global attention in the Longformer architecture:

	  1. Splice donor sites: canonical (GT) and non-canonical (GC, AT) dinucleotides
	     with surrounding context (AGGT, AGGC, AGAT)
	  2. Spliceosome branch point motifs: degenerate YNYURAY sequences that mark
	     the branch point A used in lariat formation during splicing

	Both forward and reverse strand patterns are scanned. The final mask
	merges both orientations (union of detected positions).

	Args:
		vocab: Token-to-ID dictionary from the encoder tokenizer.
		tokenized_sequence: List of integer token IDs.
		seqlen: Length of the original DNA sequence (for informational purposes).

	Returns:
		List of global attention masks, segmented into 1024-token chunks
		(matching the encoder's segment structure). Each element is 0 or 1.
	"""

	# Canonical splice donor dinucleotides in forward and reverse orientation
	spliceSites_fwd = ["AGGT", "AGGC", "AGAT"]  # Forward strand donors
	spliceSites_rev = ["ACCT", "GCCT", "ATCT"]    # Reverse strand donors

	# Generate all possible spliceosome branch point sequences
	spliceBranchSites = getSpliceBranchSeqs()
	spliceBranchSites_fwd = spliceBranchSites["forward"]  # Forward: YNYURAY variants
	spliceBranchSites_rev = spliceBranchSites["reverse"]  # Reverse complement variants

	# Build reverse vocab lookup: token_id → token_string
	id2vocab = {v: k for k, v in vocab.items()}

	# Initialize separate masks for forward and reverse strand
	global_attention_mask_fwd = [0 for i in range(len(tokenized_sequence))]
	global_attention_mask_rev = [0 for i in range(len(tokenized_sequence))]

	fwd_splice_count = 0
	rev_splice_count = 0
	fwd_branch_count = 0
	rev_branch_count = 0

	for i, token in enumerate(tokenized_sequence):
		if i == 0:
			continue  # Skip first token (no previous token for double-token matching)

		# Concatenate current and previous token strings for cross-token pattern matching
		double_token = f"{id2vocab[token]}{id2vocab[tokenized_sequence[i - 1]]}"

		# ── Forward splice donor scanning ──
		for site in spliceSites_fwd:
			if site in id2vocab[token]:
				# Pattern found within a single token
				global_attention_mask_fwd[i] = 1
				fwd_splice_count += 1
				break
			elif (site in double_token) and (site not in id2vocab[tokenized_sequence[i - 1]]):
				# Pattern spans two adjacent tokens
				global_attention_mask_fwd[i] = 1
				global_attention_mask_fwd[i - 1] = 1
				fwd_splice_count += 1
				break

		# ── Reverse splice donor scanning ──
		for site in spliceSites_rev:
			if site in id2vocab[token]:
				global_attention_mask_rev[i] = 1
				rev_splice_count += 1
				break
			elif (site in double_token) and (site not in id2vocab[tokenized_sequence[i - 1]]):
				global_attention_mask_rev[i] = 1
				global_attention_mask_rev[i - 1] = 1
				rev_splice_count += 1
				break

		# ── Forward branch point scanning ──
		# Branch point motifs always span two tokens (7-mer > typical token size)
		for site in spliceBranchSites_fwd:
			if site in double_token:
				global_attention_mask_fwd[i] = 1
				global_attention_mask_fwd[i - 1] = 1
				fwd_branch_count += 1
				break

		# ── Reverse branch point scanning ──
		for site in spliceBranchSites_rev:
			if site in double_token:
				global_attention_mask_rev[i] = 1
				global_attention_mask_rev[i - 1] = 1
				rev_branch_count += 1
				break

	# Merge forward and reverse masks: mark position as global if detected in either strand
	# TODO: is there a heuristic we can use to determine the correct direction?
	picked = []
	for x in zip(global_attention_mask_fwd, global_attention_mask_rev):
		if sum(x) >= 1:
			picked.append(1)
		else:
			picked.append(0)

	# Segment the mask into 1024-token chunks (matching encoder segment structure)
	return segmentSequence(picked, piece_size=1024)


def getSpliceBranchSeqs() -> List[str]:
	"""Generate all possible spliceosome branch point motif sequences.

	The branch point consensus motif is YNYURAY (IUPAC notation):
	  Y = pyrimidine (C or T)
	  N = any nucleotide (A, C, G, T)
	  U = uracil → T in DNA
	  R = purine (A or G)
	  A = adenine (the catalytic branch point nucleotide)

	Generates all 128 possible combinations for both forward and
	reverse complement orientations.

	Returns:
		Dict with "forward" and "reverse" keys, each containing a list
		of 7-mer DNA strings representing branch point motifs.
	"""
	from itertools import product

	# IUPAC degenerate nucleotide definitions
	motif_positions = {
		'Y': ['C', 'T'],           # Pyrimidine
		'N': ['A', 'C', 'G', 'T'], # Any nucleotide
		'R': ['A', 'G'],           # Purine
		'A': ['A'],                 # Adenine only
		'U': ['T']}                 # Uracil → Thymine in DNA

	# Branch point consensus motif: YNYURAY
	motif = 'YNYURAY'
	# Generate all possible nucleotides for each position
	possible_nucleotides = [motif_positions[char] for char in motif]
	# Cartesian product of all positions → all possible 7-mer sequences
	all_possible_sequences = [''.join(seq) for seq in product(*possible_nucleotides)]

	seqs = []
	for sequence in all_possible_sequences:
		seqs.append(sequence)
	# Generate reverse complement versions for scanning the opposite strand
	rc = [reverseComplement(seq) for seq in seqs]
	return {"forward": seqs, "reverse": rc}


def mask_sequences(seqs, num_masked_nucleotides=921, mask_token=3):
	"""Apply codon-aware random masking for Masked Language Modeling (MLM).

	Masks groups of 3 contiguous nucleotides (preserving codon boundaries)
	to prevent the model from exploiting partial codon information. This is
	biologically motivated: masking partial codons would leak frame information.

	Default: masks 921 nucleotides per sequence (307 groups of 3), which is
	approximately 15% of a 6144-nt window — following the BERT masking ratio.

	Args:
		seqs: 2D tensor of shape (batch_size, seq_len). Sequence length must
		      be divisible by 3.
		num_masked_nucleotides: Number of nucleotides to mask per sequence.
		                        Rounded down to nearest multiple of 3.
		mask_token: Integer token ID to use for masked positions (default 3 = <unk>).

	Returns:
		Tuple of:
		  - masked_seqs: Tensor with mask_token at masked positions.
		  - mask_bool: Boolean tensor (True at masked positions) for loss computation.

	Raises:
		ValueError: If sequence length is not divisible by 3.
	"""
	batch_size, seq_len = seqs.size()
	if seq_len % 3 != 0:
		raise ValueError("Sequence length must be divisible by 3.")

	num_3mers = seq_len // 3         # Total number of codon-sized groups
	num_groups = num_masked_nucleotides // 3  # Number of groups to mask

	masked_seqs = seqs.clone()        # Don't modify the original tensor
	mask_bool = torch.zeros_like(seqs, dtype=torch.bool)  # Track which positions are masked

	# For each sequence in the batch, randomly select non-overlapping groups to mask
	for i in range(batch_size):
		# Randomly sample group indices without replacement
		group_indices = random.sample(range(num_3mers), num_groups)
		for g in group_indices:
			start = g * 3                          # Start position of this 3-mer group
			masked_seqs[i, start:start + 3] = mask_token  # Replace with mask token
			mask_bool[i, start:start + 3] = True           # Mark as masked for loss

	return masked_seqs, mask_bool
