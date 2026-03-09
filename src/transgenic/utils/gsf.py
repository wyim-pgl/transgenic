"""
GSF (Gene Sentence Format) Utilities for TransGenic.

GSF is TransGenic's custom compact annotation format used to represent gene
structures as concise strings suitable for sequence-to-annotation model output.

GSF Format Overview:
    A GSF string has the structure: feature_list>transcript_list

    Feature List:
        A semicolon-separated list of features, where each feature is a
        pipe-delimited tuple of five fields:
            start|type|end|strand|phase
        - start:  0-indexed start coordinate relative to the input sequence
        - type:   feature identifier (e.g., CDS1, CDS2, five_prime_UTR1,
                  three_prime_UTR1)
        - end:    0-indexed end coordinate relative to the input sequence
        - strand: "+" for forward strand, "-" for reverse strand
        - phase:  reading frame offset encoded as A (0), B (1), C (2),
                  or "." (not applicable, used for UTRs)

    Transcript List:
        A semicolon-separated list of transcripts, where each transcript is a
        pipe-delimited list of feature type references (e.g., CDS1|CDS2|CDS3).
        Multiple alternative transcripts are separated by semicolons.

    Example:
        100|CDS1|200|+|A;200|CDS2|400|+|B;50|five_prime_UTR1|100|+|.>CDS1|CDS2|five_prime_UTR1

    This module provides utilities for:
        - Transforming GSF strings for reverse-complemented sequences
        - Converting GSF strings to standard GFF3 format for use with
          bioinformatics tools
"""

import re, sys
from typing import List, Optional, Tuple

def reverseComplement_gffString(gff, length):
	"""
	Transform a GSF string's coordinates and strand for a reverse-complemented sequence.

	When a genomic sequence is reverse-complemented, all annotations must be
	transformed accordingly: coordinates are mirrored, strands are flipped, and
	the order of features is rearranged because the 5'-to-3' direction reverses.

	Specifically, this function:
	  - Re-orders features so that CDS, five_prime_UTR, and three_prime_UTR
	    features are swapped in positional order (since what was the 5' end
	    becomes the 3' end and vice versa).
	  - Flips coordinates using: new_start = length - old_end,
	    new_end = length - old_start.
	  - Swaps strand orientation: "+" becomes "-", "-" becomes "+".
	  - Reverses the order of CDS features within each transcript, since the
	    5'-to-3' reading direction is inverted by reverse complementation.

	Args:
		gff (str): A GSF-formatted string (feature_list>transcript_list).
		length (int): The length of the sequence, used for coordinate
			transformation (new_coord = length - old_coord).

	Returns:
		str: A new GSF-formatted string with transformed coordinates, flipped
			strands, and reordered features appropriate for the reverse
			complement of the original sequence.
	"""
	# Split the GSF string into its two major components:
	# feature_list (individual genomic features) and mRNA_list (transcript definitions)
	feature_list, mRNA_list = gff.split('>')
	features = feature_list.split(';')


	# Build a reordering index that determines how features should be rearranged
	# after reverse complementation. The insertion positions (begin_cds, begin_utr5,
	# begin_utr3) track where each feature type should be placed in the new order.
	# This effectively swaps the positional ordering of CDS, 5'UTR, and 3'UTR
	# features since the 5'->3' direction reverses on the complementary strand.
	order = []
	begin_cds = 0
	begin_utr5 = 0
	begin_utr3 = 0
	phase_lookup = {}
	for i, feature in enumerate(features):
		if "CDS" in feature:
			# Insert CDS at its current insertion point; advance the UTR insertion points
			order.insert(begin_cds, i)
			begin_utr5 = i+1
			begin_utr3 = i+1
			# Record the phase (reading frame) for each CDS, keyed by feature type ID
			phase_lookup[feature.split("|")[1]] = feature.split("|")[4]
		elif "five_prime_UTR" in feature:
			# Insert 5'UTR at its current insertion point; advance CDS and 3'UTR points
			order.insert(begin_utr5, i)
			begin_cds = i+1
			begin_utr3 = i+1
		elif "three_prime_UTR" in feature:
			# Insert 3'UTR at its current insertion point; advance CDS and 5'UTR points
			order.insert(begin_utr3, i)
			begin_cds = i+1
			begin_utr5 = i+1

	# Parse each feature into its five component fields: [start, type, end, strand, phase]
	features = [feature.split('|') for feature in features]
	new_features = []
	# Maps old feature type IDs to new ones after reordering (e.g., CDS1 -> CDS3)
	feature_swap = {}
	for index, i in enumerate(order):
		# Mirror coordinates: new_start = length - old_end, new_end = length - old_start
		old_start = int(features[i][0])
		old_end = int(features[i][2])
		new_start = str(int(length) - old_end)
		new_end = str(int(length) - old_start)
		# The feature type ID is taken from the feature at the current index position,
		# not from the reordered feature -- this reassigns type labels to maintain
		# sequential numbering in the new order (e.g., CDS1 stays first in the list)
		new_type = features[index][1]
		# Record the mapping from old type ID to new type ID for transcript rewriting
		feature_swap[features[i][1]] = new_type
		phase = features[i][4]

		# Flip strand orientation: + becomes -, - becomes +
		if features[i][3] == "-":
			new_strand = "+"
		elif features[i][3] == "+":
			new_strand = "-"

		new_features.append([new_start, new_type, new_end, new_strand, phase])

	# Rebuild transcript definitions using the feature type mapping.
	# CDS order within each transcript is reversed (insert at position 0) because
	# the 5'->3' reading direction flips on the reverse complement strand.
	transcripts = [t.split("|") for t in mRNA_list.split(';')]
	new_transcripts = []
	for transcript in transcripts:
		new_utrs = []
		new_cds = []
		phase = ''
		for feature in transcript:
			if "UTR" in feature:
				# UTR features are appended in their encountered order
				new_utrs.append(feature_swap[feature])
			else:
				# CDS features are prepended (insert at 0) to reverse their order,
				# reflecting the reversed reading direction on the complementary strand
				new_cds.insert(0, feature_swap[feature])
				phase += phase_lookup[feature]
		# Concatenate reversed CDS list followed by UTRs for the new transcript
		new_transcripts.append(new_cds + new_utrs)

	# Reassemble the GSF string: features joined by ; with fields joined by |,
	# separated from transcripts by >
	return f"{';'.join(['|'.join(feature) for feature in new_features])}>{';'.join(['|'.join(transcript) for transcript in new_transcripts])}"

def get_cds_fingerprint(gff_lines: List[str]) -> Optional[Tuple]:
	"""
	Generate a fingerprint from a GFF3 gene prediction for deduplication.

	Builds a per-transcript fingerprint from CDS coordinates grouped by their
	parent mRNA.  Two predictions are duplicates when they share the same
	chromosome, strand, and identical set of per-transcript CDS intervals.

	For genes without CDS features (e.g. UTR-only), the mRNA boundaries are
	used instead so that they are still eligible for deduplication.

	Args:
		gff_lines: List of GFF3-formatted lines from gffString2GFF3().

	Returns:
		A hashable tuple that uniquely identifies the gene structure, or None
		if no parseable features are found.
	"""
	strand = None
	chrom = None
	# CDS coords keyed by parent mRNA ID
	transcript_cds = {}
	# mRNA boundaries as fallback for CDS-less genes
	mrna_bounds = {}

	for line in gff_lines:
		cols = line.split("\t")
		if len(cols) < 9:
			continue
		feat_type = cols[2]
		if feat_type == "gene":
			chrom = cols[0]
			strand = cols[6]
		elif feat_type == "mRNA":
			attrs = cols[8]
			mid = attrs.split("ID=")[1].split(";")[0] if "ID=" in attrs else None
			if mid:
				mrna_bounds[mid] = (int(cols[3]), int(cols[4]))
		elif feat_type == "CDS":
			attrs = cols[8]
			parent = attrs.split("Parent=")[1].split(";")[0] if "Parent=" in attrs else None
			if parent:
				transcript_cds.setdefault(parent, []).append((int(cols[3]), int(cols[4])))

	if chrom is None:
		return None

	if transcript_cds:
		# Sorted tuple of per-transcript CDS sets (each transcript's CDS sorted)
		per_tx = tuple(sorted(
			tuple(sorted(coords)) for coords in transcript_cds.values()
		))
		return (chrom, strand, per_tx)

	# Fallback: no CDS, use mRNA boundaries
	if mrna_bounds:
		bounds = tuple(sorted(mrna_bounds.values()))
		return (chrom, strand, bounds)

	return None


def get_utr_span(gff_lines: List[str]) -> int:
	"""
	Calculate total UTR span (in bases) from GFF3 gene prediction lines.

	Sums the length of all five_prime_UTR and three_prime_UTR features.
	Used to break ties when two predictions share the same CDS fingerprint:
	the prediction with longer UTR coverage is preferred because it captures
	more of the gene's regulatory regions.

	Args:
		gff_lines: List of GFF3-formatted lines from gffString2GFF3().

	Returns:
		Total UTR span in bases (0 if no UTR features are present).
	"""
	total = 0
	for line in gff_lines:
		cols = line.split("\t")
		if len(cols) >= 9 and "UTR" in cols[2]:
			total += int(cols[4]) - int(cols[3]) + 1
	return total


def dedup_gff3_file(output_path: str) -> int:
	"""
	Remove superseded gene blocks from a GFF3 file, keeping the last
	occurrence of each CDS fingerprint (which has the longest UTR).

	This is a post-hoc cleanup pass for the streaming write design: during
	inference, a prediction with longer UTR replaces the fingerprint entry
	but both versions exist in the file.  This pass removes the older ones.

	Args:
		output_path: Path to the GFF3 output file to deduplicate in-place.

	Returns:
		Number of superseded gene blocks removed.
	"""
	gene_blocks = []
	current_block = []
	header_lines = []

	with open(output_path) as f:
		for line in f:
			line = line.rstrip("\n")
			if line.startswith("#"):
				header_lines.append(line)
				continue
			cols = line.split("\t")
			if len(cols) >= 9 and cols[2] == "gene":
				if current_block:
					gene_blocks.append(current_block)
				current_block = [line]
			elif len(cols) >= 9:
				current_block.append(line)
	if current_block:
		gene_blocks.append(current_block)

	# For each fingerprint, the last occurrence has the longest UTR
	# (guaranteed by the caller logic that only writes a replacement
	# when the new UTR is strictly longer).
	last_occurrence = {}
	for i, block in enumerate(gene_blocks):
		fp = get_cds_fingerprint(block)
		if fp is not None:
			last_occurrence[fp] = i

	removed = 0
	with open(output_path, "w") as f:
		for line in header_lines:
			f.write(line + "\n")
		for i, block in enumerate(gene_blocks):
			fp = get_cds_fingerprint(block)
			if fp is None or last_occurrence[fp] == i:
				for line in block:
					f.write(line + "\n")
			else:
				removed += 1

	return removed


def gffString2GFF3(gff:str, chr:str, region_start:int, extra_attributes:str) -> List[str]:
	"""
	Convert a GSF string (decoded model output) to standard GFF3 format.

	Translates TransGenic's compact GSF annotation format into GFF3, the widely
	used tab-delimited format for genome annotations understood by bioinformatics
	tools (e.g., genome browsers, gene prediction pipelines).

	The conversion involves:
	  - Stripping special tokens (<s>, </s>) that may be present from the model
	    decoder output.
	  - Converting 0-indexed relative coordinates (within the input sequence) to
	    1-indexed absolute genomic coordinates using the region_start offset.
	  - Mapping encoded phase values to GFF3 phase: A->0, B->1, C->2, "."->"."
	  - Generating unique UUIDs for hierarchical gene/mRNA/feature identifiers.
	  - Constructing a proper GFF3 hierarchy: gene -> mRNA -> CDS/UTR features,
	    linked via ID/Parent attributes.
	  - Handling malformed features gracefully by skipping invalid entries and
	    continuing with valid ones wherever possible.

	Args:
		gff (str): A GSF-formatted string (feature_list>transcript_list),
			potentially containing special tokens from model output.
		chr (str): The chromosome or scaffold name for the GFF3 seqid column.
		region_start (int): The 0-indexed start position of the input sequence
			region within the chromosome. Used to convert relative feature
			coordinates to absolute genomic coordinates.
		extra_attributes (str): Additional GFF3 attribute key=value pairs to
			append to every feature line (e.g., "source=model_v1").

	Returns:
		list[str]: A list of GFF3-formatted tab-separated strings, one per
			annotation line. The list follows the hierarchy: gene line first,
			then mRNA lines, then CDS/UTR feature lines for each mRNA.
			Returns [""] if the GSF string cannot be parsed or contains no
			valid transcripts.
	"""
	# Phase encoding lookup: GSF uses A/B/C to represent GFF3 phases 0/1/2
	# (the number of bases to skip at the start of a CDS feature to reach the
	# first complete codon). "." means phase is not applicable (e.g., UTRs).
	phaseLookup = {"A":0,"B":1,"C":2,".":"."}

	# Strip special tokens that may remain from the model's decoder output.
	# <s> is the start-of-sequence token; </s> is the end-of-sequence token.
	# The "|</s>" variant handles cases where </s> is pipe-delimited with a feature.
	gff = gff.replace ("<s>", "").replace("|</s>", "").replace("</s>", "")

	# Convert from 0-indexed (model output) to 1-indexed (GFF3 convention)
	# by incrementing region_start; this offset is added to all feature coordinates later.
	region_start += 1

	# Parse the GSF string into features and transcripts.
	# Build a dictionary of features keyed by their type ID (e.g., "CDS1", "CDS2")
	# for quick lookup when processing transcripts.
	try:
		features,transcripts = gff.split(">")
		features = [feat.split("|") for feat in features.split(";")]
		# Key features by their type ID (field index 1, e.g., "CDS1") for lookup
		features = {feat[1]:feat for feat in features}
		transcripts = transcripts.split(";")
		# Determine the gene strand from the first CDS feature (CDS1)
		geneStrand = features["CDS1"][3]
	except:
		# If parsing fails (e.g., malformed GSF, missing ">", missing CDS1),
		# report the error and return an empty result
		print(f"Error parsing GFF string, skipping.\n{gff}", file=sys.stderr)
		return [""]

	# Determine the genomic bounds (start, end) for each transcript by iterating
	# over all referenced features and tracking the min start and max end.
	# Invalid or missing features are skipped gracefully rather than aborting.
	validTranscripts = []
	transcriptBounds = []
	for transcript in transcripts:
		geneStart = None
		geneEnd = None
		validFeats = []
		for feat in transcript.split("|"):
			if feat not in features:
				# Feature referenced in transcript but not defined in feature list -- skip it
				continue
			# Validate that the feature has all 5 required fields: start|type|end|strand|phase
			featureData = features[feat]
			if len(featureData) < 5:
				print(f"Warning: malformed feature '{feat}' has {len(featureData)} elements, expected 5. Skipping.", file=sys.stderr)
				continue
			# Validate that start and end coordinates are valid integers
			try:
				featStart = int(featureData[0])
				featEnd = int(featureData[2])
			except ValueError:
				print(f"Warning: feature '{feat}' has non-integer coordinates: start='{featureData[0]}', end='{featureData[2]}'. Skipping.", file=sys.stderr)
				continue
			validFeats.append(feat)
			# Track the minimum start coordinate across all features in this transcript
			if geneStart is None:
				geneStart = featStart
			elif featStart < geneStart:
				geneStart = featStart

			# Track the maximum end coordinate across all features in this transcript
			if geneEnd is None:
				geneEnd = featEnd
			elif featEnd > geneEnd:
				geneEnd = featEnd

		# Only keep transcripts that have at least one valid feature
		if not validFeats or geneStart is None:
			continue
		validTranscripts.append("|".join(validFeats))
		# Convert relative coordinates to absolute genomic coordinates
		transcriptBounds.append((geneStart+region_start, geneEnd+region_start))

	# If no valid transcripts remain after filtering, return empty result
	if not validTranscripts:
		return [""]
	transcripts = validTranscripts

	# Construct the GFF3 output lines, building the hierarchical structure:
	# gene -> mRNA(s) -> CDS/UTR features
	import uuid
	# Generate a unique identifier for this gene; used as the root ID in the hierarchy
	id = str(uuid.uuid4())
	# The gene-level bounds span the minimum start and maximum end across all transcripts
	geneStart = min([x[0] for x in transcriptBounds])
	geneEnd = max([x[1] for x in transcriptBounds])
	# Gene line: the top-level feature encompassing all transcripts.
	# End coordinate is adjusted by -1 to convert from half-open to closed interval.
	geneModel = [f"{chr}\ttransgenic\tgene\t{geneStart}\t{geneEnd-1}\t.\t{geneStrand}\t.\tID={id};{extra_attributes}"]

	# Add mRNA (transcript) lines and their child features (CDS, UTR)
	for i,transcript in enumerate(transcripts):
		# mRNA line: represents one transcript isoform, parented to the gene.
		# Transcript IDs are suffixed with .t1, .t2, etc. for each isoform.
		geneModel.append(f"{chr}\ttransgenic\tmRNA\t{transcriptBounds[i][0]}\t{transcriptBounds[i][1]-1}\t.\t{geneStrand}\t.\tID={id}.t{i+1};Parent={id};{extra_attributes}")
		transcript = transcript.split("|")
		for featureID in transcript:
			# Skip empty feature IDs that may arise from trailing pipe characters
			if featureID == "":
				continue
			featureModel = features[featureID]
			# Validate feature structure and phase before emitting GFF3 line
			if len(featureModel) < 5 or featureModel[4] not in phaseLookup:
				print(f"Error: malformed feature '{featureID}', skipping.", file=sys.stderr)
				return [""]
			# Extract the feature type name (e.g., "CDS") by stripping numeric suffix
			featureType = re.sub(r'\d+', '', featureID)
			# Extract the numeric suffix (e.g., "1" from "CDS1") for the feature sub-ID
			featureNum = re.sub(r'\D+', '', featureID)
			# Feature line: CDS or UTR, parented to the mRNA.
			# Coordinates: add region_start for absolute position; subtract 1 from end
			# to convert from half-open interval to GFF3 closed interval.
			# Phase is mapped from GSF encoding (A/B/C) to GFF3 numeric (0/1/2).
			geneModel.append(f"{chr}\ttransgenic\t{featureType}\t{int(featureModel[0])+region_start}\t{int(featureModel[2])+region_start-1}\t.\t{geneStrand}\t{phaseLookup[featureModel[4]]}\tID={id}.t{i+1}.{featureType}{featureNum};Parent={id}.t{i+1};{extra_attributes}")

	return geneModel
