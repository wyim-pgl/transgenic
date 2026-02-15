"""
Post-processing pipeline for TransGenic gene structure predictions.

This module refines raw model predictions of gene structures (in GSF format)
by validating and correcting biological features against the actual genomic
sequence. The pipeline performs the following corrections in order:

  1. Start codon correction  -- Ensures the first CDS begins with ATG, using
     segmentation model probabilities or nearest-neighbor sequence search.
  2. Splice junction correction -- Aligns CDS and UTR exon boundaries to
     canonical donor/acceptor dinucleotides (GT-AG, GC-AG, AT-AC), selecting
     the best site by probability score and reading-frame compatibility.
  3. Stop codon correction -- Ensures the last CDS ends at a valid stop codon
     (TAA, TAG, TGA) while preserving the reading frame.
  4. UTR splice junction correction -- Adjusts UTR intron boundaries using the
     same splice-site logic but without reading-frame constraints.
  5. Unused feature analysis -- Identifies high-confidence segmentation signals
     that were not consumed by the primary transcript, which may indicate
     alternative splicing events or missed exons.

GSF format overview
-------------------
A GSF string has the form ``feature_list>transcript_list`` where:
  - ``feature_list`` is semicolon-delimited, each feature encoded as
    ``start|type|end|strand|phase``  (e.g. ``100|CDS_1|400|+|A``).
  - ``transcript_list`` is semicolon-delimited, each transcript encoded as
    pipe-delimited feature type references (e.g. ``CDS_1|CDS_2|three_prime_UTR_1``).
  - Phase is encoded as A=0, B=1, C=2 representing the number of bases to
    skip at the start of a CDS feature to reach the first complete codon.

Key conventions
---------------
- Coordinates are 0-based, half-open (start inclusive, end exclusive).
- All corrections are performed on the plus strand; minus-strand predictions
  are reverse-complemented before processing and converted back afterward.
- The ``buff`` parameter on each method controls the search window radius
  (in bases) around a predicted coordinate.
- ``seqFeatures`` are biological landmarks (start codons, stop codons, splice
  donors, splice acceptors) detected from segmentation model probability
  tracks with confidence above configurable thresholds.
"""

import sys, re
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_dilation, binary_erosion, label

from .sequence import reverseComplement
from .gsf import reverseComplement_gffString


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def classifyGeneSignal(
		probs,
		threshold=0.65,
		sigma = 100,
		#rising_threshold = 0.1,
		#falling_threshold = -0.1,
		window_length=6144):
	"""Identify contiguous gene-containing regions from a probability signal.

	The function smooths the raw gene-probability track with a 1-D Gaussian
	filter, thresholds it to produce a binary mask, applies morphological
	closing (dilation then erosion) to bridge small gaps, and finally labels
	each connected component as an independent gene segment.

	Each segment is classified as:
	  - 'Complete'  -- fully contained within the window (not touching edges).
	  - 'Cutoff'    -- extends to the right boundary, indicating the gene
	                   continues beyond this window.
	  - None        -- segment touches only the left boundary (partial, not
	                   yet classified).

	Parameters
	----------
	probs : array-like
		1-D array of per-base gene-region probabilities (channel 0 of the
		segmentation model output).
	threshold : float, optional
		Probability cutoff for the binary signal (default 0.65).
	sigma : float, optional
		Standard deviation for the Gaussian smoothing kernel (default 100).
	window_length : int, optional
		Length of the prediction window in bases (default 6144).  Used to
		determine whether a segment is cut off at the right edge.

	Returns
	-------
	segments : list of dict
		Each dict has keys 'start', 'end', 'length', 'area', and 'class'.
	binary_closed : ndarray
		The final binary mask after morphological closing.
	"""

	smoothed_signal = gaussian_filter1d(probs, sigma=sigma)			 # Smooth the signal
	binary_signal = smoothed_signal > threshold					 # Threshold the signal
	binary_closed = binary_dilation(binary_signal, iterations=2) # Apply morphological closing (dilation followed by erosion)
	binary_closed = binary_erosion(binary_closed, iterations=2)

	labeled_array, num_features = label(binary_closed)			# Label connected components in the binary signal

	segments = []												# Extract gene segments
	for seg in range(1, num_features+1):
		indices = np.where(labeled_array == seg)[0]
		start, end = indices[0], indices[-1]
		length = end - start + 1
		area = np.sum(smoothed_signal[indices])
		segments.append({'start': start, 'end': end, 'length': length, 'area': area, 'class':None})

	for segment in segments: #TODO this logic needs work! Length filter?
		# Complete gene: segment is fully within the window boundaries.
		if segment['start'] > 10 and segment['end'] < window_length - 10:
			segment['class'] = 'Complete'
		# Gene cut-off: segment touches the right boundary.
		elif segment['end'] >= window_length - 10:
			segment['class'] = 'Cutoff'

	return segments, binary_closed


def checkPhase(gff):
	"""Debug utility: print cumulative reading frame length for each CDS.

	Parses a GSF string, iterates over each transcript, and prints the
	running total of CDS bases modulo 3 after each CDS feature.  A valid
	protein-coding transcript should end with a cumulative length that is
	a multiple of 3.

	Parameters
	----------
	gff : str
		A GSF-formatted gene structure string.
	"""
	features, transcripts = gff.split('>')
	features = [f.split('|') for f in features.split(';')]
	transcripts = [m.split('|') for m in transcripts.split(';')]

	# Build a lookup from feature type name to its index in the feature list
	feature_index_dict = {}
	for i, feature in enumerate(features):
		feature_index_dict[feature[1]] = i

	for transcript in transcripts:
		seqlen = 0
		for i, typ in enumerate(transcript):
			if 'CDS' in typ:
				# Accumulate CDS length (end - start) and report remainder mod 3
				seqlen += int(features[feature_index_dict[typ]][2])-int(features[feature_index_dict[typ]][0])
				print(f"{typ}, {seqlen%3}")


# ---------------------------------------------------------------------------
# PredictionProcessor -- main post-processing class
# ---------------------------------------------------------------------------

class PredictionProcessor():
	"""Validates and corrects a single TransGenic gene structure prediction.

	On construction, the processor parses the GSF string, reverse-complements
	minus-strand predictions so that all corrections operate on the plus
	strand, and pre-computes lookup structures for efficient feature access.

	If segmentation model probability tracks are provided (``probs``), the
	processor also identifies high-confidence biological landmarks
	(``seqFeatures``) that guide correction.

	Attributes
	----------
	gff : str
		The (potentially tidied) GSF string, oriented to plus strand.
	sequence : str
		The genomic DNA sequence, reverse-complemented if the prediction
		was on the minus strand.
	probs : torch.Tensor or None
		Segmentation model probability matrix of shape (seq_len, n_channels).
		Channels: 0=gene region, 1=start codon, 4=splice donor,
		5=splice acceptor, 8=stop codon.
	strand : str
		'+' or '-', the original strand of the prediction.
	parsable : bool
		False if the GSF string could not be parsed; post-processing is
		skipped in that case.
	features : list of list of str
		Parsed feature records ``[start, type, end, strand, phase]``.
	transcripts : list of list of str
		Parsed transcript records (lists of feature type references).
	start_cds : list of str
		The first CDS feature type name for each transcript.
	end_cds : list of str
		The last CDS feature type name for each transcript (immediately
		before the 3'-UTR, or the final feature if no UTR exists).
	start_pairs : list of tuple
		``(first_CDS, five_prime_UTR_or_None)`` for each transcript.
	end_pairs : list of tuple
		``(last_CDS, three_prime_UTR_or_None)`` for each transcript.
	cds_pairs : list of tuple
		Pairs of consecutive CDS features across introns (donor, acceptor).
	utr_pairs : list of tuple
		Pairs of consecutive UTR features across introns (donor, acceptor).
	seqFeatures : dict
		High-confidence biological landmarks from the probability tracks:
		``{'startCodons': [...], 'stopCodons': [...], 'donors': [...],
		'acceptors': [...]}``.
	usedSeqFeatures : dict
		Subset of ``seqFeatures`` that were consumed during correction.
	feature_index_dict : dict
		Maps feature type name -> index in ``self.features``.
	feature_flag_dict : dict
		Maps feature type name -> string of flags indicating which
		corrections fell back to sequence search (e.g. 'start', 'stop',
		'donor', 'acceptor').

	Parameters
	----------
	gff : str
		GSF-formatted gene structure string from the model.
	sequence : str
		The genomic DNA sequence for this prediction window.
	probs : torch.Tensor or None
		Segmentation probability matrix, or None to skip probability-based
		corrections.
	"""

	def __init__(self, gff:str, sequence:str, probs):
		self.gff = gff
		self.sequence = sequence
		self.probs = probs
		# Standard genetic code stop codons
		self.stopCodons = ["TAA", "TAG", "TGA"]
		# Splice donor motifs ranked from highest to lowest quality.
		# The list reflects biological prevalence: AGGT (canonical extended
		# context) is preferred over shorter or non-canonical motifs.
		self.spliceDonors = ["AGGT", "GGT","GT","AGGC", "AGAT","GGC","GAT"]
		# Canonical donor -> acceptor dinucleotide pairings.
		# GT donors pair with AG acceptors, GC donors pair with AG acceptors,
		# and AT donors pair with AC acceptors (U12 minor spliceosome).
		self.spliceCanonical = {"GT":"AG", "GC":"AG","AT":"AC"}
		self.spliceAcceptors = ["AG","AC"]
		self.parsable = True

		# ---- Tidy the raw prediction (remove malformed features) ----
		self.gff = self.tidyPrediction()

		# ---- Determine strand by counting '+' vs '-' annotations ----
		# The strand with more occurrences in the feature list is taken as
		# the overall gene strand.
		if len(gff.split("-")) > len(gff.split("+")):
			self.strand = "-"
		else:
			self.strand = "+"

		# ---- Reverse-complement minus-strand predictions ----
		# All downstream corrections assume plus-strand orientation so that
		# coordinate arithmetic and sequence lookups are straightforward.
		if self.strand == '-':
			try:
				self.reverseComplement()
			except:
				print(f"Error: sequence could not be reverse complemented.", file=sys.stderr)
				self.parsable = False
				return

		# ---- Parse feature list and transcript list from GSF string ----
		try:
			feature_list, mRNA_list = self.gff.split('>')
			self.features = [f.split('|') for f in feature_list.split(';')]
			self.transcripts = [m.split('|') for m in mRNA_list.split(';')]
		except:
			print(f"Error: prediction could not be parsed.\n{self.gff}", file=sys.stderr)
			self.parsable = False
			return

		# ---- Identify initial and terminal CDS for each transcript ----
		# The first element of each transcript is always the first CDS.
		self.start_cds = [t[0] for t in self.transcripts]
		# The last CDS is the feature immediately before the first UTR, or
		# the very last feature if no 3'-UTR is present.
		self.end_cds = []
		for transcript in self.transcripts:
			for i, typ in enumerate(transcript):
				if 'UTR' in typ:
					# The CDS just before the UTR boundary is the terminal CDS
					self.end_cds.append(transcript[i-1])
					break
				elif i == len(transcript)-1:
					# No UTR found; the last feature itself is the terminal CDS
					self.end_cds.append(transcript[-1])

		# ---- Build feature lookup dictionaries ----
		# feature_index_dict: type name -> index in self.features (for O(1) access)
		self.feature_index_dict = {}
		for i, feature in enumerate(self.features):
			self.feature_index_dict[feature[1]] = i

		# feature_flag_dict: type name -> string of correction flags (populated
		# during processing to record which corrections fell back to sequence search)
		self.feature_flag_dict = {}
		for i, feature in enumerate(self.features):
			self.feature_flag_dict[feature[1]] = ""

		# ---- Build start codon / 5'-UTR pairs ----
		# Each pair links the first CDS of a transcript with its associated
		# 5'-UTR (if any), so that moving the start codon also updates the
		# UTR end coordinate to avoid overlap or gaps.
		self.start_pairs = []
		for i, cds in enumerate(self.start_cds):
			new_pair = (cds,None)
			for feature in self.transcripts[i]:
				if 'five_prime_UTR' in feature:
					new_pair = (cds, feature)
			self.start_pairs.append(new_pair)

		# ---- Build stop codon / 3'-UTR pairs ----
		# Same idea: the terminal CDS is paired with its 3'-UTR so that
		# moving the stop codon also updates the UTR start coordinate.
		self.end_pairs = []
		for i, cds in enumerate(self.end_cds):
			new_pair = (cds,None)
			for feature in self.transcripts[i]:
				if 'three_prime_UTR' in feature:
					new_pair = (cds, feature)
					break
			self.end_pairs.append(new_pair)

		# ---- Build splice junction pairs ----
		# cds_pairs: consecutive CDS features across an intron (donor CDS, acceptor CDS).
		# utr_pairs: consecutive UTR features of the same type across an intron.
		# These pairs define the intron boundaries that need splice-site correction.
		self.cds_pairs = []
		self.utr_pairs=[]
		for transcript in self.transcripts:
			for i, typ in enumerate(transcript):
				if i == 0:
					continue
				# CDS-CDS intron: both current and previous features must be CDS
				if 'CDS' in typ and (transcript[i-1], typ) not in self.cds_pairs and 'CD' in transcript[i-1]:
					self.cds_pairs.append((transcript[i-1], typ))
				# 5'-UTR intron: both current and previous features must be five_prime_UTR
				if 'five_prime_UTR' in typ and (transcript[i-1], typ) not in self.utr_pairs and 'five_prime_UTR' in transcript[i-1]:
					self.utr_pairs.append((transcript[i-1], typ))
				# 3'-UTR intron: both current and previous features must be three_prime_UTR
				if 'three_prime_UTR' in typ and (transcript[i-1], typ) not in self.utr_pairs and 'three_prime_UTR' in transcript[i-1]:
					self.utr_pairs.append((transcript[i-1], typ))

		# ---- Identify biological features from segmentation probabilities ----
		# These are high-confidence landmarks that guide coordinate correction.
		# Also initialise a tracking dict to record which features get consumed,
		# so that analyzeUnused() can later identify potential alternative splicing.
		if self.probs != None:
			self.seqFeatures = self.findSeqFeatures()
			self.usedSeqFeatures = {"startCodons": [],"stopCodons": [],"donors": [],"acceptors": []}

	def tidyPrediction(self):
		"""Remove malformed features and transcripts from the GSF string.

		Validates each feature record for:
		  - Recognised type (CDS, five_prime_UTR, three_prime_UTR)
		  - Parseable integer start/end coordinates with start != end
		  - Valid phase character (A, B, C, or '.')
		  - Valid strand character ('+' or '-')

		Invalid features are removed from the feature list.  Transcript
		records are then cleaned of references to removed features, duplicate
		entries, and unrecognised types.  If a transcript becomes empty after
		cleaning, it is replaced with a default transcript containing all
		remaining valid features.

		Returns
		-------
		str
			The cleaned GSF string.
		"""
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

		# ---- Validate individual features ----
		try:
			validFeatures = []
			invalidFeatureIndexes = []
			for i, feature in enumerate(features):
				valid = True
				# Check that the feature type contains at least one recognised type string
				typeCount = [typ in feature[1] for typ in types]
				try:
					start = int(feature[0])
					end = int(feature[2])
				except:
					valid = False
					start = 0
					end = 0

				# Zero-length features are invalid
				if start == end:
					valid = False
				# Phase must be A, B, C, or '.' (dot for non-CDS features)
				elif feature[4] not in phases:
					valid = False
				elif feature[3] not in strands:
					valid = False
				# Feature type must match at least one known type
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

		# Remove invalid features in reverse order to preserve indices
		for i in sorted(invalidFeatureIndexes, reverse=True):
			del features[i]

		# ---- Validate transcript feature references ----
		for i, transcript in enumerate(transcripts):
			invalidFeatureIndexes = []
			for j,feature in enumerate(transcript):
				# Check that each transcript entry references a known type
				typCount = [typ in feature for typ in types]
				if sum(typCount) < 1:
					invalidFeatureIndexes.append(j)
				# Check that the referenced feature still exists after validation
				elif feature not in validFeatures:
					if j not in invalidFeatureIndexes:
						invalidFeatureIndexes.append(j)
				# Remove duplicate references within the same transcript
				elif feature in transcript[0:j]:
					if j not in invalidFeatureIndexes:
						invalidFeatureIndexes.append(j)
			for entry in sorted(invalidFeatureIndexes, reverse=True):
				del transcripts[i][entry]

		# If a transcript is empty after cleaning, create a default transcript
		# containing all valid features in order
		for i,transcript in enumerate(transcripts):
			if len(transcript) == 0:
				transcripts[i] = [feature[1] for feature in features]

		# De-duplicate transcripts (order within each transcript is preserved)
		transcripts = [list(x) for x in set(tuple(x) for x in transcripts)]

		return f"{';'.join(['|'.join(feature) for feature in features])}>{';'.join(['|'.join(transcript) for transcript in transcripts])}"

	def reverseComplement(self):
		"""Reverse-complement the sequence, GSF coordinates, and probability tracks.

		After this call, all data is oriented to the plus strand so that
		downstream corrections can use simple left-to-right coordinate
		arithmetic.  The GSF coordinate reversal is handled by
		``reverseComplement_gffString``, which mirrors all start/end
		positions relative to the sequence length.
		"""
		self.sequence = reverseComplement(self.sequence)
		self.gff = reverseComplement_gffString(self.gff, len(self.sequence))
		if self.probs != None:
			# Flip the probability matrix along the sequence axis so that
			# position i in the reversed sequence maps to the same probability
			self.probs = torch.flip(self.probs, dims=[0])

	def findLargestGeneRegion(self, lst, threshold=0.4):
		"""Find the longest contiguous region above a probability threshold.

		Scans the probability list for runs of values exceeding ``threshold``
		and returns the start and end indices of the longest such run.

		Parameters
		----------
		lst : list of float
			Per-base probability values (typically the gene-region channel).
		threshold : float, optional
			Minimum probability to include a position in a region (default 0.4).

		Returns
		-------
		tuple of (int, int) or (None, None)
			(start, end) indices of the largest region, or (None, None) if
			no region exceeds the threshold.
		"""
		max_start = max_end = -1
		current_start = current_length = max_length = 0

		for i, value in enumerate(lst):
			if value > threshold:
				if current_length == 0:
					current_start = i  # Start a new region
				current_length += 1
			else:
				if current_length > max_length:
					max_length = current_length
					max_start = current_start
					max_end = i - 1  # End of the last valid region

				# Reset current length as we hit a value below the threshold
				current_length = 0

		# Final check in case the largest region is at the end of the list
		if current_length > max_length:
			max_length = current_length
			max_start = current_start
			max_end = len(lst) - 1

		if max_start != -1:
			return (max_start, max_end)
		else:
			return (None, None)

	def findSeqFeatures(self, junctionThresh=0.6, startThresh=0.6):
		"""Detect biological landmarks from segmentation probability tracks.

		Scans four probability channels to identify high-confidence positions
		for start codons, stop codons, splice donors, and splice acceptors.
		Each candidate is validated against the actual DNA sequence to confirm
		it carries the expected motif.  Results are sorted by descending
		probability so that higher-confidence candidates are tried first
		during correction.

		Probability channels used:
		  - Channel 1: start codon (ATG). A 3-base sliding average is
		    computed and positions above ``startThresh`` with a matching
		    ATG in the sequence are retained.
		  - Channel 8: stop codon (TAA/TAG/TGA). Same 3-base averaging.
		    Coordinates are stored as the position immediately AFTER the
		    stop codon (i.e. the end coordinate for half-open intervals).
		  - Channel 4: splice donor. A 2-base sliding average is computed.
		    Only positions matching a canonical donor dinucleotide (GT, GC,
		    or AT) are retained.
		  - Channel 5: splice acceptor. Same 2-base averaging. Only
		    positions matching a canonical acceptor dinucleotide (AG or AC)
		    are retained.  Coordinates are stored as the position immediately
		    AFTER the acceptor dinucleotide (the exon start).

		Parameters
		----------
		junctionThresh : float, optional
			Minimum average probability for splice site detection (default 0.6).
		startThresh : float, optional
			Minimum average probability for start/stop codon detection
			(default 0.6).

		Returns
		-------
		dict
			Keys: 'startCodons', 'stopCodons', 'donors', 'acceptors'.
			Values: lists of 0-based sequence positions, sorted by descending
			probability score.
		"""

		# Find the central gene region
		#segments = classifyGeneSignal(self.probs[:, 0], threshold=geneThresh, sigma=200)
		#prev_len = 0
		#for segment in segments[0]:
		#	if int(segment["length"]) > prev_len:
		#		geneStart = segment["start"]
		#		geneEnd = segment["end"]
		#		prev_len = int(segment["length"])

		#geneStart, geneEnd = self.findLargestGeneRegion(self.probs[:,0], threshold=geneThresh)

		# ---- Start codons (channel 1) ----
		# Compute a 3-base sliding average of the start-codon probability,
		# matching the triplet nature of the ATG motif.
		means = []
		lst = self.probs[:,1].tolist() #self.probs[geneStart:geneEnd,1].tolist()
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0) + (lst[i + 2] if i + 2 < len(lst) else 0)
			means.append(sum_value/3)
		# Filter by threshold, then confirm ATG in sequence, then rank by probability
		startCodons = [i for i, x in enumerate(means) if x > startThresh]
		startCodons = [i for i in startCodons if self.sequence[i:i+3] == "ATG"]
		startCodons = sorted(startCodons, key=lambda i: means[i], reverse=True)

		# ---- Stop codons (channel 8) ----
		# Same 3-base sliding average for stop codon probability.
		# Coordinates stored as end position (i+3) for half-open interval convention.
		means = []
		lst = self.probs[:,8].tolist()
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0) + (lst[i + 2] if i + 2 < len(lst) else 0)
			means.append(sum_value/3)
		stopCodons = [i for i, x in enumerate(means) if x > startThresh]
		# +3 converts to half-open end coordinate (position after the stop codon)
		stopCodons = [i+3 for i in stopCodons if self.sequence[i:i+3] in self.stopCodons]
		# Sort by probability at the start of the codon (i-3 because we already added 3)
		stopCodons = sorted(stopCodons, key=lambda i: means[i-3], reverse=True)

		# ---- Splice donors (channel 4) ----
		# 2-base sliding average for the donor dinucleotide (GT, GC, or AT).
		means = []
		lst = self.probs[:,4].tolist()
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0)
			means.append(sum_value/2)
		# Only keep positions where the sequence carries a canonical donor dinucleotide
		donors = [i for i, x in enumerate(means) if (x > junctionThresh) and (self.sequence[i:i+2] in self.spliceCanonical.keys())]
		donors = sorted(donors, key=lambda i: means[i], reverse=True)

		# ---- Splice acceptors (channel 5) ----
		# 2-base sliding average for the acceptor dinucleotide (AG or AC).
		# Coordinates stored as i+2: the first base of the downstream exon.
		means = []
		lst = self.probs[:,5].tolist()
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0)
			means.append(sum_value/2)
		acceptors = [i+2 for i, x in enumerate(means) if x > junctionThresh and (self.sequence[i:i+2] in self.spliceCanonical.values())]
		# Sort by probability at the dinucleotide start (i-2 because we already added 2)
		acceptors = sorted(acceptors, key=lambda i: means[i-2], reverse=True)

		return {
			"startCodons": startCodons,
			"stopCodons": stopCodons,
			"donors": donors,
			"acceptors": acceptors
		}

	def checkStartCodons(self, buff=50, usingSeqFeature=False):
		"""Correct start codon positions for all transcripts.

		Strategy (two-tier):
		  1. If segmentation features are available (``usingSeqFeature=True``),
		     pick the highest-probability ATG within ``buff`` bases of the
		     predicted start that does not introduce an in-frame stop codon
		     before the end of the first CDS.
		  2. If no segmentation-based ATG is found (or ``usingSeqFeature`` is
		     False), fall back to the nearest ATG in the sequence within the
		     buffer window.

		The first CDS of every transcript always receives phase 'A' (= 0),
		because the reading frame starts at the first base of the start codon.

		After adjusting start codons, the end coordinate of the associated
		5'-UTR (if any) is updated to match, so the UTR and CDS remain
		abutting with no gap or overlap.

		Parameters
		----------
		buff : int, optional
			Half-width of the search window around the predicted start
			coordinate (default 50).
		usingSeqFeature : bool, optional
			Whether to use segmentation probability features for correction
			(default False).
		"""

		if usingSeqFeature:
			startCodons = self.seqFeatures["startCodons"]

		# ---- Process each transcript's first CDS ----
		# Pick the highest probability (lowest index) start which doesn't
		# introduce in-frame stop codons.
		# Phase is always set to 'A' (= 0) for the first CDS because the
		# reading frame starts at the first base of the start codon.
		for i, cds in enumerate(self.start_cds):
			self.features[self.feature_index_dict[cds]][4] = 'A'
			start = int(self.features[self.feature_index_dict[cds]][0])
			if usingSeqFeature:
				if len(startCodons) > 0:
					# Find all segmentation start codons within the buffer window
					new_starts = [i for i in startCodons if i in range(start-buff,start+buff)]
					found=False
					if len(new_starts)>0:
						# If only one CDS (single-exon gene), don't constrain by
						# internal stop codons since the stop codon is at the end
						if self.start_cds == self.end_cds:
							new_start = new_starts[0]
							found = True
						else:
							# For multi-exon genes, reject candidates that create
							# an in-frame stop codon within the first CDS
							for new_start in new_starts:
								containsStop = self.containsStopCodon(self.sequence[new_start:int(self.features[self.feature_index_dict[cds]][2])], "A")
								if not containsStop:
									found = True
									break

						if found:
							self.features[self.feature_index_dict[cds]][0] = str(new_start)
							if new_start not in self.usedSeqFeatures["startCodons"]:
								self.usedSeqFeatures["startCodons"].append(new_start)
						else:
							# No stop-codon-free candidate; use the top-ranked one
							# anyway and flag for sequence-based fallback
							self.features[self.feature_index_dict[cds]][0] = str(new_starts[0])
							if new_starts[0] not in self.usedSeqFeatures["startCodons"]:
								self.usedSeqFeatures["startCodons"].append(new_starts[0])
							usingSeqFeature = False
					else:
						# No segmentation start codons in the buffer window
						usingSeqFeature = False
				else:
					# Segmentation found no start codons anywhere
					usingSeqFeature = False

			# ---- Fallback: sequence-based ATG search ----
			# Search the buffer window for ATG triplets and pick the one
			# closest to the originally predicted start coordinate.
			if not usingSeqFeature:
				self.feature_flag_dict[cds] += "start"
				buffer = self.sequence[start-buff:start+buff]
				startCodons = [m.start() for m in re.finditer("ATG", buffer)]
				if startCodons:
					# Select the ATG whose distance from the center of the
					# buffer (i.e. the predicted position) is smallest
					pick = startCodons[np.argmin([np.abs(loc-buff) for loc in startCodons])]
					# Convert buffer-relative offset back to absolute coordinate
					new_start = start + pick-buff
					self.features[self.feature_index_dict[cds]][0] = str(new_start)

		# ---- Update 5'-UTR end coordinates to match corrected start codons ----
		# The UTR end should abut the CDS start with no gap or overlap
		for cds, utr in self.start_pairs:
			if utr:
				self.features[self.feature_index_dict[utr]][2] = self.features[self.feature_index_dict[cds]][0]

	def checkStopCodons(self, buff=50, usingSeqFeature=False):
		"""Correct stop codon positions for all transcripts.

		Strategy (two-tier, analogous to ``checkStartCodons``):
		  1. If segmentation features are available, pick the highest-
		     probability stop codon within ``buff`` bases of the predicted
		     end that maintains a valid reading frame (total CDS length
		     divisible by 3).
		  2. If no segmentation-based stop is found, fall back to the nearest
		     stop codon in the sequence within the buffer window.

		After adjusting stop codons, the start coordinate of the associated
		3'-UTR (if any) is updated to match the new CDS end.

		Parameters
		----------
		buff : int, optional
			Half-width of the search window around the predicted stop
			coordinate (default 50).
		usingSeqFeature : bool, optional
			Whether to use segmentation probability features (default False).
		"""
		if usingSeqFeature:
			stopCodons = self.seqFeatures["stopCodons"]

		# ---- Process each transcript's terminal CDS ----
		# Pick the highest probability (lowest index) stop which maintains
		# a valid frame (total CDS length across all exons must be divisible by 3).
		for i, cds in enumerate(self.end_cds):
			stop = int(self.features[self.feature_index_dict[cds]][2])
			if usingSeqFeature:
				if len(stopCodons) > 0:
					# Find all segmentation stop codons within the buffer window
					new_stops = [i for i in stopCodons if i in range(stop-buff, stop+buff)]
					found = False
					if len(new_stops)>0:
						# Determine the transcript that this terminal CDS belongs to
						# and sum up all upstream CDS lengths to check frame compatibility
						working_transcript_index = [i for i,t in enumerate(self.transcripts) if cds in self.transcripts[i]][0]
						upstream_cds = [i for i in self.transcripts[working_transcript_index] if "CDS" in i]
						upstream = upstream_cds[:-1]
						# Sum lengths of all CDS features upstream of the terminal CDS
						length = 0
						for feat in upstream:
							length += int(self.features[self.feature_index_dict[feat]][2]) - int(self.features[self.feature_index_dict[feat]][0])

						for new_stop in new_stops:
							# Total coding length = upstream CDS length + terminal CDS length
							# For a valid ORF this must be a multiple of 3
							full_length = length + (new_stop - int(self.features[self.feature_index_dict[cds]][0]))
							if full_length % 3 == 0:
								found = True
								break

						if found:
							self.features[self.feature_index_dict[cds]][2] = str(new_stop)
							if new_stop not in self.usedSeqFeatures["stopCodons"]:
								self.usedSeqFeatures["stopCodons"].append(new_stop)
						else:
							# No frame-compatible stop found; use top-ranked anyway
							self.features[self.feature_index_dict[cds]][2] = str(new_stops[0])
							if new_stops[0] not in self.usedSeqFeatures["stopCodons"]:
								self.usedSeqFeatures["stopCodons"].append(new_stops[0])
							usingSeqFeature = False
					else:
						usingSeqFeature = False
				else:
					usingSeqFeature = False

			# ---- Fallback: sequence-based stop codon search ----
			# Search for TAA/TAG/TGA in the buffer window and pick the closest.
			if not usingSeqFeature:
				self.feature_flag_dict[cds] += "stop"
				buffer = self.sequence[stop-buff:stop+buff]
				# +3 on match start to get the half-open end coordinate
				stopCodons = [m.start() + 3 for m in re.finditer("(TAA)|(TAG)|(TGA)", buffer)]
				if stopCodons:
					pick = stopCodons[np.argmin([np.abs(loc-buff) for loc in stopCodons])]
					# Convert buffer-relative offset back to absolute coordinate
					new_stop = stop + pick-buff
					self.features[self.feature_index_dict[cds]][2] = str(new_stop)

		# ---- Update 3'-UTR start coordinates to match corrected stop codons ----
		# The UTR start should abut the CDS end with no gap or overlap
		for cds, utr in self.end_pairs:
			if utr:
				self.features[self.feature_index_dict[utr]][0] = self.features[self.feature_index_dict[cds]][2]

	def containsStopCodon(self, seq, phase):
		"""Check whether a sequence contains an in-frame stop codon.

		Splits the sequence into codons according to the given phase offset
		and checks for TAA, TAG, or TGA among the resulting triplets.

		Parameters
		----------
		seq : str
			The DNA subsequence to check.
		phase : str
			Phase code ('A', 'B', 'C', or '.') indicating the reading frame
			offset.  A/. = 0 bases skipped, B = 1 base skipped, C = 2 bases
			skipped before the first complete codon.

		Returns
		-------
		bool
			True if at least one in-frame stop codon is present.
		"""
		phaseLookup = {"A":0,"B":1,"C":2,".":0}
		# Split into codons: first element is the partial codon before the
		# reading frame starts (may be empty for phase A), followed by
		# complete triplets from the phase offset onward.
		seq = [seq[:phaseLookup[phase]]] + [seq[i:min(i+3, len(seq)-phaseLookup[phase])] for i in range(phaseLookup[phase], len(seq), 3)]
		return "TAG" in seq or "TAA" in seq or "TGA" in seq


	def checkProbableSpliceJunctions(self, pairs, buff=100, cds=True):
		"""Correct splice junctions using segmentation probability features.

		For each intron (defined by a donor/acceptor feature pair), this
		method searches within a buffer window for the highest-probability
		donor and acceptor sites from ``self.seqFeatures``.

		For CDS introns (``cds=True``):
		  - The donor site must not introduce an in-frame stop codon in the
		    upstream exon.
		  - The acceptor's phase is recalculated based on the new donor
		    position to maintain reading frame continuity.

		For UTR introns (``cds=False``):
		  - No reading-frame constraints; the top-ranked site is used directly.

		Acceptor sites are further constrained to match the canonical pairing
		of the selected donor dinucleotide (GT->AG, GC->AG, AT->AC).

		If no segmentation feature is found in the buffer window, the method
		falls back to the nearest matching dinucleotide in the raw sequence.

		Parameters
		----------
		pairs : list of tuple of str
			List of (donor_feature_type, acceptor_feature_type) pairs
			defining the introns to correct.
		buff : int, optional
			Half-width of the search window (default 100).
		cds : bool, optional
			If True, apply reading-frame constraints (default True).
		"""
		phase_lookup = {"A":0, "B":1,"C":2}
		index_lookup = {0:"A", 1:"B",2:"C"}

		donors = self.seqFeatures["donors"]
		acceptors = self.seqFeatures["acceptors"]

		for donor, acceptor in pairs:
			donor_start = int(self.features[self.feature_index_dict[donor]][0])
			donor_end = int(self.features[self.feature_index_dict[donor]][2])
			donor_phase = self.features[self.feature_index_dict[donor]][4]
			acceptor_start = int(self.features[self.feature_index_dict[acceptor]][0])

			# ---- Splice donor correction ----
			# Find segmentation donors within the buffer window around the
			# predicted exon end, then select the best one.
			donor_type = None
			if len(donors) > 0:
				new_donors = [i for i in donors if i in range(donor_end-buff,donor_end+buff)]
				found = False
				if (len(new_donors)>0) and cds: # Working on CDS feature
					# Try each candidate in descending probability order;
					# pick the first that doesn't create an in-frame stop codon
					for new_donor in new_donors:
						containsStop = self.containsStopCodon(self.sequence[donor_start:new_donor], donor_phase) # Maybe just use whole seq instead of tracking phase?
						if not containsStop:
							found = True
							break

					if found:
						self.features[self.feature_index_dict[donor]][2] = str(new_donor)
						donor_type  = self.sequence[new_donor:new_donor+2]
						# Calculate the acceptor phase:
						# Phase = (donor_phase + bases_remaining_in_last_codon) mod 3
						# (3 - exon_length%3) gives bases needed to complete the codon;
						# adding donor_phase accounts for the upstream frame offset.
						acceptor_phase = index_lookup[(phase_lookup[donor_phase] + (3-((new_donor-donor_start)%3)))%3]
						self.features[self.feature_index_dict[acceptor]][4] = acceptor_phase
						if new_donor not in self.usedSeqFeatures["donors"]:
							self.usedSeqFeatures["donors"].append(new_donor)
					else:
						# No stop-codon-free candidate; use top-ranked donor anyway
						self.features[self.feature_index_dict[donor]][2] = str(new_donors[0])
						donor_type  = self.sequence[new_donors[0]:new_donors[0]+2]
						# Phase calculation without the frame-safety inversion
						acceptor_phase = index_lookup[(phase_lookup[donor_phase] + ((new_donors[0]-donor_start)%3))%3]
						self.features[self.feature_index_dict[acceptor]][4] = acceptor_phase
						if new_donors[0] not in self.usedSeqFeatures["donors"]:
							self.usedSeqFeatures["donors"].append(new_donors[0])

				elif (len(new_donors)>0): # Working on UTR feature (no phase constraints)
					self.features[self.feature_index_dict[donor]][2] = str(new_donors[0])
					donor_type  = self.sequence[new_donors[0]:new_donors[0]+2]
					if new_donors[0] not in self.usedSeqFeatures["donors"]:
							self.usedSeqFeatures["donors"].append(new_donors[0])

				else: # If no donor site from segmentation, look for nearest donor in the buffer window
					self.feature_flag_dict[donor] += "donor"
					buffer = self.sequence[donor_end-buff:donor_end+buff]
					# Search for the GT dinucleotide (most common canonical donor)
					donors = [m.start() for m in re.finditer("GT", buffer)]
					if donors:
						# Pick the GT closest to the predicted exon end
						pick = donors[np.argmin([np.abs(loc-buff) for loc in donors])]
						new_donor = donor_end + pick-buff
						self.features[self.feature_index_dict[donor]][2] = str(new_donor)
						donor_type  = self.sequence[new_donor:new_donor+2]
						if cds:
							# Update acceptor phase based on the new donor position
							acceptor_phase = index_lookup[(phase_lookup[donor_phase] + ((new_donor-donor_start)%3))%3]
							self.features[self.feature_index_dict[acceptor]][4] = acceptor_phase

			# ---- Splice acceptor correction ----
			# Find segmentation acceptors within the buffer window around the
			# predicted exon start, constrained by canonical donor-acceptor pairing.
			if len(acceptors) > 0:
				new_acceptors = [i for i in acceptors if i in range(acceptor_start-buff,acceptor_start+buff)]
				found = False
				if len(new_acceptors)>0:
					# Try each candidate; pick the first whose dinucleotide
					# matches the canonical partner of the chosen donor
					for new_acceptor in new_acceptors:
						if donor_type and (donor_type in self.spliceCanonical.keys()):
							if self.sequence[new_acceptor-2:new_acceptor] == self.spliceCanonical[donor_type]:
								found = True
								break

					if found:
						self.features[self.feature_index_dict[acceptor]][0] = str(new_acceptor)
						if new_acceptor not in self.usedSeqFeatures["acceptors"]:
							self.usedSeqFeatures["acceptors"].append(new_acceptor)
					else:
						# No canonical match found; use top-ranked acceptor anyway
						self.features[self.feature_index_dict[acceptor]][0] = str(new_acceptors[0])
						if new_acceptors[0] not in self.usedSeqFeatures["acceptors"]:
							self.usedSeqFeatures["acceptors"].append(new_acceptors[0])
				else:
					# ---- Fallback: sequence-based acceptor search ----
					self.feature_flag_dict[acceptor] += "acceptor"
					buffer = self.sequence[acceptor_start-buff:acceptor_start+buff]
					acceptors = [m.start() for m in re.finditer("AG", buffer)]
					if acceptors:
						pick = acceptors[np.argmin([np.abs(loc-buff) for loc in acceptors])]
						# +2 because the exon starts after the acceptor dinucleotide
						new_acceptor = acceptor_start + pick-buff +2
						self.features[self.feature_index_dict[acceptor]][0] = str(new_acceptor)


	def checkCanonicalSpliceJunctions(self, buff=50):
		"""Correct splice junctions using canonical splice site pattern matching.

		This method provides an alternative splice correction strategy that
		does not rely on segmentation probabilities.  Instead, it searches
		the sequence around each predicted intron boundary for canonical
		splice site motifs and selects the best match using a quality hierarchy.

		Donor site quality ranking (highest to lowest):
		  AGGT > GGT > GT > AGGC > AGAT > GGC > GAT

		The extended context (e.g. AGGT vs GT) captures the exonic nucleotides
		immediately upstream of the splice site, which contribute to splice
		efficiency in vivo.

		For each quality tier, the method calls ``selectDonorByPhase`` to find
		a candidate that maintains reading frame continuity (cumulative CDS
		length up to and including the donor matches the expected phase of
		the downstream acceptor).

		If no phase-compatible donor is found at any quality tier, the method
		falls back to the closest site at the highest available quality tier
		without phase constraints.

		Acceptor sites are paired canonically with the chosen donor
		dinucleotide (GT->AG, GC->AG, AT->AC).

		Parameters
		----------
		buff : int, optional
			Half-width of the search window around predicted intron
			boundaries (default 50).

		Notes
		-----
		The ``sjLog`` attribute is populated during this method to record all
		candidate sites found at each quality tier, which can be useful for
		debugging splice site selection.

		The ``decisionTensors`` mentioned in the code comments refer to a
		planned (but not yet implemented) tensor-based approach with dimensions
		for stop-codon presence and frame congruence across all donor-acceptor
		combinations.
		"""
		# Use canonical splice junctions to adjust feature cordinates.
		# Search for donor and acceptor splice sites
		# Find the highest quality donor site closest to prediction. Preference order (AGGT/GGT/AGGC/AGAT/GGC/GAT)
		# Select the matching acceptor site closest to the prediction which does not introduce in-frame stop codons
		# buff: search buffer for cds boundary markers
		# note: decisionTensors are (2, #donor sites,#acceptor sites). Dimension 0[0] is stop codon presence, Dimension 0[1] frame congruence

		# TODO: Need a way to mitigate the cumulative effect of incorrectly predicted splice junctions when using phase
		#       to update the prediction. I need to balance the phase correction approach with the fact that some phase
		#       calls will be wrong and I will not find the correct splice junction everytime.

		# ---- Initialize the splice junction log ----
		# For each CDS feature, record all candidate donors/acceptors found
		# at each quality tier, plus which one was ultimately chosen.
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

		# ---- Process each CDS intron ----
		for donor, acceptor in self.cds_pairs:
			donor_start = int(self.features[self.feature_index_dict[donor]][0])
			donor_end = int(self.features[self.feature_index_dict[donor]][2])
			acceptor_start = int(self.features[self.feature_index_dict[acceptor]][0])
			acceptor_end = int(self.features[self.feature_index_dict[acceptor]][2])

			# ---- Find candidate donor sites in the buffer window ----
			# If the exon is shorter than the buffer, shrink the upstream
			# search to avoid extending past the exon start.
			if donor_end-donor_start < buff:
				buf = donor_end-donor_start
			else:
				buf = buff
			buffer = self.sequence[donor_end-buf:donor_end+buff]
			for site in self.spliceDonors:
				# re.finditer returns match objects; .end() gives the position
				# after the full motif match, then we subtract buf+2 to get
				# the offset relative to the predicted donor_end, aligned to
				# the core donor dinucleotide position.
				HC = [m.end() for m in re.finditer(site, buffer)]
				pick = [loc-buf-2 for loc in HC]
				# Sort by absolute offset from the predicted position (closest first)
				pick = sorted(pick, key=abs)
				self.sjLog[donor]["donor"][site] = pick

			# ---- Find candidate acceptor sites in the buffer window ----
			if acceptor_end-acceptor_start < buff:
				buf = acceptor_end-acceptor_start
			else:
				buf = buff
			buffer = self.sequence[acceptor_start-buff:acceptor_start+buf]
			for site in self.spliceAcceptors:
				HC = [m.start() for m in re.finditer(site, buffer)]
				# Offset relative to predicted acceptor_start; +2 because the
				# exon starts after the acceptor dinucleotide
				pick = [loc-buf+2 for loc in HC]
				pick = sorted(pick, key=abs)
				self.sjLog[acceptor]["acceptor"][site] = pick


			# ---- Select the highest quality donor site with phase correction ----
			# Try each quality tier in descending order.  selectDonorByPhase
			# returns True if a phase-compatible donor was found at this tier.
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

			# ---- Fallback: select closest donor without phase constraint ----
			# If no phase-compatible donor was found at any quality tier,
			# pick the closest site at the highest available quality tier.
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

			# ---- Select the matching acceptor site ----
			# Pop the closest candidate from the chosen acceptor type.
			# If no acceptor has been found, don't update.
			try:
				closestAcceptor = self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]].pop(0)
			except:
				closestAcceptor = None

			#if closestAcceptor:
			#	self.features[self.feature_index_dict[acceptor]][0] = str(int(acceptor_start + closestAcceptor))
			#	if self.containsStopCodon((donor,acceptor)):
			#		if len(self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]]) > 1:
			#			while self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]]:
			#				self.features[self.feature_index_dict[acceptor]][0] = str(int(acceptor_start + self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]].pop(0)))
			#				if not self.containsStopCodon((donor,acceptor)):
			#					break

	def selectDonorByPhase(self, donor:str, donor_end:int, acceptor:str, dsite:str, asite:str) -> bool:
		"""Select a donor site that maintains reading frame continuity.

		For each candidate at the given quality tier (``dsite``), this method
		computes the cumulative CDS length from the start of the first
		transcript up to and including the adjusted donor exon.  A candidate
		is accepted if this cumulative length modulo 3 equals the phase
		value recorded for the downstream acceptor feature.

		The phase_lookup here maps phase codes to the expected modulo-3
		remainder of cumulative CDS length at an exon-intron boundary:
		  - A -> 0 : exon ends on a complete codon boundary
		  - B -> 2 : 2 bases of the last codon are in this exon (1 base
		             carries over to the next exon)
		  - C -> 1 : 1 base of the last codon is in this exon (2 bases
		             carry over to the next exon)

		Note that phase_lookup values differ from the standard phase
		encoding (A=0, B=1, C=2) because phase represents the offset
		at the START of the next exon, not the remainder at the end.
		Specifically:
		  - Phase B means skip 1 base at the next exon start, which
		    means 2 bases of the codon ended in the current exon
		    (cumulative length mod 3 = 2).
		  - Phase C means skip 2 bases, so 1 base ended in the current
		    exon (cumulative length mod 3 = 1).

		Parameters
		----------
		donor : str
			Feature type name of the donor (upstream) CDS.
		donor_end : int
			Originally predicted end coordinate of the donor CDS.
		acceptor : str
			Feature type name of the acceptor (downstream) CDS.
		dsite : str
			Donor motif quality tier being tested (e.g. 'AGGT', 'GT').
		asite : str
			Canonical acceptor dinucleotide paired with ``dsite``
			(e.g. 'AG' for GT donors).

		Returns
		-------
		bool
			True if a phase-compatible donor was found and applied.
			False otherwise.
		"""
		# Select the donor site with the lowest distance to the predicted site which maintains phase in the first transcript
		phase_lookup = {"A":0, "B":2,"C":1, ".":0} # modulo of cumulative previous sequence
		moveOn = False
		if len(self.sjLog[donor]["donor"][dsite]) > 0:
			chosen = None
			for loc in self.sjLog[donor]["donor"][dsite]:
				# Compute the absolute position of this candidate donor
				pick = str(int(donor_end + loc))
				# Use the first transcript for phase checking
				transcript = self.transcripts[0]
				if (donor in transcript) & (acceptor in transcript):
					# Sum lengths of all CDS features preceding the donor exon
					transcript_length = 0
					for typ in transcript:
						if typ == donor:
							break
						if 'CDS' in typ:
							transcript_length += int(self.features[self.feature_index_dict[typ]][2])-int(self.features[self.feature_index_dict[typ]][0])
					# Add the donor exon length (from its start to the candidate donor position)
					transcript_length += int(pick)-int(self.features[self.feature_index_dict[typ]][0])
					# Check if cumulative length mod 3 matches the expected phase of the acceptor
					if (transcript_length % 3) == phase_lookup[self.features[self.feature_index_dict[acceptor]][4]]:
						chosen = dsite
						break

			if chosen:
				# Record the chosen donor and acceptor types in the log
				self.sjLog[donor]["donor"]["chosen"] = dsite
				self.sjLog[acceptor]["acceptor"]["chosen"] = asite
				# Update the donor end coordinate to the selected position
				self.features[self.feature_index_dict[donor]][2] = pick
				moveOn = True

			return moveOn

	def checkSpliceJunctions(self, pairs, buff=50, cds=True):
		"""Dispatch splice junction correction to the probability-based method.

		This is a thin wrapper that calls ``checkProbableSpliceJunctions``.
		The commented-out call to ``checkCanonicalSpliceJunctions`` is an
		alternative strategy that uses sequence pattern matching without
		probability scores.

		Parameters
		----------
		pairs : list of tuple of str
			Intron boundary pairs (donor_type, acceptor_type).
		buff : int, optional
			Search window half-width (default 50).
		cds : bool, optional
			Whether pairs are CDS introns requiring phase constraints
			(default True).
		"""
		self.checkProbableSpliceJunctions(pairs, buff=buff, cds=cds)
		#self.checkCanonicalSpliceJunctions(buff=buff)

	def stitchGFF(self, originalStrand=True):
		"""Reassemble the corrected features into a GSF string.

		Joins the (potentially modified) feature and transcript lists back
		into GSF format.  If the original prediction was on the minus strand
		and ``originalStrand`` is True, the coordinates are reverse-
		complemented back to minus-strand orientation.

		Parameters
		----------
		originalStrand : bool, optional
			If True (default), convert back to the original strand
			orientation before returning.

		Returns
		-------
		str
			The corrected GSF string.
		"""
		updatedGFF = f"{';'.join(['|'.join(feature) for feature in self.features])}>{';'.join(['|'.join(transcript) for transcript in self.transcripts])}"
		if originalStrand and (self.strand == "-"):
			updatedGFF = reverseComplement_gffString(updatedGFF, len(self.sequence))
		return updatedGFF

	def analyzeUnused(self):
		"""Identify segmentation features not consumed by the primary annotation.

		After post-processing, some high-confidence segmentation landmarks
		may remain unused.  These can indicate:
		  - Alternative transcription start sites (unused start codons)
		  - Alternative polyadenylation / stop codon usage (unused stop codons)
		  - Skipped exons or alternative splicing (unused donors/acceptors)

		For each unused feature, the method attempts to pair it with the
		nearest complementary feature (e.g. an unused acceptor is paired
		with the nearest downstream donor to define a potential exon).

		Returns
		-------
		dict
			Keys and their meanings:
			  - 'extraStarts' : list of (start, end) tuples for potential
			    alternative first exons (unused start codon paired with the
			    nearest downstream donor).
			  - 'extraStops' : list of (start, end) tuples for potential
			    alternative last exons (nearest upstream acceptor paired with
			    the unused stop codon).
			  - 'extraAs' : list of (start, end) tuples for potential exons
			    defined by unused acceptors paired with the nearest downstream
			    donor.
			  - 'extraDs' : list of (start, end) tuples for potential exons
			    defined by the nearest upstream acceptor paired with unused
			    donors.
		"""
		numcds = len(self.cds_pairs) + 1

		# ---- Identify potential alternative start sites ----
		# For each unused start codon, find the nearest downstream donor
		# to define a potential alternative first exon.
		unusedStart = list(set(self.seqFeatures["startCodons"])-set(self.usedSeqFeatures["startCodons"]))
		extraStarts = []
		for newStart in unusedStart:
			candidates = [i for i in self.seqFeatures["donors"] if i > newStart]
			if candidates:
				newStart_end = candidates[0]
				extraStarts.append((newStart,newStart_end))
				if newStart_end not in self.usedSeqFeatures["donors"]:
					self.usedSeqFeatures["donors"].append(newStart_end)

		# ---- Identify potential exons from unused acceptor junctions ----
		# For each unused acceptor, find the nearest downstream donor
		# to define a potential internal exon.
		unusedAcceptors = list(set(self.seqFeatures["acceptors"])-set(self.usedSeqFeatures["acceptors"]))
		extraAs = []
		for newA in unusedAcceptors:
			candidates = [i for i in self.seqFeatures["donors"] if i > newA]
			if candidates:
				newA_end = candidates[0]
				extraAs.append((newA, newA_end))
				if newA_end not in self.usedSeqFeatures["donors"]:
					self.usedSeqFeatures["donors"].append(newA_end)

		# ---- Identify potential exons from unused donor junctions ----
		# For each unused donor, find the nearest upstream acceptor
		# to define a potential internal exon.
		unusedDonors = list(set(self.seqFeatures["donors"])-set(self.usedSeqFeatures["donors"]))
		extraDs = []
		for newD in unusedDonors:
			candidates = [i for i in self.seqFeatures["acceptors"] if i < newD]
			if candidates:
				newD_start = [i for i in self.seqFeatures["acceptors"] if i < newD][-1]
				extraDs.append((newD_start, newD))
				if newD_start not in self.usedSeqFeatures["acceptors"]:
					self.usedSeqFeatures["acceptors"].append(newD_start)

		# ---- Identify potential alternative stop sites ----
		# For each unused stop codon, find the nearest upstream acceptor
		# to define a potential alternative last exon.
		unusedStop = list(set(self.seqFeatures["stopCodons"])-set(self.usedSeqFeatures["stopCodons"]))
		extraStops = []
		for newStop in unusedStop:
			candidates = [i for i in self.seqFeatures["acceptors"] if i < newStop]
			if candidates:
				newStop_start = candidates[-1]
				extraStops.append((newStop_start, newStop))
				if newStop_start not in self.usedSeqFeatures["acceptors"]:
					self.usedSeqFeatures["acceptors"].append(newStop_start)

		return {
			"extraStarts":extraStarts,
			"extraStops":extraStops,
			"extraAs":extraAs,
			"extraDs":extraDs
		}




	def postProcessPrediction(self, utr_splice_buf = 100, splice_buffer=10, start_buffer=150, stop_buffer=150, seqFeature=True) -> str:
		"""Run the full post-processing pipeline on this prediction.

		Executes the correction steps in the biologically required order:
		  1. **Start codon correction** (``checkStartCodons``) -- Establishes
		     the reading frame origin.
		  2. **CDS splice junction correction** (``checkSpliceJunctions`` on
		     ``cds_pairs``) -- Fixes intron boundaries while maintaining the
		     reading frame established in step 1.
		  3. **Stop codon correction** (``checkStopCodons``) -- Ensures the
		     ORF terminates at a valid stop codon in the correct frame.
		  4. **UTR splice junction correction** (``checkSpliceJunctions`` on
		     ``utr_pairs``) -- Fixes UTR intron boundaries (no frame
		     constraints).

		If the prediction was not parseable during construction, the original
		GSF string is returned unchanged.

		Parameters
		----------
		utr_splice_buf : int, optional
			Search window half-width for UTR splice junctions (default 100).
			UTR junctions use a larger buffer because they are typically less
			precisely predicted than CDS junctions.
		splice_buffer : int, optional
			Search window half-width for CDS splice junctions (default 10).
			CDS junctions use a small buffer because incorrect adjustments
			can break the reading frame.
		start_buffer : int, optional
			Search window half-width for start codon correction (default 150).
		stop_buffer : int, optional
			Search window half-width for stop codon correction (default 150).
		seqFeature : bool, optional
			Whether to use segmentation probability features for start/stop
			codon correction (default True).

		Returns
		-------
		str
			The corrected GSF string, converted back to the original strand
			orientation.
		"""
		# Use start/stop codon, splice junction, and valid protein translation to adjust feature cordinates and phases.
		# Enforce that each transcript is a valid protein-coding sequence.
		# Ensure that updates do not make overlapping exons
		# buff: search buffer for cds boundary markers

		if self.parsable:
			self.checkStartCodons(buff=start_buffer, usingSeqFeature=seqFeature)
			self.checkSpliceJunctions(self.cds_pairs, buff=splice_buffer)
			self.checkStopCodons(buff=stop_buffer, usingSeqFeature=seqFeature)
			self.checkSpliceJunctions(self.utr_pairs, buff=utr_splice_buf, cds=False)
			return self.stitchGFF()
		else:
			return self.gff
