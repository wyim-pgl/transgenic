import torch, random
from typing import List, Tuple

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

def validateCDS(gff:str, seq:str, geneModel:str) -> Tuple[bool, str]:
	# Purpose:
	#      Validate that the CDS features in the gff produce a valid coding sequence
	# Inputs:
	#      gff: str, the gff string
	#      seq: str, the sequence
	#      geneModel: str, the gene model name
	# Outputs:
	#      (bool, str) - (valid, error message)

	try:
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
	except Exception as e:
		return (False, f"Error validating {geneModel}, skipping: {e}")
	return (True, None)

def segmentSequence(seq, piece_size = 6144):
	# Segment the sequence into evenly sized chunks smaller than 4092bp (encoder max length of 1024 tokens)
	seqs = [seq[i:min(i+piece_size, len(seq))] for i in range(0, len(seq), piece_size)]
	#windowed_seqs = []
	#for i in range(len(seqs)):
	#	windowed_seqs.append(seqs[i])
	#	if i+1 < len(seqs):
	#		windowed_seqs.append(seqs[i][piece_size//2:len(seqs[i])] + seqs[i+1][0:piece_size//2 ])
	return seqs #windowed_seqs

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

def mask_sequences(seqs, num_masked_nucleotides=921, mask_token=3):
	"""
	Masks groups of 3 contiguous nucleotides in each sequence of a batch and returns a boolean mask.
	
	Args:
		seqs (torch.Tensor): 2D tensor of shape (batch_size, seq_len) representing the batch of sequences.
							Each sequence should have a length divisible by 3.
		num_masked_nucleotides (int): Desired number of nucleotides to mask per sequence.
									Will be rounded down to the nearest multiple of 3.
		mask_token (int): The token to use for masking.
	
	Returns:
		tuple: (masked_seqs, mask_bool) where:
			- masked_seqs (torch.Tensor): The sequences after masking.
			- mask_bool (torch.BoolTensor): Boolean tensor of the same shape as seqs,
			with True at masked positions.
	"""
	batch_size, seq_len = seqs.size()
	if seq_len % 3 != 0:
		raise ValueError("Sequence length must be divisible by 3.")
	
	# Calculate number of 3-mer groups per sequence.
	num_3mers = seq_len // 3
	
	# Determine the number of groups to mask per sequence.
	num_groups = num_masked_nucleotides // 3 
	
	# Create copies for the outputs.
	masked_seqs = seqs.clone()
	mask_bool = torch.zeros_like(seqs, dtype=torch.bool)
	
	# Iterate over each sequence in the batch.
	# Randomly select non-overlapping group indices for the sequence.
	for i in range(batch_size):
		group_indices = random.sample(range(num_3mers), num_groups)
		for g in group_indices:
			start = g * 3
			masked_seqs[i, start:start+3] = mask_token
			mask_bool[i, start:start+3] = True
			
	return masked_seqs, mask_bool

