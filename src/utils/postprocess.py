import sys, re
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_dilation, binary_erosion, label

from .sequence import reverseComplement
from .gsf import reverseComplement_gffString

def classifyGeneSignal(
		probs, 
		threshold=0.65,
		sigma = 100, 
		#rising_threshold = 0.1, 
		#falling_threshold = -0.1, 
		window_length=6144):
	
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


class PredictionProcessor():
	def __init__(self, gff:str, sequence:str, probs):
		self.gff = gff
		self.sequence = sequence
		self.probs = probs
		self.stopCodons = ["TAA", "TAG", "TGA"]
		self.spliceDonors = ["AGGT", "GGT","GT","AGGC", "AGAT","GGC","GAT"]
		self.spliceCanonical = {"GT":"AG", "GC":"AG","AT":"AC"}
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
		
		self.feature_flag_dict = {}
		for i, feature in enumerate(self.features):
			self.feature_flag_dict[feature[1]] = ""

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
	
		# Get pairs for splice junction validation
		self.cds_pairs = []
		self.utr_pairs=[]
		for transcript in self.transcripts:
			for i, typ in enumerate(transcript):
				if i == 0:
					continue
				if 'CDS' in typ and (transcript[i-1], typ) not in self.cds_pairs and 'CD' in transcript[i-1]:
					self.cds_pairs.append((transcript[i-1], typ))
				if 'five_prime_UTR' in typ and (transcript[i-1], typ) not in self.utr_pairs and 'five_prime_UTR' in transcript[i-1]:
					self.utr_pairs.append((transcript[i-1], typ))
				if 'three_prime_UTR' in typ and (transcript[i-1], typ) not in self.utr_pairs and 'three_prime_UTR' in transcript[i-1]:
					self.utr_pairs.append((transcript[i-1], typ))
		
		# Identify features in the segmentation probabilities
		if self.probs != None:
			self.seqFeatures = self.findSeqFeatures()
			self.usedSeqFeatures = {"startCodons": [],"stopCodons": [],"donors": [],"acceptors": []}

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
		if self.probs != None:
			self.probs = torch.flip(self.probs, dims=[0])
	
	def findLargestGeneRegion(self, lst, threshold=0.4):
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
		
		# Find the central gene region
		#segments = classifyGeneSignal(self.probs[:, 0], threshold=geneThresh, sigma=200)
		#prev_len = 0
		#for segment in segments[0]:
		#	if int(segment["length"]) > prev_len:
		#		geneStart = segment["start"]
		#		geneEnd = segment["end"]
		#		prev_len = int(segment["length"])
		
		#geneStart, geneEnd = self.findLargestGeneRegion(self.probs[:,0], threshold=geneThresh)

		# Find all valid start codons over the threshold
		# Rank by probability score 
		means = []
		lst = self.probs[:,1].tolist() #self.probs[geneStart:geneEnd,1].tolist()
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0) + (lst[i + 2] if i + 2 < len(lst) else 0)
			means.append(sum_value/3)
		startCodons = [i for i, x in enumerate(means) if x > startThresh]
		startCodons = [i for i in startCodons if self.sequence[i:i+3] == "ATG"]
		startCodons = sorted(startCodons, key=lambda i: means[i], reverse=True)

		# Find all high probability stop codons in the gene region
		means = []
		lst = self.probs[:,8].tolist()
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0) + (lst[i + 2] if i + 2 < len(lst) else 0)
			means.append(sum_value/3)
		stopCodons = [i for i, x in enumerate(means) if x > startThresh]
		stopCodons = [i+3 for i in stopCodons if self.sequence[i:i+3] in self.stopCodons]
		stopCodons = sorted(stopCodons, key=lambda i: means[i-3], reverse=True)
		
		# Pick out all splice donors and acceptors
		# limit to canonical splice junctions
		means = []
		lst = self.probs[:,4].tolist() 
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0)
			means.append(sum_value/2)
		donors = [i for i, x in enumerate(means) if (x > junctionThresh) and (self.sequence[i:i+2] in self.spliceCanonical.keys())]
		donors = sorted(donors, key=lambda i: means[i], reverse=True)

		means = []
		lst = self.probs[:,5].tolist() 
		for i in range(len(lst)):
			sum_value = lst[i] + (lst[i + 1] if i + 1 < len(lst) else 0)
			means.append(sum_value/2)
		acceptors = [i+2 for i, x in enumerate(means) if x > junctionThresh and (self.sequence[i:i+2] in self.spliceCanonical.values())]
		acceptors = sorted(acceptors, key=lambda i: means[i-2], reverse=True)

		return {
			"startCodons": startCodons,
			"stopCodons": stopCodons,
			"donors": donors,
			"acceptors": acceptors
		}

	# Use a segmentation model start codon if it is reasonably close to the prediction
	# Otherwise, look for an 'ATG' within the buffer window 
	# Reset 5'-UTR end to match start codon
	def checkStartCodons(self, buff=50, usingSeqFeature=False):
		
		if usingSeqFeature:
			startCodons = self.seqFeatures["startCodons"]

		# Identify start codons within the buffer region
		# Pick the highest probability (lowest index) start which doesn't introduce in-frame stop codons
		# Set phase to 'A' (Always 'A' for the first cds)
		for i, cds in enumerate(self.start_cds):
			self.features[self.feature_index_dict[cds]][4] = 'A'
			start = int(self.features[self.feature_index_dict[cds]][0])
			if usingSeqFeature:
				if len(startCodons) > 0:
					new_starts = [i for i in startCodons if i in range(start-buff,start+buff)]
					found=False
					if len(new_starts)>0:
						# If only one CDS, don't constrain by stop codons
						if self.start_cds == self.end_cds:
							new_start = new_starts[0]
							found = True
						else:
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
							self.features[self.feature_index_dict[cds]][0] = str(new_starts[0])
							if new_starts[0] not in self.usedSeqFeatures["startCodons"]:
								self.usedSeqFeatures["startCodons"].append(new_starts[0])
							usingSeqFeature = False
					else:
						usingSeqFeature = False
				else:
					usingSeqFeature = False
			
			# If no start codons from segmentation, look for start codons in the buffer window
			# Pick the closest start codon to the predicted start
			if not usingSeqFeature:
				self.feature_flag_dict[cds] += "start"
				buffer = self.sequence[start-buff:start+buff]
				startCodons = [m.start() for m in re.finditer("ATG", buffer)]
				if startCodons:
					pick = startCodons[np.argmin([np.abs(loc-buff) for loc in startCodons])]
					new_start = start + pick-buff
					self.features[self.feature_index_dict[cds]][0] = str(new_start)
		
		# Update utr starts
		for cds, utr in self.start_pairs:
			if utr:
				self.features[self.feature_index_dict[utr]][2] = self.features[self.feature_index_dict[cds]][0]

	# Check end_cds for stop codons, if missing search sequence for new stop codon
	# Identify all stop codons in the buffer window and select the closest stop codon which maintains phase
	def checkStopCodons(self, buff=50, usingSeqFeature=False):
		if usingSeqFeature:
			stopCodons = self.seqFeatures["stopCodons"]
		
		# Identify stop codons within the buffer region
		# Pick the highest probability (lowest index) stop which maintains a valid frame (multiple of 3)
		for i, cds in enumerate(self.end_cds):
			stop = int(self.features[self.feature_index_dict[cds]][2])
			if usingSeqFeature:
				if len(stopCodons) > 0:
					new_stops = [i for i in stopCodons if i in range(stop-buff, stop+buff)]
					found = False
					if len(new_stops)>0:
						# Only matching frame to the first transcript
						working_transcript_index = [i for i,t in enumerate(self.transcripts) if cds in self.transcripts[i]][0]
						upstream_cds = [i for i in self.transcripts[working_transcript_index] if "CDS" in i]
						upstream = upstream_cds[:-1]
						length = 0
						for feat in upstream:
							length += int(self.features[self.feature_index_dict[feat]][2]) - int(self.features[self.feature_index_dict[feat]][0]) 
						
						for new_stop in new_stops:
							full_length = length + (new_stop - int(self.features[self.feature_index_dict[cds]][0]))
							if full_length % 3 == 0:
								found = True
								break

						if found:
							self.features[self.feature_index_dict[cds]][2] = str(new_stop)
							if new_stop not in self.usedSeqFeatures["stopCodons"]:
								self.usedSeqFeatures["stopCodons"].append(new_stop)
						else:
							self.features[self.feature_index_dict[cds]][2] = str(new_stops[0])
							if new_stops[0] not in self.usedSeqFeatures["stopCodons"]:
								self.usedSeqFeatures["stopCodons"].append(new_stops[0])
							usingSeqFeature = False
					else:
						usingSeqFeature = False
				else:
					usingSeqFeature = False

			# If no stop codons from segmentation, look for stop codons in the buffer window
			# Pick the closest stop codon to the predicted stop
			if not usingSeqFeature:
				self.feature_flag_dict[cds] += "stop"
				buffer = self.sequence[stop-buff:stop+buff]
				stopCodons = [m.start() + 3 for m in re.finditer("(TAA)|(TAG)|(TGA)", buffer)]
				if stopCodons:
					pick = stopCodons[np.argmin([np.abs(loc-buff) for loc in stopCodons])]
					new_stop = stop + pick-buff
					self.features[self.feature_index_dict[cds]][2] = str(new_stop)

		# Amend utr starts
		for cds, utr in self.end_pairs:
			if utr:
				self.features[self.feature_index_dict[utr]][0] = self.features[self.feature_index_dict[cds]][2]
	
	# Used to check if a subsequence makes internal stop codons
	def containsStopCodon(self, seq, phase):
		phaseLookup = {"A":0,"B":1,"C":2,".":0}
		seq = [seq[:phaseLookup[phase]]] + [seq[i:min(i+3, len(seq)-phaseLookup[phase])] for i in range(phaseLookup[phase], len(seq), 3)]
		return "TAG" in seq or "TAA" in seq or "TGA" in seq


	def checkProbableSpliceJunctions(self, pairs, buff=100, cds=True):
		phase_lookup = {"A":0, "B":1,"C":2}
		index_lookup = {0:"A", 1:"B",2:"C"}

		donors = self.seqFeatures["donors"]
		acceptors = self.seqFeatures["acceptors"]

		for donor, acceptor in pairs:
			donor_start = int(self.features[self.feature_index_dict[donor]][0])
			donor_end = int(self.features[self.feature_index_dict[donor]][2])
			donor_phase = self.features[self.feature_index_dict[donor]][4]
			acceptor_start = int(self.features[self.feature_index_dict[acceptor]][0])

			# Identify splice donor sites within the buffer region
			# Pick the highest probability donor (lowest index) which doesn't introduce an inframe stop codon
			# Update acceptor phase based on selection
			donor_type = None
			if len(donors) > 0:
				new_donors = [i for i in donors if i in range(donor_end-buff,donor_end+buff)]
				found = False
				if (len(new_donors)>0) and cds: # Working on CDS feature
					for new_donor in new_donors:
						containsStop = self.containsStopCodon(self.sequence[donor_start:new_donor], donor_phase) # Maybe just use whole seq instead of tracking phase?
						if not containsStop:
							found = True
							break
					
					if found:
						self.features[self.feature_index_dict[donor]][2] = str(new_donor)
						donor_type  = self.sequence[new_donor:new_donor+2]
						acceptor_phase = index_lookup[(phase_lookup[donor_phase] + (3-((new_donor-donor_start)%3)))%3]
						self.features[self.feature_index_dict[acceptor]][4] = acceptor_phase
						if new_donor not in self.usedSeqFeatures["donors"]:
							self.usedSeqFeatures["donors"].append(new_donor)
					else:
						self.features[self.feature_index_dict[donor]][2] = str(new_donors[0])
						donor_type  = self.sequence[new_donors[0]:new_donors[0]+2]
						acceptor_phase = index_lookup[(phase_lookup[donor_phase] + ((new_donors[0]-donor_start)%3))%3]
						self.features[self.feature_index_dict[acceptor]][4] = acceptor_phase
						if new_donors[0] not in self.usedSeqFeatures["donors"]:
							self.usedSeqFeatures["donors"].append(new_donors[0])
				
				elif (len(new_donors)>0): # Working on UTR feature
					self.features[self.feature_index_dict[donor]][2] = str(new_donors[0])
					donor_type  = self.sequence[new_donors[0]:new_donors[0]+2]
					if new_donors[0] not in self.usedSeqFeatures["donors"]:
							self.usedSeqFeatures["donors"].append(new_donors[0])
				
				else: # If no donor site from segmentation, look for nearest donor in the buffer window
					self.feature_flag_dict[donor] += "donor"
					buffer = self.sequence[donor_end-buff:donor_end+buff]
					donors = [m.start() for m in re.finditer("GT", buffer)]
					if donors:
						pick = donors[np.argmin([np.abs(loc-buff) for loc in donors])]
						new_donor = donor_end + pick-buff
						self.features[self.feature_index_dict[donor]][2] = str(new_donor)
						donor_type  = self.sequence[new_donor:new_donor+2]
						if cds:
							acceptor_phase = index_lookup[(phase_lookup[donor_phase] + ((new_donor-donor_start)%3))%3]
							self.features[self.feature_index_dict[acceptor]][4] = acceptor_phase
			
			# Identify splice acceptor sites within the buffer region
			# Pick the highest probability acceptor (lowest index) which matches the canonical splice site pairing of the donor site
			if len(acceptors) > 0:
				new_acceptors = [i for i in acceptors if i in range(acceptor_start-buff,acceptor_start+buff)]
				found = False
				if len(new_acceptors)>0:
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
						self.features[self.feature_index_dict[acceptor]][0] = str(new_acceptors[0])
						if new_acceptors[0] not in self.usedSeqFeatures["acceptors"]:
							self.usedSeqFeatures["acceptors"].append(new_acceptors[0])
				else:
					self.feature_flag_dict[acceptor] += "acceptor"
					buffer = self.sequence[acceptor_start-buff:acceptor_start+buff]
					acceptors = [m.start() for m in re.finditer("AG", buffer)]
					if acceptors:
						pick = acceptors[np.argmin([np.abs(loc-buff) for loc in acceptors])]
						new_acceptor = acceptor_start + pick-buff +2
						self.features[self.feature_index_dict[acceptor]][0] = str(new_acceptor)
			

	def checkCanonicalSpliceJunctions(self, buff=50):
		# Use canonical splice junctions to adjust feature cordinates.
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

			#if closestAcceptor:
			#	self.features[self.feature_index_dict[acceptor]][0] = str(int(acceptor_start + closestAcceptor))
			#	if self.containsStopCodon((donor,acceptor)):
			#		if len(self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]]) > 1:
			#			while self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]]:
			#				self.features[self.feature_index_dict[acceptor]][0] = str(int(acceptor_start + self.sjLog[acceptor]["acceptor"][self.sjLog[acceptor]["acceptor"]["chosen"]].pop(0)))
			#				if not self.containsStopCodon((donor,acceptor)):
			#					break

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
	def checkSpliceJunctions(self, pairs, buff=50, cds=True):
		self.checkProbableSpliceJunctions(pairs, buff=buff, cds=cds)
		#self.checkCanonicalSpliceJunctions(buff=buff)
	
	def stitchGFF(self, originalStrand=True):
		updatedGFF = f"{';'.join(['|'.join(feature) for feature in self.features])}>{';'.join(['|'.join(transcript) for transcript in self.transcripts])}"
		if originalStrand and (self.strand == "-"):
			updatedGFF = reverseComplement_gffString(updatedGFF, len(self.sequence))
		return updatedGFF
	
	def analyzeUnused(self):
		numcds = len(self.cds_pairs) + 1

		# Identify missed alternate starts 
		unusedStart = list(set(self.seqFeatures["startCodons"])-set(self.usedSeqFeatures["startCodons"]))
		extraStarts = []
		for newStart in unusedStart:
			candidates = [i for i in self.seqFeatures["donors"] if i > newStart]
			if candidates:
				newStart_end = candidates[0]
				extraStarts.append((newStart,newStart_end))
				if newStart_end not in self.usedSeqFeatures["donors"]:
					self.usedSeqFeatures["donors"].append(newStart_end)

		# Identify missed Acceptor junctions
		unusedAcceptors = list(set(self.seqFeatures["acceptors"])-set(self.usedSeqFeatures["acceptors"]))
		extraAs = []
		for newA in unusedAcceptors:
			candidates = [i for i in self.seqFeatures["donors"] if i > newA]
			if candidates:
				newA_end = candidates[0]
				extraAs.append((newA, newA_end))
				if newA_end not in self.usedSeqFeatures["donors"]:
					self.usedSeqFeatures["donors"].append(newA_end)

		# Identify missed Donor junctions
		unusedDonors = list(set(self.seqFeatures["donors"])-set(self.usedSeqFeatures["donors"]))
		extraDs = []
		for newD in unusedDonors:
			candidates = [i for i in self.seqFeatures["acceptors"] if i < newD]
			if candidates:
				newD_start = [i for i in self.seqFeatures["acceptors"] if i < newD][-1]
				extraDs.append((newD_start, newD))
				if newD_start not in self.usedSeqFeatures["acceptors"]:
					self.usedSeqFeatures["acceptors"].append(newD_start)

		# Identify unused Stop codons
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