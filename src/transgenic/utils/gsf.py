import re, sys
from typing import List

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

def gffString2GFF3(gff:str, chr:str, region_start:int, extra_attributes:str) -> List[str]:
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
	geneModel = [f"{chr}\ttransgenic\tgene\t{geneStart}\t{geneEnd-1}\t.\t{geneStrand}\t.\tID={id};{extra_attributes}"]
	
	# Add mRNA models
	for i,transcript in enumerate(transcripts):
		geneModel.append(f"{chr}\ttransgenic\tmRNA\t{transcriptBounds[i][0]}\t{transcriptBounds[i][1]-1}\t.\t{geneStrand}\t.\tID={id}.t{i+1};Parent={id};{extra_attributes}")
		transcript = transcript.split("|")
		for featureID in transcript:
			if featureID == "":
				continue
			featureModel = features[featureID]
			featureType = re.sub(r'\d+', '', featureID)
			featureNum = re.sub(r'\D+', '', featureID)
			geneModel.append(f"{chr}\ttransgenic\t{featureType}\t{int(featureModel[0])+region_start}\t{int(featureModel[2])+region_start-1}\t.\t{geneStrand}\t{phaseLookup[featureModel[4]]}\tID={id}.t{i+1}.{featureType}{featureNum};Parent={id}.t{i+1};{extra_attributes}")
	
	return geneModel

