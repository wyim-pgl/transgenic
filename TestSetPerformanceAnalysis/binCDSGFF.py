import sys, re, os

file = sys.argv[1]
out_prefix = sys.argv[2]
outdir = sys.argv[3]
#file = "TestSetPerformanceAnalysis/temp/TAIR10_hyenaTest_prediction_noPost.gff3"
#out_prefix = "TAIR10_hyenaTest_prediction_noPost"
#outdir = "binByCDS"

os.makedirs(outdir, exist_ok=True)
outfile = None

gene_models = {}
cds_count = 0

with open(file, 'r') as infile:
	mrna_model = ''
	gene_model = ''
	mrna_models = []
	for line in infile:
		typ = line.split("\t")[2]
		if (typ == "gene"):
			mrna_models.append(mrna_model)
			if cds_count > 0:
				with open(f"{outdir}/{out_prefix}-{cds_count}.gff3", 'a') as outfile:
					outfile.write(gene_model)
					for models in mrna_models:
						outfile.write(mrna_model)
			
			mrna_models = []
			mrna_model = ''
			gene_model = line
			cds_count = 0
			continue
		
		elif typ == "mRNA":
			if mrna_model:
				mrna_models.append(mrna_model)
			mrna_model = line
		elif typ == "CDS":
			cds_count += 1
			mrna_model += line
		else:
			mrna_model += line

# Write final gene model
mrna_models.append(mrna_model)
if cds_count > 0:
	with open(f"{outdir}/{out_prefix}-{cds_count}.gff3", 'a') as outfile:
		outfile.write(gene_model)
		for models in mrna_models:
			outfile.write(mrna_model)
