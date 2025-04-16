import sys, re, os

file = sys.argv[1]
out_prefix = sys.argv[2]
#file = "TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15_Hyena_SegmentFocalDice_E13-21/TAIR10_hyenaTest_labels.gff3"
#out_prefix = "TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15_Hyena_SegmentFocalDice_E13-21/TAIR10_hyenaTest_labels"

try:
	os.system(f"rm {out_prefix}.ASonly.gff3")
except Exception as e:
	print(e)

gene_models = {}

with open(file, 'r') as infile:
	mrna_model = ''
	gene_model = ''
	mrna_models = []
	for line in infile:
		typ = line.split("\t")[2]
		if (typ == "gene"):
			mrna_models.append(mrna_model)
			if len(mrna_models) > 1:
				with open(f"{out_prefix}.ASonly.gff3", 'a') as outfile:
					outfile.write(gene_model)
					for models in mrna_models[1:]:
						outfile.write(mrna_model)
			
			mrna_models = []
			mrna_model = ''
			gene_model = line
			continue
		
		elif typ == "mRNA":
			if mrna_model:
				mrna_models.append(mrna_model)
			mrna_model = line

		else:
			mrna_model += line

# Write final gene model
mrna_models.append(mrna_model)
if len(mrna_models) > 1:
	with open(f"{out_prefix}.ASonly.gff3", 'a') as outfile:
		outfile.write(gene_model)
		for models in mrna_models[1:]:
			outfile.write(mrna_model)
