import sys, re, os

file = sys.argv[1]
out_prefix = sys.argv[2]
pred_file = sys.argv[3]
pred_prefix = sys.argv[4]
#file = "TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15_Hyena_SegmentFocalDice_E13-21/TAIR10_hyenaTest_labels.gff3"
#out_prefix = "TAIR10_hyenaTest_labels"
#pred_file = "TestSetPerformanceAnalysis/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15_Hyena_SegmentFocalDice_E13-21/TAIR10_hyenaTest_prediction_post.gff3"
#pred_prefix = "TAIR10_hyenaTest_prediction_post"

gene_models = {}

with open(file, 'r') as infile:
	gene_cds_count = 0
	mrna_cds_count = 0
	gene_model = ''
	for line in infile:
		typ = line.split("\t")[2]
		if (typ == "gene"):
			if (gene_cds_count != 0) and (mrna_cds_count != 0):
				gene_cds_count = max(gene_cds_count, mrna_cds_count)
				gene_models[GM] = gene_cds_count
				with open(f"{out_prefix}.{str(gene_cds_count)}.gff3", 'a') as outfile:
					outfile.write(gene_model)
			
			gene_cds_count = 0
			mrna_cds_count = 0
			gene_model = line
			GM = re.search(r"GM=([A-Za-z0-9\.\-]+);{0,1}", line.split("\t")[8].strip())[1]
			continue
		
		if typ == "mRNA":
			gene_cds_count = max(gene_cds_count, mrna_cds_count)
			mrna_cds_count = 0

		if typ == "CDS":
			mrna_cds_count += 1
		
		gene_model = gene_model + line

# Write final gene model
gene_cds_count = max(gene_cds_count, mrna_cds_count)
gene_models[GM] = gene_cds_count
with open(f"{out_prefix}.{str(gene_cds_count)}.gff3", 'a') as outfile:
	outfile.write(gene_model)


for gm in gene_models:
	os.system(f"grep {gm} {pred_file} >> {pred_prefix}.{gene_models[gm]}.gff3")

for count in list(set(gene_models.values())):
	os.system(f"gffcompare -r {out_prefix}.{count}.gff3 {pred_prefix}.{count}.gff3 -T -o {pred_prefix}_{count}")