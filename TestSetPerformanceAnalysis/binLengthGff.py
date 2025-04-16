import os, sys, math

file = sys.argv[1]
out_prefix = sys.argv[2]
outdir = sys.argv[3]
bin_size = sys.argv[4]
#file = "TestSetPerformanceAnalysis/temp/TAIR10_hyenaTest_prediction_noPost.gff3"
#out_prefix = "TAIR10_hyenaTest_prediction_noPost"
#outdir = "binByLength"

os.makedirs(outdir, exist_ok=True)
outfile = None

with open(file, "r") as infile:
	for line in infile:
		if line[0] == "#":
			continue
		
		line = line.split("\t")
		
		if line[2] == "gene":
			if outfile:
				outfile.close()
			
			gene_length = int(line[4]) - int(line[3])
			bin = math.ceil(gene_length / int(bin_size)) * int(bin_size)
			outfile = open(f"{outdir}/{out_prefix}-{bin}.gff3", "a")

		outfile.write("\t".join(line))

outfile.close()
