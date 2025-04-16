import sys
import zlib
import duckdb
import torch
import pandas as pd
from tqdm import tqdm

from utils.sequence import loadGenome, reverseComplement, validateCDS
from utils.gsf import reverseComplement_gffString

#TODO: Add functionality to make a single input from a genome sequence and a single gene model

def genome2GSFDataset(genome, gff3, db, maxLen=49152, addExtra=0, staticSize=6144, addRC=False, addRCIsoOnly=False, clean=False):
	# Purpose: 
	#   Read a genome assembly and gff3 annotation file into a 
	#   duckdb database for training or inference. For each gene 
	#   model, the corresponding nucleotide sequence is extracted
	#   from the genome and the target annotation string is created.
	#   The function may be called multiple times to append add 
	#   multiple genomes to the database
	# Inputs:
	#   genome: path to a fasta file containing the genome assembly
	#   gff3:		path to a gff3 file containing gene annotations
	#   db:			path to a duckdb database file (will be created if it does not exist)
	#   maxLen:		Maximum size of gene model sequence to include in the database (larger gene models are skipped)
	#   addExtra:	Max size of random buffer to add to the gene model sequence (used to capture UTR start and end during training)
	#   staticSize:	Extracted sequences will be this size (SegmentNT performs best with static sizes)
	#   addRC:		Add reverse complement of gene model sequence to the database (Used to augment training data)
	#   addRCIsoOnly: When adding RC seqs, only add gene models with alternative splicing
	#   clean: If true only sequences will be added that have start and stop codons and are a multiple of 3
	# Outputs:
	#   A duckdb database used to load data into the model
	# TODO:
	#   - Add support for bed files
	#   - Add inference mode with just target sequences for prediction

	# Load genome into a dictionary
	genome_dict = loadGenome(genome)

	# Connect to isoform database and create geneList table
	con = duckdb.connect(db)
	con.sql("SET enable_progress_bar = false;")
	con.sql(
		"CREATE TABLE IF NOT EXISTS geneList ("
			"geneModel VARCHAR, "
			"start INT, "
			"fin INT, "
			"strand VARCHAR, "
			"chromosome VARCHAR, "
			"sequence VARCHAR, "
			"gff VARCHAR, "
			"static_fpb INT, "
			"static_tpb INT, "
			"five_prime_buf INT, "
			"three_prime_buf INT, "
			"rn INT PRIMARY KEY)")
	con.sql("CREATE SEQUENCE IF NOT EXISTS row_id START 1;")

	
	print(f"\nProcessing {gff3}...")
	num_lines = sum(1 for line in open(gff3, "r"))
	with open(gff3, 'r') as gff3file:
		region_start = None
		feature_list = ''
		mRNA_list = ''
		skipGene = False
		five_ps = {}
		three_ps = {}
		cds_num = {}
		
		for line in tqdm(gff3file, total=num_lines):
			if line.startswith('#') | (line == '\n'):
				continue
			else:
				line = line.strip().split('\t')
				chr, _, typ, start, fin, _, strand, phase, attributes = line
				if typ == 'gene':
					# Add previous gene model to the database
					if (region_start != None) & (not skipGene):
						mRNA_list = mRNA_list[1:-1]
						gff = f"{feature_list[:-1]}>{mRNA_list}"
						valid = True
						if clean:
							valid, error = validateCDS(gff, sequence, geneModel)
							if not valid:
								print(f"{error}. Skipping", file=sys.stderr)
								region_start = 0
								region_end = 0
								geneModel = None
								feature_list = ''
								mRNA_list = ''
								five_ps = {}
								three_ps = {}
								cds_num = {}
					
						try:
							if valid:
								con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf) VALUES (nextval('row_id'), '{geneModel}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence}', '{gff}', '{five_prime_buffer}', '{three_prime_buffer}','{int(torch.randint(addExtra, (1,)))}', '{int(torch.randint(addExtra, (1,)))}')")
						except Exception as e:
							print(f"{geneModel=}")
							print(f"{sequence=}")
							print(f"{gff=}")
							print(f"Error inserting {geneModel} into database: {e}", file=sys.stderr)
							con.close()
							sys.exit()
						
						if addRC:
							gff_rc = reverseComplement_gffString(gff, region_end - region_start)
							if addRCIsoOnly:
								if ';' in mRNA_list:
									try:
										if valid:
											con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf) VALUES (nextval('row_id'), '{geneModel + "-rc"}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence_rc}', '{gff_rc}', '{five_prime_buffer}', '{three_prime_buffer}', '{int(torch.randint(addExtra, (1,)))}', '{int(torch.randint(addExtra, (1,)))}')")
									except Exception as e:
										print(f"{geneModel=}-rc")
										print(f"{sequence_rc=}")
										print(f"{gff_rc=}")
										print(f"Error inserting {geneModel+"-rc"} into database: {e}", file=sys.stderr)
										con.close()
										sys.exit()
							else:
								try:
									if valid:
										con.sql(f"INSERT INTO geneList (rn, geneModel, start, fin, strand, chromosome, sequence, gff, static_fpb, static_tpb, five_prime_buf, three_prime_buf) VALUES (nextval('row_id'), '{geneModel + "-rc"}', {region_start}, {region_end}, '{strand}', '{chr}', '{sequence_rc}', '{gff_rc}', '{five_prime_buffer}', '{three_prime_buffer}', {int(torch.randint(addExtra, (1,)))}', '{int(torch.randint(addExtra, (1,)))}')")
								except Exception as e:
									print(f"{geneModel=}-rc")
									print(f"{sequence_rc=}")
									print(f"{gff_rc=}")
									print(f"Error inserting {geneModel+"-rc"} into database: {e}", file=sys.stderr)
									con.close()
									sys.exit()
								
						geneModel = None
						region_start = 0
						region_end = 0
						feature_list = ''
						mRNA_list = ''
						five_ps = {}
						three_ps = {}
						cds_num = {}
					
					# Construct current gene model
					gene_length = int(fin) - int(start) + 1
					if gene_length <= staticSize:
						additional_sequence = staticSize - (gene_length % staticSize)
					else:
						additional_sequence = ((gene_length // staticSize) + 1)*staticSize - gene_length
					three_prime_buffer = additional_sequence//2
					if not (additional_sequence % 2):
						five_prime_buffer = additional_sequence//2
					else:
						five_prime_buffer = additional_sequence//2 + 1
						

					skipGene = False
					geneModel = attributes.split(';')[0].split('=')[1]
					if (int(start) - five_prime_buffer - 1) < 0:
						five_prime_buffer = int(start) - 1
					region_start = int(start) - five_prime_buffer - 1 # Gffs are 1-indexed

					if (five_prime_buffer + gene_length + three_prime_buffer) <= staticSize:
						three_prime_buffer = staticSize - (five_prime_buffer + gene_length)
					region_end = int(fin) + three_prime_buffer        # End not subtracted because python slicing is exclusive
					
					# 49,152bp corresponds to 8,192 6-mer tokens (Max input)
	 				# 25,002 -> 4,167 6-mer tokens (Max output)
					if (region_end - region_start > maxLen):
						print(f"Skipping {geneModel} because gene length > {maxLen}", file=sys.stderr)
						region_start = None
						region_end = None
						geneModel = None
						skipGene = True
						continue
					if ((region_end - region_start)% staticSize) != 0:
						print(f"Warning: {geneModel} not a multiple of {staticSize=}", file=sys.stderr)

					# Get forward strand sequence
					sequence = genome_dict[chr][region_start:region_end]
					if addRC:
						sequence_rc = reverseComplement(sequence)
				
				elif skipGene:
					continue
				
				elif typ == 'lncRNA':
					skipGene = True
					continue

				elif typ == 'mRNA':
					mRNA_list = mRNA_list[:-1] + ";"

				# Build gff string - start and end coordinates are relative to the gene model sense strand (inclusive on start, exclusive on end)
				elif typ == 'CDS': 
					start = (int(start) - 1) - region_start 
					end = int(fin) - region_start
					if phase == '0':
						phase = 'A'
					elif phase == '1':
						phase = 'B'
					elif phase == '2':
						phase = 'C'
					else:
						phase = '.'

					try:
						num = cds_num[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(cds_num) + 1)
						cds_num[f"{start}-{end}-{strand}-{phase}"] = num
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"
					
					mRNA_list += f"{typ+num}|"
				
				elif typ == 'five_prime_UTR':
					start = (int(start) - 1) - region_start
					end = int(fin) - region_start
					try:
						num = five_ps[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(five_ps) + 1)
						five_ps[f"{start}-{end}-{strand}-{phase}"] = num
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"
					
					mRNA_list += f"{typ+num}|"
					
				elif typ == 'three_prime_UTR':
					start = (int(start) - 1) - region_start
					end = int(fin) - region_start
					try:
						num = three_ps[f"{start}-{end}-{strand}-{phase}"]
					except:
						num = str(len(three_ps) + 1)
						three_ps[f"{start}-{end}-{strand}-{phase}"] = num
						feature_list += f"{start}|{typ+num}|{end}|{strand}|{phase};"
					
					mRNA_list += f"{typ+num}|"

	con.close()

# Note: Legacy implementation (very slow during dataloading). Recommended to use 'genome2PreprocessedSegmentationDataset' instead
def genome2SegmentationDataset(genome_file, gff_file, organism, db):

	# load gff into database with organism name
	table = organism
	gff3 = pd.read_csv(gff_file, sep='\t', header=None, comment='#')
	gff3.columns = ['chromosome', 'source', 'feature', 'start', 'fin', 'score', 'strand', 'frame', 'attribute']
	gff3['organism'] = organism
	with duckdb.connect(db) as con:
		con.sql(
			f'CREATE TABLE IF NOT EXISTS {table}_gff ('
			'chromosome VARCHAR, '
			'source VARCHAR, '
			'feature VARCHAR, '
			'start INT, '
			'fin INT, '
			'score VARCHAR, '
			'strand VARCHAR, '
			'frame VARCHAR, '
			'attribute VARCHAR, '
			'organism VARCHAR)')

		con.sql(
			f'INSERT INTO {table}_gff '
			'SELECT * '
			'FROM gff3; '
		)

	# load chromosome sequences into database with organism name
	genome_dict = loadGenome(genome_file)
	genome_df = pd.DataFrame(genome_dict.items(), columns=['chromosome', 'sequence'])
	genome_df['organism'] = organism
	genome_df['length'] = genome_df['sequence'].apply(len)
	with duckdb.connect(db) as con:
		con.sql(
			f"CREATE TABLE IF NOT EXISTS {table}_genome ("
			'chromosome VARCHAR, '
			'sequence VARCHAR, '
			'organism VARCHAR, '
			'length INT)')

		con.sql(
			f'INSERT INTO {table}_genome '
			'SELECT * '
			'FROM genome_df; '
		)


def genome2PreprocessedSegmentationDt(db, genome, gff, table, window_size, step_size):
	#dataset = segmentationDataset(table, window_size, step_size, olddb, encoder_model=encoder_model, preprocess=True)
	#dl = makeDataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=False, sampler=None, num_workers=workers, collate_fn=preprocessing_collate_fn)
	
	classes = ['protein_coding_gene', 
				'lncRNA', 
				'exon', 
				'intron', 
				'splice_donor', 
				'splice_acceptor', 
				'5UTR', 
				'3UTR', 
				'CTCF-bound', 
				'polyA_signal', 
				'enhancer_Tissue_specific', 
				'enhancer_Tissue_invariant', 
				'promoter_Tissue_specific', 
				'promoter_Tissue_invariant']

	gffClassMap = {'gene': 'protein_coding_gene',  
					'exon': 'exon', 
					'intron': 'intron',
					'five_prime_cis_splice_site': 'splice_donor', 
					'three_prime_cis_splice_site': 'splice_acceptor', 
					'five_prime_UTR': '5UTR', 
					'three_prime_UTR': '3UTR'}
	
	# Read genome sequence and gff into memory
	genome = loadGenome(genome)
	gff = pd.read_csv(gff, sep='\t', header=None, comment='#')
	gff.columns = ['chromosome', 'source', 'feature', 'start', 'fin', 'score', 'strand', 'frame', 'attributes']

	
	for chr in genome:
		# Encode each chromosome
		sequence = genome[chr]
		chr_gff = gff[gff['chromosome'] == chr].reset_index(drop=True)
		class_tensor = torch.zeros((len(sequence), len(classes)), dtype=torch.float32)
		
		print(f"Processing chr {chr}...", file=sys.stderr)
		skip = False
		for i, row in tqdm(chr_gff.iterrows()):
			start = row['start']
			end = row['fin']
			feature = row['feature']
			nextfeature = chr_gff.loc[i+1, 'feature'] if i+1 < len(chr_gff) else None
			if feature == 'gene':
				skip = False
			if nextfeature == 'lncRNA':
				skip = True
			if not skip:			
				if feature in gffClassMap:
					class_idx = classes.index(gffClassMap[feature])
					class_tensor[start:end, class_idx] = 1
		
		# Split sequence into windows with step size
		print(f"Adding {chr} windows...", file=sys.stderr)
		for i in tqdm(range(0, len(sequence), step_size)):
			start = i
			end = i + window_size
			if end > len(sequence):
				end = len(sequence)
			window_seq = sequence[start:end]
			window_class = class_tensor[start:end]
			window_class = zlib.compress(window_class.numpy().tobytes())
			with open(db, 'a') as f:
				f.write(f"{window_seq}\t{window_class}\t{table}\t{chr}\t{start}\t{end}\n")

