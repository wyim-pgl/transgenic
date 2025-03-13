#!/usr/bin/env python3

# Modified from HiSat2 to include UTR introns and work with GFF3 formatting
# https://github.com/DaehwanKimLab/hisat2/blob/master/hisat2_extract_splice_sites.py

from sys import stderr, exit
import os
from collections import defaultdict as dd, Counter
from argparse import ArgumentParser, FileType


def extract_splice_sites(gff_file, feature_type='exon', verbose=False, add=False):
	genes = dd(list)
	trans = {}

	# Parse valid exon lines from the GTF file into a dict by transcript_id
	for line in gff_file:
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		if '#' in line:
			line = line.split('#')[0].strip()

		try:
			chrom, source, feature, left, right, score, \
				strand, frame, attributes = line.split('\t')
		except ValueError:
			continue
		left, right = int(left), int(right)

		if feature != feature_type or left >= right:
			continue

		# Parse the attributes column in GFF3 format
		values_dict = {}
		for attr in attributes.split(';'):
			if attr:
				key, value = attr.strip().split('=')
				values_dict[key] = value

		if 'Parent' not in values_dict:
			continue

		# Extract the transcript_id from the Parent attribute (assuming it's a single value)
		transcript_id = values_dict['Parent']
		if transcript_id not in trans:
			trans[transcript_id] = [chrom, strand, [[left, right]]]
			# Assuming the gene_id can be deduced from the Parent or ID fields in a related entry
			if 'ID' in values_dict:
				gene_id = values_dict['ID'].split('.')[0]
			else:
				gene_id = transcript_id.split('.')[0]
			genes[gene_id].append(transcript_id)
		else:
			trans[transcript_id][2].append([left, right])

	# Sort spliced features and merge where separating introns are <=5 bps
	for tran, [chrom, strand, exons] in trans.items():
			exons.sort()
			tmp_exons = [exons[0]]
			for i in range(1, len(exons)):
				if exons[i][0] - tmp_exons[-1][1] <= 5:
					tmp_exons[-1][1] = exons[i][1]
				else:
					tmp_exons.append(exons[i])
			trans[tran] = [chrom, strand, tmp_exons]

	# Calculate and print the unique junctions
	junctions = set()
	for chrom, strand, exons in trans.values():
		for i in range(1, len(exons)):
			junctions.add((chrom, exons[i-1][1], exons[i][0], strand))
	junctions = sorted(junctions)

	if add:
		os.system(f"cp {gff_file.name} {gff_file.name.split('gff')[0]}splice.gff3")

	for chrom, left, right, strand in junctions:
		# Zero-based offset
		print('{}\t{}\t{}\t{}'.format(chrom, left-1, right-1, strand))
		if add:
			with open(f"{gff_file.name.split('gff')[0]}splice.gff3", 'a') as f:
				if strand == '+':
					donor_start = left + 1
					donor_end = left + 2
					acceptor_start = right - 2
					acceptor_end = right -1
				else:
					donor_start = right - 2
					donor_end = right - 1
					acceptor_start = left + 1
					acceptor_end = left + 2

				f.write(f"{chrom}\tHisat2_Splice\tfive_prime_cis_splice_site\t{donor_start}\t{donor_end}\t.\t{strand}\t.\tID=splice_donor_{chrom}_{donor_start}\n")
				f.write(f"{chrom}\tHisat2_Splice\tthree_prime_cis_splice_site\t{acceptor_start}\t{acceptor_end}\t.\t{strand}\t.\tID=splice_acceptor_{chrom}_{donor_start}\n")

	# Print some stats if asked
	if verbose:
		exon_lengths, intron_lengths, trans_lengths = \
			Counter(), Counter(), Counter()
		for chrom, strand, exons in trans.values():
			tran_len = 0
			for i, exon in enumerate(exons):
				exon_len = exon[1]-exon[0]+1
				exon_lengths[exon_len] += 1
				tran_len += exon_len
				if i == 0:
					continue
				intron_lengths[exon[0] - exons[i-1][1]] += 1
			trans_lengths[tran_len] += 1

		print('genes: {}, genes with multiple isoforms: {}'.format(
				len(genes), sum(len(v) > 1 for v in genes.values())),
			file=stderr)
		print('transcripts: {}, transcript avg. length: {:.0f}'.format(
				len(trans), sum(trans_lengths.elements())//len(trans)),
			file=stderr)
		print('exons: {}, exon avg. length: {:.0f}'.format(
				sum(exon_lengths.values()),
				sum(exon_lengths.elements())//sum(exon_lengths.values())),
			file=stderr)
		print('introns: {}, intron avg. length: {:.0f}'.format(
				sum(intron_lengths.values()),
				sum(intron_lengths.elements())//sum(intron_lengths.values())),
			file=stderr)
		print('average number of exons per transcript: {:.0f}'.format(
				sum(exon_lengths.values())//len(trans)),
			file=stderr)


if __name__ == '__main__':
	parser = ArgumentParser(
		description='Extract splice junctions from a GTF file')
	parser.add_argument('gff_file',
		nargs='?',
		type=FileType('r'),
		help='input GFF file (use "-" for stdin)')
	parser.add_argument('-v', '--verbose',
		dest='verbose',
		action='store_true',
		help='also print some statistics to stderr')
	parser.add_argument('-f', '--feature',
		dest='feature_type',
		default='exon',
		help='feature type to extract splice junctions (default: exon)')
	parser.add_argument('-a', '--add',
		dest='add',
		action='store_true',
		help='Add the splice junctions to the gff. Must sort afterwards.')

	args = parser.parse_args()
	if not args.gff_file:
		parser.print_help()
		exit(1)
	extract_splice_sites(args.gff_file, feature_type=args.feature_type, verbose=args.verbose, add=args.add)