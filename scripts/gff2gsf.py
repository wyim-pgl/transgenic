#!/usr/bin/env python3
"""
Convert GFF3 annotations to Gene Sentence Format (GSF).

Usage:
    python gff2gsf.py input.gff3 > output.gsf
    python gff2gsf.py input.gff3 -o output.gsf
"""

import argparse
import sys
from typing import TextIO


def convert_phase(phase: str) -> str:
    """Convert GFF3 phase (0,1,2) to GSF phase (A,B,C)."""
    phase_map = {'0': 'A', '1': 'B', '2': 'C'}
    return phase_map.get(phase, '.')


def gff2gsf(gff_file: TextIO, output: TextIO, relative_coords: bool = True):
    """
    Convert GFF3 to GSF format.
    
    Args:
        gff_file: Input GFF3 file handle
        output: Output file handle
        relative_coords: If True, coordinates are relative to gene start (0-indexed)
    """
    gene_start = None
    gene_id = None
    feature_list = ''
    mRNA_list = ''
    
    # Track unique features by their coordinates
    cds_num = {}
    five_ps = {}
    three_ps = {}
    
    for line in gff_file:
        if line.startswith('#') or line.strip() == '':
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        chrom, source, typ, start, end, score, strand, phase, attributes = fields
        start = int(start)
        end = int(end)
        
        if typ == 'gene':
            # Output previous gene if exists
            if gene_start is not None and feature_list and mRNA_list:
                mRNA_list = mRNA_list.rstrip('|')
                if mRNA_list.endswith(';'):
                    mRNA_list = mRNA_list[:-1]
                gsf = f"{feature_list.rstrip(';')}>{mRNA_list}"
                output.write(f"{gene_id}\t{gsf}\n")
            
            # Start new gene
            gene_start = start - 1  # Convert to 0-indexed
            gene_id = attributes.split(';')[0].split('=')[1] if '=' in attributes else attributes
            feature_list = ''
            mRNA_list = ''
            cds_num = {}
            five_ps = {}
            three_ps = {}
        
        elif typ == 'mRNA':
            # New transcript - add separator if not first
            if mRNA_list and not mRNA_list.endswith(';'):
                mRNA_list = mRNA_list.rstrip('|') + ';'
        
        elif typ == 'CDS':
            if relative_coords:
                feat_start = start - 1 - gene_start
                feat_end = end - gene_start
            else:
                feat_start = start - 1
                feat_end = end
            
            gsf_phase = convert_phase(phase)
            key = f"{feat_start}-{feat_end}-{strand}-{gsf_phase}"
            
            if key not in cds_num:
                num = str(len(cds_num) + 1)
                cds_num[key] = num
                feature_list += f"{feat_start}|CDS{num}|{feat_end}|{strand}|{gsf_phase};"
            else:
                num = cds_num[key]
            
            mRNA_list += f"CDS{num}|"
        
        elif typ == 'five_prime_UTR':
            if relative_coords:
                feat_start = start - 1 - gene_start
                feat_end = end - gene_start
            else:
                feat_start = start - 1
                feat_end = end
            
            key = f"{feat_start}-{feat_end}-{strand}"
            
            if key not in five_ps:
                num = str(len(five_ps) + 1)
                five_ps[key] = num
                feature_list += f"{feat_start}|five_prime_UTR{num}|{feat_end}|{strand}|.;"
            else:
                num = five_ps[key]
            
            mRNA_list += f"five_prime_UTR{num}|"
        
        elif typ == 'three_prime_UTR':
            if relative_coords:
                feat_start = start - 1 - gene_start
                feat_end = end - gene_start
            else:
                feat_start = start - 1
                feat_end = end
            
            key = f"{feat_start}-{feat_end}-{strand}"
            
            if key not in three_ps:
                num = str(len(three_ps) + 1)
                three_ps[key] = num
                feature_list += f"{feat_start}|three_prime_UTR{num}|{feat_end}|{strand}|.;"
            else:
                num = three_ps[key]
            
            mRNA_list += f"three_prime_UTR{num}|"
    
    # Output last gene
    if gene_start is not None and feature_list and mRNA_list:
        mRNA_list = mRNA_list.rstrip('|')
        if mRNA_list.endswith(';'):
            mRNA_list = mRNA_list[:-1]
        gsf = f"{feature_list.rstrip(';')}>{mRNA_list}"
        output.write(f"{gene_id}\t{gsf}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GFF3 annotations to Gene Sentence Format (GSF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python gff2gsf.py annotation.gff3 > output.gsf
    python gff2gsf.py annotation.gff3 -o output.gsf
    python gff2gsf.py annotation.gff3 --absolute  # Use absolute coordinates

Output format (tab-separated):
    gene_id    GSF_string

GSF format:
    <feature_list>><transcript_list>
    
    Feature: start|type|end|strand|phase
    Example: 0|CDS1|150|+|A;200|CDS2|300|+|B>CDS1|CDS2
"""
    )
    parser.add_argument('input', help='Input GFF3 file (use - for stdin)')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--absolute', action='store_true',
                        help='Use absolute coordinates instead of relative to gene start')
    
    args = parser.parse_args()
    
    # Open input
    if args.input == '-':
        infile = sys.stdin
    else:
        infile = open(args.input, 'r')
    
    # Open output
    if args.output:
        outfile = open(args.output, 'w')
    else:
        outfile = sys.stdout
    
    try:
        gff2gsf(infile, outfile, relative_coords=not args.absolute)
    finally:
        if args.input != '-':
            infile.close()
        if args.output:
            outfile.close()


if __name__ == '__main__':
    main()
