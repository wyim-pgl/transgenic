# scripts/standardize_gff.py

- load_chrom_lengths · function · L100-L115 — def load_chrom_lengths(genome_dir, species_prefix)
- merge_intervals · function · L118-L130 — def merge_intervals(intervals)
- parse_attributes_gff3 · function · L133-L140 — def parse_attributes_gff3(attr_string)
- parse_attributes_gtf · function · L143-L155 — def parse_attributes_gtf(attr_string)
- parse_attributes · function · L158-L165 — def parse_attributes(attr_string)
- format_attributes · function · L168-L185 — def format_attributes(attributes)
- infer_species · function · L188-L214 — def infer_species(tool_name, filename)
- _check_intron_lengths · function · L217-L226 — def _check_intron_lengths(exons_sorted)
- _fix_inverted_cds · function · L229-L262 — def _fix_inverted_cds(cds_features)
- standardize_gff · function · L265-L851 — def standardize_gff(input_file, output_file, species_prefix, tool_name, genome_dir=None)
- _feature_line · function · L854-L859 — def _feature_line(feature)
- main · function · L862-L908 — def main(): # Parse command line arguments
