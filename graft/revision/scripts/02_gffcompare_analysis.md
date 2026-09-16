# revision/scripts/02_gffcompare_analysis.py

- GFFCompareResults · class · L41-L59 — class GFFCompareResults
- run_gffcompare · function · L62-L123 — def run_gffcompare( reference_gtf: Path, predicted_gtf: Path, output_prefix: Path, genome_fasta: Optional[Path] = None ) -> Path
- parse_tmap_file · function · L126-L155 — def parse_tmap_file(tmap_path: Path) -> pd.DataFrame
- parse_stats_file · function · L158-L182 — def parse_stats_file(stats_path: Path) -> dict
- calculate_metrics · function · L185-L254 — def calculate_metrics(tmap_df: pd.DataFrame, total_reference_all: int = 0) -> GFFCompareResults
- count_reference_transcripts · function · L257-L278 — def count_reference_transcripts(reference_gtf: Path) -> int
- analyze_isoform_distribution · function · L281-L368 — def analyze_isoform_distribution( reference_gtf: Path, predicted_gtf: Path, output_dir: Path ) -> pd.DataFrame
- count_isoforms_per_gene · function · L295-L342 — def count_isoforms_per_gene(gtf_path: Path) -> dict
- norm · function · L304-L305 — def norm(gid)
- categorize_prediction · function · L371-L382 — def categorize_prediction(ref_count: int, pred_count: int) -> str
- generate_summary_report · function · L385-L424 — def generate_summary_report( results: GFFCompareResults, isoform_dist: pd.DataFrame, output_path: Path )
- print_results_table · function · L427-L481 — def print_results_table(results: GFFCompareResults, isoform_dist: pd.DataFrame)
- main · function · L495-L566 — def main(reference: str, predicted: str, output_dir: str, genome: str, prefix: str)
