# revision/scripts/04_generate_figure6.py

- load_gffcompare_results · function · L26-L30 — def load_gffcompare_results(results_dir: Path) -> dict
- load_splice_event_results · function · L33-L37 — def load_splice_event_results(results_dir: Path, prefix: str = 'splice_events') -> dict
- load_isoform_distribution · function · L40-L43 — def load_isoform_distribution(results_dir: Path) -> pd.DataFrame
- plot_panel_a · function · L46-L84 — def plot_panel_a(ax, gffcompare_results: dict)
- plot_panel_b · function · L87-L125 — def plot_panel_b(ax, isoform_dist: pd.DataFrame)
- plot_panel_c · function · L128-L167 — def plot_panel_c(ax, splice_results: dict)
- plot_panel_d · function · L170-L204 — def plot_panel_d(ax, gffcompare_results: dict)
- create_figure6 · function · L207-L251 — def create_figure6( gffcompare_dir: Path, splice_event_dir: Path, output_path: Path, figsize: tuple = (14, 12) )
- main · function · L263-L284 — def main(gffcompare_dir: str, splice_event_dir: str, output: str, width: int, height: int)
