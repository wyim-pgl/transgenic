# examples/testing_ch4.py

- _set_seed · function · L39-L44 — def _set_seed(seed: int = 42)
- _generate_batch · function · L47-L75 — def _generate_batch(model, ii, am, max_length=2048)
- _parse_gffcompare_stats · function · L78-L115 — def _parse_gffcompare_stats(stats_path: str) -> Dict[str, float]
- _count_isoforms_gffread · function · L118-L141 — def _count_isoforms_gffread(gff_path: str) -> int
- _build_subset_ref_gff · function · L144-L200 — def _build_subset_ref_gff(ref_gff_path: str, gene_names: set, output_path: str) -> str
- evaluate_chr4 · function · L203-L370 — def evaluate_chr4( model: Any, device: torch.device, *, n_samples: int = 500, batch_size: int = 4, max_gen_len: int = 2048, work_dir: Optional[str] = None, seed: int = 42, verbose: bool = True, ) -> Dict[str, float]
- main · function · L374-L409 — def main()
