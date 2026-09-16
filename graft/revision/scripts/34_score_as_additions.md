# revision/scripts/34_score_as_additions.py

- _load_sibling · function · L140-L147 — def _load_sibling(filename: str, name: str) -> ModuleType
- chain · function · L166-L174 — def chain(struct: tuple) -> tuple
- _agi · function · L177-L181 — def _agi(gene_id: str) -> str
- gtf_cds_structures · function · L184-L215 — def gtf_cds_structures(path: Path) -> tuple[dict, dict]
- gff3_gene_strands · function · L218-L236 — def gff3_gene_strands(path: Path) -> dict[str, str]
- input_gm_keys · function · L239-L259 — def input_gm_keys(path: Path) -> tuple[dict[str, str], int]
- pair_input_to_output · function · L262-L320 — def pair_input_to_output(inp: dict, out: dict, input_gff: Path, output_gff: Path) -> tuple[dict, dict]
- _reference_context · function · L323-L359 — def _reference_context(ref: dict, ref_strand: dict, art_by_gene: dict, primary_ids: set[str] | None) -> tuple[dict, int, int]
- _tally · function · L362-L464 — def _tally(inp: dict, out: dict, io_pairs: dict, in_to_ref: dict, context: dict, out_strand: dict) -> Counter
- _pct · function · L467-L470 — def _pct(numerator: int, denominator: int) -> float | None
- score · function · L473-L652 — def score(input_gff: Path, output_gff: Path, tair10_gtf: Path = TAIR10_GTF, atrtd3_gtf: Path = ATRTD3_GTF, primary_ids_path: Path | None = PRIMARY_IDS, tool: str | None = None, cli: dict | None = None) -> dict
- _print_result · function · L655-L661 — def _print_result(res: dict, indent: str = " ") -> None
- main · function · L664-L696 — def main(argv: list[str] | None = None) -> int
