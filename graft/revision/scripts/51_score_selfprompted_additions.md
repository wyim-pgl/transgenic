# revision/scripts/51_score_selfprompted_additions.py

- _load_sibling · function · L101-L108 — def _load_sibling(filename: str, name: str) -> ModuleType
- chain · function · L116-L122 — def chain(struct: tuple) -> tuple
- _agi · function · L125-L129 — def _agi(gene_id: str) -> str
- gtf_cds_by_gene · function · L132-L165 — def gtf_cds_by_gene(path: Path) -> tuple[dict, dict, dict]
- gff3_prediction · function · L168-L234 — def gff3_prediction(path: Path) -> tuple[dict, dict, dict, dict]
- _attr · function · L237-L242 — def _attr(attributes: str, key: str) -> str | None
- _augustus_order · function · L245-L256 — def _augustus_order(txs: dict) -> dict
- key · function · L253-L255 — def key(t: str) -> int
- _dedup_records · function · L259-L290 — def _dedup_records(loci: dict, records: dict) -> tuple[dict, dict]
- _flatten · function · L293-L301 — def _flatten(loci: dict, records: dict) -> dict
- _span · function · L304-L307 — def _span(txs: dict, seq: str) -> tuple
- map_loci_to_reference · function · L310-L351 — def map_loci_to_reference(pred: dict, pred_seq: dict, ref: dict, ref_seq: dict, locus_key: str) -> tuple[dict, dict]
- score · function · L354-L504 — def score(pred_gff: Path, locus_key: str, prompt_source: str, tair10_gtf: Path = TAIR10_GTF, atrtd3_gtf: Path = ATRTD3_GTF, primary_ids_path: Path = PRIMARY_IDS, tool: str | None = None, dedup: bool = True, cli: dict | None = None) -> dict
- pct · function · L435-L438 — def pct(n: int, d: int) -> float | None
- _print_result · function · L507-L513 — def _print_result(res: dict, indent: str = " ") -> None
- main · function · L516-L550 — def main(argv: list[str] | None = None) -> int
