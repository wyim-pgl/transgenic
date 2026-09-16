# revision/scripts/58_headline_additions_structural.py

- load_primary · function · L79-L84 — def load_primary(path: Path) -> dict[str, str]
- additions_with_flags · function · L87-L118 — def additions_with_flags(completed_gff: Path, supplied: dict, genome: dict, locus_of_gene) -> list[dict]
- attach_matches · function · L121-L129 — def attach_matches(records: list[dict], ref: dict, art: dict, primary: dict) -> None
- pct · function · L132-L133 — def pct(n: int, d: int) -> float | None
- summarize · function · L136-L152 — def summarize(records: list[dict], denominator: int, flag: str | None) -> dict
- structural_counts · function · L155-L171 — def structural_counts(records: list[dict]) -> dict
- check · function · L174-L181 — def check(name: str, got: dict, want: dict) -> bool
- transfer_arm · function · L184-L203 — def transfer_arm(tool: str, genome: dict, ref: dict, art: dict, primary: dict) -> dict
- locus_of_gene · function · L192-L193 — def locus_of_gene(gene: str | None) -> str
- main · function · L206-L319 — def main() -> int
- locus_of_gene · function · L225-L228 — def locus_of_gene(gene: str | None)
