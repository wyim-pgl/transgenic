# revision/scripts/46_identity_baseline.py

- evaluation_loci · function · L62-L79 — def evaluation_loci(path: Path) -> set[str]
- primary_of_locus · function · L82-L88 — def primary_of_locus(path: Path) -> dict[str, str]
- write_echo · function · L91-L113 — def write_echo(loci: set[str], primary: dict[str, str], dst: Path) -> tuple[int, int]
- run_gffcompare · function · L116-L123 — def run_gffcompare(query: Path, ref: Path, prefix: Path) -> Path
- score_tmap · function · L126-L148 — def score_tmap(tmap: Path, total_reference: int) -> dict
- raw_stats · function · L151-L159 — def raw_stats(path: Path) -> dict[str, float]
- main · function · L162-L247 — def main() -> int
