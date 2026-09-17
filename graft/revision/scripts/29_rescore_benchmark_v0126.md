# revision/scripts/29_rescore_benchmark_v0126.py

- top_beam · function · L59-L91 — def top_beam(src: Path, dst: Path) -> None
- seqids · function · L94-L106 — def seqids(path: Path, limit: int = 200000) -> set[str]
- harmonise_seqids · function · L109-L140 — def harmonise_seqids(query: Path, ref: Path, dst: Path) -> Path
- parse_stats · function · L143-L151 — def parse_stats(path: Path) -> dict[str, float]
- score · function · L154-L182 — def score(job: tuple[str, str, Path]) -> dict | None
- main · function · L185-L207 — def main() -> int
