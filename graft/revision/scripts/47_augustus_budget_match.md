# revision/scripts/47_augustus_budget_match.py

- cds_by_transcript · function · L56-L75 — def cds_by_transcript(path: Path) -> dict
- read_augustus · function · L78-L102 — def read_augustus(path: Path) -> tuple[dict, dict]
- chain · function · L105-L106 — def chain(struct: tuple) -> tuple
- build_candidates · function · L109-L143 — def build_candidates(pred: dict, posterior: dict, ref: dict, art: dict, primary: dict) -> tuple[list[dict], int, int]
- score · function · L146-L164 — def score(subset: list[dict], alt_total: int, loci_scored: int) -> dict
- tie_bounds · function · L167-L175 — def tie_bounds(ordered: list[dict], budget: int, above: int, tied: int, flag: str) -> list[float]
- main · function · L178-L298 — def main() -> int
