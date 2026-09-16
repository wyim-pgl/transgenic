# revision/scripts/57_score_additions_shared_loci.py

- load_primary · function · L77-L82 — def load_primary(path: Path) -> dict[str, str]
- first_prediction · function · L85-L88 — def first_prediction(txs: dict) -> tuple | None
- locus_records · function · L91-L106 — def locus_records(pred: dict, ref: dict, art: dict, primary: dict, augustus: bool) -> dict
- score · function · L109-L138 — def score(records: dict, loci, denominator: int | None = None) -> dict
- check · function · L141-L148 — def check(name: str, got: dict, want: dict) -> bool
- budget_matched_augustus · function · L151-L173 — def budget_matched_augustus(pred: dict, posterior: dict, ref: dict, art: dict, primary: dict, loci: set, budget: int, denominator: int) -> tuple[dict, dict]
- main · function · L176-L301 — def main() -> int
