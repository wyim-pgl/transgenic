# revision/scripts/48_score_zmays_additions.py

- wilson95 · function · L73-L82 — def wilson95(hits: int, n: int) -> list | None
- chain · function · L85-L87 — def chain(struct: tuple) -> tuple
- read_prediction · function · L90-L157 — def read_prediction(path: Path) -> tuple[dict, dict]
- read_reference · function · L160-L203 — def read_reference(path: Path) -> tuple[dict, dict, dict]
- score · function · L206-L261 — def score(pred: dict, ref: dict, primary: dict, loci: list, label: str, refname: str = "RefGenV4") -> dict
- prompt_identity_check · function · L264-L294 — def prompt_identity_check(pred: dict, ref_order: dict, struct: dict, longest: dict) -> dict
- read_reference_ordered · function · L297-L322 — def read_reference_ordered(path: Path) -> tuple[dict, dict, dict]
- read_athaliana · function · L325-L363 — def read_athaliana() -> tuple[dict, dict, dict]
- verify_athaliana · function · L366-L374 — def verify_athaliana() -> dict
- two_proportion_z · function · L377-L390 — def two_proportion_z(h1: int, n1: int, h2: int, n2: int) -> dict
- main · function · L393-L512 — def main() -> int
