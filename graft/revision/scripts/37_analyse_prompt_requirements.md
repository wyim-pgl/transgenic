# revision/scripts/37_analyse_prompt_requirements.py

- agi · function · L53-L55 — def agi(value: str) -> str
- utr_outside · function · L58-L73 — def utr_outside(exons: list, cds: tuple) -> tuple
- load_reference · function · L76-L107 — def load_reference(gtf: Path, primary_ids_path: Path) -> dict
- load_prompt · function · L110-L156 — def load_prompt(gff: Path) -> dict
- load_emitted · function · L159-L188 — def load_emitted(gff: Path) -> tuple[dict, dict]
- _pct · function · L191-L192 — def _pct(numerator: int, denominator: int)
- utr_accuracy · function · L195-L214 — def utr_accuracy(prompt, emitted, reference) -> dict
- utr_closeness · function · L217-L250 — def utr_closeness(prompt, emitted, reference) -> dict
- covered · function · L219-L223 — def covered(segments)
- generated_utr · function · L253-L277 — def generated_utr(prompt, emitted, utr_of, reference) -> dict
- main · function · L280-L309 — def main(argv: list[str] | None = None) -> int
