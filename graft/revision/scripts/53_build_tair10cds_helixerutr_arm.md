# revision/scripts/53_build_tair10cds_helixerutr_arm.py

- _load_arm_builder · function · L101-L113 — def _load_arm_builder()
- load_helixer_first_transcript_utr · function · L116-L157 — def load_helixer_first_transcript_utr(gff: Path) -> dict
- donate_utr · function · L160-L169 — def donate_utr(cds: list, donor: list, strand: str) -> tuple[list, list]
- write_arm · function · L172-L234 — def write_arm(rows: list, path: Path) -> dict
- build · function · L237-L280 — def build(tair10_gtf: Path, primary_ids_path: Path, helixer_gff: Path, out_path: Path) -> dict
- main · function · L283-L305 — def main(argv: list[str] | None = None) -> int
