# revision/scripts/35_build_frame_shift_prompts.py

- load_primary_ids · function · L60-L61 — def load_primary_ids(path: Path) -> set[str]
- load_tair10 · function · L64-L102 — def load_tair10(gtf: Path, primary_ids: set[str]) -> dict
- load_helixer_genes · function · L105-L119 — def load_helixer_genes(gff: Path) -> dict
- pair_and_filter · function · L122-L152 — def pair_and_filter(tair: dict, helixer: dict) -> tuple[list, dict]
- write_arm · function · L155-L176 — def write_arm(rows: list, path: Path, frame: str) -> int
- utr_segments · function · L179-L204 — def utr_segments(exons: list, cds: list, strand: str) -> tuple
- load_tair10_with_utr · function · L207-L242 — def load_tair10_with_utr(gtf: Path, primary_ids: set) -> dict
- write_arm_with_utr · function · L245-L265 — def write_arm_with_utr(rows: list, path: Path, frame: str) -> int
- _merge · function · L268-L276 — def _merge(intervals: list) -> list
- place_donated_utr · function · L279-L317 — def place_donated_utr(cds: list, donor: list, strand: str) -> tuple[list, list]
- load_helixer_first_cds · function · L320-L358 — def load_helixer_first_cds(gff: Path) -> dict
- write_reframed · function · L361-L381 — def write_reframed(genes: dict, path: Path) -> int
- write_helixer_with_tair_utr · function · L384-L447 — def write_helixer_with_tair_utr(helixer: dict, tair_utr: dict, path: Path) -> dict
- main · function · L450-L554 — def main(argv: list[str] | None = None) -> int
