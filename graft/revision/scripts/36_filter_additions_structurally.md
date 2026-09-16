# revision/scripts/36_filter_additions_structurally.py

- load_genome · function · L67-L90 — def load_genome(path: Path) -> dict[str, str]
- store · function · L72-L76 — def store(seq_name: str, seq: str) -> None
- spliced_cds · function · L93-L97 — def spliced_cds(chromosome: str, segments: tuple, strand: str) -> str
- has_complete_orf · function · L100-L105 — def has_complete_orf(cds: str) -> bool
- has_canonical_introns · function · L108-L120 — def has_canonical_introns(chromosome: str, segments: tuple, strand: str) -> bool
- read_cds_by_transcript · function · L123-L143 — def read_cds_by_transcript(path: Path) -> tuple[dict, dict, dict]
- supplied_structures · function · L146-L174 — def supplied_structures(prompt_gff: Path) -> dict
- filter_predictions · function · L177-L233 — def filter_predictions(prompt_gff: Path, completed_gff: Path, genome_path: Path, out_gff: Path | None) -> dict
- write_filtered · function · L236-L270 — def write_filtered(completed_gff: Path, keep: set, out_gff: Path) -> None
- orf_audit · function · L273-L315 — def orf_audit(annotation_gff: Path, genome_path: Path) -> dict
- main · function · L318-L342 — def main(argv: list[str] | None = None) -> int
