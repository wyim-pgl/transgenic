# revision/scripts/62_geenuff_qc.py

- is_hard · function · L34-L36 — def is_hard(flag: str) -> bool
- read_flags_tsv · function · L39-L51 — def read_flags_tsv(path: str) -> Dict[Tuple[str, str], Dict[str, Set[str]]]
- loss_mask_decision · function · L54-L69 — def loss_mask_decision(gene_flags: Dict[str, Set[str]], transcript_ids: Iterable[str]) -> Tuple[float, List[str], List[str]]
- extract_from_geenuff_db · function · L81-L114 — def extract_from_geenuff_db(db_path: str, species_id: str, seen: Optional[Dict] = None) -> List[Tuple[str, str, str, str, int, int]]
- main · function · L117-L140 — def main(argv=None)
