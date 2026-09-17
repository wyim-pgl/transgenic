# scripts/make_orthogroup_splits.py

- _load_contract · function · L36-L43 — def _load_contract()
- read_manifest · function · L46-L48 — def read_manifest(path: str) -> List[Dict[str, str]]
- gene_keys · function · L51-L63 — def gene_keys(gc, species_id: str, gff_path: str) -> Tuple[List[str], Dict[str, str]]
- read_orthogroups · function · L66-L81 — def read_orthogroups(path: str) -> Dict[str, Dict[str, List[str]]]
- build_rows · function · L84-L154 — def build_rows(gc, manifest_rows, gff_override: Dict[str, str], orthogroups: Dict[str, Dict[str, List[str]]], unassigned: Dict[str, Dict[str, List[str]]], holdout_names: Set[str], holdout_species: str, seed: int, source_version: str, fractions=FRACTIONS)
- _parse_kv · function · L157-L162 — def _parse_kv(items)
- main · function · L165-L198 — def main(argv: Optional[List[str]] = None) -> int
