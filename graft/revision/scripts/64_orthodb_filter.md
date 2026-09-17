# revision/scripts/64_orthodb_filter.py

- taxid_of_header · function · L33-L37 — def taxid_of_header(header: str) -> int
- _open · function · L40-L41 — def _open(path: str, mode: str = "rt")
- read_fasta · function · L44-L55 — def read_fasta(path: str) -> Iterator[Tuple[str, str]]
- read_lineage · function · L58-L68 — def read_lineage(path: str) -> Dict[int, List[int]]
- parse_taxonomy_xml · function · L71-L83 — def parse_taxonomy_xml(xml: str) -> Dict[int, List[int]]
- _efetch_taxonomy_xml · function · L86-L90 — def _efetch_taxonomy_xml(ids: List[int]) -> str
- fetch_lineage · function · L93-L113 — def fetch_lineage(taxids: Iterable[int], cache_path: str, batch: int = 100, pause: float = 0.4, fetch=None) -> Dict[int, List[int]]
- excluded_taxids · function · L116-L123 — def excluded_taxids(exclude: Set[int], lineage: Dict[int, List[int]], present: Optional[Set[int]] = None, allow_missing: bool = False) -> Set[int]
- file_md5 · function · L126-L131 — def file_md5(path: str) -> str
- _parse_kv · function · L134-L141 — def _parse_kv(items: Optional[List[str]]) -> Dict[str, str]
- main · function · L144-L198 — def main(argv: Optional[List[str]] = None) -> int
