# revision/scripts/61_univec_trim.py

- open_any · function · L37-L38 — def open_any(path: str, mode: str = "rt")
- read_fasta · function · L41-L54 — def read_fasta(path: str) -> Iterator[Tuple[str, str, str]]
- classify · function · L57-L63 — def classify(qstart: int, qend: int, qlen: int, score: int) -> Tuple[str, str]
- read_hits · function · L66-L77 — def read_hits(path: str) -> Dict[str, List[Tuple[int, int, int, int]]]
- plan_record · function · L80-L117 — def plan_record(qlen: int, rec_hits: Iterable[Tuple[int, int, int, int]], min_len: int) -> Tuple[str, List[Tuple[int, int]], List[str]]
- run · function · L120-L142 — def run(fasta: str, hits_path: str, out_fasta: str, report: str, summary: str, min_len: int = DEFAULT_MIN_LEN) -> Dict
- main · function · L145-L156 — def main(argv=None)
