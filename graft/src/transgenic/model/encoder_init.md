# src/transgenic/model/encoder_init.py

- expectation_key · function · L36-L37 — def expectation_key(checkpoint: str, d_model: int, n_layer: int) -> str
- _layer_index · function · L40-L45 — def _layer_index(key: str) -> Optional[int]
- summarise · function · L48-L58 — def summarise(rows: Sequence[Dict]) -> Dict
- check_expected · function · L61-L76 — def check_expected(summary: Dict, expected: Optional[Dict[str, int]]) -> List[str]
- write_report · function · L79-L91 — def write_report(path: str, rows: Sequence[Dict], checkpoint: str, target: Dict, source: Dict, violations: Optional[Sequence[str]] = None) -> str
