# revision/scripts/19_verify_manuscript_numbers.py

- check · function · L32-L38 — def check(name: str, claimed, derived, source: str, tol: float = 0.051) -> None
- load_csv · function · L41-L43 — def load_csv(path: Path) -> list[dict]
- jload · function · L46-L48 — def jload(path: Path) -> dict
- stats_metrics · function · L51-L59 — def stats_metrics(path: Path) -> dict[str, tuple[float, float]]
- stats_counts · function · L62-L71 — def stats_counts(path: Path) -> dict[str, int]
- f1 · function · L318-L319 — def f1(sn: float, pr: float) -> float
- _chains · function · L342-L357 — def _chains(path: Path, gtf: bool) -> dict[str, tuple]
- panel_counts · function · L360-L375 — def panel_counts(locus: str) -> dict
- _cds_by_tx · function · L447-L461 — def _cds_by_tx(path, gff3=True)
- _events · function · L536-L542 — def _events(pred_key: str) -> tuple[int, float]
- _matched · function · L554-L556 — def _matched(pred_key: str) -> int
- _f1 · function · L627-L632 — def _f1(species: str, tool: str, level: str) -> float | None
- _spearman · function · L896-L920 — def _spearman(x: list[float], y: list[float]) -> float
- _ranks · function · L902-L914 — def _ranks(v: list[float]) -> list[float]
