# revision/scripts/60_epoch_budget.py

- Run · class · L30-L34 — class Run
- parse_trainer_log · function · L37-L48 — def parse_trainer_log(text: str, name: str = "log") -> Run
- load_run · function · L51-L71 — def load_run(path: Path, kind: Optional[str] = None) -> Run
- fit_power_law · function · L74-L93 — def fit_power_law(epochs: List[int], loss: List[float]) -> Dict[str, float]
- epoch_where_gain_below · function · L96-L105 — def epoch_where_gain_below(fit: Dict[str, float], rel_tol: float, max_search: int = 200) -> Optional[int]
- analyse · function · L108-L172 — def analyse(run: Run, patience: int, min_delta: float, plateau_tol: float, margin: int, max_epochs: int) -> Dict
- main · function · L175-L221 — def main(argv: Optional[List[str]] = None) -> int
