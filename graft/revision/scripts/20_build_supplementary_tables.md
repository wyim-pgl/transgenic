# revision/scripts/20_build_supplementary_tables.py

- load_csv · function · L84-L86 — def load_csv(path: Path) -> list[dict]
- jload · function · L89-L91 — def jload(path: Path) -> dict
- stats_metrics · function · L94-L101 — def stats_metrics(path: Path) -> dict[str, tuple[float, float]]
- stats_counts · function · L104-L113 — def stats_counts(path: Path) -> dict[str, int]
- f1 · function · L116-L117 — def f1(sn: float, pr: float) -> float
- md_table · function · L120-L125 — def md_table(header: list[str], rows: list[list]) -> str
- Out · class · L128-L147 — class Out
- __init__ · method · L131-L134 — def __init__(self, outdir: Path)
- table · method · L136-L144 — def table(self, slug: str, title: str, note: str, header: list[str], rows: list[list]) -> None
- text · method · L146-L147 — def text(self, s: str) -> None
- supplemental_figure_legends · function · L150-L171 — def supplemental_figure_legends() -> str
- build · function · L174-L784 — def build(out: Out) -> None
- split_variant · function · L476-L480 — def split_variant(key: str) -> tuple[str, str]
- predicted_transcripts · function · L797-L811 — def predicted_transcripts(result_dir: str) -> int
- main · function · L814-L829 — def main() -> int
