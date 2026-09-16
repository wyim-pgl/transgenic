# revision/scripts/72_mvp_select_loci.py

- SelectError · class · L50-L51 — class SelectError(RuntimeError)
- digest · function · L54-L59 — def digest(path: Path) -> str
- read_eligible · function · L62-L85 — def read_eligible(path: Path) -> list
- read_est_counts · function · L88-L98 — def read_est_counts(path: Path) -> dict
- select_panels · function · L101-L142 — def select_panels(pool: list, counts: dict, mvp: int, core: int, pilot: int, max_len: int, min_est: int, report: list) -> list
- write_tsv · function · L145-L156 — def write_tsv(rows, columns, path: Path) -> str
- main · function · L159-L188 — def main(argv=None) -> int
