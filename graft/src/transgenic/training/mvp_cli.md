# src/transgenic/training/mvp_cli.py

- CandidateRecords · class · L32-L36 — class CandidateRecords
- _sha256 · function · L39-L44 — def _sha256(path: Path) -> str
- _digest · function · L47-L51 — def _digest(value: Any, label: str) -> str
- _regular_file · function · L54-L62 — def _regular_file(value: os.PathLike[str] | str, label: str) -> Path
- _json_record · function · L65-L77 — def _json_record(path_value: os.PathLike[str] | str, expected_sha: str, label: str) -> tuple[Path, dict]
- _validate_inventory · function · L80-L119 — def _validate_inventory(value: dict) -> None
- validate_candidate_records · function · L122-L186 — def validate_candidate_records( acceptance_record: os.PathLike[str] | str, acceptance_sha256: str, inventory_record: os.PathLike[str] | str, receipts_record: os.PathLike[str] | str, ) -> CandidateRecords
- build_parser · function · L189-L211 — def build_parser() -> argparse.ArgumentParser
- _required · function · L214-L217 — def _required(args: argparse.Namespace, names: tuple[str, ...]) -> None
- main · function · L220-L239 — def main(argv: list[str] | None = None) -> int
