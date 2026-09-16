# src/transgenic/training/mvp_cli.py

- CandidateRecords · class · L26-L30 — class CandidateRecords
- _sha256 · function · L33-L38 — def _sha256(path: Path) -> str
- _digest · function · L41-L45 — def _digest(value: Any, label: str) -> str
- _regular_file · function · L48-L56 — def _regular_file(value: os.PathLike[str] | str, label: str) -> Path
- _json_record · function · L59-L71 — def _json_record(path_value: os.PathLike[str] | str, expected_sha: str, label: str) -> tuple[Path, dict]
- _validate_inventory · function · L74-L114 — def _validate_inventory(value: dict) -> None
- validate_candidate_records · function · L117-L180 — def validate_candidate_records( acceptance_record: os.PathLike[str] | str, acceptance_sha256: str, inventory_record: os.PathLike[str] | str, receipts_record: os.PathLike[str] | str, ) -> CandidateRecords
- build_parser · function · L183-L205 — def build_parser() -> argparse.ArgumentParser
- _required · function · L208-L211 — def _required(args: argparse.Namespace, names: tuple[str, ...]) -> None
- main · function · L214-L233 — def main(argv: list[str] | None = None) -> int
