# revision/scripts/mvp/make_launch_readiness.py

- accelerate_config_binding · function · L54-L58 — def accelerate_config_binding() -> dict
- optional_sidecars · function · L61-L72 — def optional_sidecars(checkpoint: Path) -> dict
- require_allocation · function · L75-L80 — def require_allocation() -> str
- _sha256 · function · L83-L92 — def _sha256(path: Path) -> str
- json_snapshot · function · L95-L99 — def json_snapshot(path, expected)
- tooling_identity · function · L102-L106 — def tooling_identity()
- _canonical_bytes · function · L109-L110 — def _canonical_bytes(value: dict) -> bytes
- _atomic_write · function · L113-L138 — def _atomic_write(destination: Path, payload: bytes) -> None
- _regular · function · L141-L145 — def _regular(path_value: str, label: str) -> Path
- _load_receipt · function · L148-L183 — def _load_receipt(preflight_dir: Path, name: str, bindings_sha256: str) -> dict
- _consumed_code_manifest · function · L186-L193 — def _consumed_code_manifest() -> dict
- _source_checkpoint_binding · function · L196-L232 — def _source_checkpoint_binding(init_from: str) -> dict
- generate · function · L235-L318 — def generate(args: argparse.Namespace) -> int
- check · function · L321-L353 — def check(args: argparse.Namespace) -> int
- _add_input_arguments · function · L356-L361 — def _add_input_arguments(parser: argparse.ArgumentParser) -> None
- build_parser · function · L364-L379 — def build_parser() -> argparse.ArgumentParser
- main · function · L382-L391 — def main(argv: list[str] | None = None) -> int
