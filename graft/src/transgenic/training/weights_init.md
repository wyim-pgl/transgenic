# src/transgenic/training/weights_init.py

- _canonical · function · L52-L53 — def _canonical(value: Any) -> bytes
- _sha256_bytes · function · L56-L57 — def _sha256_bytes(data: bytes) -> str
- _snapshot · function · L60-L69 — def _snapshot(path: Path) -> dict[str, Any]
- _regular_file · function · L72-L76 — def _regular_file(path: Path, parent: Path, label: str) -> Path
- _resolve · function · L79-L92 — def _resolve(init_from: os.PathLike[str] | str) -> tuple[Path, Path]
- _json_file · function · L95-L97 — def _json_file(path: Path) -> Any
- _value · function · L100-L107 — def _value(value: Any) -> Any
- _manifest · function · L110-L127 — def _manifest(config: Any, recipe: dict[str, Any] | None) -> dict[str, dict[str, Any]]
- _manifest_record · function · L130-L131 — def _manifest_record(fields: dict[str, dict[str, Any]]) -> dict[str, Any]
- _check_equal · function · L134-L140 — def _check_equal(source: dict[str, dict[str, Any]], target: dict[str, dict[str, Any]]) -> None
- load_weights_only · function · L143-L286 — def load_weights_only(model: torch.nn.Module, init_from: os.PathLike[str] | str) -> dict[str, Any]
