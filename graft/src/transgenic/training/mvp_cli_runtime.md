# src/transgenic/training/mvp_cli_runtime.py

- _snapshot_hash · function · L23-L34 — def _snapshot_hash(value: os.PathLike[str] | str, label: str) -> tuple[Path, str]
- _reject_wal · function · L37-L40 — def _reject_wal(path: Path, label: str) -> None
- verify_candidate_artifacts · function · L43-L81 — def verify_candidate_artifacts(candidate: CandidateRecords, db_value, evidence_value, panel_value) -> dict
- _manifest_files · function · L84-L97 — def _manifest_files(path: Path) -> None
- validate_launch_readiness · function · L100-L161 — def validate_launch_readiness(readiness_value, readiness_sha256: str, candidate: CandidateRecords, paths, *, seed: int, recipe: MVPRecipe) -> dict
- _source_paths · function · L164-L169 — def _source_paths(init_from) -> tuple[Path, Path]
- _validate_output · function · L172-L179 — def _validate_output(path_value) -> Path
- _versions · function · L182-L186 — def _versions() -> dict[str, str]
- require_matching_identity · function · L189-L197 — def require_matching_identity(accelerator, value: Any) -> None
- compile_bound_forward · function · L200-L205 — def compile_bound_forward(model, backend="inductor")
- execute · function · L208-L228 — def execute(args, candidate: CandidateRecords, *, validate_only: bool) -> dict
- _train · function · L231-L320 — def _train(args, candidate, artifacts, final, recipe, weights, run_config, readiness) -> dict
