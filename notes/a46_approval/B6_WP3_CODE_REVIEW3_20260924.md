**APPROVE**

**A. Residual MEDIUM: FIXED.** [Zero-list validation](/tmp/b6-wp3-tracks/src/transgenic/evidence/b6_bindings.py:294) now checks species, requires the exact producer-key set, and matches each hash to its own `(species, source)` merge entry. The previous `{"protein_c0": EST_HASH}` reproduction is rejected, as are missing, swapped, extra, and wrong-species entries. Training requires a merged protein C0 for every covered species; EST-only zero lists remain accepted when evaluation coverage has no protein C0.

**B. No new CRITICAL/HIGH found in the reviewed changes.**

- **Provenance is rehashed.** Each verification rebuilds the map from current file bytes. The final canonical `record_sha256` includes that map. A memory-only change to provenance-file bytes changed both the record hash and WP4’s binding identity; missing C0 provenance failed closed.
- **WP4 matches the contract.** Its [training entry point](/tmp/b6-wp4-trainer/train/train_HyenaTransgenic_B6.py:531) passes the freeze file path and validates the canonical record hash. The actual WP4 runtime accepted a record returned by current WP3 verification and rejected subsequent provenance-map tampering. Both provenance fields participate in resume comparison. The shared specification files are byte-identical.
- **WP2 matches the contract.** Its [freeze producer](/tmp/b6-wp2-corpus/src/transgenic/evidence/b6_corpus_repair.py:538) supplies the species-indexed hashes and manifest paths required by WP3.

Verification used existing fixtures opened read-only and in-memory negative probes. No full suite or training run was performed. No files were modified.