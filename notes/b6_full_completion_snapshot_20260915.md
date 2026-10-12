# Completion snapshot — 2026-09-15 17:26 PDT

Read-only remote checks, no jobs submitted/changed. squeue6164049/6164118 empty.

Vvinifera6164049_7 allocation/batch COMPLETED0:0 elapsed1h00m05s, batchRSS8941560K. Both source DONE and batch.exitcode file present (contents not reread in this snapshot). Together with prior snapshots all9 species have both DONE; full provenance/raw/timing acceptance audit still pending. Raw HSP acceptance/phase projection not performed.

Full MVP build6164118 allocation/batch COMPLETED0:0 elapsed2h22m10s, batchRSS1266852K. .extern OUT_OF_MEMORY0:125 on both; cause unresolved, not silently clean accounting.

Read actual completion artifacts under /data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_full_20260915:
- payload.exitcode0, builder exit_code0, wall8450.3604s including post-source hashes.
- SUCCESS statusPASS/full_build_success=true, scope FULL SOURCE candidate MVP; NOT_FOR_PRODUCTION_TRAINING; production_training_accepted=false.
- DB1068773376B, recorded SHA256ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d (not freshly rehashed by controller yet).
- reads_in1050677, members_in1046825, molecules_after_dedup986381, retained733812; collapsed60444; excluded panel67293/test113262/unmapped72014.
- panel protected2079, retained_overlapping_panel0, retained_overlapping_test0.
- valid tiles1447, with evidence1442.
- Four train/valid forward/RC raster results record PASS for N masking/allN/fraction0/sourceoff/RC; [30720,8]float16. Validation sample contains2861N bases; availability channel27859 nonzero.
- input_identities records expected corpus6840c93f..., BAMf4598638..., FASTA63d56962... matching prior identities.

Remaining: verify full source count against independent BAM receipt/header and final output hash/manifest before training-candidate acceptance. Inspect exact before/after source receipts, full accounting/provenance. These artifact readings establish reported build/QC completion, not production-training approval, full scientific leakage proof, or GPU certification.
