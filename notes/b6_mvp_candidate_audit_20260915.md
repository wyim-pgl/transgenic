# Full MVP candidate audit — 2026-09-15

Read-only audit agent11e30180-f282-426 reported artifact integrity PASS; technical candidate eligibility PASS limited to exploratory MVP runtime/GPU readiness preparation. Production/scientific/A46 approval NOT granted. Remote SUCCESS remains production_training_accepted=false; do not rewrite it.

Independent final DB hash: ed2d0eef2b205626db0983ab9f029f84c30514babc77c330bdb250b2fdfe3d0d,1068773376B, /data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_full_20260915/fullc0lite.db. No output/corpus WAL observed. Artifact mutable0644, consumers must bind digest.

Input BAM/reference/panel rehashed and matched prior receipts. Corpus23.5GB not rehashed in this audit, prior freeze/prebuild/postbuilder hashes agree. Counts: flagstat1050677 incl3852supplementary; members1046825, collapse60444,molecules986381. idxstats978663mapped+72014unmapped=1050677. retained733812+panel67293+test113262+unmapped72014=986381. Header is reference/command evidence, not record-count evidence.

QC panel2079, retained panel/test overlap0, valid evidence1442/1447 (five without evidence explicitly retained). Four train/valid forward/RC raster checks PASS incl Nmask/allN/off/fraction0. Archived approved code abca40fb4941207fa9ad5bbeba211d11f3942748, builder/raster/spec/panel hashes agree with Git bytes; operational scripts0444 match receipt. Payload0/batch0; externOOM0:125 unresolved.

Consolidated receipt hashes and ordered train/valid RN inventory still to publish locally; cannot treat colocated mutable JSON receipts as cryptographically bound by SUCCESS alone. No realHyena/bf16/compile/GPU certification.

## Scope decision
Author previously approved exploratory MVP execution, not A46/full-production training. Technical acceptance, exploratory launch readiness, and scientific/production approval are separate. CLI implementation and synthetic CPU tests may proceed without production approval. Real exploratory data use requires digest-bound technical record; real GPU launch requires separate checkpoint/SIF/model/runtime readiness gates. No GPU jobs submitted. Existing finalize helper identity_scope='production' denotes its fixed250/effective96 profile only, NOT author authorization; CLI will expose runtime_profile=mvp_fixed_250x96 and authorization_scope=exploratory_mvp with production_authorized=false. Do not change reviewed helper solely to rename this legacy internal term.
