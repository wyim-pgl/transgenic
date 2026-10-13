# Legacy bootstrap transport regression

Proof3369338 failed1:0 after32m39s ongh063 (parent andbatch; extern0:0), before any replay forward. NGC source57tests passed withone skip and proof harness7tests passed; allocation input verification completed. Allfour ranks raised `ValueError: unsupported JSON value: tuple` at b6_compile_replay.bootstrap_legacy → CompileHistory._gather. Logs, manifest and verification record are preserved in3369338/.

The real manifest validator returns `(provenance, rows)`, while CompileHistory's deliberately strict bounded JSON encoder rejects tuples. Existing tests separately covered manifest validation and legacy replay, but missed their dispatch boundary. The repair converts only this validated pair to a JSON list at the transport call; non-main None remains JSON null. Strict journal encoding and validator return API remain unchanged.

Regression `LegacyManifestTests.test_bootstrap_transports_validated_manifest_and_all_rank_companions` uses real manifest/checkpoint/draw-prefix validation, checkpoint coordination, journal serialization and bootstrap dispatch, replacing only GPU byte transport and final replay execution. It asserts exact provenance and allfour ranks'16companions reach replay. RED reproduced the exact production exception; GREEN focused six tests passed16.260s, including both actualtwo-rankGloo replay tests. `git diff --check` passed. This focused test substitutes GPU byte exchange; actualfour-rankNGC acceptance remains a required gate.

Originalcheckpoint200, diagnostic256, immutable188bf3e/v1 source/helpers and failed3369338 outputs are preserved. No promotion or full pilot4099 start occurred. A reviewed new source snapshot and new proof job are required; failed job is not reused or silently retried.

Scoped independent review approved with no findings. Source repair committed and pushed as `198ce14ce53266f53b7bda5d0c4cdfd94deb43b4`; source graft rebuilt (422files). New immutable snapshot staging is separate from existing188bf3e.
