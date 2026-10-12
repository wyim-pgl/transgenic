# Independent Task 2 source/deployment review

## Verdict

**APPROVE bounded CPU diagnostic**, subject to the controller staging conditions below. This approves an attempt, not runtime compatibility, successful 50-panel production, scientific reference acceptance, decoder readiness, or submission itself. No demonstrated deterministic missing runtime package/API was found on the exercised import path. No SSH, remote imports, jobs, source changes, or subagents were used in this review.

Only **`final_release/`** is covered. Independently calculated SHA256 of `final_release/transport.json`:

`b4bc567d6bdae382318fd0061186bf83c5077b12ac024b5cc36a6dcf9bb54eef`

| Payload | Independently verified SHA256 |
|---|---|
| source.tar | afe0bf0cbe60397167bfe64f878db4c8472053f652272308e8772e3ba562fc2f |
| reference_authority.tar | 7002445d7007c6495d5fc00406faa2c7b373ef10b7c5ed93674bc9b04aae6ec3 |
| cpu_site.tar | 939cd7ab2480c03c057be6817e5e9076d0bb6bd453f7a39ba849a58e7d6d6f79 |
| authority.py | 0b3917b7fa33df0d1818bdbad992f664e8dc6b959639a09aa998f75a97e8b2dc |
| run_panel_cpu.py | 3e62ae91e83b874b2c32d453d2e7001a9834930304ca57159eb4194f21acf3bd |
| panel_cpu.sbatch | 9f236df0ff39ce3b6282e729d577f932b89cd0fac301cb4bff406616ea1abb26 |
| cpu_dependency.json | cf4d509e8914d6bcf44e6a96a79fe127e4521f7592e4a768a7e2659760ec3435 |

## Independent packed-byte checks

These were performed against tar members, not inferred from REPORT.md or an unpacked source tree:

- All **100** source inventory entries matched their SHA256 and direct `git show b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6:<path>` bytes. Fixed inventory SHA256 `7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689` matched. All 113 regular source archive members also matched the extracted `final_release/source` files. Companion and component checks passed; component pin is `555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44`. Canonical decoder bytes are unchanged; this does not reauthorize decoder execution or alter the reported reviewed `b53c31756d4fba10b144afd87e6ff8bc7db0bfbeaf4d346d085243a726512cfb` closure.
- Independently decompressed every packed Git object, checked type/length headers and SHA1 object IDs, and compared raw bodies with the original repository's `git cat-file` bytes. The five original objects are commit `92d8020ee79eeedd7f1d50167b1c8dc94700fb16`, root tree `ffff3bcb4c0a54e1893e637747f432ccb0d5369e`, data tree `4b71774b05a533bb68fe37183f96ef3ab98efbc1`, manifests tree `25dd0a4520ad2d37e43af36562c660b96fbbd466`, and blob `d41b313a230ecde4b51f045a91e5ea39b00c40d9`. No fabricated replacement trees. The authority-only and source archives contain identical authority files.
- Actual sparse-root `git show 92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv` and `authority.verify` passed. Manifest SHA256 is `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`. This authorizes that path traversal only, not complete history or `git fsck`.
- All three tar archives have unique member names, only regular files/directories, no absolute/traversal names, symlinks, hardlinks, or devices. The dependency archive has exactly the **54** receipt-listed files. All matched both hashes and the locally retained wheel's actual ZIP bytes. Wheel SHA256 is `e238060db5ca59879882a6e9b015e2c65d5c64ddf281ba1d7a9a2033764152cf`; ABI is CPython 3.10 x86-64. PyPI attribution is from the supplied receipt; no fresh network query was made. No installation or environment extraction was performed.
- Fresh local rerun: **9 unit tests PASS, 21.141 s**; `bash -n final_release/panel_cpu.sbatch` PASS. These supersede neither the controller's earlier 22.032 s run nor actual compute testing. One auxiliary AST-display command used unavailable local `ast.unparse`; rerunning with `ast.get_source_segment` succeeded. No implementation failure was hidden.

## Runtime/import finding

`run_panel_cpu.py:46–67,132–136` checks the dependency receipt and Python/package versions, then performs real imports before the expensive candidate/reference work. The metadata observation is Python **3.10.20**, Torch **1.13.1+cpu**, NumPy **1.26.4**; only DuckDB **1.5.5 cp310** is added. No cp311 site is prepended, no script installs/downloads packages, and no environment/source replacement is present.

Finite source inspection: `mvp_prepare_panel_inputs.py:30–45` requires `mvp_cli_runtime`; that module's import at `src/transgenic/training/mvp_cli_runtime.py:15` reaches `mvp_runtime.py:14–15`, importing Torch and `torch.utils.data.distributed.DistributedSampler`. Module-level statements, decorators and defaults were inspected. Accelerate imports at `mvp_runtime.py:57–79` are inside the uncalled training factory, not a mandatory import for this diagnostic. Model/Safetensors/finalizer/compile paths are likewise not called by candidate verification. The producer/bootstrap/track and strict legacy-parser path requires NumPy/DuckDB plus standard-library utilities, not the decoder stack. No proven missing required API/package warrants a blocker. This is **not evidence that Torch 1.13 executes successfully**: native loading, transitive environment dependencies and filesystem/container behavior remain compute-time gates. A real import failure is a bounded negative outcome, not permission to bypass the canonical verifier.

## Execution and scientific boundaries

- `panel_cpu.sbatch:4–9,34–41`: one node/task, four CPUs, 16 GB, one hour, no requeue/GPU; outer timeout 3500 s with 20 s kill grace. `/data/gpfs` is read-only, with only the dedicated output parent overlaid writable. Source, sparse `.git`, databases, reference and existing environment are therefore read-only **when staged canonically under that bind**. This is project/data write protection, not a claim of a fully hermetic container: normal container temporary/home mounts are not explicitly contained.
- `run_panel_cpu.py:120–160`: exclusive output directory, source/object verification, actual canonical producer call, no checkpoint/model operation. Exceptions preserve incomplete diagnostics; timeout/native failure need not emit FAILED. Absence of OBSERVATIONS_COMPLETE is never success.
- `run_panel_cpu.py:77–117` invokes the actual frozen-reference gate and canonical restriction/strict GFF parser, reports per-locus failures and duplicate structures without changing multiplicity. Canonical `mvp_produce_counts.py:54–70,222–303` binds actual reference bytes and checks parser inventories/geometry. Restricted output is an audit derivative, not a replacement freeze. A completion marker can coexist with reported reference incompatibilities and is not scientific approval.
- Canonical `mvp_prepare_panel_inputs.py:30–54,102–170` binds accepted candidate files, uses read-only DuckDB and WAL checks, compares seeds 1/2/3 with `rn=None`, `excluded_heldout`, full retention, and genomic-forward windows. Biological channels 0–6 are counted separately from availability channel 7; zero support is reported. No retained/training fallback or fabricated support.
- Existing recorded stats are FASTA **121183059**, GFF **60503119**, corpus **23534252032**, evidence **1068773376** bytes at the four canonical paths in `input_inventory.json`. These are historical stats, not fresh comparisons or independent remote hashes. No actual 50-locus/reference results were produced by this review.

## Controller conditions and Task 3 interface

Before any submission, independently preserve/check the exact transport pin above; use a **new canonical, non-symlink release directory under `/data/gpfs`**, with no preexisting source/site/output tree. Reject extra executable/site files and extract only the reviewed members without overwriting existing paths. The batch script does not implement transport verification or safe extraction; those remain controller responsibilities. Use a dedicated empty outputs parent, approved account/CPU partition and absolute fresh log paths; check live queue status before submission. Do not overlay an existing checkout or retry into a failed namespace.

**Task 3 Git-authority interface: APPROVE.** Transfer the pinned `reference_authority.tar` into a new exact-code root with no `.git`, verify with the pinned `authority.py`, then invoke Task 3's own canonical frozen-reference gate against actual bytes. The manifest blob has the stated `3ca7…f783` identity; this review made no Task 3 folder edits and does not use the working TSV as a substitute for Git authority. Future Task 2 reference/request receipts are observations for comparison, not authority replacements.

Future decoder consumption remains explicitly deferred: canonical `mvp_verify_panel_inputs.py:34` requires exact NumPy/DuckDB versions, and paths/bytes are also bound. NumPy1.26.4/DuckDB1.5.5 production here does not establish decoder compatibility. No weakening of those gates is approved.

Advisor consultation was attempted but unavailable due to its client/model version error; no advisor approval is claimed. Graft-reported context savings: approximately 1,087,887 tokens (estimate).
