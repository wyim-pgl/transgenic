# Reference identifier diagnosis — job 6165559

## Decision

**Root contract error identified: the evaluation paths confuse corpus builder keys with raw reference GFF IDs. An authoritative mapping already exists: `gene_key_map.gene_id → gene_id_original`, keyed by species. Use it; do not invent a coordinate or prefix heuristic.**

The canonical count producer and CPU audit both restrict the raw GFF with panel `gene_id` values (`Ath014705`, etc.), although panel generation deliberately publishes builder keys and uses `gene_id_original` internally when reading reference annotations. The resulting header-only GFF is the observed failure. This is not evidence that the biological loci are absent.

The actual 50 mapping rows were **not queried** in this investigation. Their completeness, identity binding, and all 50 downstream strict-parser results remain a compute-side validation gate. A real frozen-reference first-gene experiment below establishes the namespace failure mechanism, not a claim that all 50 raw genes already pass strict parsing. Duplicate-isoform author policy remains unresolved and separate.

No implementation, reference/label/denominator change, remote write, job submission, cancellation, merge, push, or subagent was performed. Training/P1 source `b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6` and job 3163986 were not changed or revalidated here.

## Evidence scope and exact source

All source spans below refer to **b660**, not main, unless explicitly marked CPU wrapper. Export: `source_b660/source/`, relative to this report. It was unpacked from the reviewed `final_release_bindfix1/source.tar`; missing historical creation scripts were exported with `git show b660bf1:<path>`. `source_b660_verification.json` records SHA256 for **96 Python files**, each independently compared byte-for-byte to the b660 Git blob. Graft was built/queried in this export before source investigation. Graft caches and its export-local ignore configuration are not production source changes.

Existing observations remain in `RESULT_REVIEW.md` and `result_watch/metadata/outputs/job-6165559/`:

- Payload completed; all 50 EST inputs had biological nonzero support. Allocation/batch COMPLETED 0:0 does not resolve extern OUT_OF_MEMORY 0:125.
- `reference_compatibility.json`: `panel_strict_parser={genes:0,ok:true,transcripts:0}`, compatible loci 0/50. First failure: `reference missing panel gene Ath014705`; gene geometry `(Chr3,11921197,11924574,-)` is 0-based half-open.
- `reference.panel.gff3`: exactly 16 bytes, `##gff-version 3\n`, SHA256 `897007ceaf55680e4fa9d1d9e6ebac641ede633c0a9f2426db36ecbc66177df9`.
- Actual request: corpus `/data/gpfs/assoc/pgl/data/Transgenic/training_input/db/b5_full_a40_v1.db`; reference `/data/gpfs/assoc/pgl/data/Transgenic/genomes/Athaliana_167_TAIR10.gene.clean.gff3`.
- Compute-produced full reference SHA256 `529510b60756c9a1ae5a590ec4097fd4b9773649c29e6a5543553118bb042c38`, size 60,503,119. Full FASTA SHA256 `63d56962ccbbdcc974950e5ccdc3e1d3d37a665f7161f658edcd3d19e9c63973`.
- Frozen authority: `92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`; manifest SHA256 `3ca7a3b9a5793f62d25f503c93cfd7167eb17f7c5b90de7830e6a8777117f783`. Athaliana GFF MD5 `0264c349bc61ccceb546648edec46ebb`, FASTA MD5 `513e5ef30845ed754b00816a99abbf8e`. Locally read immutable Git row agrees with actual request.

## Exact origin and loss of the mapping

| Frozen file:span | Contract |
|---|---|
| `src/transgenic/utils/gsf_contract.py:89–104` | `species_code('Athaliana') == 'Ath'`. `gene_key` takes ID first, then Name only if ID is empty. A candidate longer than 10 characters or with more than one dot becomes species code plus six-digit **ordinal in GFF order**. An invalid long ID does NOT fall back to a shorter Name. |
| `src/transgenic/utils/gsf_contract.py:157–200` | `parse_gff3` increments the counter for each gene, generates the builder key, and retains original `ID` and `Name` separately in `Gene`. Geometry converts `(start,end)` to `(start-1,end)`. |
| `src/transgenic/datasets/build_b5.py:146–163` | Defines `gene_key_map(species_id,gene_id,gene_id_original,name_original,chromosome,start0,end0)` and reference hashes in `build_manifest`. The map has no strand column. |
| `src/transgenic/datasets/build_b5.py:291–304,499–526` | Both gene and tile build paths consume this parser and persist the original mapping. In tile-v3, mapping is stored before QC/transcript changes. Do not infer locus identity from tile `geneList.gene_id` or window boundaries: use `gene_key_map`. |
| `src/transgenic/datasets/preprocess.py:99–122` | Legacy `genome2GSFDataset` delegates to `build_species`; not another canonical ID authority for this corpus. |
| `revision/scripts/71_b6_panels.py:230–260` | Loads `build_manifest` and joins `gene_key_map` with `gene_split`, including both builder ID and original ID. Checks map/join cardinality. |
| `revision/scripts/71_b6_panels.py:281–300,349–361,392–409` | Hash-checks source GFF against corpus manifest. Looks up raw genes by **g[1] = gene_id_original**, verifies chrom/start/end, takes strand from raw GFF, then publishes **g[0] = builder key** as panel `gene_id`. This is intentional, not broken panel creation. |
| `revision/scripts/72_mvp_select_loci.py:101–142` | Filters/labels eligible rows by length and EST support; preserves input `gene_id` and geometry. No translation to raw GFF ID. |
| `src/transgenic/evidence/mvp_windows.py:94–136` | Hash-pinned panel loader preserves the panel key and interprets panel start/fin as 1-based inclusive. |
| `revision/scripts/mvp/mvp_produce_counts.py:418–430` | **Fault:** sends `{l.gene_id}` straight to raw-GFF restrictor and then calls reference selection with the same builder key. |
| `revision/scripts/mvp/mvp_produce_counts.py:284–294` | **Second necessary fix site:** reference selector requires `locus.gene_id in genes` and indexes that ID before exact geometry validation. Mapping only restriction would still fail here. Prediction selection is overlap-based and should not change. |
| CPU wrapper `final_release_bindfix1/run_panel_cpu.py:76–118` | Repeats the same unconverted restriction and reference selection. Must use the same future helper as count production, including its fallback per-locus path. |

`standardize_gff.py` is not a justification to transform this reference: `scripts/standardize_gff.py:804–829` renames IDs to `<prefix>_g000001` and rewrites Parent relationships. That is neither the builder's `Ath000001` rule nor the actual raw IDs observed here. No standardization or reference-preprocessing rerun is appropriate.

## Bounded raw-reference observation

Retained artifacts: `reference_prefix_observation.json` (exact SSH argv, stat, stderr and prefix hash), `reference_prefix_128KiB.bin` (131,072 bytes only). Remote command body:

```python
import os,json,base64,stat
p='/data/gpfs/assoc/pgl/data/Transgenic/genomes/Athaliana_167_TAIR10.gene.clean.gff3'
s=os.stat(p)
assert stat.S_ISREG(s.st_mode)
with open(p,'rb') as f: data=f.read(131072)
print(json.dumps({'path':p,'size':s.st_size,'mtime':s.st_mtime,
 'prefix_bytes':len(data),'base64':base64.b64encode(data).decode()}))
```

Executed through `ssh -o BatchMode=yes -o ConnectTimeout=20 pronghorn '/usr/bin/python -c ...'`. Stat size 60,503,119; mtime 1765586389.1804478. Prefix SHA256 computed **locally**: `ee62582867993bb5a31a5f665027fbd93f1331e51729ebe4f0faa5ac967cf19c`. Initial attempt used `s.st_mtime_ns` and returned exit 1 before metadata was recovered; retry used portable `s.st_mtime`. Each attempt was bounded to the same first 128 KiB, never a later range. Initial stderr was not retained, so its precise exception is not asserted.

First actual complete records:

```text
##gff-version 3
##annot-version TAIR10
ChrM phytozomev10 gene 273 734 . - . ID=ATMG00010.TAIR10;Name=ATMG00010
ChrM phytozomev10 mRNA 273 734 . - . ID=ATMG00010.1.TAIR10;Parent=ATMG00010.TAIR10;Name=ATMG00010.1;longest=1;pacid=19643678
ChrM phytozomev10 exon 273 734 . - . ID=nbis-exon-207326;Parent=ATMG00010.1.TAIR10;pacid=19643678
ChrM phytozomev10 CDS 273 734 . - 0 ID=ATMG00010.1.TAIR10.CDS.1;Parent=ATMG00010.1.TAIR10;pacid=19643678
```

Whitespace above is for display; binary artifact preserves tabs. Next gene ID is `ATMG00030.TAIR10`, Name `ATMG00030`. These are mitochondrial prefix examples, **not** any nuclear panel locus. No whole-reference scan/hash, remote DuckDB connection/query, or raw full-reference download was made. The prefix stat is not a new full-byte identity verification; full identity evidence comes from job6165559.

## Local mechanism check, not a fix

Ran unchanged b660 `gc.parse_gff3`, restrictor, and `p.read_gff` on the **first complete actual gene block only**, in a local temporary directory. Python bytecode writing disabled. `reference_id_local_observations.json` retains results and first-block SHA256. Procedure:

```python
# source = original header + first complete gene and descendants, before next gene
[g] = list(gc.parse_gff3(source_text.splitlines(), species_code=gc.species_code('Athaliana')))
restrictor.ID_RE = re.compile(r'(?:^|;)ID=([^;\s]+)')
restrictor.PARENT_RE = re.compile(r'(?:^|;)Parent=([^;\s]+)')
# Run restrict(source,{g.gene_id},out) and restrict(source,{g.gene_id_original},out)
# Then run unchanged p.read_gff on each output.
```

Observed:

| Selection key | Builder/original result | restrict `(genes,tx,features)` | Output | Strict parser |
|---|---|---|---|---|
| internal | `Ath000001` | `(0,0,0)` | 16B | 0 genes |
| raw original | `ATMG00010.TAIR10` | `(1,1,2)` | 416B | 1 gene |

Builder also retained Name `ATMG00010` and geometry `(ChrM,272,734,-)`. No hardcoded guessed prefix conversion was used.

`revision/scripts/24_restrict_reference_to_test_genes.py:72–114` performs literal ID/Parent matching. Both current callers already tighten its regex to exclude whitespace/newlines. The prefix has canonical ID/Parent keys and supported feature types; using actual original ID works unchanged. Thus a field-key/parser/feature-filter failure is not needed to explain the empty subset, and changing the regex again would not repair namespace translation. `read_gff:261` intentionally permits header-only empty predictions; the missing-reference error is appropriately raised later. Preserve empty **prediction** support, but require complete **reference** mapping/inventory before evaluation.

Limits: this check does not certify all panel transcript types, ordering, exon geometry, multi-parent handling, or duplicate structures. Once IDs are resolved, any such errors must remain visible, not be repaired/deduplicated silently.

## Controller-ready minimum implementation plan — NOT executed

### 1. Isolate evaluation and retrieve the existing authority

Create a separate eval-only immutable branch/release from b660 after review; pin its Git SHA, operator entrypoint and source manifest separately from training. Leave queued/running P1 3163986, its source manifest, finalizer and eventual training consumed-code pin unchanged. Do not force training source and evaluation source to have the same Git HEAD or replace a training-source check with the new evaluation pin.

On an authorized **compute allocation**, use the existing accepted `candidate.corpus` read-only, after existing candidate/inventory hash checks (no WAL). Read exactly the Athaliana `build_manifest` row and the requested 50 map rows. Query shape:

```sql
SELECT species_id,gene_id,gene_id_original,name_original,chromosome,start0,end0
FROM gene_key_map
WHERE species_id = ? AND gene_id IN (/* 50 parameter placeholders */);
SELECT species_id,fasta_sha256,gff_sha256,build_version,window_policy
FROM build_manifest WHERE species_id = ?;
```

No production `geneList` label query, DB update, reference rebuild, or split redraw. Validate:

- Map is total over the exact ordered 50 panel IDs; no missing/duplicate source key, null/empty original ID, wrong species, or many-to-one target ID. Do not hide duplicates with dict construction or SQL DISTINCT.
- Corpus manifest full GFF/FASTA SHA256 matches the bytes verified by `frozen_reference`, whose MD5 authority stays pinned to 92d8020. Reject changed corpus, panel or reference hashes before using mappings.
- Map chromosome/start0/end0 equals panel `(chromosome,start-1,fin)` exactly. Raw gene identified by `gene_id_original` must then match those coordinates **and panel strand**. Strand comes from the raw gene, not an imagined map column.
- Persist a small mapping receipt containing both IDs, geometry, species, method `corpus_gene_key_map`, corpus/panel/reference hashes and evaluation code identity. Recheck input identities at publication. This receipt records a verified mapping; it is not a substitute for verifying its source.

### 2. One shared evaluation adapter, both callers

Add a small shared reference-preparation helper (in the eval producer or an adjacent eval-only module): resolve the validated mapping, restrict with **original raw IDs**, parse strict raw GFF, require exact raw reference gene inventory, and select transcripts by the mapped ID with unchanged exact panel geometry checks. Return per-panel-key transcript sets plus mapping/restriction receipt. Keep raw GFF ID/Parent text unchanged; do not rekey annotations or mutate `PanelLocus.gene_id`.

Replace producer `:418–430` and CPU wrapper `:86–109` duplicate preparation with this helper. Make mapped reference lookup explicit at `select_transcripts:284–292`, or have the helper own reference selection while leaving the prediction selector unchanged. CPU diagnostics must report both identifiers and distinguish mapping failure, restriction failure and strict parsing failure. Both callers must fail closed for missing reference coverage; an observation wrapper may still publish a clearly failed diagnostic, never decoder-ready/PASS.

Check restrictor counts and parsed inventory against all expected IDs before scoring. Do not weaken grammar/geometry checks or turn missing reference into an empty reference denominator.

### 3. Preserve decoder and publication contracts

Keep frozen TSV bytes, ordered 50 keys, core20 membership, 30,720nt windows, seed/fraction keys and **350 identities** unchanged: 3 seeds × 2 fractions × 50 plus baseline seed123/off × 50. Decoder `mvp_decode_panel.py:123–158` and producer `prediction_inventory:349–367` continue using internal panel IDs. Add raw reference ID only in evaluation provenance. No changes to EST channels, heldout rules, labels, scoring denominator or duplicate policy.

**Important deployment coupling:** panel-input `preflight` (`mvp_prepare_panel_inputs.py:30–45`) hashes `mvp_produce_counts.py`; `mvp_verify_panel_inputs.py:29–39` compares exact inventory and live source identities. Editing the producer means old job6165559 panel publication cannot simply be declared valid under the new evaluation source. Do not relax this gate or overwrite its manifest/COMPLETE. Preferred straightforward rollout: publish a fresh immutable panel-input/CPU-audit namespace under the new eval pin and pass the unchanged verifier, retaining old failure evidence. If controller instead wants reuse of existing arrays, it needs a separately reviewed explicit provenance bridge retaining original producing source identity; that is additional work, not an implicit exception. Actual decoder NumPy/DuckDB equality also remains a separate gate.

No decoder implementation change is required merely to rename reference lookup. If a mapping sidecar is passed through a request, extend evaluation-only transport/provenance explicitly; do not silently assume `mvp_decode_panel.py:173–174` forwards unknown fields.

### 4. Tests before code, then bounded real-data gate

Add focused tests around the shared helper and both call sites, using actual-prefix-derived small fixtures plus synthetic adversarial maps:

1. Failing regression: long GFF ID + short Name yields generated internal key; old internal-ID restriction gives header only. New shared path resolves via explicit stored original ID and selects exact same raw transcripts.
2. An already-valid raw ID case is identity-mapped; no special `Ath` handling. Include counter/order cases to prevent magic ordinal/prefix reconstruction.
3. Missing map row, duplicate source row, reused target ID, wrong species, null ID and duplicate raw gene ID reject before scoring.
4. Corpus/build-manifest/raw-reference/panel hash mismatch rejects. Exact coordinate or strand mismatch rejects (including off-by-one, adjacent and nearest-coordinate candidates).
5. Reference subset must cover all panel IDs; empty prediction remains legal. Malformed/orphan/unsupported reference features still reject; duplicate isoforms remain reported and unmodified.
6. CPU and producer use the same helper and agree on mapping, transcript inventory and failure categories. Exercise the CPU per-locus diagnostic fallback too.
7. Assert frozen panel hash, ordered keys, windows, 350 prediction key set and original GFF bytes unchanged. Assert training/finalizer code pin unchanged; assert stale panel publication is not accepted after producing-code changes.
8. After independent local review, authorize a new bounded compute-only compatibility run. Record all 50 exact ID pairs, 50/50 map/geometry coverage, actual transcript inventories, remaining strict parser/duplicate findings and source/input hashes. Mapping success alone is not scientific approval or resolution of duplicate policy.

### Coordinate bridge contingency — not the preferred fix

An existing explicit map has been found, so a coordinate-only bridge is unnecessary here. If the accepted corpus unexpectedly lacks it, STOP and report that contradiction first. A future reviewed fallback could match only **unique exact `(chrom,start0,end0,strand)`** across the *same hash-bound frozen GFF* and frozen panel, because panel creation sourced geometry/strand from the corresponding raw gene after corpus/reference agreement. It must establish those provenance bindings, one-to-one total coverage and uniqueness over the full relevant reference population, reject ambiguous/coincident genes and all mismatches, and record that coordinates rather than explicit aliases supplied identity. Never use nearest/overlap matches, strip `.TAIR10`, derive an AT ID from Ath digits, recount ordinals on a subset, or choose the first candidate. This investigation has not validated such a fallback for 50 loci.

## Commands and remaining boundaries

Local context/identity operations: `graft map`; `graft ask` for producer/restrictor/panel; build/query graft inside exact b660 export; `graft skeleton`/`graft grep gene_id_original`/`graft callers select_transcripts`; `git rev-parse b660bf1`; `git show b660bf1:<creation-script>`; byte comparison of exported Python files to b660 blobs; `git show 92d8020:data/manifests/b5_species_v1.tsv`; local prefix replay above. Retained JSON files carry source hashes, raw remote command and observations. No whole DB/reference was read on login. Advisor consultation failed with client/model-version incompatibility; it supplied no advice or review approval.

**Next controller action:** review this diagnosis, then authorize the small shared eval mapping adapter and its tests on a new eval-only pin. Do not retry the unchanged job: it deterministically supplies the wrong namespace. Do not treat this report as authorization to launch or change scientific policy.
