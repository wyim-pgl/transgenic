# Evidence scorer readings (issue #24)

No real addition/evidence overlap has been inspected. Protocol §§3–9 and all evidence/driver files are unchanged.

## Decided by the author — 2026-09-05

Both decisions were supplied by the manuscript author in the issue #24 continuation instruction on **2026-09-05**. The **primary was declared before any real addition was scored**; this implementation and its tests inspect synthetic fixtures only.

1. **Complete-chain unit threshold: report both readings, with the primary declared now.**
   - **Primary — `chain_support_single_read`:** one accepted read satisfying §6's corrected whole-chain predicate gives complete support. No §5 constituent-junction threshold is added to this predicate.
   - **Sensitivity — `chain_support_threshold`:** whole-chain witnesses themselves must reach the source unit threshold: ONT ≥3, PacBio ≥2, EST/FL-cDNA ≥2 A11 units (distinct clone/accession identities, subject to the existing fallback and cap).
   - **Checked justification:** `PROTOCOL_B1_frozen_v1.md` §6 (line 150) says “An evidence read gives complete intron-chain support”; §8 (line 178) defines T1 through a tier-1 read, singular; A18.2's ONT support-threshold cell (line 249) explicitly says “§5 counts read as units.” §5 is junction calling. A16 (line 618) changes ONT counting to independent units and repeats the thresholds, but does not explicitly attach them to the §6 witness predicate. A17's chain discussion (line 225) concerns training evidence and does not impose a numerical B1 chain threshold. No direct contradiction to the author's reading was found in these passages.
   - **Why paired:** A37 item 3 (line 749) requires both EST arms wherever an EST-derived quantity appears; workspace `quarantine.md` §1e records why outcome-selected A36 was withdrawn. The same paired-reporting pattern is adopted here so the chain reading cannot be chosen after seeing the rates. Both readings accompany every chain-support result, tier, S12 input and rate, crossed with both A37 arms. Differing tiers are explicitly flagged.
2. **Non-transitive ONT PCR equivalence: retain the stop.** Only mutually-within-10-nt cliques form units. A same-library/strand/corrected-chain path with endpoints at 0, 9 and 18 nt is not a clique; connected-component merging or greedy partitioning would invent a linkage rule. The author adopts `RuleUnresolved` for this configuration. A non-raising pre-scoring diagnostic counts all candidate components and non-cliques globally, per run and per addition; counting components for diagnosis does not make them molecule units.

## Status

This remains a **provisional implementation, not a completed issue #24 or a production B1 scorer**. The `purpose=b1` author-decision block is removed. Existing input hashes, orientation audits, role/QC/control checks and PCR-clique stops remain. The incomplete-work list below is now also an explicit B1 refusal gate, not a claim of completion. Synthetic and C0 diagnostic scoring report both readings; differing chain readings no longer stop scoring. No real primary outcome has been computed and no junction threshold has changed.

## Explicit provisional readings for review

- **§5 coordinates/strand:** intervals are 1-based inclusive genomic low/high coordinates, also on minus-strand transcripts. Chains are sorted in genomic order and compared only on the same strand (equivalent to reversing both into transcription order). `ts` is composed with FLAG 0x10 per A39. Audited `-uf` permits read-strand fallback; contradictory `ts:-` stops. Without either strand source, only a separate strand-ambiguous coverage flag is populated, not a same-strand denominator.
- **§4 acceptance:** the entire read group is examined before its primary is retained. Any supplementary record or SA tag rejects the read; multiple primaries, cross-contig/cross-strand primaries and observed equal-best secondary placements reject it. Secondary records do not contribute observations. MAPQ is 20. A suppressed secondary placement cannot be reconstructed from raw BAM; C0 must supply an explicit `mapping_ambiguous` audit state.
- **§5 anchors:** nearest 20 (ONT/PacBio) or 15 (EST/FL-cDNA) aligned exonic bases, at most 2/1 mismatches. Another intron or the alignment end cannot supply an anchor. Indels within the nearest 6/8 aligned bases fail; an indel beyond that zone does not itself fail. Deleted bases never supply anchor length. This is a provisional operational reading of the unspecified treatment of gaps outside the exclusion zone.
- **§5 correction:** enumerate donor and acceptor offsets independently in the frozen radius, including zero. Exactly one canonical coordinate pair permits correction; multiple pairs reject even if the raw pair is canonical. No canonical candidate preserves a flagged raw noncanonical junction, implementing “non-canonical predicted introns … scored with the same rules.” EST coordinates are never corrected. If the intended neighborhood is a joint equal-offset shift rather than independent endpoint offsets, the author must overrule this before use. Anchor QC is on the observed alignment; corrections do not fabricate new aligned bases.
- **§6 chain identity:** all introns intersecting the CDS span remain in the comparison, including failed/ambiguous calls; removing a failed extra intron must not manufacture a chain. Terminal chain anchors need aligned blocks extending 20 nt beyond the corrected boundaries. Introns outside the CDS span do not disqualify a read. Terminal completeness codes do not gate chain support. Linked clone reads are never concatenated into a whole-chain witness or an unsequenced adjacency.
- **§7 spanning:** callability uses the accepted alignment's outer genomic extent regardless of its splice structure, as stated by “any structure” and “irrespective of splicing.” Junction-level anchor failures do not remove an otherwise accepted read from this denominator. Read-level rejection does. Each novel junction may have a different callable read.
- **§7 empty chains:** mono-exonic additions have no defined first donor/last acceptor, so `chain_applicable=false`, neither chain nor novel-junction callable, but remain in the frozen all-addition denominator. No novel junction means `novel_junction_support=N/A` and no junction-callable denominator, rather than vacuous truth.
- **§8 union:** apply thresholds within each source first; union supported features afterward. One ONT unit plus one PacBio unit cannot satisfy a source threshold. Callability is the union of eligible observations and may span different sources for different novel junctions. Run-level counts and pooled counts are both emitted. Union independence is a list of states; declared source independence is retained even for uncovered additions.
- **§8 partial/tier:** `partial` requires at least one threshold-supported junction, not just one raw junction observation. Combination-novel constituent-only support is T4, never T3 or complete. Unsupported callable rows have explicit negative flags; all non-callable negative flags are null. All-addition rate denominators do not turn their non-callable failures into negative evidence.
- **A11:** newest parsed accession version is retained for counting; clone normalization removes punctuation/case and separated read-direction suffixes. Clone keys include biological library. With no clone, the exact contig/strand/CIGAR block signature is the counting key within library. Ten-unit cap is applied per feature/library. The cap also applies to sensitivity whole-chain unit counts; the primary requires one accepted whole-chain read. Clone/library aliases must already be resolved in C0 metadata; the scorer does not guess them.
- **A16/A18.2:** library/sample/UMI/locus/strand identifies UMI units; direct RNA uses run/read identity; PacBio requires source FLNC/ZMW identity and cannot use polished-cluster IDs. Unknown ONT UMI/protocol metadata stops, rather than asserting independence. Endpoint PCR components are accepted only when all members are mutually within 10 nt; non-cliques stop. This is a detection gate, not a new clustering rule.
- **A18.5/genotypes:** training-role observations apply aligned fraction 0.80, identity 0.95 (ONT/EST) or 0.98 (PacBio) and canonical junction acceptance. EST validation retains A8's fraction/divergence checks. Maize nonreference/pooled/unknown observations require the A7 divergence and competing-paralog metadata. Other B1 validation long reads do not acquire A18.5 training-only filters. Genotype labels use `reference`, `known_nonreference`, `hybrid_pooled`, `unknown`, plus a separately labelled `species_union` output; no genotype weights modify validation rates.
- **§9–§10:** frozen addition/filter/reference-match fields are input data, not recomputed from evidence. Controls are scored before additions. Wilson intervals use the ordinary two-sided 95% binomial formula; zero denominator yields null rate and interval. No full-length expression or experimental-validation claim is emitted.

## Draft input and output contract

Run `59_evidence_support.py --config CONFIG.json --out NEW_DIRECTORY` with Python 3.10+ and samtools on PATH for BAM input. Synthetic SAM files need no samtools. PAF-only ingestion is deliberately rejected: it cannot establish the required SAM-flag exclusions. `--cs=long` BAM tags are consumed directly; no sam2paf conversion is used.

The JSON config has `schema: "b1-evidence-v1"`, `purpose` (`synthetic`, `c0_diagnostic`, or `b1`, which still refuses on the incomplete-work gate), `assembly`, `genome`, `role_manifests` (list), `additions`, and `runs`. File entries have `path` (relative to config) and `sha256`. Additions are a JSON list of `Addition` dataclass fields; intron lists are genomic low/high pairs. The synthetic CLI test constructs a complete minimal example.

Every run names `dataset`, `run`, `species`, `assembly`, `source`, `genotype_stratum`, boolean `uf`, both independence fields, `status: "complete"`, hashed `alignment` and hashed `metadata`. EST/FL-cDNA runs require both `arm: "primary"` and `arm: "min121"`. Nonsynthetic runs additionally require hashed DONE/provenance, and ONT orientation-audit JSON. Metadata is a JSON mapping from read ID to biological library, BioProject, genotype, explicit ingestion-QC and mapping-ambiguity states, plus applicable molecule-unit fields. EST metadata additionally carries post-trim length and optional accession/clone. Upstream ingestion QC is an explicit prerequisite covering vector/masking/contamination/paralog checks; it is not inferred from an existing DONE marker.

Successful diagnostic scoring emits `per_addition.csv`, `S12_inputs.csv`, `S12c_runs.csv`, `C0_observations.jsonl`, `PCR_diagnostic_runs.csv`, `PCR_diagnostic_additions.csv`, `PROVENANCE.json`, and a hash-bound `DONE`.

`per_addition.csv` includes `chain_support_single_read` and `chain_support_threshold`, plus `chain_reading_report` holding both labelled tiers, complete sources, high-confidence sources and callable-negative flags. There is no unqualified `tier` or `chain_support` result. `tier_disagreement` flags a difference within each addition/arm/genotype/scope. S12 rates include both `chain_reading` and `chain_reading_label` for **every** metric (even unchanged junction and exact-CDS rates), crossed with A37's `arm`. Raw witness/unit counts remain shared measurements accompanied by both support readings.

PCR diagnostics count connected candidate groups with the exact same enumeration used by molecule assignment. Per-run counts include global groups containing any member of that run; per-addition counts include groups with any same-strand overlapping member. This detects cross-run non-transitivity too. Such counts are not additive across runs or additions. Zero rows are emitted for unaffected runs/additions. Only accepted, known non-UMI ONT cDNA observations enter these groups. Unknown protocol/UMI metadata still fails molecule assignment; zero detected non-cliques is not clearance of other gates. The pass counts without raising or assigning units.

A PCR stop or the B1 incomplete-work gate preserves **only** the two diagnostic CSVs and `PROVENANCE.json` with `status=scoring_refused`, the reason, counts, hashes and incomplete-work list. It emits no scores and no `DONE`, and the CLI exits 2. Counts are also in provenance on successful diagnostic scoring. The provenance hashes config, code, declared inputs and outputs. Neither the filesystem location of inputs nor a manifest role substitutes for an input hash comparison. An existing output directory is refused. Hash mismatch and earlier input/audit/QC failures leave no output directory; the two documented post-ingestion refusals retain diagnostics only.

## Work remaining after the author decisions — all still incomplete

These are implementation limits, not amendments or interpretations adopted by default:

- Replace the provisional in-memory read grouping/observation collection with disk-backed indexing before hundreds-of-GB production use. This draft has only been run on synthetic fixtures.
- Finish manifest-wide expected-run reconciliation, frozen-reference identity binding, per-species A4 attestation, P4 seed/sample verification and P5 +9-nt decoy correspondence validation. Existing draft gates are not a substitute for those checks.
- Complete A37 per-length-bin callability/junction/intron-count/gene-length reporting. Both arms are required and gene length/intron count are exported, but length-bin tables are not implemented yet.
- Complete S12 P2 alternative-recall inputs and retained-match counts; A7 labelled MAPQ10/placement sensitivity outputs. No absent result may be reported as zero. The current provenance explicitly names missing outputs.
- Validate C0 normalization for accession-version aliases across runs, conflicting clone/library metadata, source-molecule provenance, and complete S12d terminal-completeness inputs. No real read-level metadata was invented or populated in this task.

## Verification

Synthetic-only command, from `/data/gpfs/assoc/pgl/data/Transgenic` (uv supplies pytest without changing the shared conda environment):

```sh
/data/gpfs/home/wyim/.local/bin/uv run --no-project --with pytest --python /data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python python -m pytest -q transgenic/revision/scripts/tests/test_evidence_support.py
```

Coverage includes single-witness primary support without threshold-supported constituents for all four sources; sensitivity thresholds and union non-pooling; tier disagreement; both readings crossed with A37 arms in rates; non-clique counts, cross-run membership, zero counts and permutation invariance; retained diagnostic outputs on refusal; and B1 QC, orientation and incomplete-work gates.

Output:

```text
........................................... [100%]
43 passed in 0.52s
```

`git diff --check` reported no errors. Only the scorer, its synthetic tests and this readings document are modified. No real evidence was scored. No commit is made; evidence/, *.sbatch and frozen §§3–9 text are unchanged.
