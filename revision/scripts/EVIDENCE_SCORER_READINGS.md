# Evidence scorer readings (issue #24)

No real addition/evidence overlap has been inspected. Protocol §§3–9 and all evidence/driver files are unchanged.

## Author decisions pending — do not select a numerical reading

1. **Complete-chain unit threshold (§§5–6, §8, A16, A18.2).** §5 places minimum support in the junction-calling table; §6 says one read witnesses a complete chain and §8 derives T1/T2 from a read. A16 refers to independent-unit support thresholds. These permit different numerical implementations: one full-chain witness plus threshold-supported constituent junctions, or the source threshold on full-chain witnesses themselves. Example: one ONT read spans two junctions, each independently seen on two additional truncated reads. The former passes and the latter fails. Pending author decision; neither interpretation may become a primary outcome by default.
2. **Non-transitive ONT PCR equivalence (A16/A18.2).** Three same-library, same-chain reads with both endpoints offset by 0, 9 and 18 nt satisfy the 10-nt rule pairwise along a path, but the endpoints of that path do not. Connected components merge ends 18 nt apart; greedy or complete-linkage partitioning chooses which pair counts together. No linkage/tie rule is frozen for this 10-nt unit (design §3's 25-nt model-end clustering is a different operation). Stop on this configuration rather than choosing a partition or treating ambiguous units as negatives.

## Status

This is a **provisional implementation, not a completed issue #24 or a production B1 scorer**. The `purpose=b1` entry point raises `RuleUnresolved` before scoring. Synthetic and C0 diagnostic execution are available for review. Where the two chain-threshold readings above give different answers, diagnostic scoring also stops. No primary outcome has been computed and no threshold has been changed. The author questions were sent before implementing the affected scoring path.

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
- **A11:** newest parsed accession version is retained for counting; clone normalization removes punctuation/case and separated read-direction suffixes. Clone keys include biological library. With no clone, the exact contig/strand/CIGAR block signature is the counting key within library. Ten-unit cap is applied per feature/library. For the provisional whole-chain counts the cap also applies to chain witnesses; this cannot settle the pending chain-threshold reading. Clone/library aliases must already be resolved in C0 metadata; the scorer does not guess them.
- **A16/A18.2:** library/sample/UMI/locus/strand identifies UMI units; direct RNA uses run/read identity; PacBio requires source FLNC/ZMW identity and cannot use polished-cluster IDs. Unknown ONT UMI/protocol metadata stops, rather than asserting independence. Endpoint PCR components are accepted only when all members are mutually within 10 nt; non-cliques stop. This is a detection gate, not a new clustering rule.
- **A18.5/genotypes:** training-role observations apply aligned fraction 0.80, identity 0.95 (ONT/EST) or 0.98 (PacBio) and canonical junction acceptance. EST validation retains A8's fraction/divergence checks. Maize nonreference/pooled/unknown observations require the A7 divergence and competing-paralog metadata. Other B1 validation long reads do not acquire A18.5 training-only filters. Genotype labels use `reference`, `known_nonreference`, `hybrid_pooled`, `unknown`, plus a separately labelled `species_union` output; no genotype weights modify validation rates.
- **§9–§10:** frozen addition/filter/reference-match fields are input data, not recomputed from evidence. Controls are scored before additions. Wilson intervals use the ordinary two-sided 95% binomial formula; zero denominator yields null rate and interval. No full-length expression or experimental-validation claim is emitted.

## Draft input and output contract

Run `59_evidence_support.py --config CONFIG.json --out NEW_DIRECTORY` with Python 3.10+ and samtools on PATH for BAM input. Synthetic SAM files need no samtools. PAF-only ingestion is deliberately rejected: it cannot establish the required SAM-flag exclusions. `--cs=long` BAM tags are consumed directly; no sam2paf conversion is used.

The JSON config has `schema: "b1-evidence-v1"`, `purpose` (`synthetic`, `c0_diagnostic`, or currently blocked `b1`), `assembly`, `genome`, `role_manifests` (list), `additions`, and `runs`. File entries have `path` (relative to config) and `sha256`. Additions are a JSON list of `Addition` dataclass fields; intron lists are genomic low/high pairs. The synthetic CLI test constructs a complete minimal example.

Every run names `dataset`, `run`, `species`, `assembly`, `source`, `genotype_stratum`, boolean `uf`, both independence fields, `status: "complete"`, hashed `alignment` and hashed `metadata`. EST/FL-cDNA runs require both `arm: "primary"` and `arm: "min121"`. Nonsynthetic runs additionally require hashed DONE/provenance, and ONT orientation-audit JSON. Metadata is a JSON mapping from read ID to biological library, BioProject, genotype, explicit ingestion-QC and mapping-ambiguity states, plus applicable molecule-unit fields. EST metadata additionally carries post-trim length and optional accession/clone. Upstream ingestion QC is an explicit prerequisite covering vector/masking/contamination/paralog checks; it is not inferred from an existing DONE marker.

Successful diagnostic execution emits `per_addition.csv`, `S12_inputs.csv`, `S12c_runs.csv`, `C0_observations.jsonl`, `PROVENANCE.json`, and a hash-bound `DONE`. The provenance hashes config, code, declared inputs and outputs. Neither the filesystem location of inputs nor a manifest role substitutes for an input hash comparison. An existing output directory is refused. Hash mismatch or an unresolved rule leaves no result directory.

## Work remaining after the author decisions

These are implementation limits, not amendments or interpretations adopted by default:

- Replace the provisional in-memory read grouping/observation collection with disk-backed indexing before hundreds-of-GB production use. This draft has only been run on synthetic fixtures.
- Finish manifest-wide expected-run reconciliation, frozen-reference identity binding, per-species A4 attestation, P4 seed/sample verification and P5 +9-nt decoy correspondence validation. Existing draft gates are not a substitute for those checks.
- Complete A37 per-length-bin callability/junction/intron-count/gene-length reporting. Both arms are required and gene length/intron count are exported, but length-bin tables are not implemented yet.
- Complete S12 P2 alternative-recall inputs and retained-match counts; A7 labelled MAPQ10/placement sensitivity outputs. No absent result may be reported as zero. The current provenance explicitly names missing outputs.
- Validate C0 normalization for accession-version aliases across runs, conflicting clone/library metadata, source-molecule provenance, and complete S12d terminal-completeness inputs. No real read-level metadata was invented or populated in this task.

## Verification

Command (uv supplies pytest without changing the shared conda environment):

```sh
/data/gpfs/home/wyim/.local/bin/uv run --no-project --with pytest --python /data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python python -m pytest -q transgenic/revision/scripts/tests/test_evidence_support.py
```

Output:

```text
................................ [100%]
32 passed in 1.56s
```

`git diff --check` reported no errors. Only the three requested files are new/untracked; no commit was made.
