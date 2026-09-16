# MVP evaluation producer contract (Track B, controller review required)

Interface: `revision/scripts/mvp/mvp_produce_counts.py --input evaluation.json --out NEW_DIRECTORY`.
Python: `produce(evaluation_manifest_path, new_output_directory)` in that script.
Training does not wait for this evaluator. No GPU, alignment, fetching, or jobs here.

Decoder writes a JSON transport manifest, **not an approval or trusted count receipt**:

```json
{
  "panels": "/absolute/b6_panels_v1.tsv",
  "fasta": "/absolute/Athaliana_167_TAIR10.fa",
  "reference": "/absolute/Athaliana_167_TAIR10.gene.clean.gff3",
  "candidate": {"acceptance": "/absolute/acceptance.json", "acceptance_sha256": "...", "inventory": "/absolute/inventory.json", "receipts": "/absolute/receipts.json", "corpus": "/absolute/corpus.db", "evidence": "/absolute/c0lite.db"},
  "runs": {
    "1": {"final": "/absolute/mvp_seed1/final", "code_manifest": "/absolute/consumed_code_manifest.json"},
    "2": {"final": "/absolute/mvp_seed2/final", "code_manifest": "/absolute/consumed_code_manifest.json"},
    "3": {"final": "/absolute/mvp_seed3/final", "code_manifest": "/absolute/consumed_code_manifest.json"}
  },
  "decoder": {"files": {"/absolute/decoder.py": "sha256"}, "commands": [["python", "decoder.py", "..."]]},
  "predictions": [
    {"seed": 1, "fraction": 0, "gene_id": "EXACT_PANEL_ID", "window": ["Chr1", 10000, 40720], "path": "/absolute/locus.gff3", "sha256": "...", "size": 1234, "checkpoint_sha256": "..."}
  ]
}
```

Exactly 350 prediction rows: each frozen panel locus × seeds 1/2/3 × fractions
0/1, plus seed **123**, fraction **0** for each locus. Integers, not booleans or
strings. Window is chromosome, **0-based half-open** start/end from
`mvp_windows.evaluation_window`. GFF3 coordinates are **1-based inclusive,
genomic forward reference**, known genomic strands; gene/mRNA/exon/CDS hierarchy.
Empty header-only GFF3 is a completed zero-prediction decode, never a missing file.
Each row binds the actual checkpoint bytes: final `model.safetensors` for 1/2/3;
initialization audit's source model for 123. One independent file per row.

Final directories use the existing `mvp_finalize` five-payload `manifest.json`,
`TRAINING_DONE`, metadata, paired records/summary unchanged. No separate invented
seed receipt. Candidate records use existing CLI schemas and actual DB inventory
verification. Code manifests are those hashed in final metadata; their listed
source bytes must remain locally available. Initialization source files also
remain locally available. Decoder files/commands are recorded provenance, not
proof a command ran. Counts are derived afresh from raw GFF, never accepted from
this JSON.

Reference authority: repository commit
`92d8020ee79eeedd7f1d50167b1c8dc94700fb16:data/manifests/b5_species_v1.tsv`,
Athaliana row. Missing git object or mismatch fails closed. No caller reference
hash can replace this anchor. GFFCompare **0.12.6** must be on PATH. The producer
runs only local GFFCompare comparisons, records executable bytes/version, exact
commands and output bytes. It does not run alignments.

Outputs: `counts.json`, `statistics.json`, `receipt.json`, per-locus restricted
GTF and GFFCompare outputs. Independent read-splice P/R is **absent**, not negative,
until the separate alignment provenance/author coverage policy is resolved.
Final decisions remain blocked; withholding and HiFi policy are not inferred.
Duplicate structure ambiguity is rejected rather than silently introducing a
new isoform deduplication rule (see scoring report). Operational use requires
controller review and real GFFCompare smoke validation.
