# B6 in plain language — teaching TransGenic to use transcript evidence, without giving up de novo annotation

한글판: [`B6_WORKPLAN_README.ko.md`](B6_WORKPLAN_README.ko.md). This page explains the B6 plan for anyone in the lab. The exact, frozen specification lives in
[`B6_EVIDENCE_MODE_WORKPLAN_v1.md`](B6_EVIDENCE_MODE_WORKPLAN_v1.md) (v1.2, reviewed three times by Codex and once
by Kimi K3 on 2026-09-14). Nothing in B6 changes the seed-123 run that is training now.

## 1. Why we are doing this (one paragraph)

Today TransGenic learns only from the reference annotation of nine plant genomes: it reads DNA and writes gene
structures. Tools such as BRAKER, MAKER or PASA also use transcript evidence (ESTs, long reads) and proteins, so a
comparison between them and a model that never saw evidence is not a fair one. The author's decision: the model
must be able to learn from, and use, that evidence. At the same time we do not want to lose what makes TransGenic
special, which is annotating from sequence alone. So B6 builds **one model with two modes**.

## 2. The idea in pictures

```
              DNA sequence ─────────────┐
                                        ├──► HyenaDNA encoder ──► GSF decoder ──► gene structures (text)
  evidence tracks (may be all zero) ────┘
```

- **Evidence tracks**: for every nucleotide, a few numbers that say "an EST/long read covers this base",
  "a splice donor/acceptor is supported here by N molecules", "a protein aligns here". Fragments count exactly
  for the bases they cover; nothing has to be assembled into a full transcript. (This is why we dropped the earlier
  idea of adding EST-derived isoforms to the labels: more than 90 % of ESTs are fragments.)
- **de novo mode**: the tracks are all zero. The model behaves like today's TransGenic.
- **evidence mode**: the tracks are filled from the user's aligned ESTs, long reads and proteins.
- **Training trick**: during training we randomly zero the tracks for half of the windows and randomly thin the
  evidence in the other half (keeping 1 %, 10 %, 50 % or 100 % of the molecules). One set of weights therefore
  learns to work with no evidence, little evidence and full evidence.
- What we predict (the GSF text) and how we decode it (grammar constraints, tiling, stitching) do not change.

## 3. What we will be able to say afterwards

1. Turning evidence off does not make the model worse than a model that never saw evidence (gate G1).
2. Turning evidence on makes annotation better, and we can show how much better as a function of how much
   evidence is supplied, measured at the same loci (the "dose-response" curve; gate G2).
3. Reference-prompted completion (the paper's main mode) works in both modes.

We will NOT claim that B6 is "the same run as B5 plus evidence": the architecture and the random streams differ,
so a dedicated control run (same model, evidence permanently off) is part of the plan.

## 4. Which evidence goes where

| Data | Used for training? | Used for scoring the results? |
|---|---|---|
| ESTs, ONT, PacBio (Sequel II+), proteins of the nine training species (incl. *A. thaliana*) | yes | *A. thaliana* ESTs only at held-out loci, labelled "in-domain" |
| *Z. mays* and *S. lycopersicum* (withheld species) | never | yes (maize is the primary validation species) |
| *A. thaliana* validation-only long reads (A-ONT1, A-HiFi) | never | yes (the independent tier) |

Two rules protect the results from circularity:
- **No leakage into training.** Any EST/read/protein that touches a held-out locus, a test block, a test orthogroup
  gene or a masked gene — anywhere in the genome, with any of its alignments — is removed from the training tracks
  as a whole molecule. This is checked with a separate "audit" alignment that keeps secondary placements, and the
  counts go into a QC table (the number retained at held-out loci must be zero).
- **Input evidence ≠ scoring evidence.** At every validation locus the evidence we feed the model and the evidence we
  score it against are different, declared sets (for maize: input = ESTs + one long-read set, scoring = the other
  long-read sets). Otherwise the model would be scored against what it was just shown.

## 5. The order of work (and what each step proves)

| Step | What | Roughly | Proves |
|---|---|---|---|
| **MVP pilot** (1 week, exploratory) | *A. thaliana* only, ~20 held-out loci, evidence channel wired onto the current seed-123 checkpoint, short fine-tune, evidence on vs off, 3 seeds (~20 GPU-h) | week 1 | does the channel help at all? If not, stop here. |
| WP0 | Write amendment A46 with every value fixed (seeds, hashes, commands, locus panels, thresholds); author approves | 1 day | nothing is built before the rules are frozen |
| WP1 | Evidence tables (C0) from the existing alignments: molecules, blocks, junctions, chains, roles, provenance | 4–5 days | also needed for the maize validation of the paper |
| WP2 | Leakage exclusion + repair of a masking gap at window edges; rebuild the training corpus | 3 days | QC table with zero leakage |
| WP3 | Evidence tracks, thinning, memory benchmark at the largest window (129 kb) | 2 days | the fused model fits in GPU memory |
| WP4 | Model change (one linear projection, no bias) + trainer + 4-GPU tests incl. decoding with cache | 2–3 days | training and decoding behave identically with tracks on/off |
| WP5 | Pilot on the pre-registered panel: B6 vs evidence-off control | 3–4 days | go/no-go on the gates |
| WP6 | Full runs (B6, evidence-off control; two short controls) | 1–2 weeks incl. queue | the reportable models |
| WP7 | Evaluation in both modes, dose-response, maize/A. thaliana validation | 1 week | the tables and figures |

Total ≈ 5–6 weeks after the MVP. GPU: with seed 123 as the sequence-only comparator (author decision 2026-09-14) the
full runs fit the current allocation with ≈ 50 GPU-hours to spare.

## 6. Decisions the author still has to make (before WP6)

1. ~~Top up ≈ 350 GPU-hours, or drop the rebuilt sequence-only control (C1') and use seed 123 as the comparator.~~ Decided 2026-09-14: seed 123 is the sequence-only comparator; C1' is not run.
2. Which model leads the abstract: B6 de novo mode, B6 evidence mode, or B5.
3. For maize: which long-read libraries are model input and which are scoring (proposal: input = M-EST + Wang 2018
   HQ isoforms; scoring = Wang 2020 FLNC + root-tip ONT).
4. Whether the 121-nt EST arm is also used as an input track (proposal: scoring tables only).

## 7. How to check progress

- Seed 123 (B5) status and the B6 steps are recorded in the run log and in this repository's protocol amendments.
- Every B6 step above has a verification column in the detailed plan; a step is "done" only when that check passed.
- Reviews: Codex rounds 1–3 and the Kimi K3 challenge are summarised at the top of the detailed plan; the raw
  transcripts are kept with the session notes.
