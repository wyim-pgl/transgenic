# PROTOCOL M — Exploratory protocol: prompt-free and evidence-prompted completion, and an evidence-assisted candidate pool

**Version 1.0 — 2026-09-01. Status: exploratory, gated.** Separate from the frozen B1 validation protocol. Nothing here changes B1's primary outcomes. Scope decision (Codex-reviewed): the candidate-pool system is follow-up-paper material unless the oracle gate below passes within its one-week budget, in which case one paragraph and one supplementary table may enter the revision as "evidence-assisted candidate pool", never as a released whole-genome or prompt-free mode.

## 1. Questions
- **M(a)** Can prompt-free decoding plus evidence selection yield a usable isoform layer at EST-callable loci?
- **M(b)** Does the transformer add anything beyond an evidence-only reconstruction (EST splice-graph paths + ORF rule; PASA-lite) at the same loci and candidate budget?
- **M(c)** Can an EST-derived primary serve as the prompt ("evidence prompt"), and does it beat self-prompting (0.1%)?
- **N** Does widening the candidate pool with leakage-filtered cross-species protein-to-genome structures, EST graph paths and AUGUSTUS samples raise oracle recall enough for reranking to matter, and is any gain attributable to transformer candidates?

## 2. Frozen loci and gold
- A. thaliana: the 3,429 strict held-out loci (no RC twin in training); gold alternatives = 124 distinct alternative CDS structures at these loci (compositionally depleted, 0.036 alt/locus — report macro locus recall alongside).
- Z. mays: non-leaked RefGen_V4 loci (174 legacy GRMZM models excluded) with a completion prediction; the exact locus and alternative-structure denominators (L_Zm, T_Zm) are computed and published on day 1 before any candidate is generated.
- Pilot subsets for M(a)/M(c): 500 loci per species sampled (seed 123) from the C callability map with ≥ 2 splice-bearing EST accessions and ≥ 1 independent ONT/Iso-Seq molecule at the locus; the long-read molecules stay hidden until final evaluation.
- Gold for final evaluation = independent ONT/Iso-Seq complete chains (B1 §6 definition); reference structures are used for oracle recall only.

## 3. Candidate sources (all deduplicated to strand-aware ordered CDS intervals, source-tagged)
- M_K: deterministic TransGenic de novo beams, K = 1, 4, 8, 16 (`num_beams=K, num_return_sequences=K`), plus for M(a) 16 fixed-seed samples (`top_p=0.95, temperature=0.8`).
- Protein: miniprot on a leakage-safe protein set (§5): `miniprot -I -t64 --gff -N 10 --outn 10 --outs 0.90 --outc 0.70`; keep ≥ 30% identity, ≥ 70% query coverage, no frameshift/internal stop.
- EST: ≤ 32 splice-graph paths per locus ranked by support product, termini only from qualified poly(A)/5′ evidence, longest complete ORF ≥ 90 aa, canonical motifs.
- AUGUSTUS: manuscript setting unchanged (`--species=arabidopsis --sample=100 --alternatives-from-sampling=true --noInFrameStop=true`; maize configuration recorded by checksum, not tuned).
- U = M_16 ∪ protein ∪ EST paths ∪ AUGUSTUS.

## 4. Oracle gate (day 4) and kill rules
Exact-CDS oracle recall@K by maximum bipartite matching (candidate ↔ gold, identical ordered CDS intervals), also exact intron-chain and overlap (reciprocal CDS coverage ≥ 0.80, junction F1 ≥ 0.80) variants; per-locus recall; CDS-exon and junction recall.
- **Go**: at K = 16, U improves alternative exact-CDS oracle recall over M_16 by ≥ 10 points and ≥ 1.5× in both species, locus-bootstrap 95% CI lower bound > +5 (A. thaliana: ≥ 13 extra matches of 124).
- **Conditional go**: 5–10 points in one species, ≥ 10 in the other, and ≥ 3 points uniquely attributable to transformer candidates.
- **Kill candidate expansion**: < 5 points in either species, or gains only at overlap level.
- **Kill the transformer claim**: U passes but removing transformer candidates costs < 3 points in both species.
- M(a) pilot pass: oracle recall ≥ 20% and ≥ 2× top-1 in each species; selected output independent exact-chain precision ≥ 20%, alternative-chain recall ≥ 5% among callable loci, paired-bootstrap CI over model-score-only excluding zero.

## 5. Circularity and leakage rules
- Exclude every protein of the evaluated species (synonyms, cultivars, organelles, and any UniProt entry cross-referenced to TAIR10/Araport or RefGen_V4/V5), regardless of release date; one representative protein per source gene.
- Primary leakage-safe analysis additionally excludes the evaluated gene's orthogroup (OrthoFinder groups built before evaluation; reciprocal-best hits; ≥ 40% identity over ≥ 70% bidirectional coverage) and close paralogs by the same rule; a secondary "homology-assisted" analysis may keep pre-cutoff cross-species orthologs and is labeled as such.
- Proposed snapshot cutoffs to verify from database metadata: Brassicaceae proteomes released after 2010 excluded for A. thaliana; Poaceae proteomes released after 2017 excluded for maize.
- Post-alignment second screen against the evaluated reference proteome; any source meeting the identity/coverage rule is discarded with its candidates. Every exclusion is logged.
- ESM-2 pseudo-log-likelihood (`esm2_t12_35M_UR50D`, exact masked PLL, FP16, windows of 1,022 with stride 511 for long proteins, computed for ≤ 4 finalists per locus) is a generic plausibility feature only; its pretraining corpus cannot be purged, so it is never validation, never evidence of novelty, and always ablated.
- Reranker weights are trained on allowed training species only (A. thaliana excluded when its strict set is evaluated; maize never used), frozen before scoring, and never touch ONT/Iso-Seq.

## 6. Arms and ablations (identical loci, gold, deduplication, filters, budget)
Protein-only; EST-only; AUGUSTUS-only; Protein+EST; Protein+EST+AUGUSTUS (strong no-transformer baseline); Transformer-only (M_16); Full pool without the transformer-score feature; Full pool ranked by transformer score + structural validity only; Full system; Full system minus ESM. Generation credit = R_full − R_(protein+EST+AUGUSTUS) and transformer-unique recall; selection credit = full reranker vs no-transformer-score on identical candidates at one candidate per eligible locus and at the top-1,000 candidates genome-wide (paired locus bootstrap + permutation test; positive requires ≥ 3-point precision gain with CI excluding zero).
M(c) arms: self-prompt; evidence CDS-only prompt; evidence CDS + qualified-UTR prompt; no-prompt evidence-assisted (M(a)); reference prompt as ceiling. Evidence primary constructible only when one chain has ≥ 2 accessions, ≥ 2 libraries and no conflicting supported junction; 3′ UTR only with non-templated poly(A) passing the internal-priming filter; 5′ UTR only from oriented full-length/RAFL/cap evidence; otherwise CDS-only. Pass: evidence prompt beats self-prompt with CI excluding zero and reaches ≥ 5% independent exact-chain precision in both species; CDS+UTR arm counts as better than CDS-only only if its lower CI is higher.

## 7. Scoring axes
CDS exact match (additions definition); CDS intron-chain match; UTR structure (only among candidates and references with a UTR call, 5′ and 3′ separately); termini (median |ΔTSS|, |ΔTES| and fractions within ±30/±50/±100 nt). A correct CDS stays correct with an absent/incorrect UTR; no full-transcript credit for a correct CDS with a wrong claimed UTR. Gold alternatives are stratified: protein-homolog-supported / EST-supported / both / neither, with exact-CDS recall per stratum.

## 8. Seven-day schedule and cost (RTX 4090 + 64-core node; estimates, not measurements)
Day 1 freeze loci, gold, protein snapshot, exclusions, thresholds, L_Zm/T_Zm. Day 2 miniprot (8–24 h/genome), EST path enumeration, AUGUSTUS (12–36 node-h/species). Days 2–4 beams (A. thaliana 3–8 GPU-h; maize 30–60 GPU-h at beam 8; beam 16 only where beam 8 adds candidates). Day 4 oracle gate; stop on kill. Days 5–6 reranker on allowed species, pre-rank to 4/locus, ESM PLL (24–72 GPU-h both species). Day 7 ablations, bootstrap, leakage audit, go/no-go.

## 9. Reporting
Positive: "evidence-assisted candidate pool … increased exact-CDS oracle recall at a fixed per-locus budget, while ablation of transformer-generated candidates reduced recall by X … reported as an experimental system, not a released mode." Negative: "expanding the candidate pool … did not improve exact-CDS oracle recall by the pre-specified five-point minimum in both held-out evaluations … transformer candidates contributed fewer than three points … we report this as a boundary on candidate reranking and defer decoder changes to future work." Negative M(c): "EST-derived prompts increased structural constraint but did not recover the reference-prompt completion behaviour; the limitation lies in candidate generation and/or prompt-feature accuracy, not merely reranking." The manuscript's statement that completion currently requires a curated CDS+UTR prompt stands unless the pass criteria are met.
