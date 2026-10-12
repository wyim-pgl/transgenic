# Cap/CAGE TSS evidence inventory for the A16/A49 post-C0 completeness pass (2026-09-23)

Read-only inventory. No file other than this note was written; no commit, no Slurm job.

## Why

`PROTOCOL_B1_frozen_v1.md` A49 item 1 (line 1137): `complete_5p_code` is computed in a
post-C0 pass that needs, per species, a cap/CAGE input; "the cap/CAGE input is to be named
per species before the pass; where none exists, 5C is unreachable and reads default to 5I as
A16 states." Design `EVIDENCE_TRANSCRIPT_MODEL_DESIGN_v1.md` §2 (line 12): 5C requires either
(a) a cap-trapped/TeloPrime library with the 5' primer detected, or an ONT strand-switch
adapter AND an end within 50 nt of a cap/CAGE TSS, or (b) a TSS cluster (≥ 5 units, ≥ 2
libraries, ≥ 2 cap/CAGE-associated units) whose medoid is within ±50 nt of a same-strand CAGE
peak. Both routes need a same-species cap/CAGE peak set; route (a)'s first clause needs a
cap-trapped long-read library.

Species: nine training species = the `c2_training_eligible` species of
`DATASET_ROLES.est_v1.tsv` / `DATASET_ROLES.longread_v1.tsv` (Athaliana, Bdistachyon, Gmax,
Osativa, Ppatens, Ptrichocarpa, Sbicolor, Sitalica, Vvinifera); two scoring species =
`b1_validation_only` (Zmays, Slycopersicum). A. thaliana additionally carries the
validation-only A-ONT1 (`ont/Athaliana_FLIC_PRJNA1087576`) and A-CCS23
(`pacbio/Athaliana_zhang2023_PRJNA911826`) sets (A46 rule 10 / A47.2).

## Summary table

| species | taxid | on pronghorn? | public cap/CAGE-class dataset(s) (accession, strategy, tissue, year) | TeloPrime / cap-trapped long-read set in manifest | recommendation |
|---|---|---|---|---|---|
| Athaliana | 3702 | **no** (no CAGE/PEAT/TSS-seq/TeloPrime files or manifest rows under `/data/gpfs/assoc/pgl/data/Transgenic`) | **PRJNA1087576** SRR28341940/41/42 — CAGE (NextSeq 500), mature leaf, 2024; same BioProject and tissue as A-ONT1 (SRR28341943/44/45). PRJNA253067 SRR1425301 — PEAT, whole root, 2014. PRJNA270670 SRR2097904 — nanoCAGE, root, 2015. PRJDB5793 (6 runs), PRJDB3455 (8), PRJDB5045 (1) — CAGE, whole plant / cultured cells, 2018–2019. PRJNA562289 (9 runs) — CAGE, seedling, 2020. PRJNA600990 (15) — CAGE, seedling, 2021. PRJDB8904 (22) — CAGE, 2020. PRJNA846134 (12) — CAGE, rosette leaf, 2022. PRJDB12660 (11) — CAGE, seedling, 2022. PRJNA658605 SRR12494050/51 — TSS-seq (nanoCAGE-XL), root/shoot, 2024. PRJDB35587 (4) — CAGE, seedling, 2025. TSS-seq (Marquardt lab, "other"): PRJNA453570 (24), PRJNA488719 (16), PRJNA531657 (6), PRJNA544565 (8), seedlings, 2018–2019; PRJDB9678 (32) drought TSS, 2022. | none in the manifests. Public TeloPrime v2 Iso-Seq exists: **PRJNA754773 SRR15462255** (SRX11761448, Sequel I, FL-cDNA, inflorescence + leaves, 2022, 32.6 M reads) — not in the manifest; Sequel I is excluded from training by A14, but as a *5C evidence* input it is route (a) material. | **5C reachable.** Name `PRJNA1087576` CAGE (SRR28341940–42) as the primary input: it is the tissue-matched partner of the A-ONT1 scoring set. Add a training-side broad set (e.g. PRJNA562289 + PRJDB5793) so the ±50 nt clustering has ≥ 2 CAGE libraries. Decide separately whether SRR15462255 (TeloPrime, Sequel I) is admitted as a cap-trapped 5C source under A14's Sequel I exclusion. |
| Bdistachyon | 15368 | no | **none found** (ENA `tax_tree(15368)` with CAGE/PEAT/TSS-seq/STRIPE/RAMPAGE/TeloPrime/oligo-cap terms → 0 runs; `scientific_name="Brachypodium*"` → 0; web search → no dataset) | none (EST only in the manifests) | **5C unreachable → all reads 5I** (A16 default). |
| Gmax | 3847 | no | **PRJNA757465** SRR15601584–91 — STRIPE-seq, 8 tissues (leaf, stem tip, seed, pod, nodule, root, flower, stem), Purdue, 2022. **PRJDB7219** DRR146878–83 — nAnT-iCAGE, leaf/stem/root, 2020. **PRJNA1223007** (8 runs, SRR32325593–96 + SRR34414589–92) — nanoCAGE, root/shoot, 2025 (Sci Data 2025). | none | **5C reachable.** Primary: PRJNA757465 (tissue breadth) + PRJDB7219 (independent library); PRJNA1223007 as a third library. |
| Osativa | 4530 | no | **PRJDB6857** DRR127211–18 — CAGE (CLONEEND-labelled), Japonica, 2018 (Nara; polysome/5'UTR study). **PRJNA695000** SRR13552962–66 — RAMPAGE, Indica IRBB7 leaves, 2021. *Mislabelled, not CAGE:* PRJNA767776, PRJNA768961, PRJNA767540, PRJNA1082621, PRJNA1104531 (Ghent; `library_selection=CAGE` but descriptions say mRNA-seq) — exclude. | none (ONT direct RNA + cDNA only) | **5C reachable** with PRJDB6857 (Japonica, matches Nipponbare sets) and PRJNA695000 (Indica, matches `indica_flagleaf_PRJNA1291274` only). Genotype stratum must be respected (A18.5). |
| Ppatens | 3218 | no | **none found** (ENA `tax_tree(3218)` → 6 raw hits, none CAGE-class; `scientific_name="Physcomitr*"` → 0; web search → only RNA-seq atlases) | none (direct RNA only) | **5C unreachable → all reads 5I.** |
| Ptrichocarpa | 3694 | no | **none found** (ENA `tax_tree(3694)` → 14 raw hits, none CAGE-class; `scientific_name="Populus*"` → 36 raw, none CAGE-class; web search → none) | none (direct RNA + Sequel II Iso-Seq, no TeloPrime) | **5C unreachable → all reads 5I.** |
| Sbicolor | 4558 | no | **PRJEB42222** ERR5021711–14 — RAMPAGE, Tx2783 and RTx436 root + shoot, CSHL, 2021. | none | **5C reachable**, but only one BioProject (two genotypes, ≥ 2 libraries so §2(b) is satisfiable); genotypes differ from the manifest's BTx623/wray/baijiu sets, so A18.5 genotype acceptance applies. |
| Sitalica | 4555 | no | **none found** for *S. italica* (ENA `tax_tree(4555)` → 22 raw, none CAGE-class; web search → none). Adjacent species only: *S. viridis* PRJNA354250 (43 runs, `library_selection=CAGE`, 13 tissues, 2017; study states "transcription start sites were mapped" but runs are labelled RNA-Seq) — different species, not a same-genome peak set. | none | **5C unreachable → all reads 5I** (unless the author decides to admit cross-species *S. viridis* TSS via lift-over, which the design does not provide for). |
| Vvinifera | 29760 | no | **none found.** PRJNA1164469 (12 runs, `library_selection=CAGE`, 2024) is a mislabelled metformin/nitrogen mRNA-seq study (Kunming) — exclude. | none | **5C unreachable → all reads 5I.** |
| Zmays (scoring) | 4577 | no | **PRJNA288042** SRR2078285–92 — CAGE, B73 and A632 root + shoot, 2015 (Mejía-Guerra et al.). **PRJNA438108** (10 runs) — RAMPAGE B73 root tips etc., CSHL, 2018. **PRJNA437131** (10) — RAMPAGE W22, 2018. **PRJNA431779** (10) — RAMPAGE NC350, 2018. PRJNA498523 (6) — RAMPAGE teosinte Til11, 2019. | none | **5C reachable.** B73 sets (PRJNA288042 + PRJNA438108) match the B73 scoring manifest. |
| Slycopersicum (scoring) | 4081 | no | **PRJDB14685** DRR415368–73 — TSS-seq, seedling, Kyushu, 2023. PRJDB13239 DRR353754 — TSS-seq, *S. lycopersicum* var. *cerasiforme* (plus DRR353753 *S. pimpinellifolium*, DRR353755 *S. cheesmaniae*), 2024. *Mislabelled, not CAGE:* PRJNA1095548, PRJNA1172399 (Ghent mRNA-seq) — exclude. | none (`ont/Slycopersicum_heinz_PRJEB37834` is ordinary full-length cDNA) | **5C reachable** with PRJDB14685 (6 libraries, cultivar not stated in ENA; check against Heinz 1706). PRJDB13239 only as a second library if var. *cerasiforme* is accepted under A18.5. |

Legend: "on pronghorn?" = any CAGE/PEAT/TSS-seq/TeloPrime/cap-trap data or manifest row under
`/data/gpfs/assoc/pgl/data/Transgenic`. Answer is **no for every species**: the only name hits
are the protocol's own `feature_tss` scripts, and the only text hits are design/brainstorm
notes (`codex_fulllength_20260901.md`, `codex_brainstorm_est.md`, `codex_est_brainstorm_20260901.md`,
`integrity_report_20260817.md`, `peer_review_report_20260817.md`).

## TeloPrime / cap-trapped long-read sets in the manifests

None. `ONT_PROTOCOL.b6_v1.draft.tsv` (143 rows) column `ena_kit_or_protocol_text` never mentions
TeloPrime, cap-trap, CapTrap, SMARTer or a strand-switch kit; protocol classes are `cDNA`
(97), `direct_RNA` (45), `unresolved` (1). `PACBIO_LIBRARY.b6_v1.draft.tsv` (72 rows) carries only
ENA experiment accessions in `library`, no kit text, and no note mentions TeloPrime.
`DATASET_ROLES.longread_v1.tsv` note column: no hits. Consequently §2 route (a) first clause
("cap-trapped/TeloPrime library with 5' primer detected") is unavailable for every manifest set;
only the second clause (ONT strand-switch adapter + CAGE TSS within 50 nt) and route (b)
(cluster medoid within ±50 nt of a CAGE peak) can produce 5C, and both need the CAGE inputs above.

## Caveats

- ENA `library_selection=CAGE` is unreliable: at least eight studies (Ghent rice/tomato, Kunming
  grape, Tottori Arabidopsis PRJNA562704, Copenhagen PRJEB13223, NAIST PRJDB5913 degradome) carry
  the tag on ordinary RNA-seq or non-TSS libraries. Every row in the table above was checked
  against the study description; the mislabelled ones are listed explicitly so they are not
  re-discovered.
- NCBI E-utilities (GEO/SRA esearch) timed out twice from this host; the GEO cross-check was
  therefore not completed. ENA mirrors SRA, so ENA `tax_tree` coverage is the basis of "none found".
- Web search (Brachypodium, Physcomitrium, Populus, Setaria italica) returned no CAGE/TSS-seq
  datasets; a literature-level PEAT/CAGE paper could still exist without a public run, but none
  surfaced.
- The A. thaliana TeloPrime run SRR15462255 is on Sequel I. A14/A46 exclude RS II/Sequel I data
  from *training*; whether it may serve as a *cap-evidence* input for the completeness pass is an
  author decision, not something this inventory settles.

## Commands and queries used

Local (repo `transgenic_local`):

```
python3: csv.DictReader over data/manifests/DATASET_ROLES.longread_v1.tsv, DATASET_ROLES.est_v1.tsv,
         DATASET_ROLES.b6_v1.tsv, ONT_PROTOCOL.b6_v1.draft.tsv, PACBIO_LIBRARY.b6_v1.draft.tsv
         (species/role counts; regex telo|cap-?trap|CAGE|SMARTer|strand.?switch|SSP over every column)
grep -rniE 'CAGE|PEAT|TSS-seq|TSSseq|TeloPrime|cap[- ]?trap|oligo[- ]?capp|STRIPE|nanoCAGE' data/manifests/ revision/protocols/ notes/
grep -nE 'A16|A46|A49|rule 10|scoring species|5C|5I' revision/protocols/PROTOCOL_B1_frozen_v1.md
grep -niE 'CAGE|cap|TSS|medoid|50 nt|5C|5I' revision/protocols/EVIDENCE_TRANSCRIPT_MODEL_DESIGN_v1.md
```

Pronghorn (`ssh pronghorn`, `D=/data/gpfs/assoc/pgl/data/Transgenic`):

```
ls $D; find $D -maxdepth 2 -type d
find $D/evidence -maxdepth 3 -type d
find $D/evidence $D/training_input -maxdepth 5 \( -iname "*cage*" -o -iname "*peat*" -o -iname "*tss*" -o -iname "*telo*" -o -iname "*captrap*" -o -iname "*stripe*" -o -iname "*rampage*" \)
grep -liE "\bCAGE\b|\bPEAT\b|TSS-seq|TeloPrime|cap-trap|RAMPAGE|STRIPE-seq" $D/*.md $D/evidence/*.md $D/evidence/*.tsv $D/evidence/*/*.md $D/evidence/*/*.tsv
```
(a whole-tree `find $D -maxdepth 7` timed out at 200 s on GPFS and was replaced by the scoped
version above.)

ENA portal API (`https://www.ebi.ac.uk/ena/portal/api/search`, `result=read_run`, JSON, limit 2000–3000), per taxid
3702, 15368, 3847, 4530, 3218, 3694, 4558, 4555, 29760, 4577, 4081:

```
tax_tree(<taxid>) AND ( library_selection="CAGE"
  OR experiment_title="*<T>*" OR study_title="*<T>*" OR library_name="*<T>*" ... )
  for T in CAGE, PEAT, TSS-seq, TSS seq, TSSseq, nanoCAGE, STRIPE, TeloPrime, oligo-cap, oligocap,
           CapTrap, cap-trap, cap trap, cap analysis, capped RNA, 5' end, transcription start site,
           RAMPAGE, CAGEscan, TSS
```
then the same term list with `scientific_name="<Genus>*"` for Setaria, Brachypodium, Physcomitr,
Populus, Glycine, Sorghum, Vitis, Oryza, Solanum lycopersicum, Zea, Arabidopsis; post-filtered in
Python (word-boundary CAGE/PEAT, "repeat"/"PEAT complex" removed). Study descriptions fetched with
`result=study&query=study_accession="PRJ..."&fields=study_title,study_description,center_name,first_public`
for every candidate listed above; run lists with `study_accession="PRJ..."` for PRJNA1087576,
PRJNA354250, PRJDB13239, PRJDB14685, PRJNA695000, PRJDB6857, PRJEB42222, PRJNA757465,
PRJNA1223007, PRJDB7219, PRJNA288042 and `run_accession="SRR15462255"`.

NCBI: `esearch.fcgi?db=gds|sra&term="<organism>"[Organism] AND <CAGE|RAMPAGE|STRIPE-seq|TSS-seq|...>`
— timed out (two attempts), not used.

Web search: "Brachypodium distachyon CAGE OR TSS-seq OR STRIPE-seq", "Physcomitrella patens OR
Physcomitrium patens CAGE transcription start sites", "Populus trichocarpa CAGE OR TSS-seq OR
RAMPAGE", "Setaria italica CAGE OR RAMPAGE OR TSS-seq" — no dataset hits.
