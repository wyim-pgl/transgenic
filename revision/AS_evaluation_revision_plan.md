# TransGenic Revision Plan: Alternative Splicing Evaluation

## Reviewer Core Criticism (One-line Summary)
**Base-level F1 is inappropriate for evaluating alternative splicing (AS)**

### Specific Issues:
1. "58% base-level accuracy" - meaning unclear
2. AS requires **isoform-level** and **splice event-level** metrics

---

## Revision Goals

| Goal | Description |
|------|-------------|
| Transcript-level accuracy | Did we correctly predict transcripts? |
| Isoform count accuracy | How well do we predict the number of isoforms per gene? |
| Splice event recovery | How well do we recover splice events (exon skipping, etc.)? |

---

## STEP 1. Define Evaluation Criteria (TEXT + METHODS)

### 1.1 Reference Definition
- **Arabidopsis**: TAIR10 + AtRTD3
- **Rice/Maize**: TAIR-based AS (optional, TAIR-based only is sufficient)

### 1.2 "Correct Isoform" Definition (Methods section text)

> **Transcript-level exact match:**
> - All exon-intron boundaries identical
> - CDS frame consistency maintained
> - Partial matches reported as separate metric

---

## STEP 2. Transcript-level Accuracy (KEY METRIC)

### 2.1 Isoform Recovery Rate

**Definition:** Proportion of reference isoforms completely reproduced by TransGenic

**Method:**
- Use GFFCompare class codes
- Match against reference transcripts

**Metrics:**
```
Isoform Recall    = (# exact matched isoforms) / (total reference isoforms)
Isoform Precision = (# exact matched isoforms) / (total predicted isoforms)
```

**Owner:** STATISTICIAN

---

## STEP 3. Isoform Count Distribution Analysis

*Reviewer R2 will particularly appreciate this*

### 3.1 Isoforms per Gene Distribution

**Comparison:** Reference vs TransGenic

**Visualization:** Histogram or violin plot

**Metrics:**
- Mean isoforms per gene
- Overprediction rate
- Underprediction rate

**Example sentence for paper:**
> "TransGenic captures the overall distribution of isoform counts per gene, while slightly underestimating highly complex loci."

---

## STEP 4. Splice Event-level Evaluation (True AS Evaluation)

*Technical core of revision*

### 4.1 Event Type Definitions (based on AtRTD3)

| Event Type | Abbreviation |
|------------|--------------|
| Exon Skipping | SE |
| Alternative 5' Splice Site | A5SS |
| Alternative 3' Splice Site | A3SS |
| Intron Retention | IR |

### 4.2 Event Recovery

**Per event type:**
- Proportion of reference events predicted by TransGenic

**Metrics:**
```
Event Recall (per type)
Event Precision (per type)
```

**Note:** Use rMATS-style definitions only; no need to run rMATS itself.

---

## STEP 5. Reinterpret Base-level AS Metric (Save the 58%)

*Don't discard existing results - reposition them*

### 5.1 Redefine the Meaning of 58%

> "This reflects cumulative nucleotide overlap across alternative regions.
> Low recall is driven by missing low-abundance isoforms, not incorrect splice boundaries."

### 5.2 Separate Precision vs Recall Presentation

**Figure components:**
- Base-level Precision
- Base-level Recall
- Base-level F1

**Result:** Abstract becomes defensible.

---

## STEP 6. Figure 6 Design (Multi-panel)

| Panel | Content |
|-------|---------|
| **A** | Transcript-level precision/recall bar plot |
| **B** | Isoform count distribution (Reference vs TransGenic) |
| **C** | Splice event recovery by type |
| **D** (optional) | Example gene schematic: recovered vs missed isoforms |

**Impact:** This single figure addresses both Reviewer 2 and Reviewer 3.

---

## STEP 7. Supplementary Methods (Required)

Must include:
- [ ] Transcript matching criteria
- [ ] Event definition rules
- [ ] Tools and versions
- [ ] Handling of duplicated or permuted isoforms

*If omitted, R2 will raise again*

---

## Role Assignment

### BIOINFORMATICIAN
- [ ] GFFCompare transcript matching
- [ ] Isoform count extraction
- [ ] Event detection scripts

### STATISTICIAN
- [ ] Metric definition text
- [ ] Precision/recall calculations
- [ ] Figure statistics summary

### LEAD AUTHOR
- [ ] Abstract sentence redesign
- [ ] Discussion framing (why this matters)

---

## Expected Outcomes

1. **Remove core criticism:** "AS evaluation is inappropriate"
2. **Transform 58%:** From weakness to contextualized result
3. **Reposition TransGenic:** Not just "general gene finder" but **isoform completion + hypothesis generator**

---

## References

- GFFCompare documentation
- AtRTD3 transcriptome (Genome Biology, 2022)
- rMATS splicing event definitions
- Recent AS evaluation practices in plant genomics literature
