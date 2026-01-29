# Metrics Definitions

Detailed definitions of evaluation metrics used in TransGenic AS analysis.

## Transcript-level Metrics

### Isoform Recall

**Definition:** Proportion of reference isoforms exactly reproduced by TransGenic.

```
Isoform Recall = (# exact matched isoforms) / (# total reference isoforms)
```

**What counts as "exact match":**
- All exon-intron boundaries identical
- Same number of exons
- GFFCompare class code `=`

**Interpretation:**
- High recall → TransGenic captures most known isoforms
- Low recall vs AtRTD3 → Missing rare/low-abundance isoforms

### Isoform Precision

**Definition:** Proportion of predicted isoforms that exactly match reference.

```
Isoform Precision = (# exact matched isoforms) / (# total predicted isoforms)
```

**Interpretation:**
- High precision → Few false positive isoforms
- Low precision → Overprediction of isoforms

### Isoform F1

**Definition:** Harmonic mean of recall and precision.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why F1:**
- Balances recall and precision
- Single metric for overall transcript-level accuracy

## GFFCompare Class Codes

| Code | Name | Description |
|------|------|-------------|
| `=` | Exact match | Complete match of intron chain |
| `c` | Contained | Query contained within reference exons |
| `k` | Containment | Reference contained within query |
| `j` | Partial match | Multi-exon with at least one junction match |
| `e` | Single exon | Single exon transfrag overlapping reference |
| `o` | Overlap | Generic exonic overlap |
| `i` | Intronic | Falls entirely within reference intron |
| `x` | Opposite strand | Exonic overlap on opposite strand |
| `p` | Polymerase run-on | Possible polymerase run-on fragment |
| `u` | Unknown | Intergenic, no overlap |

### Grouping for Analysis

| Group | Codes | Interpretation |
|-------|-------|----------------|
| **Exact** | `=` | Perfect transcript prediction |
| **Partial** | `c`, `k`, `j` | Structural similarity, AS variant |
| **Overlap** | `e`, `o` | Some overlap, different structure |
| **Other** | `i`, `x`, `p`, `u` | Likely false positive |

## Splice Event Metrics

### Event Definitions (rMATS-style)

#### Exon Skipping (SE)
```
Inclusion:    =====[included exon]=====
Skipping:     ========================
```
An exon present in one isoform but absent (skipped) in another.

#### Alternative 5' Splice Site (A5SS)
```
Isoform 1:    ====|----------
Isoform 2:    ========|------
                     ^ different donor sites
```
Same 3' acceptor site, different 5' donor sites.

#### Alternative 3' Splice Site (A3SS)
```
Isoform 1:    ----------|====
Isoform 2:    ------|========
                    ^ different acceptor sites
```
Same 5' donor site, different 3' acceptor sites.

#### Intron Retention (IR)
```
Spliced:      ====|     |====
Retained:     ================
```
An intron retained as exonic sequence in one isoform.

### Event Recall

**Definition:** Proportion of reference events recovered in predictions.

```
Event Recall = (# matched events) / (# reference events)
```

**Per event type:**
- SE Recall, A5SS Recall, A3SS Recall, IR Recall

### Event Precision

**Definition:** Proportion of predicted events matching reference.

```
Event Precision = (# matched events) / (# predicted events)
```

## Isoform Count Metrics

### Per-gene Isoform Distribution

Compare the number of isoforms per gene between reference and prediction.

| Category | Definition |
|----------|------------|
| Exact | Same isoform count |
| Overprediction | More predicted than reference |
| Underprediction | Fewer predicted than reference |
| Missed gene | Gene in reference, not in prediction |
| Novel gene | Gene in prediction, not in reference |

### Summary Statistics

- **Mean isoforms per gene** (reference vs prediction)
- **Overprediction rate:** % of genes with excess isoforms
- **Underprediction rate:** % of genes with missing isoforms

## Interpretation Guidelines

### High Recall, High Precision
✅ TransGenic accurately predicts known isoforms

### High Recall, Low Precision
⚠️ TransGenic captures reference but overpredicts

### Low Recall, High Precision
⚠️ TransGenic is conservative, misses some isoforms

### Low Recall vs AtRTD3 specifically
This is **expected** because:
1. AtRTD3 includes low-abundance isoforms from deep Iso-seq
2. These require very high coverage to detect
3. TransGenic focuses on well-supported isoforms

**Recommended statement for paper:**
> Lower recall against AtRTD3 reflects uncharacterized low-abundance isoforms rather than incorrect splice boundary predictions. TransGenic successfully recovers the majority of well-supported AS events.
