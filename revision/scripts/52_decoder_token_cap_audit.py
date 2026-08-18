#!/usr/bin/env python3
"""
Decoder token-cap audit (Reviewer 3, W8).

Table S7 tabulates per-gene feature counts against the GSF *vocabulary* limits
(150 CDS, 50 five-prime UTR, 50 three-prime UTR segments per gene). That is not
the only capacity limit. Coordinates are emitted digit by digit, so a gene can
satisfy every vocabulary limit and still not fit the decoder, whose positional
embedding table holds 2,048 positions
(`max_decoder_position_embeddings=2048`, configuration_transgenic.py:63).
`isoformDataHyena.__getitem__` (datasets.py:308-313) tokenizes the GSF label with
`GFFTokenizer._tokenize`, then truncates anything longer than `self.maxlength =
2048`, overwriting the last kept token with `</s>`.

This script measures the second limit directly. For every gene of every
reference annotation it:

  1. rebuilds the extraction window exactly as `genome2GSFDataset()` does
     (preprocess.py:303-348) -- GSF coordinates are window-relative, so the
     window determines how many digit tokens each coordinate costs;
  2. rebuilds the GSF label string exactly as `genome2GSFDataset()` does
     (preprocess.py:398-469), including feature deduplication and the
     `feature_list>transcript_list` assembly;
  3. tokenizes it with the real `GFFTokenizer` and counts tokens;
  4. reports the per-species length distribution, the number of genes over
     2,048 tokens, and the 2x2 contingency of the token-cap constraint against
     the vocabulary constraint that Table S7 reports.

`<unk>` tokens are counted in the same pass: a gene with more than 150 CDS
segments still tokenizes to the same length, but its out-of-vocabulary feature
names (CDS151, ...) collapse to `<unk>`, so the unk counter is an independent
re-derivation of the Table S7 over-limit counts.

The tokenizer is imported from the installed source by file path rather than
through `transgenic.model`, because that package's `__init__` imports torch and
this audit needs no model weights. Token counts are identical under the legacy
"v1" (272-token, published-checkpoint) and the current "v2" (288-token)
vocabularies -- v2 spends its extra <txN>/<iso> tokens in slots that v1 fills
with ';' -- and the script asserts this rather than assuming it.

Usage:
  python 52_decoder_token_cap_audit.py                    # all references
  python 52_decoder_token_cap_audit.py --species Z_mays A_thaliana
  python 52_decoder_token_cap_audit.py --validate-window  # check window math
                                                          # against the built DB
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from datetime import date

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                 # .../transgenic
PROJ = os.path.dirname(REPO)                                  # .../Transgenic
CMP = os.path.join(PROJ, "transgenic_comparison")
REF_DIR = os.path.join(CMP, "reference_annotations")
RES_DIR = os.path.join(REPO, "revision", "results", "decoder_token_cap")
TOKENIZER_SRC = os.path.join(
    REPO, "src", "transgenic", "model", "tokenization_transgenic.py")
ATRTD3_GTF = os.path.join(REPO, "revision", "data", "AtRTD3", "AtRTD3.gtf")
# Predict-mode DuckDB built from the Z. mays reference by genome2GSFDataset();
# its start/fin columns are the extraction windows, so it validates step 1.
VALIDATION_DB = os.path.join(REF_DIR, "Zmays_493_APGv4.fa.transgenic.db")

# ── Constants mirrored from the model / preprocessing code ───────────────────
DECODER_MAX_POSITIONS = 2048    # configuration_transgenic.py:63; datasets.py:194
STATIC_SIZE = 6144              # preprocess.py:62  (encoder chunk size)
MAX_REGION_LEN = 49152          # preprocess.py:60  (maxLen; larger genes skipped)
VOCAB_CDS_LIMIT = 150           # tokenization_transgenic.py:83
VOCAB_UTR_LIMIT = 50            # tokenization_transgenic.py:87

# Organelle filter, byte-identical to 08_feature_tss_stats.py:30, so that the
# gene sets of Table S7 and this audit are the same.
SKIP_CHROM = re.compile(r"^(ChrM|ChrC|MT|Pt)|chloroplast|mitochondria", re.I)

# Reference annotations, from 11_run_feature_tss_all.sh:13-26.
REFERENCES = [
    ("A_thaliana", "Athaliana_167_TAIR10.gene.clean.gff3"),
    ("B_distachyon", "Bdistachyon_314_v3.1.gene_exons.clean.gff3"),
    ("B_rapa", "BrapaO_302V_711_v1.1.gene.gff3"),
    ("G_max", "Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3"),
    ("L_sativa", "Lsativa_467_v5.gene_exons.gff3"),
    ("O_sativa", "Osativa_323_v7.0.gene_exons.exon.gff3"),
    ("P_patens", "Ppatens_318_v3.3.gene_exons.clean.gff3"),
    ("P_trichocarpa", "Ptrichocarpa_533_v4.1.gene_exons.clean.gff3"),
    ("S_bicolor", "Sbicolor_730_v5.1.gene_exons.clean.gff3"),
    ("S_italica", "Sitalica_312_v2.2.gene_exons.clean.gff3"),
    ("S_lycopersicum", "Slycopersicum_796_ITAG5.0.gene.gff3"),
    ("V_vinifera", "Vvinifera_PN40024_5.1_on_T2T_ref.exon.gff3"),
    ("Z_mays", "Zmays_493_RefGen_V4.gene_exons.exon.gff3"),
]

PER_GENE_FIELDS = [
    "gene", "chrom", "strand", "gene_start", "gene_end", "gene_length",
    "region_start", "region_end", "region_length", "preprocess_status",
    "n_cds", "n_utr5", "n_utr3", "n_transcripts",
    "n_tokens", "n_unk_tokens", "over_token_cap", "over_vocab_limit",
]


# ── Tokenizer ────────────────────────────────────────────────────────────────
def load_tokenizers():
    """Import the real GFFTokenizer by file path (skips the torch-importing
    package __init__) and return {"v1": tok, "v2": tok}."""
    spec = importlib.util.spec_from_file_location(
        "tokenization_transgenic", TOKENIZER_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {v: mod.GFFTokenizer(vocab_version=v) for v in ("v1", "v2")}


# ── Window geometry (preprocess.py:303-348) ──────────────────────────────────
def extraction_window(gene_start, gene_end, static_size=STATIC_SIZE):
    """Return (region_start, region_end) for a gene, replicating the padding
    arithmetic of genome2GSFDataset().

    gene_start/gene_end are 1-indexed inclusive GFF3 coordinates. region_start
    is 0-indexed; region_end is exclusive. Chromosome-end clamping is omitted
    on purpose: it can only shrink region_end, and GSF coordinates are measured
    from region_start, which the source already guarantees is non-negative.
    """
    gene_length = gene_end - gene_start + 1
    if gene_length <= static_size:
        additional = static_size - (gene_length % static_size)
    else:
        additional = ((gene_length // static_size) + 1) * static_size - gene_length

    three_prime_buffer = additional // 2
    if not (additional % 2):
        five_prime_buffer = additional // 2
    else:
        five_prime_buffer = additional // 2 + 1

    if (gene_start - five_prime_buffer - 1) < 0:
        five_prime_buffer = gene_start - 1
    region_start = gene_start - five_prime_buffer - 1

    if (five_prime_buffer + gene_length + three_prime_buffer) <= static_size:
        three_prime_buffer = static_size - (five_prime_buffer + gene_length)
    region_end = gene_end + three_prime_buffer
    return region_start, region_end


# ── GSF assembly (preprocess.py:398-469) ─────────────────────────────────────
PHASE_MAP = {"0": "A", "1": "B", "2": "C"}


class GeneBuilder:
    """Accumulates one gene's GSF feature and transcript sections."""

    __slots__ = ("gene_id", "chrom", "strand", "gene_start", "gene_end",
                 "region_start", "region_end", "feature_list", "mrna_list",
                 "cds_num", "five_ps", "three_ps", "n_transcripts",
                 "skip_reason")

    def __init__(self, gene_id, chrom, strand, gene_start, gene_end,
                 region_start, region_end):
        self.gene_id = gene_id
        self.chrom = chrom
        self.strand = strand
        self.gene_start = gene_start
        self.gene_end = gene_end
        self.region_start = region_start
        self.region_end = region_end
        self.feature_list = ""
        self.mrna_list = ""
        self.cds_num = {}
        self.five_ps = {}
        self.three_ps = {}
        self.n_transcripts = 0
        self.skip_reason = None

    def open_transcript(self):
        # preprocess.py:404 -- drop the trailing '|' of the previous transcript
        # and start a new one. For the first mRNA this seeds mrna_list with ';'.
        self.mrna_list = self.mrna_list[:-1] + ";"
        self.n_transcripts += 1

    def add_feature(self, typ, start, fin, strand, phase, table):
        """typ is the GSF feature prefix ('CDS' / 'five_prime_UTR' /
        'three_prime_UTR'); start/fin are 1-indexed inclusive GFF3
        coordinates."""
        s = (start - 1) - self.region_start   # 1-indexed -> 0-indexed, window-relative
        e = fin - self.region_start           # GFF3 end inclusive -> GSF end exclusive
        key = f"{s}-{e}-{strand}-{phase}"
        num = table.get(key)
        if num is None:
            num = str(len(table) + 1)
            table[key] = num
            self.feature_list += f"{s}|{typ}{num}|{e}|{strand}|{phase};"
        self.mrna_list += f"{typ}{num}|"

    def gsf(self):
        """Assemble the GSF string, or None when the gene has no coding /
        UTR children (preprocess.py:217-222)."""
        mrna = self.mrna_list[1:-1]
        if self.feature_list and mrna:
            return f"{self.feature_list[:-1]}>{mrna}"
        return None


def iter_genes_gff3(path):
    """Yield every gene of a sorted GFF3 as a GeneBuilder, following the
    line-by-line state machine of genome2GSFDataset().

    Genes the preprocessor drops are yielded too, carrying a `skip_reason`, and
    their features are accumulated rather than discarded. The preprocessor stops
    reading a dropped gene's children; this audit keeps reading them so that the
    per-gene feature counts cover every nuclear gene and can be reconciled
    against Table S7, whose denominator is the whole annotation.
    """
    current = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line == "\n":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            chrom, _, typ, start, fin, _, strand, phase, attributes = f[:9]

            if typ == "gene":
                if current is not None:
                    yield current
                gene_start, gene_end = int(start), int(fin)
                region_start, region_end = extraction_window(gene_start, gene_end)
                current = GeneBuilder(
                    attributes.split(";")[0].split("=")[1], chrom, strand,
                    gene_start, gene_end, region_start, region_end)
                if region_end - region_start > MAX_REGION_LEN:
                    # preprocess.py:339 -- oversized genes never enter the DB.
                    current.skip_reason = "region_over_maxlen"

            elif current is None:
                continue
            elif typ == "lncRNA":
                # preprocess.py:393 -- lncRNA genes are dropped entirely.
                if current.skip_reason is None:
                    current.skip_reason = "lncRNA"
            elif typ == "mRNA":
                current.open_transcript()
            elif typ == "CDS":
                current.add_feature("CDS", int(start), int(fin), strand,
                                    PHASE_MAP.get(phase, "."), current.cds_num)
            elif typ == "five_prime_UTR":
                current.add_feature("five_prime_UTR", int(start), int(fin),
                                    strand, phase, current.five_ps)
            elif typ == "three_prime_UTR":
                current.add_feature("three_prime_UTR", int(start), int(fin),
                                    strand, phase, current.three_ps)
    if current is not None:
        yield current


GTF_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')
# AtRTD3 is a GTF: no gene/mRNA lines, lowercase UTR feature names, and
# gene/transcript identity carried in attributes. This adapter rebuilds the
# gene -> transcript -> feature hierarchy that the GFF3 path gets for free.
GTF_TYPE_MAP = {"cds": "CDS", "five_prime_utr": "five_prime_UTR",
                "three_prime_utr": "three_prime_UTR"}


def iter_genes_gtf(path):
    """Yield GeneBuilder objects from a GTF (AtRTD3). Requires the file to be
    grouped by gene, which AtRTD3 is."""
    rows_by_gene = {}
    order = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line == "\n":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            chrom, _, typ, start, fin, _, strand, phase, attributes = f[:9]
            key = GTF_TYPE_MAP.get(typ.lower())
            if key is None and typ.lower() != "exon":
                continue
            a = dict(GTF_ATTR.findall(attributes))
            gid = a.get("gene_id")
            tid = a.get("transcript_id")
            if not gid or not tid:
                continue
            if gid not in rows_by_gene:
                rows_by_gene[gid] = {"chrom": chrom, "strand": strand,
                                     "min": int(start), "max": int(fin),
                                     "tx": {}}
                order.append(gid)
            g = rows_by_gene[gid]
            g["min"] = min(g["min"], int(start))
            g["max"] = max(g["max"], int(fin))
            if key is None:      # exon: bounds only, as in the GFF3 path
                continue
            g["tx"].setdefault(tid, []).append((key, int(start), int(fin), phase))

    for gid in order:
        g = rows_by_gene[gid]
        region_start, region_end = extraction_window(g["min"], g["max"])
        gb = GeneBuilder(gid, g["chrom"], g["strand"], g["min"], g["max"],
                         region_start, region_end)
        if region_end - region_start > MAX_REGION_LEN:
            gb.skip_reason = "region_over_maxlen"
            yield gb
            continue
        for tid, feats in g["tx"].items():
            gb.open_transcript()
            for key, start, fin, phase in feats:
                table = (gb.cds_num if key == "CDS" else
                         gb.five_ps if key == "five_prime_UTR" else gb.three_ps)
                ph = PHASE_MAP.get(phase, ".") if key == "CDS" else phase
                gb.add_feature(key, start, fin, g["strand"], ph, table)
        yield gb


# ── Statistics ───────────────────────────────────────────────────────────────
def pct(sorted_vals, q):
    """Percentile by the same nearest-rank convention as
    08_feature_tss_stats.py:112, so p99 means the same thing here and in
    Table S7."""
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def audit_species(species, path, tokenizers, out_dir, is_gtf=False):
    """Tokenize every gene of one annotation; write the per-gene TSV; return
    the summary dict."""
    tok_v1, tok_v2 = tokenizers["v1"], tokenizers["v2"]
    unk_id = tok_v1.vocab["<unk>"]

    lengths = []            # genes the preprocessor keeps -> reach the decoder
    lengths_excluded = []   # genes dropped upstream, tokenized for reference
    counters = Counter()
    vocab_counters = Counter()
    contingency = Counter()
    over_cap_examples = []
    tsv_path = os.path.join(out_dir, f"per_gene_token_lengths_{species}.tsv")

    with open(tsv_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(PER_GENE_FIELDS)

        for g in (iter_genes_gtf(path) if is_gtf else iter_genes_gff3(path)):
            if SKIP_CHROM.search(g.chrom):
                counters["organelle_excluded"] += 1
                continue
            # genes_nuclear is the Table S7 denominator: every nuclear gene in
            # the annotation, whether or not the preprocessor keeps it.
            counters["genes_nuclear"] += 1

            n_cds, n_u5, n_u3 = len(g.cds_num), len(g.five_ps), len(g.three_ps)
            # Vocabulary limits are tallied over every nuclear gene, matching
            # the denominator Table S7 uses.
            if n_cds > VOCAB_CDS_LIMIT:
                vocab_counters["over_150_cds"] += 1
            if n_u5 > VOCAB_UTR_LIMIT:
                vocab_counters["over_50_utr5"] += 1
            if n_u3 > VOCAB_UTR_LIMIT:
                vocab_counters["over_50_utr3"] += 1
            over_vocab = (n_cds > VOCAB_CDS_LIMIT or n_u5 > VOCAB_UTR_LIMIT
                          or n_u3 > VOCAB_UTR_LIMIT)
            if over_vocab:
                vocab_counters["over_any"] += 1

            gsf = g.gsf()
            status = g.skip_reason or ("no_cds_or_utr" if gsf is None else "retained")
            counters[status] += 1

            n_tokens = n_unk = None
            over_token = False
            if gsf is not None:
                tokens = tok_v1._tokenize(gsf)
                n_tokens = len(tokens)
                n_unk = sum(1 for t in tokens
                            if tok_v1._convert_token_to_id(t) == unk_id)
                over_token = n_tokens > DECODER_MAX_POSITIONS
                # Lengths are reported under v1, the 272-token vocabulary of the
                # published checkpoints. v2 normally matches it token for token
                # (its <txN>/<iso> tokens occupy the slots v1 fills with ';'),
                # but it caps the transcript count at 15
                # (tokenization_transgenic.py:177) and then emits <iso> only for
                # j < tx_count - 1 (line 207), so a gene with more than 15
                # isoforms loses one separator per extra isoform. Record the
                # divergence instead of assuming it away.
                n_v2 = len(tok_v2._tokenize(gsf))
                if n_v2 != n_tokens:
                    counters["v1_v2_length_mismatch"] += 1
                    counters["v1_v2_max_abs_diff"] = max(
                        counters["v1_v2_max_abs_diff"], abs(n_tokens - n_v2))
                if status == "retained":
                    lengths.append(n_tokens)
                    contingency[(over_token, over_vocab)] += 1
                    if over_token and len(over_cap_examples) < 25:
                        over_cap_examples.append({
                            "gene": g.gene_id, "n_tokens": n_tokens,
                            "n_cds": n_cds, "n_utr5": n_u5, "n_utr3": n_u3,
                            "n_transcripts": g.n_transcripts})
                else:
                    lengths_excluded.append(n_tokens)

            writer.writerow([
                g.gene_id, g.chrom, g.strand, g.gene_start, g.gene_end,
                g.gene_end - g.gene_start + 1,
                g.region_start, g.region_end, g.region_end - g.region_start,
                status, n_cds, n_u5, n_u3, g.n_transcripts,
                "" if n_tokens is None else n_tokens,
                "" if n_unk is None else n_unk,
                int(over_token), int(over_vocab)])

    lengths.sort()
    lengths_excluded.sort()
    n = len(lengths)
    denom = max(1, n)
    n_nuclear = max(1, counters["genes_nuclear"])
    n_over = sum(1 for x in lengths if x > DECODER_MAX_POSITIONS)
    both = contingency[(True, True)]
    token_only = contingency[(True, False)]
    vocab_only = contingency[(False, True)]

    def block(arr):
        d = max(1, len(arr))
        return {"n": len(arr), "mean": round(sum(arr) / d, 1),
                "p50": pct(arr, 0.50), "p90": pct(arr, 0.90),
                "p95": pct(arr, 0.95), "p99": pct(arr, 0.99),
                "p999": pct(arr, 0.999), "max": arr[-1] if arr else 0}

    return {
        "species": species,
        "annotation_file": path,
        "decoder_max_positions": DECODER_MAX_POSITIONS,
        "genes_nuclear": counters["genes_nuclear"],
        "genes_organelle_excluded": counters["organelle_excluded"],
        "genes_skipped_region_over_maxlen": counters["region_over_maxlen"],
        "genes_skipped_lncRNA": counters["lncRNA"],
        "genes_skipped_no_cds_or_utr": counters["no_cds_or_utr"],
        "genes_tokenized": n,
        "tokenizer_vocab_version_reported": "v1",
        "genes_where_v2_length_differs": counters["v1_v2_length_mismatch"],
        "v2_max_abs_length_diff": counters["v1_v2_max_abs_diff"],
        "token_length": block(lengths),
        "token_length_excluded_genes": block(lengths_excluded),
        "genes_over_token_cap": n_over,
        "pct_over_token_cap": round(100.0 * n_over / denom, 3),
        "pct_over_token_cap_of_nuclear": round(100.0 * n_over / n_nuclear, 3),
        # Vocabulary tallies over all nuclear genes: directly comparable to
        # the Table S7 columns of the same name.
        "vocab_limit_all_nuclear_genes": {
            "over_150_cds": vocab_counters["over_150_cds"],
            "over_50_utr5": vocab_counters["over_50_utr5"],
            "over_50_utr3": vocab_counters["over_50_utr3"],
            "over_any": vocab_counters["over_any"],
            "pct_over_any": round(100.0 * vocab_counters["over_any"] / n_nuclear, 3),
        },
        # Contingency is restricted to genes that reach the decoder, so the two
        # constraints are compared on one gene set.
        "constraint_contingency": {
            "over_token_cap_only": token_only,
            "over_vocab_limit_only": vocab_only,
            "over_both": both,
            "over_neither": contingency[(False, False)],
        },
        "binding_constraint": (
            "token_cap" if (token_only + both) > (vocab_only + both)
            else "vocabulary" if (vocab_only + both) > (token_only + both)
            else "tie"),
        "over_cap_examples": over_cap_examples,
        "per_gene_tsv": tsv_path,
    }


# ── Window-math validation against the built database ────────────────────────
def validate_window_math(db_path, gff_path):
    """Compare computed extraction windows against the ones stored in a
    predict-mode DuckDB built by genome2GSFDataset()."""
    try:
        import duckdb
    except ImportError:
        return {"status": "skipped", "reason": "duckdb not importable"}
    if not os.path.exists(db_path):
        return {"status": "skipped", "reason": f"missing {db_path}"}

    con = duckdb.connect(db_path, read_only=True)
    stored = dict(
        (gm, (s, f)) for gm, s, f in
        con.sql("SELECT geneModel, start, fin FROM geneList").fetchall())
    con.close()

    checked = matched = 0
    mismatches = []
    for item in iter_genes_gff3(gff_path):
        if item.skip_reason is not None:
            continue
        ref = stored.get(item.gene_id)
        if ref is None:
            continue
        checked += 1
        if (item.region_start, item.region_end) == ref:
            matched += 1
        elif len(mismatches) < 10:
            mismatches.append({
                "gene": item.gene_id,
                "computed": [item.region_start, item.region_end],
                "stored": list(ref)})
    return {
        "status": "ok",
        "database": db_path,
        "genes_in_database": len(stored),
        "genes_checked": checked,
        "windows_matched": matched,
        "pct_matched": round(100.0 * matched / max(1, checked), 3),
        "example_mismatches": mismatches,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", nargs="*", default=None,
                    help="Subset of species keys (default: all references + AtRTD3)")
    ap.add_argument("--out-dir", default=RES_DIR)
    ap.add_argument("--validate-window", action="store_true",
                    help="Check window arithmetic against the Z. mays predict DB")
    ap.add_argument("--no-atrtd3", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizers = load_tokenizers()

    targets = [(sp, os.path.join(REF_DIR, fn), False) for sp, fn in REFERENCES]
    if not args.no_atrtd3 and os.path.exists(ATRTD3_GTF):
        targets.append(("AtRTD3", ATRTD3_GTF, True))
    if args.species:
        wanted = set(args.species)
        targets = [t for t in targets if t[0] in wanted]

    summaries = []
    for species, path, is_gtf in targets:
        if not os.path.exists(path):
            print(f"[skip] {species}: missing {path}", file=sys.stderr)
            continue
        print(f"[run ] {species}: {os.path.basename(path)}", flush=True)
        s = audit_species(species, path, tokenizers, args.out_dir, is_gtf=is_gtf)
        summaries.append(s)
        tl = s["token_length"]
        print(f"       n={s['genes_tokenized']} median={tl['p50']} "
              f"p95={tl['p95']} p99={tl['p99']} max={tl['max']} "
              f"over_2048={s['genes_over_token_cap']} "
              f"({s['pct_over_token_cap']}%)", flush=True)

    report = {
        "generated": date.today().isoformat(),
        "script": os.path.basename(__file__),
        "decoder_max_positions": DECODER_MAX_POSITIONS,
        "static_size": STATIC_SIZE,
        "max_region_len": MAX_REGION_LEN,
        "vocab_limits": {"cds": VOCAB_CDS_LIMIT, "utr5": VOCAB_UTR_LIMIT,
                         "utr3": VOCAB_UTR_LIMIT},
        "tokenizer_source": TOKENIZER_SRC,
        "percentile_convention": "nearest-rank, identical to 08_feature_tss_stats.py",
        "species": summaries,
    }
    if args.validate_window:
        zm = os.path.join(REF_DIR, dict(REFERENCES)["Z_mays"])
        report["window_validation"] = validate_window_math(VALIDATION_DB, zm)

    json_path = os.path.join(args.out_dir, "decoder_token_cap_summary.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    csv_path = os.path.join(args.out_dir, "decoder_token_cap_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([
            "species", "genes_nuclear", "genes_tokenized",
            "token_median", "token_p90", "token_p95", "token_p99", "token_p999",
            "token_max", "genes_over_2048", "pct_over_2048",
            "vocab_over_150_cds", "vocab_over_50_utr5", "vocab_over_50_utr3",
            "vocab_over_any", "pct_vocab_over_any",
            "over_token_only", "over_vocab_only", "over_both",
            "binding_constraint", "genes_skipped_region_over_maxlen",
            "genes_skipped_lncRNA", "genes_skipped_no_cds_or_utr",
            "genes_where_v2_length_differs"])
        for s in summaries:
            tl, cc = s["token_length"], s["constraint_contingency"]
            vl = s["vocab_limit_all_nuclear_genes"]
            w.writerow([
                s["species"], s["genes_nuclear"], s["genes_tokenized"],
                tl["p50"], tl["p90"], tl["p95"], tl["p99"], tl["p999"],
                tl["max"], s["genes_over_token_cap"], s["pct_over_token_cap"],
                vl["over_150_cds"], vl["over_50_utr5"], vl["over_50_utr3"],
                vl["over_any"], vl["pct_over_any"],
                cc["over_token_cap_only"], cc["over_vocab_limit_only"],
                cc["over_both"], s["binding_constraint"],
                s["genes_skipped_region_over_maxlen"],
                s["genes_skipped_lncRNA"], s["genes_skipped_no_cds_or_utr"],
                s["genes_where_v2_length_differs"]])

    print(f"\nWrote {json_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
