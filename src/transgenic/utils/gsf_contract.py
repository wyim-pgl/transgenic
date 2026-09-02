"""GSF contract implementation — docs/gsf_spec_v1.md (v1.0, frozen 2026-09-02).

Pure Python, no torch/numpy. Loaded by path from tests/conftest.py and imported by the
production builder (preprocess.py / gff2gsf.py) once #12 lands. Covers: coordinates (§1),
grammar (§2), canonical ordering gsf-order-v1 (§3), caps (§4), reverse complement (§5), DB row
contract (§6), split table (§7), and the C2 label/weight contract (§10, protocol A18.4).
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

WINDOW_UNIT = 6144
MAX_WINDOW = 49152                      # sym6144-v1 (published recipe)
WINDOW_TIERS = (30720, 61440, 129024)   # tier6144-v2: 5, 10, 21 chunks (~32 / 64 / 128 kb); Helixer plant context is 106,920
MAX_WINDOW_V2 = WINDOW_TIERS[-1]
MIN_FLANK = 1000                        # nt of context required on each side of the gene before a tier is accepted
TIER_UP_PROB = 0.3                      # augmentation: probability of using the next larger tier (training builds only)
CAPS = {"CDS": 150, "five_prime_UTR": 50, "three_prime_UTR": 50, "transcripts": 15, "tokens": 2048}
ORDERING_VERSION = "gsf-order-v1"
WINDOW_POLICY = "sym6144-v1"
WINDOW_POLICY_V2 = "tier6144-v2"
BUILD_VERSION = "gsf-contract-v1"
PHASE_TO_LETTER = {0: "A", 1: "B", 2: "C"}
LETTER_TO_PHASE = {v: k for k, v in PHASE_TO_LETTER.items()}
FEATURE_TYPES = ("five_prime_UTR", "CDS", "three_prime_UTR")
RC_MODES = ("none", "all", "isoform-only")
GENE_SEP = "<gene>"          # v3: separates gene blocks inside one window label
EMPTY_LABEL = "<empty>"      # v3: window without any complete gene
CAPS_V3 = {"tokens": 8192, "genes": 96}   # from tile statistics (A. thaliana 129-kb tiles: p95 42 genes, max 80, ~120 tokens/gene)
WINDOW_POLICY_V3 = "tile6144-v3"
EMPTY_KEEP_PROB = 0.1

SEG_CLASSES = ["protein_coding_gene", "lncRNA", "exon", "intron", "splice_donor", "splice_acceptor", "5UTR", "3UTR",
               "CTCF-bound", "polyA_signal", "enhancer_Tissue_specific", "enhancer_Tissue_invariant",
               "promoter_Tissue_specific", "promoter_Tissue_invariant"]
SOURCE_WEIGHT = {"protein": 1.0, "pacbio": 1.0, "ont": 0.8, "est": 0.6}            # exon/UTR family (A18.4)
JUNCTION_SOURCE_WEIGHT = {"est": 1.0, "pacbio": 0.9, "ont": 0.7, "protein": 0.5}  # intron/boundary family (A20)
JUNCTION_FAMILY = ("intron", "splice_donor", "splice_acceptor")
WEIGHT_CAP = 4.0
RETAINED_INTRON_WEIGHT = 0.25


class CapError(ValueError):
    """A record exceeds a frozen cap (§4). Records are rejected, never truncated."""


class SplitError(KeyError):
    """A gene has no entry in the split table (§7): builders fail closed."""


@dataclass(frozen=True)
class Feature:
    type: str
    start1: int  # GFF3 1-based inclusive
    end1: int
    phase: str   # "0"/"1"/"2" for CDS, "." otherwise


@dataclass
class Gene:
    gene_id: str
    chrom: str
    strand: str
    start0: int
    end0: int
    transcripts: Dict[str, List[Feature]] = field(default_factory=dict)
    gene_id_original: str = ""   # GFF ID attribute verbatim
    name_original: str = ""      # GFF Name attribute verbatim


MAX_KEY_LEN = 10


def species_code(species_id: str) -> str:
    """Three-letter species code: first three characters of the manifest species_id (e.g. Athaliana -> Ath)."""
    return (species_id[:3] or "Unk").capitalize() if not species_id[:3].isupper() else species_id[:3]


def key_is_valid(key: str) -> bool:
    return 0 < len(key) <= MAX_KEY_LEN and key.count(".") <= 1


def gene_key(species_code_: str, gff_id: str, gff_name: str, counter: int) -> str:
    """Author rule (2026-09-02): ID first, then Name; a candidate longer than 10 characters or with more than one dot,
    or no candidate at all, becomes <3-letter species code><6-digit ordinal in GFF order>."""
    cand = gff_id or gff_name
    if cand and key_is_valid(cand):
        return cand
    return f"{species_code_}{counter:06d}"


# ----------------------------------------------------------------------------------------------
# §1 coordinates
# ----------------------------------------------------------------------------------------------
def pad_window_tiered(start0: int, end0: int, rng: Optional["random.Random"] = None, tier_up_prob: float = 0.0,
                      tiers: Sequence[int] = WINDOW_TIERS, min_flank: int = MIN_FLANK) -> Tuple[int, int, int]:
    """tier6144-v2: the smallest tier that holds the gene plus MIN_FLANK on both sides; with `rng` and
    tier_up_prob the next tier is chosen instead (variable-context augmentation) and the gene is placed at a
    random offset inside it. Returns (ws, we, tier). Genes longer than the largest tier minus flanks are rejected
    by the caller via CapError (window > MAX_WINDOW_V2)."""
    length = end0 - start0
    idx = next((i for i, t in enumerate(tiers) if length + 2 * min_flank <= t), None)
    if idx is None:
        we_ws = tiers[-1]
        return start0 - min_flank, start0 - min_flank + we_ws + WINDOW_UNIT, tiers[-1] + WINDOW_UNIT  # over cap -> rejected downstream
    if rng is not None and tier_up_prob > 0 and idx + 1 < len(tiers) and rng.random() < tier_up_prob:
        idx += 1
    tier = tiers[idx]
    slack = tier - length
    left = slack // 2 if rng is None else rng.randint(min_flank, slack - min_flank)
    ws = start0 - left
    we = ws + tier
    if ws < 0:
        we -= ws
        ws = 0
    return ws, we, tier


def pad_window(start0: int, end0: int) -> Tuple[int, int]:
    """Symmetric padding to a multiple of WINDOW_UNIT; no extra chunk at exact multiples."""
    length = end0 - start0
    pad = (WINDOW_UNIT - length % WINDOW_UNIT) % WINDOW_UNIT
    left = pad // 2
    right = pad - left
    ws, we = start0 - left, end0 + right
    if ws < 0:
        we -= ws
        ws = 0
    return ws, we


# ----------------------------------------------------------------------------------------------
# GFF3 parsing (EOF flush, per-gene chromosome/strand ownership)
# ----------------------------------------------------------------------------------------------
_ATTR_RE = re.compile(r"([^;=]+)=([^;]*)")


def _attrs(col9: str) -> Dict[str, str]:
    return {k.strip(): v.strip() for k, v in _ATTR_RE.findall(col9)}


def parse_gff3(lines: Iterable[str], species_code: Optional[str] = None) -> Iterator[Gene]:
    """Yield genes in file order. A gene is emitted when the next gene line starts or at EOF.
    With species_code the gene key rule applies (gene_key); without it the GFF ID (then Name) is used verbatim."""
    cur: Optional[Gene] = None
    tx_parent: Dict[str, str] = {}
    counter = 0
    seen_keys: set = set()
    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 9:
            continue
        chrom, _src, ftype, s, e, _sc, strand, phase, col9 = cols[:9]
        a = _attrs(col9)
        if ftype == "gene":
            if cur is not None:
                yield cur
            counter += 1
            gid, gname = a.get("ID", ""), a.get("Name", "")
            key = gene_key(species_code, gid, gname, counter) if species_code else (gid or gname or f"{chrom}:{s}-{e}")
            if key in seen_keys:
                raise ValueError(f"duplicate gene key {key} (ID={gid!r}, Name={gname!r}) at {chrom}:{s}-{e}")
            seen_keys.add(key)
            cur = Gene(key, chrom, strand, int(s) - 1, int(e), {}, gid, gname)
            tx_parent = {}
        elif ftype in ("mRNA", "transcript"):
            if cur is None:
                cur = Gene(a.get("Parent", a.get("ID", "gene")), chrom, strand, int(s) - 1, int(e))
            tid = a.get("ID", f"tx{len(cur.transcripts) + 1}")
            tx_parent[tid] = cur.gene_id
            cur.transcripts.setdefault(tid, [])
        elif ftype in FEATURE_TYPES:
            if cur is None:
                raise ValueError(f"feature before any gene/mRNA: {line}")
            tid = a.get("Parent", next(iter(cur.transcripts), "tx1"))
            if tid not in cur.transcripts:
                if cur.transcripts and tid not in tx_parent:
                    raise ValueError(f"feature {a.get('ID')} refers to transcript {tid} of another gene")
                cur.transcripts[tid] = []
            cur.transcripts[tid].append(Feature(ftype, int(s), int(e), phase if ftype == "CDS" else "."))
    if cur is not None:
        yield cur


# ----------------------------------------------------------------------------------------------
# §2/§3 canonical serialisation
# ----------------------------------------------------------------------------------------------
def _rel(f: Feature, ws: int) -> Tuple[str, int, int, str]:
    return (f.type, f.start1 - 1 - ws, f.end1 - ws, f.phase)


def _orient(strand: str):
    return (lambda x: x) if strand == "+" else (lambda x: -x)


def _transcription_order(feats: Sequence[Tuple[str, int, int, str]], strand: str):
    o = _orient(strand)
    return sorted(feats, key=lambda f: (o(f[1]), o(f[2])))


def exons_of(feats: Sequence[Tuple[str, int, int, str]]) -> List[Tuple[int, int]]:
    """Merge coordinate-contiguous features (UTR|CDS boundaries) into exons; introns are the genomic gaps between them."""
    ex: List[Tuple[int, int]] = []
    for f in sorted(feats, key=lambda f: (f[1], f[2])):
        if ex and f[1] <= ex[-1][1]:
            ex[-1] = (ex[-1][0], max(ex[-1][1], f[2]))
        else:
            ex.append((f[1], f[2]))
    return ex


def _signature(feats: Sequence[Tuple[str, int, int, str]], strand: str):
    o = _orient(strand)
    ordered = _transcription_order(feats, strand)
    exons = exons_of(feats)
    gaps = [(a[1], b[0]) for a, b in zip(exons, exons[1:])]           # genomic introns, ascending
    if strand == "-":
        gaps = [(o(b), o(a)) for a, b in reversed(gaps)]
    chain = gaps
    span = (o(ordered[0][1]), o(ordered[-1][2])) if strand == "+" else (o(ordered[0][2]), o(ordered[-1][1]))
    cds = tuple((o(f[1]), o(f[2]), f[3]) for f in ordered if f[0] == "CDS")
    utr = tuple((f[0], o(f[1]), o(f[2])) for f in ordered if f[0] != "CDS")
    return (len(chain), tuple(chain), span, cds, utr), ordered


def _serialize(transcripts: Sequence[Sequence[Tuple[str, int, int, str]]], strand: str) -> str:
    sigs = {}
    for feats in transcripts:
        if not feats:
            continue
        key, ordered = _signature(feats, strand)
        sigs.setdefault(key, tuple(ordered))  # identical signatures merge
    ordered_tx = [sigs[k] for k in sorted(sigs)]
    names: Dict[Tuple[str, int, int, str], str] = {}
    counters = {t: 0 for t in FEATURE_TYPES}
    for tx in ordered_tx:
        for f in tx:
            if f not in names:
                counters[f[0]] += 1
                names[f] = f"{f[0]}{counters[f[0]]}"
    feature_list = sorted(names, key=lambda f: (f[1], f[2], FEATURE_TYPES.index(f[0]), f[3]))
    feats_str = ";".join(f"{f[1]}|{names[f]}|{f[2]}|{strand}|{_phase_letter(f)}" for f in feature_list)
    tx_str = ";".join("|".join(names[f] for f in tx) for tx in ordered_tx)
    return f"{feats_str}>{tx_str}"


def _phase_letter(f: Tuple[str, int, int, str]) -> str:
    if f[0] != "CDS":
        return "."
    if f[3] not in ("0", "1", "2"):
        raise CapError(f"CDS feature {f[1]}-{f[2]} has no phase ({f[3]!r})")
    return PHASE_TO_LETTER[int(f[3])]


def gene_to_gsf(gene: Gene, window_start: int) -> str:
    txs = [[_rel(f, window_start) for f in feats] for feats in gene.transcripts.values()]
    return _serialize(txs, gene.strand)


def _parse(gsf: str):
    feats_str, _, tx_str = gsf.partition(">")
    feats: Dict[str, Tuple[str, int, int, str]] = {}
    strand = "+"
    strands = set()
    for item in feats_str.split(";"):
        if not item:
            continue
        s, name, e, strand, ph = item.split("|")
        strands.add(strand)
        if len(strands) > 1:
            raise ValueError("mixed strands in one GSF record")
        ftype = re.match(r"[A-Za-z_]+", name).group(0)
        phase = str(LETTER_TO_PHASE[ph]) if ftype == "CDS" else "."
        feats[name] = (ftype, int(s), int(e), phase)
    txs = [[feats[n] for n in t.split("|")] for t in tx_str.split(";") if t]
    return feats, txs, strand


def gsf_to_gene(gsf: str, window_start: int, chrom: str, strand: str, gene_id: str) -> Gene:
    _feats, txs, s = _parse(gsf)
    if s != strand:
        raise ValueError(f"GSF strand {s} does not match {strand}")
    gene = Gene(gene_id, chrom, strand, 0, 0)
    for i, tx in enumerate(txs, 1):
        gene.transcripts[f"t{i}"] = [Feature(t, rs + window_start + 1, re_ + window_start, ph) for (t, rs, re_, ph) in tx]
    allf = [f for tx in gene.transcripts.values() for f in tx]
    gene.start0 = min(f.start1 for f in allf) - 1
    gene.end0 = max(f.end1 for f in allf)
    return gene


def canonicalize(gsf: str) -> str:
    _feats, txs, strand = _parse(gsf)
    return _serialize(txs, strand)


# ----------------------------------------------------------------------------------------------
# §5 reverse complement (pure transformation, phases recomputed 5'->3')
# ----------------------------------------------------------------------------------------------
def _recompute_phases(tx: Sequence[Tuple[str, int, int, str]], strand: str, initial_phase: Optional[int] = None) -> List[Tuple[str, int, int, str]]:
    """Phases 5'->3': the first CDS keeps its own phase (0 unless the model is 5'-partial), the rest follow from the
    coding length so far. This makes RC an involution for partial models as well."""
    ordered = _transcription_order(tx, strand)
    out, cum, first = [], 0, True
    for f in ordered:
        if f[0] == "CDS":
            if first:
                p0 = int(f[3]) if initial_phase is None else initial_phase
                phase, first = p0, False
                cum = -p0                       # p0 bases are skipped before the first complete codon
            else:
                phase = (3 - cum % 3) % 3
            out.append((f[0], f[1], f[2], str(phase)))
            cum += f[2] - f[1]
        else:
            out.append(f)
    return out


def reverse_complement(gsf: str, window_len: int) -> str:
    _feats, txs, strand = _parse(gsf)
    new_strand = "-" if strand == "+" else "+"
    new_txs = []
    for tx in txs:
        first_cds = next((f for f in _transcription_order(tx, strand) if f[0] == "CDS"), None)
        p0 = int(first_cds[3]) if first_cds else 0
        mirrored = [(t, window_len - e, window_len - s, ph) for (t, s, e, ph) in tx]
        new_txs.append(_recompute_phases(mirrored, new_strand, initial_phase=p0))
    return _serialize(new_txs, new_strand)


# ----------------------------------------------------------------------------------------------
# §4 caps and v2 token counting
# ----------------------------------------------------------------------------------------------
def count_tokens_v2(gsf: str) -> int:
    feats_str, _, tx_str = gsf.partition(">")
    n = 0
    for feature in feats_str.split(";"):
        for col in feature.split("|"):
            n += len(col) if col.isdigit() else 1
        n += 1  # ';' (the last one becomes <txN>)
    n += 1  # '>'
    if tx_str:
        parts = tx_str.split(";")
        for j, tx in enumerate(parts):
            for col in tx.split("|"):
                n += len(col) if col.isdigit() else 1
            if j < len(parts) - 1:
                n += 1  # <iso>
    return n + 2  # <s> ... </s> (the tokenizer prepends <s>; verified against tokenization_transgenic.py)


def check_caps(gsf: str, window_len: Optional[int] = None) -> None:
    feats, txs, _ = _parse(gsf)
    if not txs or not any(f[0] == "CDS" for f in feats.values()):
        raise CapError("no CDS: a GSF label requires at least one coding transcript")
    if any(not any(feats[n][0] == "CDS" for n in tx) for tx in [t.split("|") for t in gsf.partition(">")[2].split(";") if t]):
        raise CapError("transcript without CDS")
    counts = {t: 0 for t in FEATURE_TYPES}
    for f in feats.values():
        counts[f[0]] += 1
    for t in FEATURE_TYPES:
        if counts[t] > CAPS[t]:
            raise CapError(f"{t} features {counts[t]} > {CAPS[t]}")
    if len(txs) > CAPS["transcripts"]:
        raise CapError(f"transcripts {len(txs)} > {CAPS['transcripts']}")
    n = count_tokens_v2(gsf)
    if n > CAPS["tokens"]:
        raise CapError(f"v2 tokens {n} > {CAPS['tokens']}")
    if window_len is not None and window_len > MAX_WINDOW:
        raise CapError(f"window {window_len} > {MAX_WINDOW}")


# ----------------------------------------------------------------------------------------------
# §6 DB rows
# ----------------------------------------------------------------------------------------------
def build_rows(genes: Iterable[Gene], species_id: str, rc: str, split_lookup: Dict[str, str], return_rejected: bool = False):
    if rc not in RC_MODES:
        raise ValueError(f"--rc must be one of {RC_MODES}, got {rc!r}")
    rows, rejected = [], []
    for gene in genes:
        if gene.gene_id not in split_lookup:
            raise SplitError(f"{species_id}:{gene.gene_id} has no split entry")
        ws, we = pad_window(gene.start0, gene.end0)
        L = we - ws
        gsf = gene_to_gsf(gene, ws)
        try:
            check_caps(gsf, window_len=L)
        except CapError as e:
            rejected.append({"species_id": species_id, "gene_id": gene.gene_id, "reason": str(e)})
            continue
        base = {
            "species_id": species_id, "gene_id": gene.gene_id, "chromosome": gene.chrom, "strand": gene.strand,
            "start": ws, "fin": we, "gsf": gsf, "is_rc": False, "split": split_lookup[gene.gene_id],
            "ordering_version": ORDERING_VERSION, "window_policy": WINDOW_POLICY, "build_version": BUILD_VERSION,
            "gsf_token_count": count_tokens_v2(gsf), "predict": None,
        }
        rows.append(base)
        want_rc = rc == "all" or (rc == "isoform-only" and len(gene.transcripts) >= 2)
        if want_rc:
            rgsf = reverse_complement(gsf, L)
            row = dict(base)
            row.update(is_rc=True, strand="-" if gene.strand == "+" else "+", gsf=rgsf, gsf_token_count=count_tokens_v2(rgsf))
            rows.append(row)
    return (rows, rejected) if return_rejected else rows


# ----------------------------------------------------------------------------------------------
# §7 split table
# ----------------------------------------------------------------------------------------------
def make_split(orthogroup_of: Dict[str, str], seed: int, fractions: Tuple[float, float, float], strict_holdout: Set[str]) -> Dict[str, str]:
    held_groups = {orthogroup_of[g] for g in strict_holdout if g in orthogroup_of}
    groups = sorted(set(orthogroup_of.values()) - held_groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * fractions[0]))
    n_valid = int(round(n * fractions[1]))
    assign = {}
    for i, g in enumerate(groups):
        assign[g] = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
    for g in held_groups:
        assign[g] = "test"
    return {gene: assign[grp] for gene, grp in orthogroup_of.items()}


def validate_split(rows: Sequence[Dict], excluded_species: Set[str] = frozenset({"Zmays"})) -> List[str]:
    v: List[str] = []
    fwd = {(r["species_id"], r["gene_id"]): r["split"] for r in rows if not r.get("is_rc")}
    by_group: Dict[str, Set[str]] = {}
    for r in rows:
        key = (r["species_id"], r["gene_id"])
        if r["species_id"] in excluded_species:
            v.append(f"excluded species present: {r['species_id']} {r['gene_id']}")
        if r.get("is_rc") and key in fwd and fwd[key] != r["split"]:
            v.append(f"rc row split {r['split']} != forward split {fwd[key]} for {r['gene_id']}")
        if r.get("strict_holdout") and r["split"] != "test":
            v.append(f"strict held-out gene {r['gene_id']} in split {r['split']}")
        if r["split"] not in ("train", "valid", "test"):
            v.append(f"invalid split {r['split']!r} for {r['species_id']}:{r['gene_id']}")
        if r.get("is_rc") and key not in fwd:
            v.append(f"rc row without a forward row for {r['species_id']}:{r['gene_id']}")
        by_group.setdefault(r["orthogroup_id"] or f"{r['species_id']}:{r['gene_id']}", set()).add(r["split"])
    for grp, splits in by_group.items():
        if len(splits) > 1:
            v.append(f"orthogroup {grp} spans splits {sorted(splits)}")
    return v


def validate_evidence_roles(labels: Sequence[Dict]) -> List[str]:
    return [f"validation-only evidence labels {l['gene_id']} in split {l['split']}"
            for l in labels if l.get("source_role") == "b1_validation_only" and l.get("split") in ("train", "valid")]


# ----------------------------------------------------------------------------------------------
# §10 C2 labels and weights (protocol A18.4)
# ----------------------------------------------------------------------------------------------
_CLASS = {c: i for i, c in enumerate(SEG_CLASSES)}
_TYPE_CLASSES = {
    "CDS": ("exon", "protein_coding_gene"), "exon": ("exon",), "intron": ("intron",),
    "five_prime_UTR": ("exon", "5UTR", "protein_coding_gene"), "three_prime_UTR": ("exon", "3UTR", "protein_coding_gene"),
}


def segmentation_labels(features: Sequence[Tuple[str, int, int, str]], L: int, window_start: int):
    labels = [[0.0] * len(SEG_CLASSES) for _ in range(L)]
    weights = [[0.0] * len(SEG_CLASSES) for _ in range(L)]
    for ftype, s1, e1, _strand in features:
        for cls in _TYPE_CLASSES.get(ftype, ()):
            ci = _CLASS[cls]
            for p in range(max(0, s1 - 1 - window_start), min(L, e1 - window_start)):
                labels[p][ci] = 1.0
                weights[p][ci] = max(weights[p][ci], 1.0)
    return labels, weights


def evidence_weight(source: str, n_molecules: int, genotype: str = "reference", retained_intron: bool = False, family: str = "exon") -> float:
    """A18.4 weight; `family` selects the source-weight table (A20: 'junction' -> EST leads)."""
    if family not in ("exon", "junction"):
        raise ValueError(f"family must be 'exon' or 'junction', got {family!r}")
    if genotype not in ("reference", "non_reference", "hybrid_pooled", "unknown"):
        raise ValueError(f"unknown genotype stratum {genotype!r}")
    if n_molecules < 1:
        raise ValueError("evidence weight needs >= 1 independent molecule")
    if retained_intron:
        return RETAINED_INTRON_WEIGHT
    table = JUNCTION_SOURCE_WEIGHT if family == "junction" else SOURCE_WEIGHT
    if family == "junction" and source == "protein" and n_molecules < 2:
        raise ValueError("protein junction support needs >= 2 organisms (A19.3)")
    sw = table[source] * (1.0 if genotype == "reference" else 0.5)
    return min(WEIGHT_CAP, 1.0 + sw * math.log1p(n_molecules))


def add_evidence(labels, weights, cells: Iterable[Tuple[int, int, float]]) -> None:
    """Evidence adds positives only; existing labels and weights are never lowered."""
    L = len(labels)
    for pos, cls, w in cells:
        if not (0 <= pos < L) or not (0 <= cls < len(SEG_CLASSES)) or w < 0:
            raise ValueError(f"evidence cell out of range: pos={pos} cls={cls} w={w}")
        labels[pos][cls] = 1.0
        weights[pos][cls] = max(weights[pos][cls], w)


def junction_weight(source: str, n_molecules: int, genotype: str = "reference") -> float:
    return evidence_weight(source, n_molecules, genotype, family="junction")


def add_junction_evidence(labels, weights, donor0: int, acceptor0: int, weight: float) -> None:
    if not (0 <= donor0 < acceptor0 < len(labels)):
        raise ValueError(f"junction out of range or reversed: donor0={donor0} acceptor0={acceptor0} L={len(labels)}")
    d, a, i = _CLASS["splice_donor"], _CLASS["splice_acceptor"], _CLASS["intron"]
    cells = [(donor0, d, weight), (acceptor0, a, weight)] + [(p, i, weight) for p in range(donor0, acceptor0 + 1)]
    add_evidence(labels, weights, cells)


# ----------------------------------------------------------------------------------------------
# GSF v3: every complete gene inside a window (author decision 2026-09-02, protocol A26)
# ----------------------------------------------------------------------------------------------
def genes_in_window(genes: Sequence[Gene], ws: int, we: int) -> Tuple[List[Gene], int]:
    """Genes fully inside [ws, we) sorted by start; second value = genes that cross a window edge (excluded)."""
    inside = [g for g in genes if g.start0 >= ws and g.end0 <= we]
    partial = sum(1 for g in genes if g.end0 > ws and g.start0 < we and not (g.start0 >= ws and g.end0 <= we))
    return sorted(inside, key=lambda g: (g.start0, g.end0)), partial


def window_to_gsf_v3(genes: Sequence[Gene], ws: int) -> str:
    """Concatenate canonical per-gene GSF blocks in coordinate order; empty window -> EMPTY_LABEL."""
    blocks = []
    for g in sorted(genes, key=lambda g: (g.start0, g.end0)):
        if not any(g.transcripts.values()):
            continue
        blocks.append(gene_to_gsf(g, ws))
    return GENE_SEP.join(blocks) if blocks else EMPTY_LABEL


def split_v3(gsf: str) -> List[str]:
    return [] if gsf == EMPTY_LABEL else gsf.split(GENE_SEP)


def canonicalize_v3(gsf: str) -> str:
    blocks = split_v3(gsf)
    if not blocks:
        return EMPTY_LABEL
    keyed = []
    for b in blocks:
        feats, _, _ = _parse(b)
        keyed.append((min(f[1] for f in feats.values()), max(f[2] for f in feats.values()), canonicalize(b)))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return GENE_SEP.join(k[2] for k in keyed)


def reverse_complement_v3(gsf: str, window_len: int) -> str:
    blocks = split_v3(gsf)
    if not blocks:
        return EMPTY_LABEL
    return canonicalize_v3(GENE_SEP.join(reverse_complement(b, window_len) for b in blocks))


def count_tokens_v3(gsf: str) -> int:
    blocks = split_v3(gsf)
    if not blocks:
        return 3  # <s> <empty> </s>
    # each block counted as v2 minus its own <s>/</s>, joined by <gene>, plus <s> and </s>
    return 2 + sum(count_tokens_v2(b) - 2 for b in blocks) + (len(blocks) - 1)


def check_caps_v3(gsf: str, window_len: Optional[int] = None) -> None:
    blocks = split_v3(gsf)
    if len(blocks) > CAPS_V3["genes"]:
        raise CapError(f"genes per window {len(blocks)} > {CAPS_V3['genes']}")
    prev_end = -1
    for b in blocks:
        check_caps(b, window_len=None)
        feats, _, _ = _parse(b)
        gs, ge = min(f[1] for f in feats.values()), max(f[2] for f in feats.values())
        if gs < prev_end:
            raise CapError("overlapping genes in one window label")
        prev_end = ge
        if window_len is not None and ge > window_len:
            raise CapError(f"gene extends beyond the window ({ge} > {window_len})")
    n = count_tokens_v3(gsf)
    if n > CAPS_V3["tokens"]:
        raise CapError(f"v3 tokens {n} > {CAPS_V3['tokens']}")
    if window_len is not None and window_len > MAX_WINDOW_V2:
        raise CapError(f"window {window_len} > {MAX_WINDOW_V2}")


def tile_windows(chrom_len: int, tier: int, offset: int = 0) -> List[Tuple[int, int]]:
    """Consecutive tiles of one tier starting at `offset` (0 <= offset < tier); the last tile is shifted back so it
    keeps the full tier length (or starts at 0 when the contig is shorter than the tier)."""
    tiles = []
    ws = offset
    while ws + tier <= chrom_len:
        tiles.append((ws, ws + tier))
        ws += tier
    if not tiles or tiles[-1][1] < chrom_len:
        start = max(0, chrom_len - tier)
        if not tiles or start > tiles[-1][0]:
            tiles.append((start, start + tier))
    return tiles
