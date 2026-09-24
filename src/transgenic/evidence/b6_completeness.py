"""WP2c steps 2-3 (plan v4.1 §6; A16, design §2): CAGE peaks (D12 recipe) and read completeness codes per population.

Codes are computed PER POPULATION, never pooled: ``train_track``, ``valid_track``, ``eval_input``, ``score``, each
with its own read set, its own TSS/TES clusters and its own sha-bound admitted-chain set.  ``b6_score`` rows never
shape an input or label code (they form only the ``score`` population); A-CCS23 never gives a 3' end.  The merged C0
snapshot is never mutated; the output ``completeness_b6_v1.db`` is keyed (population, species, source, alignment_id)
under policy ``computed_post_c0_v1``, and every lookup must name its population (``code_of``).

Populations (accepted primary read placements; EST from the A37 primary arm):
  train_track  molecule train_track, training_eligible, NOT train_tiles_excluded, no b6_score (dataset, run)
  valid_track  as train_track, train_tiles_excluded admitted
  eval_input   (dataset, run) marked b6_input in the roles manifest (any molecule state), never b6_score
  score        (dataset, run) marked b6_score

5' (design §2, frozen thresholds): 5C if (a) the protocol row is author-approved (``adapter_5c_enabled``) and the
read has the library's 5' primer, or the ONT strand-switch adapter AND a 5' end within 50 nt of a same-strand CAGE
peak; or (b) the 5' end is within 25 nt of the medoid of a TSS cluster with >= 5 molecule units from >= 2 libraries
(>= 2 units in each of two libraries), >= 2 cap/CAGE-associated units (5' end within 50 nt of a same-strand peak) and
a medoid within +-50 nt of a same-strand CAGE peak.  Otherwise 5I.  A species without CAGE peaks is all 5I on (b)
and on the adapter-plus-CAGE clause of (a).
3': 3C if the untemplated poly(A) is >= 10 nt and passes the internal-priming filter (fail: >= 12 A in the downstream
20 nt, or >= 6 consecutive A; A3 / A18.2), or the 3' end is within 25 nt of the medoid of a TES cluster with >= 2
tail-positive units.  Otherwise 3I (``internal_priming_suspect`` where the tail fails the filter).
Internal: IC = exact admitted chain with >= 20-nt terminal anchors; IP = proper contiguous sub-chain of an admitted
chain; IM = mono-exonic; IX = conflicting / ambiguous (a non-accepted junction, an unadmitted chain, or an exact chain
with a short terminal anchor).
Clusters: greedy left-to-right complete linkage on sorted end positions (diameter <= 25 nt, design §3); the medoid is
the observed position minimising the summed distance (ties: lowest coordinate).
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from transgenic.evidence import leakage as LK

VERSION = "completeness_b6_v1"
POLICY = "computed_post_c0_v1"
POPULATIONS = ("train_track", "valid_track", "eval_input", "score")
CLUSTER_RULE = "greedy_left_to_right_complete_linkage_diameter25_v1"
DIAMETER = 25
TSS_WINDOW = 25
TES_WINDOW = 25
CAGE_WINDOW = 50
TSS_MIN_UNITS, TSS_MIN_LIBS, TSS_MIN_PER_LIB, TSS_MIN_CAP = 5, 2, 2, 2
TES_MIN_TAIL = 2
POLYA_MIN = 10
IP_WINDOW_A, IP_RUN_A = 12, 6
ANCHOR_MIN = 20
NO_3PRIME_SETS = ("A-CCS23",)
CODE_COLUMNS = (
    ("population", "VARCHAR NOT NULL"), ("species", "VARCHAR NOT NULL"), ("source", "VARCHAR NOT NULL"),
    ("alignment_id", "VARCHAR NOT NULL"), ("molecule_id", "VARCHAR NOT NULL"), ("dataset", "VARCHAR NOT NULL"),
    ("run", "VARCHAR NOT NULL"), ("complete_5p_code", "VARCHAR NOT NULL"), ("complete_3p_code", "VARCHAR NOT NULL"),
    ("read_internal_code", "VARCHAR NOT NULL"), ("five_basis", "VARCHAR"), ("three_basis", "VARCHAR"),
    ("internal_basis", "VARCHAR"), ("internal_priming_suspect", "BOOLEAN NOT NULL"), ("tss_cluster_id", "VARCHAR"),
    ("tes_cluster_id", "VARCHAR"),
)
CODE_NAMES = tuple(c for c, _ in CODE_COLUMNS)


class CompletenessError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise CompletenessError(message)


# ------------------------------------------------------------------------------------------ CAGE (D12)
def load_recipe(path, expected_sha256):
    """The frozen CAGE recipe: its sha256 must equal ``expected_sha256`` (refuse without it) and its JSON block
    supplies every parameter the peak caller uses."""
    require(expected_sha256, "the CAGE recipe sha256 is required (D12: WP2c refuses to run without it)")
    sha = LK.sha256(path)
    require(sha == expected_sha256, f"{path}: sha256 {sha} != the frozen recipe sha {expected_sha256}")
    text = Path(path).read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    require(m, f"{path}: no JSON parameter block")
    params = json.loads(m.group(1))
    require(params.get("recipe_version") == "CAGE_RECIPE.b6_v1", f"{path}: unexpected recipe_version")
    return sha, params


def expand_runs(field):
    """'SRR1-SRR3' -> [SRR1, SRR2, SRR3]; comma lists; '-' -> []."""
    out = []
    for part in [p.strip() for p in (field or "").split(",") if p.strip() and p.strip() != "-"]:
        a, sep, b = part.partition("-")
        if not sep:
            out.append(part)
            continue
        ma, mb = re.fullmatch(r"([A-Z]+)(\d+)", a), re.fullmatch(r"([A-Z]+)(\d+)", b)
        require(ma and mb and ma.group(1) == mb.group(1), f"bad run range {part!r}")
        width = len(ma.group(2))
        out += [f"{ma.group(1)}{n:0{width}d}" for n in range(int(ma.group(2)), int(mb.group(2)) + 1)]
    return out


def read_cage_inputs(path):
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            for run in expand_runs(r["runs"]):
                rows[r["species"]].append((run, r["strategy"]))
    return rows


def cage_tags(line, genome, assay):
    """(chrom, strand, tag0, umi) of one CAGE SAM record, or None (filtered)."""
    f = line.rstrip("\n").split("\t")
    flag, chrom, pos, mapq, cigar, seq = int(f[1]), f[2], int(f[3]), int(f[4]), f[5], f[9]
    return flag, chrom, pos, mapq, cigar, seq, f[0]


def extract_tag(line, genome, params, assay_params):
    flag, chrom, pos, mapq, cigar, seq, name = cage_tags(line, genome, assay_params)
    if flag & params["filter"]["exclude_flags"] or mapq < params["filter"]["mapq_min"]:
        return None
    if flag & 1 and not flag & 0x40:
        return None                                              # read 1 only
    ops = [(int(n), o) for n, o in re.findall(r"(\d+)([MIDNSHP=X])", cigar) if o != "H"]
    lead = ops[0][0] if ops and ops[0][1] == "S" else 0
    trail = ops[-1][0] if len(ops) > 1 and ops[-1][1] == "S" else 0
    span = sum(n for n, o in ops if o in "MDN=X")
    start0, end0 = pos - 1, pos - 1 + span
    g = genome[chrom]
    first_m = next((n for n, o in ops if o not in "S"), 0) if ops else 0
    shift = 0
    if assay_params["g_rule"] and seq != "*":
        limit = min(params["g_rule_max"], first_m)
        for k in range(limit):
            if flag & 16:
                rb, gb = seq[len(seq) - 1 - trail - k].upper(), g[end0 - 1 - k].upper()
                if rb == "C" and gb != "C":
                    shift += 1
                    continue
            else:
                rb, gb = seq[lead + k].upper(), g[start0 + k].upper()
                if rb == "G" and gb != "G":
                    shift += 1
                    continue
            break
    strand = "-" if flag & 16 else "+"
    tag0 = (end0 - 1 - shift) if strand == "-" else (start0 + shift)
    umi = None
    if assay_params["dedup"] == "read_name_suffix_umi":
        umi = name.rsplit("_", 1)[-1] if "_" in name else ""
        require(len(umi) == assay_params["umi_len"], f"{name}: no {assay_params['umi_len']}-nt UMI suffix (recipe)")
    return chrom, strand, tag0, umi


def call_peaks(tags_by_run, params):
    """tags_by_run: run -> [(chrom, strand, tag0)] -> peaks [(chrom, strand, start0, end0, peak0, n_tags, n_libs)]."""
    by = defaultdict(list)
    for run, tags in tags_by_run.items():
        for chrom, strand, t in tags:
            by[(chrom, strand)].append((t, run))
    out = []
    gap, min_tags = params["peak"]["max_gap"], params["peak"]["min_tags"]
    for (chrom, strand), items in sorted(by.items()):
        items.sort()
        cur = [items[0]]
        for it in items[1:] + [None]:
            if it is not None and it[0] - cur[-1][0] <= gap:
                cur.append(it)
                continue
            if len(cur) >= min_tags:
                counts = Counter(t for t, _ in cur)
                top = max(counts.values())
                peak = min(t for t, c in counts.items() if c == top)
                out.append((chrom, strand, cur[0][0], cur[-1][0] + 1, peak, len(cur), len({r for _, r in cur})))
            cur = [it] if it is not None else []
    return out


def build_cage(species, cage_inputs, cage_paths, genome, params, samtools="samtools"):
    """Peaks of one species from its CAGE_INPUTS runs (every listed run must be supplied, sha recorded)."""
    from transgenic.evidence import b6_read_ends as RE

    ES = RE._es()
    runs = cage_inputs.get(species, [])
    tags_by_run, shas = {}, {}
    for run, assay in runs:
        require(run in cage_paths, f"{species}: CAGE run {run} of CAGE_INPUTS has no mapped alignment supplied")
        require(assay in params["assays"], f"{species}/{run}: assay {assay!r} has no recipe entry")
        ap = params["assays"][assay]
        shas[run] = LK.sha256(cage_paths[run])
        seen, tags = set(), []
        for line in ES.sam_lines(str(cage_paths[run]), samtools):
            if line.startswith("@") or not line.strip():
                continue
            t = extract_tag(line, genome, params, ap)
            if t is None:
                continue
            if ap["dedup"] != "none":
                if t in seen:
                    continue
                seen.add(t)
            tags.append(t[:3])
        tags_by_run[run] = tags
    return call_peaks(tags_by_run, params), shas


class PeakIndex:
    def __init__(self, peaks):
        self.by = defaultdict(list)
        for chrom, strand, a, b, peak, n, libs in peaks:
            self.by[(chrom, strand)].append(peak)
        for v in self.by.values():
            v.sort()

    def near(self, chrom, strand, pos, window=CAGE_WINDOW):
        import bisect

        v = self.by.get((chrom, strand), [])
        i = bisect.bisect_left(v, pos - window)
        return i < len(v) and v[i] <= pos + window


# ------------------------------------------------------------------------------------------ clusters
def greedy_clusters(positions):
    """Sorted positions -> [[positions]] with max - min <= DIAMETER, greedy left to right."""
    out = []
    for p in sorted(positions):
        if out and p - out[-1][0] <= DIAMETER:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def medoid(ps):
    return min(sorted(set(ps)), key=lambda m: (sum(abs(x - m) for x in ps), m))


# ------------------------------------------------------------------------------------------ populations
def read_admitted(path):
    """Admitted chains of ONE population: first line ``# population=<name>`` (declared independently of the rows, so
    a legitimately empty set is a hash-bound header-only file), then the TSV header (population, species, chrom,
    strand, introns='d-a,...' in transcription order); every row's population must equal the declaration."""
    sha = LK.sha256(path)
    side = Path(str(path) + ".sha256")
    require(side.is_file() and side.read_text().split()[:1] == [sha], f"{path}: admitted-chain sidecar missing or != sha")
    chains = defaultdict(set)
    with open(path, newline="") as fh:
        first = fh.readline().rstrip("\n")
        m = re.fullmatch(r"# population=(\S+)", first)
        require(m, f"{path}: the first line must declare '# population=<name>'")
        pop = m.group(1)
        require(pop in POPULATIONS, f"{path}: declared population {pop!r} not in {POPULATIONS}")
        r = csv.DictReader(fh, delimiter="\t")
        require(r.fieldnames == ["population", "species", "chrom", "strand", "introns"], f"{path}: bad header")
        for row in r:
            require(row["population"] == pop, f"{path}: row population {row['population']!r} != declared {pop!r}")
            introns = tuple(tuple(int(x) for x in it.split("-")) for it in row["introns"].split(",") if it)
            chains[(row["species"], row["chrom"], row["strand"])].add(introns)
    return pop, sha, chains


def population_members(con, species, roles):
    """alignment rows per population: {population: [(alignment_id, source, molecule_id, library, dataset, run)]}."""
    score = {(r.dataset, r.run) for r in roles.rows if r.b6_score}
    inputs = {(r.dataset, r.run) for r in roles.rows if r.b6_input}
    no3 = {(r.dataset, r.run) for r in roles.rows if r.set_label in NO_3PRIME_SETS}
    mol = {}
    for source, mid, lib, tt, te, tte, runs in con.execute(
            "SELECT source, molecule_id, min(library), bool_and(molecule_state = 'train_track'), "
            "bool_and(role = 'training_eligible'), bool_or(train_tiles_excluded), "
            "list(DISTINCT dataset || chr(9) || run ORDER BY dataset || chr(9) || run) FROM molecule_member "
            "WHERE species = ? GROUP BY source, molecule_id ORDER BY source, molecule_id", [species]).fetchall():
        mol[(source, mid)] = (lib, tt and te, tte, {tuple(x.split("\t", 1)) for x in runs})
    out = defaultdict(list)
    for aid, source, mid, ds, run, arm in con.execute(
            "SELECT alignment_id, source, molecule_id, dataset, run, arm FROM evidence_alignment WHERE species = ? AND "
            "accepted AND placement_kind = 'primary' AND source IN ('est', 'ont', 'pacbio') ORDER BY alignment_id",
            [species]).fetchall():
        if source == "est" and arm != "primary":
            continue
        lib, train_ok, tte, mruns = mol[(source, mid)]
        rec = (aid, source, mid, lib, ds, run, (ds, run) in no3)
        scored = bool(mruns & score) or (ds, run) in score
        if (ds, run) in score:
            out["score"].append(rec)
        if not scored and train_ok:
            out["valid_track"].append(rec)
            if not tte:
                out["train_track"].append(rec)
        if (ds, run) in inputs and not scored:
            out["eval_input"].append(rec)
    return out


def _chains(con, species):
    """alignment_id -> (strand, [introns in transcription order], all accepted) and blocks."""
    ch = defaultdict(lambda: [None, [], True])
    for aid, strand, d0, a0, acc in con.execute(
            "SELECT alignment_id, strand, donor0, acceptor0, accepted FROM junction_observation WHERE species = ? "
            "ORDER BY alignment_id, intron_index", [species]).fetchall():
        rec = ch[aid]
        rec[0] = strand
        rec[1].append((d0, a0))
        rec[2] = rec[2] and bool(acc) and d0 is not None
    blocks = defaultdict(list)
    for aid, a, b in con.execute("SELECT alignment_id, start0, end0 FROM aligned_block WHERE species = ? "
                                 "ORDER BY alignment_id, alignment_block_index", [species]).fetchall():
        blocks[aid].append((a, b))
    return ch, blocks


def internal_code(aid, chrom, strand, ch, blocks, admitted, species):
    rec = ch.get(aid)
    if rec is None or not rec[1]:
        return "IM", "mono_exonic"
    if not rec[2] or strand not in ("+", "-"):
        return "IX", "unaccepted_junction"
    introns = sorted(rec[1], key=lambda x: x[0], reverse=(strand == "-"))
    chains = admitted.get((species, chrom, strand), set())
    bl = sorted(blocks.get(aid, []))
    anchors_ok = bool(bl) and bl[0][1] - bl[0][0] >= ANCHOR_MIN and bl[-1][1] - bl[-1][0] >= ANCHOR_MIN
    t = tuple(introns)
    if t in chains:
        return ("IC", "exact_admitted") if anchors_ok else ("IX", "exact_admitted_short_anchor")
    for c in chains:
        n = len(t)
        if n < len(c) and any(c[i:i + n] == t for i in range(len(c) - n + 1)):
            return "IP", "sub_chain"
    return "IX", "unadmitted_chain"


def codes_for_population(pop, rows, ends, peaks, ch, blocks, admitted, species):
    """Codes of one population's alignments (its own clusters)."""
    tss_pos, tes_pos = defaultdict(list), defaultdict(list)
    for aid, source, mid, lib, ds, run, no3 in rows:
        e = ends[aid]
        if e["transcript_strand"] not in ("+", "-"):
            continue
        tss_pos[(e["chrom"], e["transcript_strand"])].append((e["tss0"], aid))
        if not no3:
            tes_pos[(e["chrom"], e["transcript_strand"])].append((e["tes0"], aid))
    meta = {aid: (source, mid, lib) for aid, source, mid, lib, *_ in rows}

    def tail_positive(e):
        return ((e["polyA_untemplated_len"] or 0) >= POLYA_MIN and e["downstream20_A"] is not None
                and e["downstream20_A"] < IP_WINDOW_A and e["downstream_max_consecutive_A"] < IP_RUN_A)

    def cap_assoc(aid):
        e = ends[aid]
        near = peaks.near(e["chrom"], e["transcript_strand"], e["tss0"])
        adapter = bool(e["adapter_5c_enabled"]) and (bool(e["primer5_hit"]) or (bool(e["ssp_adapter_hit"]) and near))
        return near or adapter

    tss_clusters, tes_clusters, tss_of, tes_of = [], [], {}, {}
    for (chrom, strand), items in sorted(tss_pos.items()):
        pos_to = defaultdict(list)
        for p, aid in items:
            pos_to[p].append(aid)
        for cl in greedy_clusters([p for p, _ in items]):
            aids = sorted({a for p in set(cl) for a in pos_to[p]})
            units = {meta[a][1]: meta[a][2] for a in aids}
            per_lib = Counter(units.values())
            cap_units = len({meta[a][1] for a in aids if cap_assoc(a)})
            m = medoid(cl)
            ok = (len(units) >= TSS_MIN_UNITS and sum(1 for v in per_lib.values() if v >= TSS_MIN_PER_LIB) >= TSS_MIN_LIBS
                  and cap_units >= TSS_MIN_CAP and peaks.near(chrom, strand, m))
            cid = f"{pop}:{species}:{chrom}:{strand}:tss:{cl[0]}"
            tss_clusters.append((pop, species, cid, chrom, strand, cl[0], cl[-1] + 1, m, len(units), len(per_lib),
                                 cap_units, ok))
            for a in aids:
                tss_of[a] = (cid, m, ok)
    for (chrom, strand), items in sorted(tes_pos.items()):
        pos_to = defaultdict(list)
        for p, aid in items:
            pos_to[p].append(aid)
        for cl in greedy_clusters([p for p, _ in items]):
            aids = sorted({a for p in set(cl) for a in pos_to[p]})
            tails = len({meta[a][1] for a in aids if tail_positive(ends[a])})
            m = medoid(cl)
            cid = f"{pop}:{species}:{chrom}:{strand}:tes:{cl[0]}"
            tes_clusters.append((pop, species, cid, chrom, strand, cl[0], cl[-1] + 1, m, len({meta[a][1] for a in aids}),
                                 tails, tails >= TES_MIN_TAIL))
            for a in aids:
                tes_of[a] = (cid, m, tails >= TES_MIN_TAIL)
    out = []
    for aid, source, mid, lib, ds, run, no3 in sorted(rows):
        e = ends[aid]
        strand = e["transcript_strand"]
        five, fb, three, tb, suspect = "5I", None, "3I", None, False
        if strand in ("+", "-"):
            near = peaks.near(e["chrom"], strand, e["tss0"])
            if e["adapter_5c_enabled"] and (bool(e["primer5_hit"]) or (bool(e["ssp_adapter_hit"]) and near)):
                five, fb = "5C", "adapter"
            else:
                cid, m, ok = tss_of.get(aid, (None, None, False))
                if ok and abs(e["tss0"] - m) <= TSS_WINDOW:
                    five, fb = "5C", "tss_cluster"
            suspect = (e["polyA_untemplated_len"] or 0) >= POLYA_MIN and not tail_positive(e)
            if no3:
                tb = "a_ccs23_never_3prime"
            elif tail_positive(e):
                three, tb = "3C", "polya"
            else:
                cid, m, ok = tes_of.get(aid, (None, None, False))
                if ok and abs(e["tes0"] - m) <= TES_WINDOW:
                    three, tb = "3C", "tes_cluster"
        ic, ib = internal_code(aid, e["chrom"], strand, ch, blocks, admitted, species)
        out.append((pop, species, source, aid, mid, ds, run, five, three, ic, fb, tb, ib, suspect,
                    tss_of.get(aid, (None,))[0], tes_of.get(aid, (None,))[0]))
    return out, tss_clusters, tes_clusters


# ------------------------------------------------------------------------------------------ build
def build(*, merged, read_ends, roles, admitted_chains, cage_recipe, cage_recipe_sha256, cage_inputs, cage_paths,
          reference, species, out, samtools="samtools", argv=()):
    import duckdb

    from transgenic.evidence import b6_read_ends as RE
    from transgenic.evidence import b6_roles as R

    out = Path(out).absolute()
    for p in (out, Path(str(out) + ".sha256"), Path(str(out) + ".PROVENANCE.json")):
        require(not os.path.lexists(p), f"{p} exists; refusing to overwrite")
    recipe_sha, params = load_recipe(cage_recipe, cage_recipe_sha256)
    merged_sha = RE._sidecar_sha(merged)
    re_sha, re_prov = RE.open_read_ends(read_ends)
    require(re_prov["species"] == species, f"{read_ends}: read-end supplement is for {re_prov['species']}")
    require(re_prov["inputs"]["merged"]["sha256"] == merged_sha, f"{read_ends}: built against another merged snapshot")
    roles_obj = R.load_roles(roles)
    ci_sha = LK.sha256(cage_inputs)
    side = Path(str(cage_inputs) + ".sha256")
    require(side.is_file() and side.read_text().split()[:1] == [ci_sha], f"{cage_inputs}: sidecar missing or != sha")
    admitted, admitted_shas = {}, {}
    for p in admitted_chains:
        pop, sha, chains = read_admitted(p)
        require(pop in POPULATIONS and pop not in admitted, f"{p}: population {pop!r} unknown or given twice")
        admitted[pop], admitted_shas[pop] = chains, {"path": str(Path(p).absolute()), "sha256": sha}
    missing = [p for p in POPULATIONS if p not in admitted]
    require(not missing, f"admitted-chain sets missing for populations {missing} (each population has its own)")
    require(len({v["sha256"] for v in admitted_shas.values()}) == len(POPULATIONS),
            "admitted-chain files of different populations must be distinct files (own hashed set each)")
    genome = LK.load_build_b5().load_fasta(str(reference))
    require(LK.sha256(reference) == re_prov["inputs"]["reference"]["sha256"], "reference sha256 != the read-end reference")
    peaks, cage_shas = build_cage(species, read_cage_inputs(cage_inputs), cage_paths, genome, params, samtools)
    pidx = PeakIndex(peaks)
    rcon = duckdb.connect(str(read_ends), read_only=True)
    ends = {r[0]: dict(zip(("alignment_id", "chrom", "transcript_strand", "tss0", "tes0", "polyA_untemplated_len",
                            "downstream20_A", "downstream_max_consecutive_A", "primer5_hit", "ssp_adapter_hit",
                            "adapter_5c_enabled"), r))
            for r in rcon.execute("SELECT alignment_id, chrom, transcript_strand, tss0, tes0, polyA_untemplated_len, "
                                  "downstream20_A, downstream_max_consecutive_A, primer5_hit, ssp_adapter_hit, "
                                  "adapter_5c_enabled FROM read_end ORDER BY alignment_id").fetchall()}
    rcon.close()
    mcon = duckdb.connect(str(merged), read_only=True)
    try:
        pops = population_members(mcon, species, roles_obj)
        ch, blocks = _chains(mcon, species)
    finally:
        mcon.close()
    tmp = Path(str(out) + ".partial")
    require(not os.path.lexists(tmp), f"{tmp} exists; remove it")
    qc = Counter()
    dst = duckdb.connect(str(tmp))
    try:
        dst.execute("SET threads = 1")
        dst.execute("CREATE TABLE completeness (" + ", ".join(f"{c} {t}" for c, t in CODE_COLUMNS)
                    + ", PRIMARY KEY (population, species, source, alignment_id))")
        dst.execute("CREATE TABLE tss_cluster (population VARCHAR, species VARCHAR, cluster_id VARCHAR PRIMARY KEY, "
                    "chrom VARCHAR, strand VARCHAR, start0 BIGINT, end0 BIGINT, medoid0 BIGINT, units INTEGER, "
                    "libraries INTEGER, cap_units INTEGER, qualifies_5c BOOLEAN)")
        dst.execute("CREATE TABLE tes_cluster (population VARCHAR, species VARCHAR, cluster_id VARCHAR PRIMARY KEY, "
                    "chrom VARCHAR, strand VARCHAR, start0 BIGINT, end0 BIGINT, medoid0 BIGINT, units INTEGER, "
                    "tail_units INTEGER, qualifies_3c BOOLEAN)")
        dst.execute("CREATE TABLE cage_peak (species VARCHAR, chrom VARCHAR, strand VARCHAR, start0 BIGINT, end0 BIGINT, "
                    "peak0 BIGINT, n_tags INTEGER, n_libraries INTEGER)")
        dst.execute("CREATE TABLE completeness_meta (key VARCHAR NOT NULL PRIMARY KEY, value VARCHAR NOT NULL)")
        if peaks:
            dst.executemany("INSERT INTO cage_peak VALUES (?,?,?,?,?,?,?,?)", [(species,) + p for p in peaks])
        for pop in POPULATIONS:
            rows = pops.get(pop, [])
            for r in rows:
                require(r[0] in ends, f"alignment {r[0]} has no read-end row")
            codes, tss, tes = codes_for_population(pop, rows, ends, pidx, ch, blocks, admitted[pop], species)
            if codes:
                dst.executemany(f"INSERT INTO completeness VALUES ({','.join('?' * len(CODE_COLUMNS))})", codes)
            if tss:
                dst.executemany("INSERT INTO tss_cluster VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", tss)
            if tes:
                dst.executemany("INSERT INTO tes_cluster VALUES (?,?,?,?,?,?,?,?,?,?,?)", tes)
            for c in codes:
                qc[f"{pop}:{c[2]}:{c[7]}"] += 1
                qc[f"{pop}:{c[2]}:{c[8]}"] += 1
                qc[f"{pop}:{c[2]}:{c[9]}"] += 1
        prov = {"version": VERSION, "policy": POLICY, "species": species, "populations": list(POPULATIONS),
                "cluster_rule": CLUSTER_RULE,
                "thresholds": {"tss_window": TSS_WINDOW, "tes_window": TES_WINDOW, "cage_window": CAGE_WINDOW,
                               "tss_min_units": TSS_MIN_UNITS, "tss_min_libraries": TSS_MIN_LIBS,
                               "tss_min_units_per_library": TSS_MIN_PER_LIB, "tss_min_cap_units": TSS_MIN_CAP,
                               "tes_min_tail_units": TES_MIN_TAIL, "polya_min": POLYA_MIN,
                               "internal_priming": {"window_A": IP_WINDOW_A, "run_A": IP_RUN_A},
                               "anchor_min": ANCHOR_MIN, "no_3prime_sets": list(NO_3PRIME_SETS)},
                "inputs": {"merged": {"path": str(Path(merged).absolute()), "sha256": merged_sha},
                           "read_ends": {"path": str(Path(read_ends).absolute()), "sha256": re_sha},
                           "roles": {"path": roles_obj.path, "sha256": roles_obj.sha256},
                           "cage_recipe": {"path": str(Path(cage_recipe).absolute()), "sha256": recipe_sha},
                           "cage_inputs": {"path": str(Path(cage_inputs).absolute()), "sha256": ci_sha},
                           "cage_alignments": cage_shas, "admitted_chains": admitted_shas,
                           "reference": {"path": str(Path(reference).absolute()), "sha256": LK.sha256(reference)}},
                "cage_peaks": len(peaks), "code_sha256": {Path(__file__).name: LK.sha256(__file__)},
                "qc": dict(sorted(qc.items())), "argv": list(argv), "duckdb_version": duckdb.__version__}
        dst.executemany("INSERT INTO completeness_meta VALUES (?, ?)",
                        sorted((k, json.dumps(v, sort_keys=True)) for k, v in prov.items()))
        dst.execute("CHECKPOINT")
    finally:
        dst.close()
    require(not os.path.exists(str(tmp) + ".wal"), f"{tmp}.wal remains; refusing to publish")
    os.link(tmp, out)
    tmp.unlink()
    digest = LK.sha256(out)
    with open(str(out) + ".sha256", "x") as fh:
        fh.write(f"{digest}  {out.name}\n")
    prov["output"], prov["output_sha256"] = str(out), digest
    with open(str(out) + ".PROVENANCE.json", "x") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return prov


def code_of(con, *, population, species, source, alignment_id):
    """(5', 3', internal) code of one alignment IN ONE POPULATION; a population is mandatory (never pooled)."""
    require(population in POPULATIONS, f"population {population!r} must be one of {POPULATIONS}")
    row = con.execute("SELECT complete_5p_code, complete_3p_code, read_internal_code FROM completeness WHERE "
                      "population = ? AND species = ? AND source = ? AND alignment_id = ?",
                      [population, species, source, alignment_id]).fetchone()
    require(row is not None, f"no completeness code for {alignment_id} in population {population}")
    return row
