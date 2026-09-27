#!/usr/bin/env python3
"""Independent audit of the WP2 rule 5 repair (plan v4.1 §2 acceptance): A40 base vs B6 corpus.

    b6_validate_corpus.py --base b5_full_a40.db --corpus b5_full_b6_v1.db --freeze freeze.json
                          --splits <split TSV> --flags SPECIES=<A22> [...] --build-flags SPECIES=<A22> SPECIES=<A30>
                          [--build-flags ...] --seed <corpus seed>
                          --interval-manifest <species>.jsonl [...] [--c0 <C0 or merged DB> ...] [--report out.json]

--flags / --build-flags take the per-species form of b6_build_corpus.py (the bare single-file form only for a
single-species base).

Checks (any failure exits 1):
  * freeze / file shas: base, corpus, manifests; ``b6_corpus_meta`` agrees; no WAL; every species' flag files hash
    to the freeze ``species_flags``.
  * the manifest is recomputed from the base with ``leakage.build_leakage_model`` and must be identical.
  * row audit (streamed, ORDER BY rn): same rows; untouched columns identical; ACGT->N only inside the row's repair
    spans and every repair span base is N; test tiles and unrepaired rows byte-identical; regenerated GSF blocks are
    a subset of the A40 blocks, one fewer per dropped gene; tokens recomputed; RC rows = revcomp / reverse_complement_v3
    of their forward row; masked fraction <= 0.60 for train/valid rows.
  * window_genes: A40 minus B6 == exactly the ``b6_label_drop`` rows; nothing added.
  * interval equality: every repair row = its manifest component interval ∩ tile (shared causes and A30-only).
  * independent recount: every seed edge crosser (leak or hard) of every kept train/valid forward tile has its
    component repaired, and no other component is; no labelled gene intersects a repair span.
  * counts by species x split x cause equal the freeze.
  * row-set drop (author decision 18): the base rows missing from the corpus are exactly the ``b6_row_drop`` rows
    and the freeze ``row_set_drop`` tiles; each is a train/valid tile whose independently recomputed repair crosses
    masked fraction 0.60 or orphans its RC twin (or that tile's RC twin), every such tile is dropped with its twin,
    their window_genes are gone, and the freeze ``train_rows`` / ``n_pilot`` equal floor(loader train rows * 2 / 96)
    of the corpus.
  * molecule consequences (``--c0``): every C0 molecule with a decision placement (``audit_placement``) overlapping a
    shared-cause span is ``excluded_heldout`` (scope all) or ``train_tiles_excluded`` (scope train_tiles, train tile);
    the two declared differences (A30-only spans, A48 edge_partial spans) are reported, not failed.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import b6_corpus_repair as CR  # noqa: E402
from transgenic.evidence import leakage as LK  # noqa: E402

MUTABLE = ("sequence", "gff", "gsf_token_count", "n_transcripts", "qc_flags", "strict_holdout")


class ValidationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def _ro(path):
    import duckdb

    require(os.path.isfile(path), f"missing {path}")
    require(not os.path.exists(str(path) + ".wal"), f"{path} has a WAL")
    return duckdb.connect(str(path), read_only=True)


def _columns(con):
    return [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'geneList' "
                                      "ORDER BY ordinal_position").fetchall()]


def _merge_spans(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _in_spans(merged, p):
    i = bisect.bisect_right([s[0] for s in merged], p) - 1
    return i >= 0 and merged[i][0] <= p < merged[i][1]


def audit_rows(base_con, b6_con, repairs, drops, qc, dropped=()):
    """Streamed row audit; ``repairs[rn]`` = [(start0, end0), ...] genomic; ``drops[rn]`` = {gene_id};
    ``dropped``: base rns the corpus no longer holds (decision 18), skipped on the base side."""
    gc = LK.load_build_b5().gc
    b5 = LK.load_build_b5()
    cols = _columns(base_con)
    require(cols == _columns(b6_con), "geneList columns differ between base and corpus")
    idx = {c: i for i, c in enumerate(cols)}
    sel = "SELECT " + ", ".join(cols) + " FROM geneList"
    ca, cb = base_con.cursor(), b6_con.cursor()
    if dropped:
        ca.execute(sel + " WHERE NOT list_contains(?, rn) ORDER BY rn", [sorted(dropped)])
    else:
        ca.execute(sel + " ORDER BY rn")
    cb.execute(sel + " ORDER BY rn")
    fwd = {}
    try:
        while True:
            ra, rb = ca.fetchmany(256), cb.fetchmany(256)
            require(len(ra) == len(rb), "row counts differ between base and corpus")
            if not ra:
                break
            for a, b in zip(ra, rb):
                rn = a[idx["rn"]]
                require(rn == b[idx["rn"]], f"row order differs at rn {rn} / {b[idx['rn']]}")
                qc["rows"] += 1
                for c in cols:
                    if c not in MUTABLE:
                        require(a[idx[c]] == b[idx[c]], f"rn {rn}: immutable column {c} changed")
                ws, we, is_rc, split = a[idx["start"]], a[idx["fin"]], a[idx["is_rc"]], a[idx["split"]]
                spans = repairs.get(rn, [])
                if not spans:
                    require(all(a[idx[c]] == b[idx[c]] for c in MUTABLE), f"rn {rn}: changed without a repair span")
                    require(rn not in drops, f"rn {rn}: label drop without a repair span")
                else:
                    require(split in ("train", "valid") or split is None, f"rn {rn}: repair in a {split} tile")
                    qc["rows_repaired"] += 1
                    L = we - ws
                    merged = _merge_spans([(s - ws, e - ws) if not is_rc else (we - e, we - s) for s, e in spans])
                    sa, sb = a[idx["sequence"]], b[idx["sequence"]]
                    require(len(sa) == len(sb) == L, f"rn {rn}: sequence length changed")
                    for p in range(L):
                        if sa[p] != sb[p]:
                            require(sb[p] == "N" and _in_spans(merged, p), f"rn {rn}: base {p} changed outside the "
                                                                           "repair spans or not to N")
                            qc["bases_masked"] += 1
                    for s, e in merged:
                        require(set(sb[s:e]) <= {"N"}, f"rn {rn}: repair span {s}-{e} is not all N")
                    ga, gb = gc.split_v3(a[idx["gff"]]), gc.split_v3(b[idx["gff"]])
                    require(Counter(gb) <= Counter(ga), f"rn {rn}: regenerated GSF adds a gene block")
                    require(len(ga) - len(gb) == len(drops.get(rn, ())),
                            f"rn {rn}: {len(ga) - len(gb)} gene blocks removed != {len(drops.get(rn, ()))} label drops")
                    require(b[idx["gsf_token_count"]] == gc.count_tokens_v3(b[idx["gff"]]), f"rn {rn}: token count")
                if split in ("train", "valid"):
                    require(gc.masked_fraction(b[idx["sequence"]]) <= gc.MASK_FRACTION_MAX,
                            f"rn {rn}: masked fraction above {gc.MASK_FRACTION_MAX}")
                if split == "test":
                    require(a == b, f"rn {rn}: test tile changed")
                if not is_rc:
                    fwd[b[idx["geneModel"]]] = (b[idx["sequence"]], b[idx["gff"]], we - ws)
                else:
                    wid = b[idx["geneModel"]][:-3]
                    require(wid in fwd, f"rn {rn}: RC row precedes or lacks its forward row {wid}")
                    s, g, L = fwd[wid]
                    require(b[idx["sequence"]] == b5.revcomp(s), f"rn {rn}: RC sequence != revcomp(forward)")
                    require(b[idx["gff"]] == gc.reverse_complement_v3(g, L), f"rn {rn}: RC gff != reverse_complement_v3")
    finally:
        ca.close()
        cb.close()


def audit_window_genes(base_con, b6_con, drop_keys, dropped_windows=()):
    """A40 minus B6 == the label drops plus every window_genes row of a dropped window (decision 18)."""
    q = "SELECT species_id, window_id, gene_id, is_rc FROM window_genes ORDER BY ALL"
    a = Counter(base_con.execute(q).fetchall())
    b = Counter(b6_con.execute(q).fetchall())
    require(not (b - a), f"window_genes rows added: {sorted((b - a).elements())[:5]}")
    removed = a - b
    gone = set(dropped_windows)
    want = Counter(drop_keys) + Counter({k: n for k, n in a.items() if (k[0], k[1]) in gone})
    require(removed == want, f"window_genes removed {sorted(removed.elements())[:5]} != label drops + dropped windows "
                             f"{sorted(want.elements())[:5]}")
    return sum(removed.values())


def audit_row_drop(base_con, b6_con, freeze, meta, qc):
    """The decision-18 drop record: table == freeze, rows gone from the corpus, each a train/valid base row whose
    forward tile's twin (if any) is dropped with it.  Returns {rn: record}."""
    require("b6_row_drop" in {r[0] for r in b6_con.execute("SELECT table_name FROM information_schema.tables").fetchall()},
            "corpus has no b6_row_drop table")
    policy = freeze.get("row_set_policy")
    require(policy in (CR.ROW_SET_REFUSE, CR.ROW_SET_DROP) and meta.get("row_set_policy") == policy,
            f"row_set_policy freeze {policy!r} / b6_corpus_meta {meta.get('row_set_policy')!r}")
    rows = b6_con.execute("SELECT " + ", ".join(CR.ROW_DROP_COLUMNS) + " FROM b6_row_drop ORDER BY rn").fetchall()
    recs = [dict(zip(CR.ROW_DROP_COLUMNS, r)) for r in rows]
    require(recs == freeze["row_set_drop"]["tiles"], "b6_row_drop != freeze row_set_drop tiles")
    require(policy == CR.ROW_SET_DROP or not recs, "rows dropped under the refuse policy")
    by_rn = {r["rn"]: r for r in recs}
    if by_rn:
        present = {x[0] for x in b6_con.execute("SELECT rn FROM geneList WHERE list_contains(?, rn)",
                                                [sorted(by_rn)]).fetchall()}
        require(not present, f"dropped rows still in the corpus: {sorted(present)[:5]}")
        base = {x[0]: x[1:] for x in base_con.execute(
            "SELECT rn, geneModel, species_id, split, is_rc FROM geneList WHERE list_contains(?, rn)",
            [sorted(by_rn)]).fetchall()}
        wids = {(r["species_id"], r["wid"]): r["rn"] for r in recs}
        for rn, r in by_rn.items():
            require(base.get(rn) == (r["wid"], r["species_id"], r["split"], r["is_rc"]),
                    f"b6_row_drop rn {rn} does not describe the base row {base.get(rn)}")
            require(r["split"] in ("train", "valid", None), f"dropped rn {rn} is a {r['split']} row")
            fwid = r["wid"][:-3] if r["is_rc"] else r["wid"]
            require(wids.get((r["species_id"], fwid)) == r["forward_rn"], f"dropped RC rn {rn} without its forward tile")
            twin = base_con.execute("SELECT rn FROM geneList WHERE species_id = ? AND geneModel = ?",
                                    [r["species_id"], fwid + "-rc"]).fetchall()
            require(not twin or (r["species_id"], fwid + "-rc") in wids, f"dropped tile rn {rn} kept its RC twin")
    for r in recs:
        qc["rows_dropped_" + ("rc" if r["is_rc"] else "forward")] += 1
    return by_rn


def audit_events(base_con, rn, wid, species, spans, gene_by_id, rc_mode):
    """Independent row-set events of one repaired forward tile: masked fraction of the base sequence with its repair
    spans set to N, and the RC twin's condition over the labelled genes the spans leave (``b6_corpus_repair``
    ``repair_tile`` semantics, recomputed from the base window_genes)."""
    gc = LK.load_build_b5().gc
    seq, ws, we = base_con.execute("SELECT sequence, start, fin FROM geneList WHERE rn = ?", [rn]).fetchone()
    chars = list(seq)
    for a, b in spans:
        chars[a - ws:b - ws] = "N" * (b - a)
    over = gc.masked_fraction("".join(chars)) > gc.MASK_FRACTION_MAX
    labelled = [gid for (gid,) in base_con.execute("SELECT gene_id FROM window_genes WHERE species_id = ? AND "
                                                   "window_id = ? AND NOT is_rc ORDER BY gene_id", [species, wid]).fetchall()]
    after = [gene_by_id[g] for g in labelled if not any(gene_by_id[g].start0 < b and gene_by_id[g].end0 > a
                                                         for a, b in spans)]
    twin = base_con.execute("SELECT count(*) FROM geneList WHERE species_id = ? AND geneModel = ?",
                            [species, wid + "-rc"]).fetchone()[0]
    orphan = bool(twin) and not CR._rc_condition(rc_mode, after)
    return ";".join(e for e, hit in zip(CR.EVENTS, (over, orphan)) if hit)


def recount(base_con, manifests_rows, repair_by_rn, drops_by_rn, args, qc, dropped=None):
    """Independent seed recount per kept train/valid forward tile, interval equality and closure; for every repaired
    forward tile, its row-set events decide exactly which tiles decision 18 dropped (``dropped``: {rn: record})."""
    dropped = dropped or {}
    gc = LK.load_build_b5().gc
    comps = CR.ComponentIndex(manifests_rows)
    b5 = LK.load_build_b5()
    for (species,) in base_con.execute("SELECT species_id FROM build_manifest ORDER BY species_id").fetchall():
        m = LK.corpus_manifest(base_con, species)
        genome = b5.load_fasta(args.references.get(species, m["fasta"]))
        a22, build = args.sflags[species]
        prep = LK.prepare_genes(species=species, gff_path=args.gffs.get(species, m["gff"]), splits_path=args.splits,
                                flags_path=a22, build_flags=build, genome=genome)
        blocks = defaultdict(list)
        for chrom, a, b, sp in base_con.execute("SELECT chromosome, start0, end0, split FROM tile_blocks WHERE species_id "
                                                "= ? ORDER BY ALL", [species]).fetchall():
            blocks[chrom].append((a, b, sp))
        rows = base_con.execute("SELECT rn, geneModel, chromosome, start, fin, is_rc FROM geneList WHERE species_id = ? "
                                "ORDER BY rn", [species]).fetchall()
        gene_by_id = {g.gene_id: g for gs in prep.by_chrom.values() for g in gs}
        rc_mode = base_con.execute("SELECT DISTINCT rc_mode FROM build_manifest WHERE species_id = ?",
                                   [species]).fetchall()[0][0]
        for rn, wid, chrom, ws, we, is_rc in rows:
            split = gc.tile_split(blocks[chrom], ws, we)
            got = {(r["component_id"], r["cause"], r["scope"], r["start0"], r["end0"]) for r in repair_by_rn.get(rn, [])}
            want = set()
            if split in ("train", "valid"):
                edge = [g for g in prep.by_chrom.get(chrom, []) if g.end0 > ws and g.start0 < we
                        and not (g.start0 >= ws and g.end0 <= we)]
                for g in edge:
                    meta = prep.gene_meta[g.gene_id]
                    leak = bool(meta["split"]) and gc.SPLIT_RANK.get(meta["split"], 0) > gc.SPLIT_RANK[split]
                    if leak or meta["weight"] == 0:
                        comp = comps.of(species, g.gene_id)
                        require(comp is not None, f"tile {rn}: seed {g.gene_id} has no manifest component")
                        for cause, scope, _ in comp.causes:
                            want.add((comp.component_id, cause, scope, max(ws, comp.start0), min(we, comp.end0)))
            if want and not is_rc:
                events = audit_events(base_con, rn, wid, species, sorted({(w[3], w[4]) for w in want}),
                                      gene_by_id, rc_mode)
                rec = dropped.get(rn)
                require(bool(events) == (rec is not None),
                        f"tile {rn} ({wid}): recomputed row-set events {events!r} but dropped={rec is not None}")
                if rec is not None:
                    require(rec["events"] == events, f"tile {rn}: dropped for {rec['events']!r}, recomputed {events!r}")
                    qc["tiles_dropped_events_confirmed"] += 1
            if rn in dropped:
                require(want, f"dropped rn {rn} ({wid}) has no repair (not an event tile)")
                continue
            require(got == want, f"tile {rn} ({wid}): repair rows {sorted(got)} != independent recount {sorted(want)}")
            if got and not is_rc:
                qc["tiles_recounted_repaired"] += 1
        # closure: no labelled gene of a repaired tile intersects its spans (B6 window_genes, forward and RC)
    return True


def audit_closure(b6_con, repair_by_rn, qc):
    genes = {}
    for sp, gid, chrom, a, b in b6_con.execute("SELECT species_id, gene_id, chromosome, start0, end0 FROM gene_key_map "
                                               "ORDER BY ALL").fetchall():
        genes[(sp, gid)] = (chrom, a, b)
    for rn, rows in sorted(repair_by_rn.items()):
        sp, wid = rows[0]["species"], rows[0]["wid"]
        window = wid[:-3] if wid.endswith("-rc") else wid
        is_rc = wid.endswith("-rc")
        for (gid,) in b6_con.execute("SELECT gene_id FROM window_genes WHERE species_id = ? AND window_id = ? AND is_rc = ? "
                                     "ORDER BY gene_id", [sp, window, is_rc]).fetchall():
            chrom, a, b = genes[(sp, gid)]
            for r in rows:
                require(not (a < r["end0"] and b > r["start0"]),
                        f"tile {rn}: labelled gene {gid} intersects repair span {r['start0']}-{r['end0']}")
        qc["tiles_closure_checked"] += 1


def audit_molecules(c0_paths, repair_rows, manifest_by_comp, qc):
    """Molecule consequences in C0 (plan v4.1 §2 acceptance)."""
    shared = defaultdict(list)          # chrom -> (start0, end0, scope, tile_split)
    a30 = defaultdict(list)
    for r in repair_rows:
        if r["is_rc"]:
            continue
        comp_rows = manifest_by_comp[(r["species"], r["chrom"], r["component_id"])]
        c0 = {(x["cause"], x["scope"]): x["c0_exclusion"] for x in comp_rows}
        target = shared if c0[(r["cause"], r["scope"])] else a30
        target[(r["species"], r["chrom"])].append((r["start0"], r["end0"], r["scope"], r["split"]))
    for path in c0_paths:
        con = _ro(path)
        try:
            states = {}
            for sp, src, mid, state, tte in con.execute(
                    "SELECT species, source, molecule_id, min(molecule_state), bool_or(train_tiles_excluded) "
                    "FROM molecule_member GROUP BY ALL ORDER BY ALL").fetchall():
                states[(sp, src, mid)] = (state, tte)
            cur = con.cursor()
            cur.execute("SELECT species, source, molecule_id, chrom, start0, end0 FROM audit_placement ORDER BY ALL")
            seen_bad, seen_ok, seen_a30 = set(), set(), set()
            while True:
                rows = cur.fetchmany(4096)
                if not rows:
                    break
                for sp, src, mid, chrom, s, e in rows:
                    key = (sp, src, mid)
                    state, tte = states.get(key, (None, None))
                    require(state is not None, f"{path}: placement of unknown molecule {key}")
                    for a, b, scope, split in shared.get((sp, chrom), ()):
                        if s < b and e > a:
                            ok = state == "excluded_heldout" if scope == "all" else (
                                state == "excluded_heldout" or (split == "train" and bool(tte)) or split != "train")
                            (seen_ok if ok else seen_bad).add(key)
                    for a, b, _, _ in a30.get((sp, chrom), ()):
                        if s < b and e > a and state == "train_track" and not tte:
                            seen_a30.add(key)
            cur.close()
            require(not seen_bad, f"{path}: molecules overlapping a shared-cause repair span are admitted: "
                                  f"{sorted(seen_bad)[:5]}")
            qc["c0_molecules_consistent"] += len(seen_ok)
            qc["declared_a30_only_admitted_molecules"] += len(seen_a30)
            ep = con.execute("SELECT count(*) FROM excluded_interval WHERE cause = 'edge_partial'").fetchone()[0]
            qc["declared_edge_partial_intervals_in_c0"] += ep
        finally:
            con.close()


def validate(args):
    qc = Counter()
    freeze = json.loads(Path(args.freeze).read_text())
    require(freeze.get("rule_id") == CR.RULE_ID, f"freeze rule_id {freeze.get('rule_id')!r} != {CR.RULE_ID}")
    require(LK.sha256(args.base) == freeze["base_sha256"], "base sha256 != freeze base_sha256")
    require(LK.sha256(args.corpus) == freeze["out_sha256"], "corpus sha256 != freeze out_sha256")
    base_con = _ro(args.base)
    try:
        species = [r[0] for r in base_con.execute("SELECT species_id FROM build_manifest ORDER BY species_id").fetchall()]
    finally:
        base_con.close()
    args.sflags = CR.species_flags(species, *CR.parse_flag_args(args.flags, args.build_flags))
    frozen = freeze.get("species_flags")
    require(isinstance(frozen, dict), "freeze records no per-species species_flags")
    got = {sp: (r["flags"]["sha256"], [b["sha256"] for b in r["build_flags"]])
           for sp, r in CR.species_flag_record(args.sflags).items()}
    want = {sp: (r["flags"]["sha256"], [b["sha256"] for b in r["build_flags"]]) for sp, r in frozen.items()}
    require(got == want, f"flag file sha256 per species {got} != freeze species_flags {want}")
    manifest_rows, shas = [], {}
    for p in args.interval_manifest:
        rows = LK.read_interval_manifest(p)
        sp = rows[0]["species"] if rows else None
        shas[sp] = LK.interval_manifest_sha256(rows)
        manifest_rows += rows
    require(shas == freeze["interval_manifest_sha256"], f"manifest shas {shas} != freeze")
    by_comp = defaultdict(list)
    for r in manifest_rows:
        by_comp[(r["species"], r["chrom"], r["component_id"])].append(r)
    base_con, b6_con = _ro(args.base), _ro(args.corpus)
    try:
        meta = dict(b6_con.execute("SELECT key, value FROM b6_corpus_meta ORDER BY key").fetchall())
        require(meta.get("rule_id") == CR.RULE_ID and meta.get("base_sha256") == freeze["base_sha256"],
                "b6_corpus_meta disagrees with the freeze")
        require(json.loads(meta["interval_manifest_sha256"]) == shas, "b6_corpus_meta manifest shas != manifests")
        if not args.no_recompute_manifest:
            for sp in sorted(shas):
                a22, build = args.sflags[sp]
                model = LK.build_leakage_model(corpus_db=args.base, species=sp, splits_path=args.splits,
                                               flags_path=a22, build_flags=build, seed=args.seed,
                                               fasta_path=args.references.get(sp), gff_path=args.gffs.get(sp))
                require(LK.interval_manifest_sha256(model.interval_manifest) == shas[sp],
                        f"{sp}: interval manifest != the one recomputed from the base corpus")
                LK.verify_interval_manifest(model.interval_manifest, model.intervals)
                qc["manifests_recomputed"] += 1
        splits = dict(b6_con.execute("SELECT rn, coalesce(split, 'train') FROM geneList ORDER BY rn").fetchall())
        repair_rows = []
        for row in b6_con.execute("SELECT " + ", ".join(CR.REPAIR_COLUMNS) + " FROM b6_dna_repair ORDER BY ALL").fetchall():
            r = dict(zip(CR.REPAIR_COLUMNS, row))
            r["species"], r["chrom"], r["wid"], r["split"] = r["species_id"], r["chromosome"], r["wid"], splits[r["rn"]]
            repair_rows.append(r)
        repairs, repair_by_rn = defaultdict(list), defaultdict(list)
        tiles = {rn: (ws, we) for rn, ws, we in b6_con.execute("SELECT rn, start, fin FROM geneList ORDER BY rn").fetchall()}
        for r in repair_rows:
            comp_rows = by_comp.get((r["species"], r["chrom"], r["component_id"]))
            require(comp_rows, f"repair row {r} names no manifest component")
            m = {(x["cause"], x["scope"]): x for x in comp_rows}
            require((r["cause"], r["scope"]) in m, f"repair row {r} cause/scope not in the manifest")
            ws, we = tiles[r["rn"]]
            x = m[(r["cause"], r["scope"])]
            require((r["start0"], r["end0"]) == (max(ws, x["start0"]), min(we, x["end0"])),
                    f"repair row rn {r['rn']} {r['component_id']}: span {r['start0']}-{r['end0']} != manifest ∩ tile")
            repairs[r["rn"]].append((r["start0"], r["end0"]))
            repair_by_rn[r["rn"]].append(r)
            qc["interval_equal_" + ("shared" if x["c0_exclusion"] else "a30_only")] += 1
        drops, drop_keys = defaultdict(set), []
        wid_of = dict(b6_con.execute("SELECT rn, geneModel FROM geneList ORDER BY rn").fetchall())
        species_of = dict(b6_con.execute("SELECT rn, species_id FROM geneList ORDER BY rn").fetchall())
        drop_counts = Counter()
        for rn, gid, cause in b6_con.execute("SELECT rn, gene_id, cause FROM b6_label_drop ORDER BY ALL").fetchall():
            require(cause in CR.DROP_CAUSES, f"label drop cause {cause!r}")
            require(rn in repair_by_rn, f"label drop on rn {rn} without a repair span")
            drops[rn].add(gid)
            wid = wid_of[rn]
            is_rc = wid.endswith("-rc")
            drop_keys.append((species_of[rn], wid[:-3] if is_rc else wid, gid, is_rc))
            if not is_rc:
                drop_counts[(species_of[rn], splits[rn], cause)] += 1
        dropped = audit_row_drop(base_con, b6_con, freeze, meta, qc)
        audit_rows(base_con, b6_con, repairs, drops, qc, dropped=set(dropped))
        qc["window_genes_removed"] = audit_window_genes(
            base_con, b6_con, drop_keys, {(r["species_id"], r["wid"]) for r in dropped.values() if not r["is_rc"]})
        recount(base_con, manifest_rows, repair_by_rn, drops, args, qc, dropped=dropped)
        audit_closure(b6_con, repair_by_rn, qc)
        counts = {}
        for r in repair_rows:
            if not r["is_rc"]:
                c = counts.setdefault(r["species"], {}).setdefault(r["split"], {}).setdefault(
                    r["cause"], {"spans": 0, "label_drops": 0})
                c["spans"] += 1
        for (sp, split, cause), n in drop_counts.items():
            counts.setdefault(sp, {}).setdefault(split, {}).setdefault(cause, {"spans": 0, "label_drops": 0})[
                "label_drops"] += n
        require(counts == freeze["counts"], f"counts {counts} != freeze {freeze['counts']}")
        require(_row_counts(b6_con) == freeze["row_counts"], "row counts != freeze")
        train_rows = CR.loader_train_rows(b6_con)
        require(freeze.get("train_rows") == train_rows and freeze.get("n_pilot") == CR.n_pilot_record(train_rows),
                f"freeze train_rows / n_pilot {freeze.get('train_rows')} / {freeze.get('n_pilot')} != corpus loader "
                f"train rows {train_rows} -> {CR.n_pilot_record(train_rows)}")
        qc["train_rows"], qc["n_pilot"] = train_rows, freeze["n_pilot"]["value"]
    finally:
        base_con.close()
        b6_con.close()
    if args.c0:
        audit_molecules(args.c0, repair_rows, by_comp, qc)
    return {"status": "PASS", "qc": dict(sorted(qc.items())), "counts": counts}


def _row_counts(con):
    return CR._row_counts(con)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--flags", action="append", required=True, help="SPECIES=<A22> per species (as b6_build_corpus)")
    ap.add_argument("--build-flags", action="append", nargs="+", required=True, help="SPECIES=PATH ... per species")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--interval-manifest", nargs="+", required=True)
    ap.add_argument("--c0", nargs="+", help="C0 DBs (per species x source, or merged) for the molecule consequences")
    ap.add_argument("--reference", action="append")
    ap.add_argument("--gff", action="append")
    ap.add_argument("--no-recompute-manifest", action="store_true",
                    help="skip recomputing the manifest from the base (default: recompute and require equality)")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    a.references = dict(v.split("=", 1) for v in a.reference or [])
    a.gffs = dict(v.split("=", 1) for v in a.gff or [])
    return a


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        out = validate(a)
    except (ValidationError, CR.RepairError, LK.LeakageError, OSError, KeyError) as exc:
        print(f"b6_validate_corpus FAILED: {exc}", file=sys.stderr)
        return 1
    if a.report:
        with open(a.report, "x") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
            fh.write("\n")
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
