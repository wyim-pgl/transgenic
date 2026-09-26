"""B6 input binding (plan §8): ``verify_inputs`` runs at every consumption and every resume.

The family is read from the VERIFIED evidence artefact's provenance policy (after its sha check), never from the
caller: ``eval_only`` iff every (species, source) of ``c0_provenance`` records policy ``eval_only_no_leakage_model``,
``training`` iff none does (a mix fails).  Each family has a fixed required-key set; a missing key fails, and there
is no "if the key exists" branch.  A consumer declares the families it accepts (training consumers accept only
``training``; A. thaliana evaluation accepts ``training``; Z. mays evaluation accepts ``eval_only``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from transgenic.evidence import b6_track_spec as TS

FAMILIES = ("training", "eval_only")
EVAL_ONLY_POLICY = "eval_only_no_leakage_model"
CACHE_FAMILY = {"training": "train", "eval_only": "eval"}
INTERVAL_MANIFEST_VERSION = "b6-intervals-v2"
INTERVAL_FIELDS = ("manifest_version", "species", "chrom", "component_id", "members", "start0", "end0", "cause",
                   "scope", "flank", "flank_source", "c0_exclusion")
C0_EXCLUSION_CAUSES = ("heldout", "test_orthogroup", "valid_orthogroup", "a22_hard")
COMPLETENESS_DEFAULT = "null_not_computed_at_c0"


class BindingError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise BindingError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recorded_sha(path):
    """The sha a snapshot's publisher recorded: ``<file>.sha256`` sidecar or PROVENANCE ``output_sha256``."""
    side = Path(str(path) + ".sha256")
    prov = Path(str(path) + ".PROVENANCE.txt")
    shas = []
    if side.is_file():
        shas.append(side.read_text().split()[0])
    if prov.is_file():
        for line in prov.read_text().splitlines():
            key, _, value = line.partition("\t")
            if key == "output_sha256":
                shas.append(value.strip())
    _require(shas, f"{path}: no .sha256 sidecar and no PROVENANCE output_sha256")
    _require(len(set(shas)) == 1, f"{path}: sidecar and PROVENANCE disagree {shas}")
    return shas[0]


def open_snapshot(path, *, expected=None):
    """Refuse a WAL, check recomputed == recorded (== expected), open read-only, re-check no WAL appeared."""
    import duckdb

    path = Path(path)
    wal = Path(str(path) + ".wal")
    _require(path.is_file(), f"{path} missing")
    _require(not wal.exists(), f"{wal} exists: refusing an unsealed snapshot")
    got = sha256_file(path)
    rec = recorded_sha(path)
    _require(got == rec, f"{path}: sha256 {got} != recorded {rec}")
    if expected is not None:
        _require(got == expected, f"{path}: sha256 {got} != bound {expected}")
    con = duckdb.connect(str(path), read_only=True)
    _require(not wal.exists(), f"{wal} appeared after opening")
    return got, con


def _prov(con):
    out = {}
    for sp, src, key, value in con.execute("SELECT species, source, key, value FROM c0_provenance").fetchall():
        try:
            v = json.loads(value)
        except (TypeError, ValueError):
            v = value
        out.setdefault((sp, src), {})[key] = v
    _require(out, "c0_provenance is empty")
    return out


def family_of(prov):
    pol = {k: v.get("policy") for k, v in prov.items()}
    ev = {k for k, p in pol.items() if p == EVAL_ONLY_POLICY}
    _require(not ev or len(ev) == len(pol), f"mixed provenance policies {sorted(pol.items())}")
    return "eval_only" if ev else "training"


def _need(d, key, where):
    _require(key in d, f"{where}: required key {key!r} missing")
    return d[key]


def _input_sha(p, name, where):
    inputs = _need(p, "inputs", where)
    _require(isinstance(inputs, dict), f"{where}: inputs is not a mapping")
    item = _need(inputs, name, f"{where} inputs")
    return _need(item, "sha256", f"{where} inputs.{name}")


def interval_manifest_sha256(rows):
    """Canonical §2.5 hash: JSON lines (sort_keys, (',', ':')), rows sorted by
    (species, chrom, start0, end0, cause, scope, component_id), manifest_version in every row."""
    lines = []
    for r in rows:
        _require(tuple(r) == INTERVAL_FIELDS, f"interval manifest row fields {tuple(r)} != {INTERVAL_FIELDS}")
        _require(r["manifest_version"] == INTERVAL_MANIFEST_VERSION, "interval manifest version mismatch")
        lines.append(r)
    lines.sort(key=lambda r: (r["species"], r["chrom"], r["start0"], r["end0"], r["cause"], r["scope"],
                              r["component_id"]))
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in lines)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def revalidate_interval_manifest(recorded_leakage_sha, recomputed, manifest_rows, legacy_sha):
    """§8.3 revalidation of a C0 lacking ``interval_manifest_sha256``.

    (a) ``legacy_sha(recomputed)`` (b6_build_c0.intervals_sha256 over the COMPLETE rerun of build_leakage_model)
        must equal the recorded ``leakage_intervals_sha256``;
    (b) the manifest rows with c0_exclusion = true, projected to (chrom, start0, end0, cause, scope), must equal as
        a multiset the recomputed genome-level component rows (tile_rn NULL, causes C0_EXCLUSION_CAUSES);
    (c) A30-only rows must have c0_exclusion = false and match no recomputed interval.
    ``recomputed`` items expose chrom, start0, end0, cause, scope, gene_id, tile_rn.  Returns the manifest sha.
    """
    _require(legacy_sha(recomputed) == recorded_leakage_sha, "revalidation (a): the recomputed legacy interval set "
                                                             "does not reproduce leakage_intervals_sha256")
    from collections import Counter

    comp = Counter((iv.chrom, iv.start0, iv.end0, iv.cause, iv.scope) for iv in recomputed
                   if iv.tile_rn is None and iv.cause in C0_EXCLUSION_CAUSES)
    proj = Counter((r["chrom"], r["start0"], r["end0"], r["cause"], r["scope"]) for r in manifest_rows
                   if r["c0_exclusion"] is True)
    _require(comp == proj, f"revalidation (b): manifest projection != recomputed components "
                           f"(missing {sorted((comp - proj).elements())[:3]}, extra {sorted((proj - comp).elements())[:3]})")
    every = {(iv.chrom, iv.start0, iv.end0) for iv in recomputed}
    for r in manifest_rows:
        if r["cause"] == "a30_only":
            _require(r["c0_exclusion"] is False, "revalidation (c): an A30-only row claims a C0 exclusion")
            _require((r["chrom"], r["start0"], r["end0"]) not in every,
                     "revalidation (c): an A30-only row matches a recomputed interval")
    return interval_manifest_sha256(manifest_rows)


AUDITED_SOURCES = ("est", "ont", "pacbio")
_HEX = set("0123456789abcdef")


def _is_sha(v):
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX


def _is_count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def coverage(con, prov):
    """Expected (species, source) coverage: the provenance keys, which must equal the merge manifest's."""
    merged = {(sp, src) for sp, src in con.execute("SELECT species, source FROM merge_manifest").fetchall()}
    _require(merged, "merge_manifest is empty: not a merged snapshot")
    _require(set(prov) == merged, f"c0_provenance coverage {sorted(prov)} != merge_manifest {sorted(merged)}")
    return sorted(merged)


def _leakage_gate(con, cover):
    """Every covered (species, source) must record leakage_nonzero = 0; every leakage* counter must be 0."""
    per = {}
    for sp, src, metric, value in con.execute("SELECT species, source, metric, value FROM c0_qc").fetchall():
        per.setdefault((sp, src), {})[metric] = value
    for key in cover:
        qc = per.get(key)
        _require(qc is not None, f"{key}: no c0_qc record")
        _require("leakage_nonzero" in qc, f"{key}: c0_qc lacks leakage_nonzero")
        bad = {m: v for m, v in qc.items() if m.startswith("leakage") and v != 0}
        _require(not bad, f"{key}: leakage QC counters not zero {bad}")
    extra = sorted(set(per) - set(cover))
    _require(not extra, f"c0_qc records outside the merged coverage: {extra}")
    return per


def _exhaustiveness_gate(con, cover):
    """A49.2 from c0_meta.exhaustiveness: every covered audited (species, source) has integer counters;
    C <= floor(0.01 M) per source and sum C / sum M <= 0.001 per species."""
    meta = dict(con.execute("SELECT key, value FROM c0_meta").fetchall())
    ex = json.loads(_need(meta, "exhaustiveness", "c0_meta"))
    _require(isinstance(ex, dict), "c0_meta.exhaustiveness is not a mapping")
    want = {}
    for sp, src in cover:
        if src in AUDITED_SOURCES:
            want.setdefault(sp, set()).add(src)
    _require(bool(want), "no audited (species, source) in the merged coverage")
    for sp, sources in sorted(want.items()):
        rec = _need(ex, sp, "c0_meta.exhaustiveness")
        by = _need(rec, "by_source", f"exhaustiveness {sp}")
        _require(isinstance(by, dict) and set(by) == sources,
                 f"A49.2: {sp} exhaustiveness sources {sorted(by) if isinstance(by, dict) else by} != {sorted(sources)}")
        tot_c = tot_m = 0
        for src in sorted(sources):
            part = by[src]
            m, below = part.get("molecules"), part.get("below_cap")
            _require(_is_count(m) and _is_count(below) and below <= m,
                     f"A49.2: {sp}/{src} counters must be integers with below_cap <= molecules, got {part}")
            c = m - below
            _require(c <= m // 100, f"A49.2: {sp}/{src} capped {c} > floor(0.01 x {m})")
            tot_c, tot_m = tot_c + c, tot_m + m
        _require(tot_c * 1000 <= tot_m, f"A49.2: {sp} capped {tot_c} / {tot_m} > 0.001")
    return ex


def legacy_intervals_sha256(intervals):
    """Byte-identical to b6_build_c0.intervals_sha256 (sid('intervals', sorted rows))."""
    rows = sorted((iv.chrom, iv.start0, iv.end0, iv.cause, iv.scope, iv.gene_id or "",
                   -1 if iv.tile_rn is None else iv.tile_rn) for iv in intervals)
    return hashlib.sha256(json.dumps(("intervals", rows), separators=(",", ":")).encode()).hexdigest()


def rerun_leakage_model(p, species):
    """§8.3 (a): rerun build_leakage_model from the C0's RECORDED inputs and seed; each input's bytes must still
    hash to the recorded sha256.

    Not relocatable, by design: the recorded inputs are the C0 builder's own absolute paths (corpus, splits, flags,
    GFF, reference), so this runs only where they exist.  It is reached only for a legacy C0 that records no
    ``interval_manifest_sha256``; every C0 built by b6_build_c0 since plan v4.1 §2.5 records it (the B6 training
    C0s do), and verify_upstream then compares that sha with the freeze instead of rerunning."""
    from transgenic.evidence import leakage as LK

    inputs = _need(p, "inputs", f"c0_provenance {species}")
    paths = {}
    order = ("corpus", "splits", "flags", "gff", "reference")
    for name, rec in sorted(inputs.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else len(order),
                                                            kv[0])):
        if name in order or name.startswith("build_flags_"):
            path = Path(_need(rec, "path", f"inputs.{name}"))
            _require(path.is_file() and sha256_file(path) == _need(rec, "sha256", f"inputs.{name}"),
                     f"revalidation: recorded input {name} {path} is missing or changed (legacy C0 revalidation "
                     "reads the builder's recorded absolute input paths; it is not relocatable)")
            paths[name] = str(path)
    for name in ("corpus", "splits", "flags"):
        _require(name in paths, f"revalidation: C0 records no {name} input")
    build_flags = [paths[k] for k in sorted((k for k in paths if k.startswith("build_flags_")),
                                            key=lambda k: int(k.rsplit("_", 1)[1]))]
    model = LK.build_leakage_model(corpus_db=paths["corpus"], species=species, splits_path=paths["splits"],
                                   flags_path=paths["flags"], build_flags=build_flags or None,
                                   gff_path=paths.get("gff"), fasta_path=paths.get("reference"),
                                   seed=_need(p, "seed", f"c0_provenance {species}"))
    return model.intervals


def read_interval_manifest(path, expected_sha256):
    """A b6-intervals-v2 JSONL: canonical serialisation, file sha == canonical sha == sidecar == expected."""
    path = Path(path)
    _require(path.is_file(), f"interval manifest {path} missing")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        _require(line.endswith("\n"), f"{path}:{n}: unterminated line")
        row = json.loads(line)
        _require(json.dumps(row, sort_keys=True, separators=(",", ":")) == line[:-1],
                 f"{path}:{n}: not the canonical serialisation")
        rows.append({k: row[k] for k in INTERVAL_FIELDS} if set(row) == set(INTERVAL_FIELDS) else row)
    digest = interval_manifest_sha256(rows)
    _require(sha256_file(path) == digest, f"{path}: file sha256 != canonical manifest sha256")
    side = Path(str(path) + ".sha256")
    _require(side.is_file() and side.read_text().split()[:1] == [digest], f"{path}: sidecar missing or wrong")
    _require(digest == expected_sha256, f"{path}: manifest sha256 {digest} != freeze {expected_sha256}")
    return rows


def _freeze_manifests(freeze, species_set):
    man = _need(freeze, "interval_manifest_sha256", "freeze")
    _require(isinstance(man, dict), "freeze interval_manifest_sha256 must be the species-indexed mapping "
                                    "{species: sha256} (b6_corpus_repair freeze format)")
    _require(all(_is_sha(v) for v in man.values()), "freeze interval_manifest_sha256 values must be sha256 hex")
    missing = sorted(species_set - set(man))
    _require(not missing, f"freeze interval_manifest_sha256 lacks species {missing}")
    return man


def freeze_manifest_path(freeze_path, rec, species):
    """Resolve a freeze ``interval_manifests[species].path`` (decision 14): a relative path is resolved against the
    directory of the freeze file as it is found now (the bundle may have moved hosts); a legacy absolute path is
    used as recorded.  A path that does not resolve to a file fails closed, naming the recorded path."""
    raw = _need(rec, "path", f"interval_manifests[{species}]")
    _require(isinstance(raw, str) and raw, f"freeze interval_manifests[{species}].path must be a non-empty string")
    path = Path(raw)
    if path.is_absolute():
        _require(path.is_file(), f"interval manifest {raw!r} (absolute path recorded in the freeze) missing")
        return path
    path = Path(freeze_path).parent / path
    _require(path.is_file(), f"interval manifest {raw!r} (relative to the freeze directory "
                             f"{Path(freeze_path).parent}) missing: {path}")
    return path


def freeze_species_flags(freeze, species_set):
    """The freeze's per-species flag shas {species: (a22 sha, [build flag shas])} (b6_corpus_repair
    ``species_flags``): every covered species present, every value a sha256."""
    sf = _need(freeze, "species_flags", "freeze")
    _require(isinstance(sf, dict), "freeze species_flags must be the per-species mapping {species: {flags, build_flags}}")
    missing = sorted(species_set - set(sf))
    _require(not missing, f"freeze species_flags lacks species {missing}")
    out = {}
    for sp in sorted(species_set):
        rec = sf[sp]
        a22 = _need(_need(rec, "flags", f"freeze species_flags[{sp}]"), "sha256", f"freeze species_flags[{sp}].flags")
        build = _need(rec, "build_flags", f"freeze species_flags[{sp}]")
        _require(isinstance(build, list) and build, f"freeze species_flags[{sp}].build_flags must be a non-empty list")
        build = [_need(b, "sha256", f"freeze species_flags[{sp}].build_flags") for b in build]
        _require(all(_is_sha(v) for v in (a22, *build)), f"freeze species_flags[{sp}] values must be sha256 hex")
        out[sp] = (a22, build)
    return out


def c0_flag_shas(p, where):
    """A C0's recorded (inputs.flags sha, effective build flag shas): ``build_flags_<i>`` in index order, or the
    A22 file alone when the C0 was built without --build-flags (leakage.prepare_genes' default)."""
    a22 = _input_sha(p, "flags", where)
    inputs = p["inputs"]
    keys = sorted((k for k in inputs if k.startswith("build_flags_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    _require([int(k.rsplit("_", 1)[1]) for k in keys] == list(range(len(keys))),
             f"{where}: build_flags_* indices {keys} are not 0..n-1")
    build = [_input_sha(p, k, where) for k in keys]
    return a22, build or [a22]


def merged_input_path(evidence, raw, key):
    """Resolve a ``merge_manifest.input_path`` (decision 14): a relative path is resolved against the directory of the
    merged evidence DB as it is found now (the merged DB and its C0 bundle may have moved hosts together, in the same
    relative layout); a legacy absolute path is used as recorded.  Returns the C0's ``.PROVENANCE.txt`` path, which
    must exist (fail closed, naming the recorded path)."""
    _require(isinstance(raw, str) and raw, f"C0 {key}: merge_manifest input_path must be a non-empty string")
    path = Path(raw)
    if path.is_absolute():
        pv = Path(str(path) + ".PROVENANCE.txt")
        _require(pv.is_file(), f"C0 {key}: PROVENANCE file {pv} missing (absolute input_path recorded by the merge)")
        return pv
    base = Path(evidence).absolute().parent
    pv = Path(os.path.normpath(base / path) + ".PROVENANCE.txt")
    _require(pv.is_file(), f"C0 {key}: PROVENANCE file of input_path {raw!r} (relative to the merged evidence "
                           f"directory {base}) missing: {pv}")
    return pv


ZERO_UPSTREAM = {"est": "est_c0", "protein": "protein_c0"}   # b6_routea_agreement QC upstream keys


def zero_list_check(zero_lists, pops, merge_inputs, species_set):
    """{population: {species: (tsv, qc)}}: every required population x covered species present; QC names the
    population and species; ``upstream`` has EXACTLY the producer keys of that species' covered route-A/B inputs
    (``est_c0`` always, ``protein_c0`` when the species has a protein C0) and each equals the merge-manifest sha
    of its own (species, source).  ``merge_inputs`` = {(species, source): input sha256}."""
    out = {}
    for pop in pops:
        per = zero_lists.get(pop) or {}
        missing = sorted(species_set - set(per))
        _require(not missing, f"no zero list for population {pop}, species {missing}")
        for sp, (tsv, qc_path) in sorted(per.items()):
            _require(Path(tsv).is_file(), f"zero list {tsv} missing")
            qc = json.loads(Path(qc_path).read_text())
            _require(qc.get("population") == pop, f"{qc_path}: population {qc.get('population')!r} != {pop!r}")
            _require(qc.get("species") == sp, f"{qc_path}: species {qc.get('species')!r} != {sp!r}")
            up = _need(qc, "upstream", str(qc_path))
            _require(isinstance(up, dict), f"{qc_path}: upstream is not a mapping")
            _require((sp, "est") in merge_inputs, f"{qc_path}: {sp} has no merged EST C0 (route B needs one)")
            want = {key: merge_inputs[(sp, src)] for src, key in ZERO_UPSTREAM.items() if (sp, src) in merge_inputs}
            _require(set(up) == set(want), f"{qc_path}: upstream keys {sorted(up)} != required {sorted(want)}")
            for key, sha in sorted(want.items()):
                _require(up[key] == sha, f"{qc_path}: upstream {key} {str(up[key])[:12]} != merged {sp} "
                                         f"{key[:-3]} C0 {sha[:12]}")
            out.setdefault(pop, {})[sp] = {"tsv_sha256": sha256_file(tsv), "qc_sha256": sha256_file(qc_path)}
    return out


def _prov_add(prov, name, path, *, required=True):
    """Re-hash one consumed PROVENANCE/sidecar file into the record; a missing required file fails."""
    path = Path(path)
    if not path.is_file():
        _require(not required, f"provenance file {name} ({path}) missing")
        return
    _require(name not in prov, f"duplicate provenance entry {name}")
    prov[name] = sha256_file(path)


def canonical_record_sha256(record):
    body = {k: v for k, v in record.items() if k != "record_sha256" and not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=list)
                          .encode("utf-8")).hexdigest()


def _snapshot_provenance(prov, name, path):
    """A published snapshot is consumed through its .sha256 sidecar and/or its PROVENANCE.txt: every one that
    exists is recorded, and at least one must exist (open_snapshot already required it)."""
    side, pv = Path(str(path) + ".sha256"), Path(str(path) + ".PROVENANCE.txt")
    _require(side.is_file() or pv.is_file(), f"{path}: no sidecar and no PROVENANCE file")
    _prov_add(prov, f"{name}.sha256", side, required=False)
    _prov_add(prov, f"{name}.PROVENANCE.txt", pv, required=False)


def verify_upstream(*, accept, evidence, roles, zero_lists, freeze=None, corpus=None, references=None, panel=None,
                    window_manifest=None, completeness=None, evlabel=None, aug_seed=None, draws=None,
                    leakage_rerun=rerun_leakage_model, stage="consume"):
    """Everything upstream of the cache: evidence snapshot + family + typed keys + gates + references + zero lists.

    ``references``: {species: object with .sha256}.  ``stage="build"`` (cache publication) requires a reference for
    EVERY covered species; ``stage="consume"`` checks every supplied one (eval consumers must supply the species
    they read DNA for).  Each (species, source) provenance inputs.reference must equal its species' reference.
    ``zero_lists``: {population: {species: (tsv, qc)}}.  The eval_only family also needs the panel and window
    manifest at consumption.
    """
    from transgenic.evidence import c0_schema as CS

    _require(evlabel is None, "evlabel is bound: B6 consumers never read evlabel (A50.2)")
    freeze_path = None
    if freeze is not None:
        _require(isinstance(freeze, (str, Path)), "freeze must be the path of the frozen corpus freeze JSON "
                                                  "(its bytes are a provenance entry)")
        freeze_path = Path(freeze)
        _require(freeze_path.is_file(), f"freeze {freeze_path} missing")
        freeze = json.loads(freeze_path.read_text())
    provenance = {}
    accept = set(accept)
    _require(accept and accept <= set(FAMILIES), f"accept {accept} not within {FAMILIES}")
    ev_sha, con = open_snapshot(evidence)
    try:
        CS.verify_schema(con, merged=True)
        prov = _prov(con)
        family = family_of(prov)
        _require(family in accept, f"evidence family {family!r} not accepted by this consumer ({sorted(accept)})")
        cover = coverage(con, prov)
        species_set = {sp for sp, _ in cover}
        _leakage_gate(con, cover)
        ident = {"family": family, "evidence_sha256": ev_sha, "roles_sha256": roles.sha256, "coverage": cover,
                 "spec": dict(TS.SPEC_SHAS), "aug_seed": aug_seed, "draws": draws, "evlabel": "absent"}
        merge_inputs = {(sp, src): sha for sha, sp, src in
                        con.execute("SELECT input_sha256, species, source FROM merge_manifest").fetchall()}
        _snapshot_provenance(provenance, "evidence", evidence)
        for path, sha, sp, src in con.execute("SELECT input_path, input_sha256, species, source FROM merge_manifest "
                                              "ORDER BY species, source").fetchall():
            pv = merged_input_path(evidence, path, f"{sp}/{src}")
            rec = dict(line.split("\t", 1) for line in pv.read_text().splitlines() if "\t" in line)
            _require(rec.get("output_sha256", "").strip() == sha,
                     f"C0 {sp}/{src}: PROVENANCE output_sha256 != merge_manifest input sha256")
            _prov_add(provenance, f"c0:{sp}/{src}.PROVENANCE.txt", pv)
        _require(stage in ("build", "consume"), f"bad stage {stage!r}")
        if stage == "build":
            _require(references is not None and species_set <= set(references),
                     f"cache build needs a reference for every covered species {sorted(species_set)}")
        if references:
            unknown = sorted(set(references) - species_set)
            _require(not unknown, f"references for species outside the evidence coverage: {unknown}")
            for key, p in sorted(prov.items()):
                ref = references.get(key[0])
                if ref is None:
                    continue
                _require(_input_sha(p, "reference", f"c0_provenance {key}") == ref.sha256,
                         f"c0_provenance {key}: inputs.reference != supplied {key[0]} reference {ref.sha256[:12]}")
            ident["reference_sha256"] = {sp: references[sp].sha256 for sp in sorted(references)}
        if family == "training":
            _require(freeze is not None, "training family: the corpus freeze is required")
            _require(corpus is not None, "training family: the B6 corpus is required")
            base = _need(freeze, "base_sha256", "freeze")
            out_sha = _need(freeze, "out_sha256", "freeze")
            no_protein = sorted(sp for sp in species_set if (sp, "protein") not in merge_inputs)
            _require(not no_protein, f"training family: species without a merged protein C0 {no_protein} (every "
                                     "training species has miniprot alignments; route A/B zero lists need it)")
            man = _freeze_manifests(freeze, species_set)
            _prov_add(provenance, "freeze", freeze_path)
            locs = _need(freeze, "interval_manifests", "freeze")
            for sp in sorted(species_set):
                rec = _need(locs, sp, "freeze interval_manifests")
                _require(rec.get("sha256") == man[sp], f"freeze interval_manifests[{sp}] sha != mapping")
                mpath = freeze_manifest_path(freeze_path, rec, sp)
                _prov_add(provenance, f"interval_manifest:{sp}", mpath)
                _prov_add(provenance, f"interval_manifest:{sp}.sha256", Path(str(mpath) + ".sha256"))
                _require(provenance[f"interval_manifest:{sp}"] == man[sp],
                         f"interval manifest file {mpath} sha256 != freeze[{sp}]")
            c_sha = sha256_file(corpus)
            _require(not Path(str(corpus) + ".wal").exists(), f"{corpus}.wal exists")
            _require(c_sha == out_sha, f"B6 corpus sha256 {c_sha} != freeze out_sha256 {out_sha}")
            revalidated = {}
            frozen_flags = freeze_species_flags(freeze, species_set)
            for key, p in sorted(prov.items()):
                sp, where = key[0], f"c0_provenance {key}"
                _require(_input_sha(p, "corpus", where) == base, f"{where}: inputs.corpus != freeze base_sha256")
                a22, build = c0_flag_shas(p, where)
                _require(a22 == frozen_flags[sp][0], f"{where}: inputs.flags != freeze species_flags[{sp}].flags "
                                                     "(the repair must use each species' own A22 file)")
                _require(build == frozen_flags[sp][1], f"{where}: build flags {build} != freeze "
                                                       f"species_flags[{sp}].build_flags {frozen_flags[sp][1]}")
                legacy = _need(p, "leakage_intervals_sha256", where)
                if "interval_manifest_sha256" in p:
                    _require(p["interval_manifest_sha256"] == man[sp],
                             f"{where}: interval_manifest_sha256 != freeze[{sp}]")
                else:
                    locs = _need(freeze, "interval_manifests", "freeze (legacy C0 revalidation)")
                    rec = _need(locs, sp, "freeze interval_manifests")
                    _require(rec.get("sha256") == man[sp], f"freeze interval_manifests[{sp}] sha != mapping")
                    rows = read_interval_manifest(freeze_manifest_path(freeze_path, rec, sp), man[sp])
                    got = revalidate_interval_manifest(legacy, leakage_rerun(p, sp), rows, legacy_intervals_sha256)
                    _require(got == man[sp], f"{where}: revalidated manifest {got} != freeze[{sp}]")
                    revalidated["/".join(key)] = {"c0_sha256": ev_sha, "leakage_intervals_sha256": legacy,
                                                  "interval_manifest_sha256": got}
                pol = _need(_need(p, "completeness_codes", where), "policy", f"{where} completeness_codes")
                if completeness is None:
                    _require(pol == COMPLETENESS_DEFAULT, f"{where}: completeness policy {pol!r} without a WP2c DB")
            ident["exhaustiveness"] = _exhaustiveness_gate(con, cover)
            ident.update(corpus_sha256=c_sha, base_sha256=base, interval_manifest_sha256=dict(sorted(man.items())),
                         revalidated=revalidated)
        elif stage == "consume":
            _require(bool(references) and panel is not None and window_manifest is not None,
                     "eval_only family: references, panel and window manifest are required")
        if panel is not None:
            ident["panel_sha256"] = panel.sha256
        if window_manifest is not None:
            ident["window_manifest_sha256"] = window_manifest.sha256
    finally:
        con.close()
    pops = ("eval_input",) if family == "eval_only" else ("train_track", "valid_track", "eval_input")
    ident["zero_lists"] = zero_list_check(zero_lists, pops, merge_inputs, species_set)
    for pop in pops:
        for sp, (tsv, qc) in sorted(zero_lists[pop].items()):
            if sp in species_set:
                _prov_add(provenance, f"zero_list:{pop}/{sp}", tsv)
                _prov_add(provenance, f"zero_qc:{pop}/{sp}", qc)
    ident["provenance"] = dict(sorted(provenance.items()))
    ident["_merge_inputs"] = merge_inputs
    ident["record_sha256"] = canonical_record_sha256(ident)
    return ident


def verify_cache(cache, ident):
    """The cache is bound to the verified upstream: family, evidence, roles, references, shards, zero lists."""
    family = ident["family"]
    _require(cache.family == CACHE_FAMILY[family], f"cache family {cache.family} != evidence family {family}")
    b = cache.manifest["bindings"]
    _require(b.get("evidence_sha256") == ident["evidence_sha256"], "cache was built from a different evidence snapshot")
    _require(b.get("roles_sha256") == ident["roles_sha256"], "cache was built with a different DATASET_ROLES")
    refs = b.get("reference_sha256")
    _require(isinstance(refs, dict), "cache binds no per-species reference_sha256 mapping")
    for sp, sha in ident.get("reference_sha256", {}).items():
        _require(refs.get(sp) == sha, f"cache reference binding for {sp} differs from the verified reference")
    if family == "training":                  # plan §7.5: a training cache is bound to one corpus and its freeze
        for key, want, what in (("corpus_sha256", ident["corpus_sha256"], "B6 corpus"),
                                ("freeze_sha256", ident["provenance"]["freeze"], "corpus freeze")):
            got = b.get(key)
            _require(_is_sha(got), f"training cache binds no {key} (the {what} it was built against)")
            _require(got == want, f"cache was built against a different {what}: {key} {got[:12]} != verified "
                                  f"{want[:12]}")
    for s in cache.manifest["shards"]:
        _require(sha256_file(cache.path / s["file"]) == s["sha256"], f"cache shard {s['file']} changed")
    zl = cache.manifest["zero_lists"]
    copies = {pop: {sp: (cache.path / m["tsv"], cache.path / m["qc"]) for sp, m in per.items()}
              for pop, per in zl.items()}
    for pop, per in zl.items():
        for sp, m in per.items():
            _require(sha256_file(cache.path / m["tsv"]) == m["tsv_sha256"]
                     and sha256_file(cache.path / m["qc"]) == m["qc_sha256"], f"zero list {m['tsv']} or its QC changed")
    species_set = {sp for sp, _ in ident["coverage"]}
    got = zero_list_check(copies, list(ident["zero_lists"]), ident["_merge_inputs"], species_set)
    _require(got == ident["zero_lists"], "cache zero lists differ from the verified upstream zero lists")
    out = {k: v for k, v in ident.items() if not k.startswith("_") and k != "record_sha256"}
    prov = dict(out["provenance"])
    _prov_add(prov, "cache_manifest", cache.path / "manifest.json")
    _prov_add(prov, "cache_manifest.sha256", cache.path / "manifest.json.sha256")
    _require(prov["cache_manifest"] == cache.manifest_sha256, "cache manifest changed since it was opened")
    out["provenance"] = dict(sorted(prov.items()))
    out["cache_manifest_sha256"] = cache.manifest_sha256
    out["record_sha256"] = canonical_record_sha256(out)
    return out


def cache_zero_lists(cache):
    return {pop: {sp: (cache.path / m["tsv"], cache.path / m["qc"]) for sp, m in per.items()}
            for pop, per in cache.manifest["zero_lists"].items()}


def verify_inputs(*, accept, evidence, roles, cache, zero_lists=None, **kw):
    """Consumer entry point (trainer, resume, evaluation): upstream verification against the cache's own zero-list
    copies (or explicitly supplied lists), then the cache binding.  Returns the identity record."""
    ident = verify_upstream(accept=accept, evidence=evidence, roles=roles,
                            zero_lists=zero_lists if zero_lists is not None else cache_zero_lists(cache), **kw)
    return verify_cache(cache, ident)
