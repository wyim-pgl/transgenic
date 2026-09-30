"""Per-chromosome observation-cache build (b6_build_obs_cache): SQL-level chrom filter in extract_rows, species-level
EST molecules, distinct-molecule drops and library unions across chromosomes, and byte identity with the one-shot
write_cache(extract_rows(...))."""
import shutil

import duckdb
import pytest

import b6_fixtures as FX
from test_b6_obs_cache_stream import BOC, _Counting, one_shot, tree_bytes
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import c0_schema as CS

SP = FX.SPECIES
CHROM_B = "Chr3"               # second chromosome: DUP molecules are placed on FX.CHROM and CHROM_B
CHROM_X = "Chr5"               # a chromosome holding only an excluded_interval row
DUP = ("e1", "n1", "s1", "o1")  # n1 (never) and s1 (b6_score) are dropped molecules, so dropped on both chromosomes
SUFFIXED = ("alignment_id", "chain_id", "observation_id", "junction_key", "hsp_id")


def _copy(con, table, where, params=(), *, chrom=CHROM_B, over=None):
    over = dict(over or {})
    if chrom is not None:
        over.setdefault("chrom", repr(chrom))
    exprs = [over.get(c, f"{c} || '~b'" if c in SUFFIXED else c) for c, _ in CS.table_columns(table)]
    con.execute(f"INSERT INTO {table} SELECT {', '.join(exprs)} FROM {table} WHERE {where}", list(params))


def multi_chrom_db(src, dst):
    """Copy of the merged fixture C0 with DUP's placements duplicated onto CHROM_B (same molecules and members, ids
    suffixed), a leakage-excluded e1 HSP on both chromosomes, e2's HSP moved to CHROM_B (e2 aligns on FX.CHROM only),
    the excluded junction coordinates duplicated onto CHROM_B and excluded intervals moved to CHROM_B / CHROM_X."""
    shutil.copyfile(src, dst)
    con = duckdb.connect(str(dst))
    try:
        marks = ", ".join("?" * len(DUP))
        for table in ("evidence_alignment", "aligned_block", "junction_observation", "partial_chain"):
            _copy(con, table, f"molecule_id IN ({marks})", DUP)
        _copy(con, "partial_chain_intron", f"chain_id IN (SELECT chain_id FROM partial_chain WHERE molecule_id IN "
                                           f"({marks}) AND chrom = ?)", [*DUP, FX.CHROM])
        _copy(con, "est_protein_hsp", "hsp_id = 'h1'", chrom=None, over={"hsp_id": "'hx1'", "leakage_excluded": "true"})
        _copy(con, "est_protein_hsp", "hsp_id IN ('h1', 'hx1')")
        con.execute("UPDATE est_protein_hsp SET chrom = ? WHERE hsp_id = 'h2'", [CHROM_B])
        _copy(con, "excluded_junction_coordinate", "true")
        con.execute("UPDATE excluded_interval SET chrom = ? WHERE cause = 'edge_partial'", [CHROM_B])
        con.execute("UPDATE excluded_interval SET chrom = ? WHERE cause = 'valid_orthogroup'", [CHROM_X])
    finally:
        con.close()
    return dst


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"))


@pytest.fixture(scope="module")
def multi(world, tmp_path_factory):
    return multi_chrom_db(world.c0, tmp_path_factory.mktemp("multi") / "multi_chrom.db")


def _connect(db):
    return duckdb.connect(str(db), read_only=True)


def test_per_chrom_extraction_equals_species_extraction_with_drops_and_libraries_unioned(world, multi):
    con = _connect(multi)
    try:
        rows, drops, libs = OC.extract_rows(con, world.roles, "train", species=SP)
        chroms = OC.species_chroms(con, SP)
        per = {c: OC.extract_rows(con, world.roles, "train", species=SP, chrom=c, raw_drops=True) for c in chroms}
    finally:
        con.close()
    assert chroms == [FX.CHROM, CHROM_B, CHROM_X]
    by_mol = {}
    for r in rows:
        by_mol.setdefault(r["molecule_id"], set()).add(r["chrom"])
    assert by_mol["e1"] == by_mol["o1"] == {FX.CHROM, CHROM_B}
    for c, (rs, _d, _l) in per.items():           # exactly the species rows of that chrom, same relative order
        assert rs and rs == [r for r in rows if r["chrom"] == c]
    assert sum(len(p[0]) for p in per.values()) == len(rows)
    merged = {}
    for _rs, d, _l in per.values():
        assert all(isinstance(ids, set) for ids in d.values())
        for k, ids in d.items():
            merged.setdefault(k, set()).update(ids)
    assert {k: len(v) for k, v in sorted(merged.items())} == drops
    for key in ((SP, "est", "never"), (SP, "ont", "b6_score"), (SP, "est", "phase_hsp_leakage_excluded")):
        # the same molecule is dropped on both chromosomes: summing per-chrom counts would double count it
        assert key in per[FX.CHROM][1] and key in per[CHROM_B][1]
        assert sum(len(p[1].get(key, ())) for p in per.values()) > drops[key]
    union = {pop: {} for pop in OC.POPULATIONS}
    for _rs, _d, lb in per.values():
        for pop, by in lb.items():
            for sp, pairs in by.items():
                union[pop].setdefault(sp, set()).update(map(tuple, pairs))
    assert {pop: {sp: sorted(v) for sp, v in by.items()} for pop, by in union.items()} == libs
    shared = set(per[FX.CHROM][2]["train_track"][SP]) & set(per[CHROM_B][2]["train_track"][SP])
    assert shared, "a library seen on both chromosomes is listed once"


def test_default_drops_stay_molecule_counts(world, multi):
    con = _connect(multi)
    try:
        _rows, drops, _libs = OC.extract_rows(con, world.roles, "train", species=SP, chrom=CHROM_B)
    finally:
        con.close()
    assert drops and all(isinstance(n, int) for n in drops.values())


def test_builder_on_multi_chrom_db_is_byte_identical_to_one_shot_write_cache(world, multi, tmp_path, capsys):
    bindings = {"evidence_sha256": FX._sha(multi), "roles_sha256": world.roles.sha256}
    con = _connect(multi)
    try:
        writer = OC.CacheWriter(tmp_path / "streamed", family="train")
        drops, libs = BOC.extract_by_species(con, world.roles, "train", [SP], writer)
    finally:
        con.close()
    man = writer.finish(libraries=libs, drops=drops, bindings=bindings, zero_lists=world.zero)
    ref = one_shot(multi, world.roles, tmp_path / "one_shot", bindings=bindings, zero_lists=world.zero)
    assert man == ref
    assert [(s["species"], s["chrom"]) for s in man["shards"]] == [(SP, FX.CHROM), (SP, CHROM_B), (SP, CHROM_X)]
    assert (tmp_path / "streamed" / "manifest.json").read_bytes() == (tmp_path / "one_shot" / "manifest.json").read_bytes()
    assert tree_bytes(tmp_path / "streamed") == tree_bytes(tmp_path / "one_shot")
    OC.ObsCache(tmp_path / "streamed")
    err = capsys.readouterr().err.splitlines()
    assert [ln for ln in err if ln.startswith("b6_build_obs_cache: chrom ")] == [
        f"b6_build_obs_cache: chrom {SP} {s['chrom']}: {s['rows']} rows" for s in man["shards"]]
    assert [ln for ln in err if ln.startswith("b6_build_obs_cache: species ")] == [
        f"b6_build_obs_cache: species {SP}: {sum(s['rows'] for s in man['shards'])} rows"]


def test_chrom_filter_restricts_alignment_block_junction_chain_queries_at_sql_level(world, multi):
    single = _Counting(_connect(world.c0))
    whole = _Counting(_connect(multi))
    chrom = _Counting(_connect(multi))
    try:
        OC.extract_rows(single, world.roles, "train")
        OC.extract_rows(whole, world.roles, "train", species=SP)
        OC.extract_rows(chrom, world.roles, "train", species=SP, chrom=FX.CHROM)
    finally:
        for c in (single, whole, chrom):
            c.con.close()
    # fetches 1-4 (after the molecule-state check: evidence_alignment, aligned_block, junction_observation,
    # partial_chain): the multi DB holds more rows, and a FX.CHROM call fetches exactly the single-chromosome DB's
    # rows (nothing of CHROM_B reaches Python)
    joined = slice(1, 5)
    assert len(single.fetched) == len(whole.fetched) == len(chrom.fetched) and single.fetched[0] == 0
    assert all(w > s > 0 for w, s in zip(whole.fetched[joined], single.fetched[joined]))
    assert chrom.fetched[joined] == single.fetched[joined]


def test_hsp_of_a_molecule_aligned_on_another_chrom_is_admitted(world, multi):
    con = _connect(multi)
    try:
        aligned = {c for (c,) in con.execute("SELECT DISTINCT chrom FROM evidence_alignment "
                                             "WHERE molecule_id = 'e2'").fetchall()}
        rows, _d, _l = OC.extract_rows(con, world.roles, "train", species=SP, chrom=CHROM_B)
        est_mol = OC.est_molecules(con, SP)
        again = OC.extract_rows(con, world.roles, "train", species=SP, chrom=CHROM_B, check_states=False,
                                est_mol=est_mol)
    finally:
        con.close()
    assert aligned == {FX.CHROM}
    phase = [r for r in rows if r["kind"] == "phase" and r["molecule_id"] == "e2"]
    assert len(phase) == 1 and phase[0]["chrom"] == CHROM_B and phase[0]["dataset"] == FX.EST_DS
    assert again[0] == rows


def test_chrom_filter_needs_a_species(world):
    con = _connect(world.c0)
    try:
        with pytest.raises(OC.CacheError, match="chrom"):
            OC.extract_rows(con, world.roles, "train", chrom=FX.CHROM)
    finally:
        con.close()


def test_species_chroms_skips_a_scaffold_with_only_rejected_alignments(world, multi, tmp_path):
    db = tmp_path / "rejected_only.db"
    shutil.copyfile(multi, db)
    con = duckdb.connect(str(db))
    try:
        cols = [c for c, _ in CS.table_columns("evidence_alignment")]
        exprs = [("alignment_id || '~rej'" if c == "alignment_id" else "'ChrRej'" if c == "chrom"
                  else "false" if c == "accepted" else c) for c in cols]
        con.execute(f"INSERT INTO evidence_alignment SELECT {', '.join(exprs)} FROM evidence_alignment "
                    "WHERE species = ? LIMIT 1", [SP])
    finally:
        con.close()
    con = _connect(db)
    try:
        assert "ChrRej" not in OC.species_chroms(con, SP)
        assert OC.species_chroms(con, SP) == OC.species_chroms(_connect(multi), SP)
    finally:
        con.close()
