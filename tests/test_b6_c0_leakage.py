"""A46 rule 4(a)-(d) leakage closure on a real tile corpus built by build_b5 (see test_b6_c0_world)."""
import pytest

import test_b6_c0_world as W
from transgenic.evidence import leakage as LK


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build_corpus(tmp_path_factory.mktemp("leak"))


@pytest.fixture(scope="module")
def model(world):
    return LK.build_leakage_model(corpus_db=world["corpus"], species=W.SPECIES, splits_path=world["split"],
                                  flags_path=world["a22"], build_flags=[world["a22"], world["a30"]], seed=W.CORPUS_SEED)


def _causes(model, chrom, a, b, scope="all"):
    return model.index.causes(chrom, a, b, scope)


def test_every_kept_tile_is_recomputed_and_verified_against_the_corpus(model):
    assert model.qc["tiles"] > 0 and model.qc["tiles_verified"] == model.qc["tiles"]


def test_rule_4a_causes_at_each_seed_gene(model):
    assert "heldout" in _causes(model, "Chr1", 40100, 40101)            # strict held-out gH
    assert "test_orthogroup" in _causes(model, "Chr1", 70100, 70101)    # gT
    assert "a22_hard" in _causes(model, "Chr1", 130100, 130101)         # gF (A22 GeenuFF)
    assert "test_block" in _causes(model, "Chr2", 20100, 20101)         # Chr2 block is test (A29)
    assert "valid_orthogroup" in _causes(model, "Chr1", 100100, 100101, "train_tiles")
    assert "valid_orthogroup" not in _causes(model, "Chr1", 100100, 100101, "all")
    assert _causes(model, "Chr1", 170100, 170101) == set()              # plain train gene gB


def test_a30_only_gene_is_not_a_rule_4a_cause_but_build_flags_are_needed_to_verify(world, model):
    assert not model.index.overlaps_any("Chr1", 150100, 150101)         # gW: Swiss-Prot caution only
    assert model.qc["tile_masks_not_rule4a_a30_only"] > 0
    with pytest.raises(LK.LeakageError, match="recomputed masking"):
        LK.build_leakage_model(corpus_db=world["corpus"], species=W.SPECIES, splits_path=world["split"],
                               flags_path=world["a22"], seed=W.CORPUS_SEED)


def test_wrong_seed_fails_closed(world):
    with pytest.raises(LK.LeakageError):
        LK.build_leakage_model(corpus_db=world["corpus"], species=W.SPECIES, splits_path=world["split"],
                               flags_path=world["a22"], build_flags=[world["a22"], world["a30"]], seed=123)


def test_heldout_component_is_padded_by_a_seeded_flank_in_range(model):
    iv = [i for i in model.intervals if i.cause == "heldout"]
    assert len(iv) == 1 and iv[0].gene_id == "gH"
    # component gH+gS spans 40000-40600 (0-based half-open); flank drawn from [50, 150]
    assert 50 <= 40000 - iv[0].start0 <= 150 and 50 <= iv[0].end0 - 40600 <= 150
    tile = [i for i in model.intervals if i.cause == "tile_leak" and i.gene_id == "gH"]
    assert tile and all(i.scope == "train_tiles" for i in tile)
    assert max(40000 - t.start0 for t in tile if t.start0 > 30000) == 40000 - iv[0].start0


def test_edge_partial_spans_match_the_builds_own_edge_counts(world, model):
    import duckdb

    con = duckdb.connect(str(world["corpus"]), read_only=True)
    stored = sum(int(p.split("=")[1]) for (q,) in con.execute(
        "SELECT qc_flags FROM geneList WHERE split IN ('train','valid') AND NOT is_rc").fetchall()
        for p in (q or "").split(";") if p.startswith("edge_partial="))
    con.close()
    edges = [i for i in model.intervals if i.cause == "edge_partial"]
    assert stored > 0 and len(edges) == stored
    assert all(i.gene_id == "gL" and i.scope == "train_tiles" for i in edges)


def test_rule_4c_coordinates_shared_by_excluded_gene_and_neighbour(model):
    assert model.coordinates[("Chr1", "+", "donor", 100300)] == ("gV", "gV2")
    assert model.coordinates[("Chr1", "+", "donor", 40200)] == ("gH", "gS")
    assert model.excluded_4c("Chr1", "+", 100300, 100499)
    assert not model.excluded_4c("Chr1", "+", 170200, 170299)


def test_coordinate_exclusions_are_strand_aware():
    b5 = LK.load_build_b5()
    gc = b5.gc
    F = gc.Feature
    seed = gc.Gene("x", "c", "-", 0, 1000, {"x.1": [F("CDS", 1, 100, "0"), F("CDS", 201, 300, "0")]})
    other = gc.Gene("y", "c", "-", 0, 1000, {"y.1": [F("CDS", 51, 100, "0"), F("CDS", 201, 400, "0")]})
    coords = LK.coordinate_exclusions([seed, other], {"x"})
    # minus strand: intron [100, 200) -> donor0 = 199 (last base), acceptor0 = 100
    assert coords == {("c", "-", "donor", 199): ("x", "y"), ("c", "-", "acceptor", 100): ("x", "y")}


def _mol(n, secondaries=None, mapq=60, chrom="c", where=(0, 10)):
    molecules = {f"m{i}": [f"r{i}"] for i in range(n)}
    placements = {f"r{i}": [LK.Placement(f"r{i}", "frozen", "primary", chrom, where[0], where[1], mapq)] for i in range(n)}
    return molecules, placements, dict(secondaries or {})


def test_rule_4b_secondary_cap_low_mapq_and_exhaustiveness_boundary():
    index = LK.IntervalIndex([LK.ExcludedInterval("c", 500, 600, "heldout", "all")])
    molecules, placements, sec = _mol(1000, {"r0": 50})
    placements["r1"].append(LK.Placement("r1", "audit", "secondary", "c", 5000, 5100, 0))    # MAPQ 0 outside
    placements["r2"].append(LK.Placement("r2", "audit", "secondary", "c", 550, 560, 0))      # MAPQ 0 inside -> (a)
    decisions, qc = LK.decide_molecules(molecules, placements, sec, index)
    assert qc["exhaustiveness"] == pytest.approx(0.999)
    assert decisions["m0"].unauditable_secondary and decisions["m0"].excluded
    assert decisions["m1"].unauditable_low_mapq and not decisions["m1"].cause_a
    assert decisions["m2"].cause_a == ("heldout",) and not decisions["m2"].unauditable_low_mapq
    assert not decisions["m2"].cause_a_direct and not decisions["m3"].excluded
    molecules, placements, sec = _mol(1000, {"r0": 50, "r1": 51})
    with pytest.raises(LK.ExhaustivenessError, match="raise N"):
        LK.decide_molecules(molecules, placements, sec, index)


def test_rule_4b_train_tiles_scope_does_not_exclude_but_flags(model):
    molecules, placements, _ = _mol(1, chrom="Chr1", where=(100050, 100250))
    d, _ = LK.decide_molecules(molecules, placements, {}, model.index)
    assert not d["m0"].excluded and d["m0"].train_tiles_excluded


def test_rule_4d_heldout_recount_is_independent_of_the_index():
    molecules, placements, _ = _mol(2, chrom="c", where=(100, 200))
    fake = {"m0": LK.MoleculeDecision("m0", 1, 0, 1, ("heldout",), True, (), False, False, True),
            "m1": LK.MoleculeDecision("m1", 1, 0, 1, (), False, (), False, False, True)}
    qc = LK.heldout_qc(fake, molecules, placements, [("g", "c", 150, 160)])
    assert qc == {"heldout_overlapping_molecules": 2, "heldout_overlapping_excluded": 1, "heldout_retained": 1}


def test_rule_4b_secondary_count_is_summed_over_the_molecule_members():
    """Frozen wording: 'a molecule with >= 50 secondary placements'; -N caps per query, so two members with 30
    secondaries each make an unauditable molecule, and exhaustiveness counts molecules, not members."""
    index = LK.IntervalIndex([LK.ExcludedInterval("c", 500, 600, "heldout", "all")])
    molecules = {"m0": ["a", "b"], "m1": ["c"]}
    placements = {k: [LK.Placement(k, "frozen", "primary", "c", 0, 10, 60)] for k in "abc"}
    decisions, qc = LK.decide_molecules(molecules, placements, {"a": 30, "b": 30, "c": 49}, index,
                                        min_exhaustiveness=0.5)
    assert decisions["m0"].n_secondary == 60 and decisions["m0"].unauditable_secondary
    assert decisions["m1"].n_secondary == 49 and not decisions["m1"].excluded
    assert qc["exhaustiveness"] == pytest.approx(0.5)


def test_low_mapq_on_secondaries_only_is_flagged_for_the_pending_ruling():
    assert LK.LOW_MAPQ_APPLIES_TO_AUDIT_SECONDARIES is True and LK.EDGE_PARTIAL_MOLECULE_WIDE is True
    index = LK.IntervalIndex([LK.ExcludedInterval("c", 500, 600, "heldout", "all")])
    d = LK.decide_molecule("m", 1, [("frozen", "primary", "c", 0, 10, 60), ("audit", "secondary", "c", 900, 950, 0)], 1, index)
    assert d.unauditable_low_mapq and d.low_mapq_secondary_only
    d = LK.decide_molecule("m", 1, [("frozen", "primary", "c", 0, 10, 3), ("audit", "secondary", "c", 900, 950, 0)], 1, index)
    assert d.unauditable_low_mapq and not d.low_mapq_secondary_only


def _null_split_corpus(world, tmp_path, where):
    import shutil

    import duckdb

    db = tmp_path / "corpus.db"
    shutil.copy(world["corpus"], db)
    con = duckdb.connect(str(db))
    rn = con.execute(f"SELECT min(rn) FROM geneList WHERE NOT is_rc AND {where}").fetchone()[0]
    con.execute("UPDATE geneList SET split = NULL WHERE rn = ?", [rn])
    con.close()
    return db, rn


def test_null_split_tile_with_genes_is_refused(world, tmp_path):
    db, rn = _null_split_corpus(world, tmp_path, "chromosome = 'Chr1' AND start <= 10000 AND fin >= 10800")
    with pytest.raises(LK.LeakageError, match=f"tile {rn} .*split NULL"):
        LK.build_leakage_model(corpus_db=db, species=W.SPECIES, splits_path=world["split"], flags_path=world["a22"],
                               build_flags=[world["a22"], world["a30"]], seed=W.CORPUS_SEED)


def test_null_split_gene_free_tile_is_resolved_from_its_block_not_skipped(world, tmp_path):
    """build_b5 writes split NULL only for a gene-free tile; such a tile is treated as its block's split (an explicit
    derivation) and still yields its edge_partial interval instead of being skipped."""
    import shutil

    import duckdb

    db = tmp_path / "corpus.db"
    shutil.copy(world["corpus"], db)
    ws, we = 210500, 230000                          # inside gL's intron, after gE: gene-free, gL crosses both edges
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO geneList SELECT * REPLACE (999 AS rn, ? AS start, ? AS fin, ? AS sequence, NULL AS split, "
                "'edge_partial=1' AS qc_flags) FROM geneList WHERE rn = 1", [ws, we, world["genome"]["Chr1"][ws:we]])
    con.close()
    m = LK.build_leakage_model(corpus_db=db, species=W.SPECIES, splits_path=world["split"], flags_path=world["a22"],
                               build_flags=[world["a22"], world["a30"]], seed=W.CORPUS_SEED)
    assert m.qc["tiles_split_null_resolved"] == 1
    assert [(i.cause, i.scope, i.start0, i.end0, i.gene_id) for i in m.intervals if i.tile_rn == 999] == \
        [("edge_partial", "train_tiles", ws, we, "gL")]
