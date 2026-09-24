"""WP3 tracks, chains, draws and corpus contexts on the synthetic evidence world (plan §7.1-7.4, §7.8)."""
import numpy as np
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_aug as AG
from transgenic.evidence import b6_chains as CH
from transgenic.evidence import b6_context as CX
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import b6_track_spec as TS
from transgenic.evidence import b6_tracks as TR

L1 = np.log1p(1.0).astype(np.float16)
L2 = np.log1p(2.0).astype(np.float16)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"), molecules=FX.MOLECULES + FX.cap_molecules())


@pytest.fixture(scope="module")
def cache(world):
    return OC.ObsCache(world.cache_dir)


@pytest.fixture(scope="module")
def corpus(world, cache):
    return CX.B6Corpus(world.corpus, cache)


def owner(cache, pop="train_track"):
    return AG.Owner.from_pairs(FX.SPECIES, cache.owner_libraries(pop, FX.SPECIES))


def full(cache, pop="train_track", aug=TS.training_aug_seed()):
    return AG.resolve_fixed(aug, owner(cache, pop), 1.0)


# ------------------------------------------------------------------------------------------------ build filter
def test_build_filter_drops_never_score_and_excluded_protein(world):
    mids = {r["molecule_id"] for r in world.rows}
    assert "n1" not in mids and "s1" not in mids and "prx" not in mids
    assert {"eh", "oh", "oh_far"} <= mids                       # b6_input excluded_heldout kept for evaluation
    assert world.drops[(FX.SPECIES, "est", "never")] == 1
    assert world.drops[(FX.SPECIES, "ont", "b6_score")] == 1


# ------------------------------------------------------------------------------------------------ tracks
def test_raster_values_on_a_train_tile(cache, corpus):
    ctx = CX.corpus_context(corpus, 1, "train_track")
    t = TR.rasterise(cache, ctx, full(cache))
    assert t.dtype == np.float16 and t.shape == (600, 31)
    assert t[150, 0] == L1 and t[250, 0] == 0                  # e1 cov+ (block union per molecule)
    assert t[200, 1] == L1 and t[120, 1] == 0                  # e2 cov-
    assert t[520, 2] == L1                                      # e3 unknown strand
    assert t[200, 3] == L1 and t[299, 4] == L1                 # e1 donor+/acceptor+
    assert t[349, 5] == L1 and t[250, 6] == L1                 # e2 donor-/acceptor- (strand-aware, inclusive)
    assert t[20, 8] == L1 and t[20, 16] == L1                  # o1 ONT cov+, p1 PacBio cov+
    assert t[40, 11] == 0 and t[59, 12] == 0                   # rule 4(c): excluded junction never counted
    assert t[90, 11] == L1 and t[90, 19] == L1                 # second intron counted for ONT and PacBio
    assert t[145, 24] == L1 and t[155, 24] == L2 and t[180, 24] == L1   # distinct organisms
    assert t[100, 25] == L1 and t[101, 26] == L1 and t[102, 27] == L1   # route-B phase, codon position walks
    assert (t[120:125, 25:31] == 0).all() and t[119].sum() > 0 # A48 zero list
    assert t[160, 30] == L1 and t[161, 29] == L1               # '-' strand phase walks backwards
    assert (t[:, list(TS.MASK_CHANNELS)] == 1).all()          # masks 1 everywhere, including N bases 590-599
    assert (t[:, 7][590:600] == 1).all()


def test_rc_twin_is_rc_of_forward_and_twice_identity(cache, corpus):
    fw = TR.rasterise(cache, CX.corpus_context(corpus, 1, "train_track"), full(cache))
    rc = TR.rasterise(cache, CX.corpus_context(corpus, 2, "train_track"), full(cache))
    assert np.array_equal(rc, TS.rc_tracks(fw))
    assert np.array_equal(TS.rc_tracks(rc), fw)
    assert rc[599 - 200, 5] == L1                               # + donor becomes - donor, role kept
    assert rc[599 - 100, TS.phase_channel(0, "-")] == L1        # phase label kept, strand swapped


def test_dropped_tile_is_all_zero(cache, corpus):
    ctx = CX.corpus_context(corpus, 1, "train_track")
    off = AG.resolve_off(owner(cache))
    assert not TR.rasterise(cache, ctx, off).any()
    tile = CH.build_chains(cache, ctx, off)
    assert not tile.chain_mask.any() and not tile.chain_pos.any() and not tile.chain_feat.any()


def test_fraction_zero_keeps_masks_and_no_molecule(cache, corpus):
    ctx = CX.corpus_context(corpus, 1, "train_track")
    t = TR.rasterise(cache, ctx, AG.resolve_fixed(1, owner(cache), 0.0))
    assert (t[:, list(TS.MASK_CHANNELS)] == 1).all()
    assert not t[:, list(TS.COUNT_CHANNELS)].any()


def test_train_tiles_excluded_molecule_only_in_valid(cache, corpus):
    t5 = TR.rasterise(cache, CX.corpus_context(corpus, 5, "train_track"), full(cache))
    assert t5[2220 - 1800, 0] == 0                               # e6 is train_tiles_excluded
    assert t5[1860 - 1800, 8] == L1                              # o3 retained
    t3 = TR.rasterise(cache, CX.corpus_context(corpus, 3, "valid_track"), full(cache, "valid_track"))
    assert t3[730 - 600, 0] == L2                                # e4 (train_tiles_excluded) + e5 admitted in valid


def test_test_tile_and_mode_mismatch_refused(corpus):
    with pytest.raises(CX.ContextError, match="test tile"):
        CX.corpus_context(corpus, 4, "valid_track")
    with pytest.raises(CX.ContextError, match="does not match"):
        CX.corpus_context(corpus, 3, "train_track")


def test_tile_exclusions_scope_and_tile_rn(corpus):
    c1 = CX.corpus_context(corpus, 1, "train_track")
    c2 = CX.corpus_context(corpus, 2, "train_track")
    assert ("edge_partial" in {x[2] for x in c1.tile_exclusions}) and c1.tile_exclusions == c2.tile_exclusions
    c5 = CX.corpus_context(corpus, 5, "train_track")
    assert any(x[2] == "valid_orthogroup" for x in c5.tile_exclusions)


def test_retained_observation_on_an_excluded_interval_aborts(tmp_path):
    bad = ("bad", "est", FX.EST_DS, FX.EST_RUN, "LE1", "train_track", False, "training_eligible", "+",
           [(2040, 2060), (2080, 2090)])
    w = FX.make_world(tmp_path, molecules=FX.MOLECULES + [bad])
    cache = OC.ObsCache(w.cache_dir)
    ctx = CX.corpus_context(CX.B6Corpus(w.corpus, cache), 5, "train_track")
    with pytest.raises(TR.LeakageError, match="valid_orthogroup"):
        TR.rasterise(cache, ctx, full(cache))


def test_per_population_zero_lists_are_used(cache, corpus):
    assert CX.corpus_context(corpus, 1, "train_track").zero_intervals == ((120, 125),)
    assert CX.corpus_context(corpus, 3, "valid_track").zero_intervals == ()


def test_modified_zero_list_fails_even_with_unchanged_cache(tmp_path):
    w = FX.make_world(tmp_path)
    (w.cache_dir / f"zero.train_track.{FX.SPECIES}.tsv").write_text(
        "species\tchrom\tstart0\tend0\taction\treason\n")
    cache = OC.ObsCache(w.cache_dir)
    with pytest.raises(OC.CacheError, match="zero list"):
        CX.corpus_context(CX.B6Corpus(w.corpus, cache), 1, "train_track")


# ------------------------------------------------------------------------------------------------ chains
def test_chain_fill_order_4c_split_and_pacbio_before_ont(cache, corpus):
    ctx = CX.corpus_context(corpus, 1, "train_track")
    tile = CH.build_chains(cache, ctx, full(cache))
    m = tile.chain_mask
    assert not m[0] and m[1:8].all() and not m[8:].any()
    src = tile.chain_feat[1:8, TS.FEAT_SOURCE].tolist()
    assert src == [2, 2, 1, 1, 0, 0, 0]                          # PacBio, ONT (identical chains kept apart), EST
    # 4(c) split: the long-read chains lost intron [40,60); fragment = (90,109),(140,159)
    assert tile.chain_pos[1].tolist() == [90, 109] and tile.chain_pos[2].tolist() == [140, 159]
    # EST: e2 (2 introns, '-' transcription order) before e1 (1 intron)
    assert tile.chain_pos[5].tolist() == [349, 250] and tile.chain_pos[6].tolist() == [149, 100]
    assert tile.chain_feat[5, [TS.FEAT_ORDER_FWD, TS.FEAT_ORDER_REV, TS.FEAT_STRAND]].tolist() == [0, 1, 1]
    assert tile.chain_pos[7].tolist() == [200, 299]
    pi = TS.slot_permutation(TS.training_aug_seed(), "1", 0)
    slots = tile.chain_feat[[1, 3, 5, 7], TS.FEAT_SLOT].tolist()
    assert slots == list(pi[:4]) and (tile.chain_feat[[1, 2], TS.FEAT_SLOT] == pi[0]).all()


def test_chain_rc_twin_and_twice_identity(cache, corpus):
    d = full(cache)
    fw = CH.build_chains(cache, CX.corpus_context(corpus, 1, "train_track"), d)
    rc = CH.build_chains(cache, CX.corpus_context(corpus, 2, "train_track"), d)
    # the twin has its own row key, so its slots differ; positions/roles/strand follow chain-rc-v1
    p, f, m = TS.rc_chain(fw.chain_pos, fw.chain_feat, fw.chain_mask, 600)
    assert np.array_equal(rc.chain_mask, m) and np.array_equal(rc.chain_pos, p)
    keep = [TS.FEAT_ORDER_FWD, TS.FEAT_ORDER_REV, TS.FEAT_SOURCE, TS.FEAT_STRAND, TS.FEAT_UNITS]
    assert np.array_equal(rc.chain_feat[:, keep], f[:, keep])
    p2, f2, m2 = TS.rc_chain(p, f, m, 600)
    assert np.array_equal(p2, fw.chain_pos) and np.array_equal(f2, fw.chain_feat)
    # negative-strand junction identity: e2's '-' donor 349 maps to a '+' donor at 599-349
    e = int(np.nonzero((fw.chain_pos[:, 0] == 349) & fw.chain_mask)[0][0])
    assert rc.chain_pos[e].tolist() == [599 - 349, 599 - 250] and rc.chain_feat[e, TS.FEAT_STRAND] == 0


def test_whole_chain_drop_and_long_reads_first(cache, corpus):
    ctx = CX.corpus_context(corpus, 7, "train_track")
    tile = CH.build_chains(cache, ctx, full(cache))
    assert tile.qc["filled"]["ont"] == 18 and tile.qc["truncated"]["ont"] == 0
    assert tile.qc["filled"]["est"] == 511 - 18 * 27 and tile.qc["truncated"]["est"] == 70 - (511 - 486)
    assert tile.chain_mask.sum() == 511 and tile.qc["elements"] == 511
    per_slot = {}
    for e in np.nonzero(tile.chain_mask)[0]:
        per_slot.setdefault(int(tile.chain_feat[e, TS.FEAT_SLOT]), []).append(e)
    assert all(len(v) in (1, 27) for v in per_slot.values())   # no chain truncated
    assert len(per_slot) == 18 + 25 <= TS.MAX_CHAINS


def test_slot_cap_63_chains(tmp_path):
    ests = [(f"q{k}", "est", FX.EST_DS, FX.EST_RUN, "LE1", "train_track", False, "training_eligible", "+",
             [(3010 + 6 * k, 3012 + 6 * k), (3014 + 6 * k, 3016 + 6 * k)]) for k in range(80)]
    w = FX.make_world(tmp_path, molecules=FX.MOLECULES + ests)
    cache = OC.ObsCache(w.cache_dir)
    tile = CH.build_chains(cache, CX.corpus_context(CX.B6Corpus(w.corpus, cache), 7, "train_track"), full(cache))
    assert tile.qc["filled"]["est"] == 63 and tile.qc["truncated"]["est"] == 17
    assert sorted(set(tile.chain_feat[tile.chain_mask, TS.FEAT_SLOT].tolist())) == list(range(1, 64))


# ------------------------------------------------------------------------------------------------ draws
def _train_draw(cache, rn, epoch=0):
    return AG.resolve(AG.make_draw(TS.training_aug_seed(), str(rn), epoch), owner(cache))


def test_nesting_tracks_and_precap_chains_across_fractions(cache, corpus):
    ctx = CX.corpus_context(corpus, 7, "train_track")
    prev_mol, prev_keys = None, None
    for f in (0.01, 0.1, 0.5, 1.0):
        d = AG.resolve_fixed(20261037, owner(cache), f)
        _, qc = TR.rasterise_with_qc(cache, ctx, d)
        mol = set().union(*map(set, qc["retained_molecules"].values()))
        keys = set(CH.build_chains(cache, ctx, d).qc["precap_keys"])
        if prev_mol is not None:
            assert prev_mol <= mol and prev_keys <= keys
        prev_mol, prev_keys = mol, keys
    assert len(prev_mol) > 50


def test_chain_molecules_subset_of_track_molecules(cache, corpus):
    for rn in (1, 5, 7):
        ctx = CX.corpus_context(corpus, rn, "train_track")
        d = _train_draw(cache, rn)
        _, qc = TR.rasterise_with_qc(cache, ctx, d)
        tracks = set().union(*map(set, qc["retained_molecules"].values()))
        merged = CH.candidate_chains(cache, ctx, d)
        assert set().union(*merged.values()) <= tracks if merged else True


def test_uniforms_pure_and_mask_semantics_of_withholding(cache):
    o = owner(cache)
    fired_src = fired_lib = None
    for rn in range(1, 4000):
        d = AG.make_draw(20261037, str(rn), 0)
        assert d == AG.make_draw(20261037, str(rn), 0)
        r = AG.resolve(d, o)
        if r.withheld_source and fired_src is None:
            fired_src = r
        if r.withheld_library and not r.withheld_source and fired_lib is None:
            fired_lib = r
    assert fired_src.mask(fired_src.withheld_source) == 0
    src, lib = fired_lib.withheld_library
    others = [p for p in o.libraries if p[0] == src and p != (src, lib)]
    assert fired_lib.mask(src) == (1 if others else 0)
    assert not fired_lib.library_admitted(src, lib)


def test_last_library_withheld_zeroes_mask():
    o = AG.Owner.from_pairs("X", [("ont", "SRX1"), ("est", "L1")])
    for rn in range(1, 20000):
        r = AG.resolve(AG.make_draw(7, str(rn), 0), o)
        if r.withheld_library == ("ont", "SRX1") and not r.withheld_source:
            assert r.mask("ont") == 0 and r.mask("est") == 1
            return
    pytest.fail("no library withholding draw found")


def test_est_withheld_removes_phase(cache, corpus):
    ctx = CX.corpus_context(corpus, 1, "train_track")
    o = owner(cache)
    d = AG.ResolvedDraw(1, FX.SPECIES, False, 1.0, (("est", 0), ("ont", 1), ("pacbio", 1)), "est", None,
                        tuple(p for p in o.libraries if p[0] != "est"), "train")
    t = TR.rasterise(cache, ctx, d)
    assert not t[:, 0:7].any() and not t[:, list(TS.PHASE_CHANNELS)].any() and t[:, 7].max() == 0
    assert t[145, 24] == L1                                      # protein never withheld (D7)


def test_arms_transform(cache, corpus):
    t = TR.rasterise(cache, CX.corpus_context(corpus, 1, "train_track"), full(cache))
    assert not AG.apply_arm_tracks(t, "C2").any()
    c4 = AG.apply_arm_tracks(t, "C4")
    assert set(np.unique(c4[:, list(TS.COUNT_CHANNELS)]).tolist()) <= {0.0, 1.0}
    assert np.array_equal(c4[:, list(TS.MASK_CHANNELS)], t[:, list(TS.MASK_CHANNELS)])
    c6 = AG.apply_arm_tracks(t, "C6")
    assert not c6[:, 25:].any() and np.array_equal(c6[:, :25], t[:, :25])
    assert AG.arm_masks_chains("C7") and AG.arm_masks_chains("C2") and not AG.arm_masks_chains("C4")


def test_slot_permutation_never_consumes_the_uniform_stream():
    d1 = AG.make_draw(20261037, "55", 2)
    TS.slot_permutation(20261037, "55", 2)
    assert AG.make_draw(20261037, "55", 2) == d1


# ------------------------------------------------------------------------------------------------ C3
def test_c3_donor_map_derangement_twins_and_singleton():
    rows = [(1, "w1", 600, "train", False), (2, "w1", 600, "train", True), (3, "w2", 600, "train", False),
            (4, "w3", 600, "train", False), (5, "v1", 600, "valid", False), (6, "v2", 600, "valid", False)]
    m = AG.c3_donor_map(rows, 20261037)
    assert m[1] == m[2] and all(m[rn] != wid for rn, wid, *_ in rows)
    assert {m[1], m[3], m[4]} == {"w1", "w2", "w3"} and {m[5], m[6]} == {"v1", "v2"}
    with pytest.raises(ValueError, match="< 2 forward rows"):
        AG.c3_donor_map(rows + [(7, "t1", 1200, "train", False)], 20261037)


def test_c3_donor_lacking_recipient_library_withholds_its_own():
    rec = AG.Owner.from_pairs("A", [("est", "LA")])
    donor = AG.Owner.from_pairs("B", [("ont", "SRXD"), ("est", "LB")])
    for rn in range(1, 20000):
        d = AG.make_draw(20261037, str(rn), 0)
        r = AG.resolve(d, donor)
        if r.withheld_library and not r.withheld_source:
            assert r.withheld_library in donor.libraries and r.species == "B"
            assert d.sha256() == AG.make_draw(20261037, str(rn), 0).sha256()     # uniform log = B6's
            return
    pytest.fail("no library withholding draw found")


def test_c3_donor_context_uses_donor_window_in_recipient_orientation(cache, corpus):
    rec = CX.corpus_context(corpus, 2, "train_track")            # RC recipient
    don = CX.donor_context(corpus, rec, f"{FX.SPECIES}:{FX.CHROM}:1800-2400")
    assert (don.ws, don.is_rc, don.row) == (1800, True, "2")
    t = TR.rasterise(cache, don, full(cache))
    assert t[599 - (1860 - 1800), 9] == L1                       # donor o3 ONT cov+, RC'd onto the '-' channel


def test_float16_phase_rc_mirrors_gsf_rc_codon_continuity(tmp_path):
    """Mirror of tests/test_gsf_rc.py:26 for the phase channels: a spliced '+' CDS of 100 + 100 nt (codon position
    0 at 100, so 1 at 300) keeps codon continuity in transcription order after RC, on the '-' channels, in float16."""
    hsps = [("hA", "e1", 100, 200, "+", 0, False, False), ("hB", "e1", 300, 400, "+", 1, False, False)]
    w = FX.make_world(tmp_path, hsps=hsps, zero={k: [] for k in FX.ZERO})
    cache = OC.ObsCache(w.cache_dir)
    corpus = CX.B6Corpus(w.corpus, cache)
    d = full(cache)
    fw = TR.rasterise(cache, CX.corpus_context(corpus, 1, "train_track"), d)
    rc = TR.rasterise(cache, CX.corpus_context(corpus, 2, "train_track"), d)
    coding = list(range(100, 200)) + list(range(300, 400))
    fw_pos = [int(np.argmax(fw[g, 25:28])) for g in coding]
    assert fw_pos == [i % 3 for i in range(200)] and (fw[coding, 25:28].max(1) == L1).all()
    rc_coding = [599 - g for g in coding]                    # transcription order on the RC tile: decreasing index
    rc_pos = [int(np.argmax(rc[p, 28:31])) for p in rc_coding]
    assert rc_pos == fw_pos and not rc[:, 25:28].any() and rc.dtype == np.float16
    assert np.array_equal(TS.rc_tracks(rc), fw)


def test_protein_thinning_ranks_the_query_id_d7(tmp_path, monkeypatch):
    """D7 literal-key regression: rank(20261037, Athaliana, '29760:4') ~ 0.253606, so a fraction-0.5 draw keeps the
    query even when its C0 molecule_id would rank >= 0.5."""
    assert abs(TS.molecule_rank(20261037, "Athaliana", "29760:4") - 0.253606) < 1e-6
    mid = next(f"pq{k}" for k in range(1000) if TS.molecule_rank(20261037, "Athaliana", f"pq{k}") >= 0.5)
    monkeypatch.setitem(FX.ORGANISM, mid, "29760:4")
    prot = (mid, "protein", FX.PROT_DS, FX.PROT_RUN, "orthodb", "train_track", False, "training_eligible", "+",
            [(3500, 3560)])
    w = FX.make_world(tmp_path, molecules=FX.MOLECULES + [prot])
    cache = OC.ObsCache(w.cache_dir)
    row = next(r for r in w.rows if r["molecule_id"] == mid and r["kind"] == "protein")
    assert row["query_id"] == "29760:4"
    ctx = CX.corpus_context(CX.B6Corpus(w.corpus, cache), 7, "train_track")
    t, qc = TR.rasterise_with_qc(cache, ctx, AG.resolve_fixed(20261037, owner(cache), 0.5))
    assert t[3530 - 3000, 24] == L1 and qc["retained_molecules"]["protein"] == [mid]
    assert not TR.rasterise(cache, ctx, AG.resolve_fixed(20261037, owner(cache), 0.01))[3530 - 3000, 24]
