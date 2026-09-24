"""Evaluation path (plan §7.4, §8): A. thaliana eval_context on the training-family cache, and synthetic Z. mays
eval C0 -> eval cache -> eval_context -> batch with no corpus file present."""
import json
from pathlib import Path

import numpy as np
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_aug as AG
from transgenic.evidence import b6_bindings as BD
from transgenic.evidence import b6_chains as CH
from transgenic.evidence import b6_context as CX
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import b6_track_spec as TS
from transgenic.evidence import b6_tracks as TR

DS = FX.load_b6_dataset()


def sample(ctx, dna_ids, tracks, tile, draw, row_meta=None):
    return {"input_ids": dna_ids, "attention_mask": np.ones(len(dna_ids), bool), "evidence_tracks": tracks,
            "labels": None, "chain_pos": tile.chain_pos, "chain_feat": tile.chain_feat, "chain_mask": tile.chain_mask,
            "meta": {"row": ctx.row, "epoch": 0, "arm": "B6", "uniforms_sha256": "0" * 64,
                     "resolved": draw.record(), "context_kind": ctx.kind}}


# ------------------------------------------------------------------------------------------ A. thaliana
@pytest.fixture(scope="module")
def ath(tmp_path_factory):
    w = FX.make_world(tmp_path_factory.mktemp("ath"))
    cache = OC.ObsCache(w.cache_dir)
    ref = CX.Reference(w.fasta)
    panel = CX.Panel(w.panel, w.panel_sha, "pilot")
    win = CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, False, "gH", "centred30720-v1")
    win_rc = CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, True, "gH", "centred30720-v1")
    bad = CX.EvalWindow(FX.SPECIES, FX.CHROM, 766, 31486, False, "gH", "centred30720-v1")
    nolocus = CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, False, "gZZ", "centred30720-v1")
    CX.WindowManifest.write(w.tmp / "windows.tsv", [win, win_rc, bad, nolocus])
    return w, cache, ref, panel, CX.WindowManifest(w.tmp / "windows.tsv"), win, win_rc, bad


def ectx(ath, window, **kw):
    w, cache, ref, panel, wm, *_ = ath
    args = dict(reference=ref, panel=panel, roles=w.roles, generator=window.generator, cache=cache,
                window_manifest=wm)
    args.update(kw)
    return CX.eval_context(window, **args)


def test_athaliana_eval_admits_only_locus_heldout_inputs(ath):
    w, cache, *_rest = ath
    win = ath[5]
    ctx = ectx(ath, win)
    assert ctx.row == "eval:Athaliana:Chr1:765-31485:fw:pilot" and ctx.kind == "eval" and not ctx.tile_exclusions
    o = AG.Owner.from_pairs(FX.SPECIES, cache.owner_libraries("eval_input", FX.SPECIES))
    d = AG.resolve_fixed(TS.dose_aug_seed(0), o, 1.0)
    t, qc = TR.rasterise_with_qc(cache, ctx, d)
    got = set().union(*map(set, qc["retained_molecules"].values()))
    assert got == {"eh", "oh"}                                  # oh_far misses the locus; train_track never admitted
    p = lambda g: g - 765                                      # noqa: E731
    assert t[p(16030), 0] == np.float16(np.log1p(1)) and t[p(16030), 8] == np.float16(np.log1p(1))
    assert t[p(16011), TS.phase_channel((1 + 1) % 3, "+")] > 0 and not t[p(16020):p(16022), 25:].any()
    assert not t[p(150), 0] and not t[p(145), 24]              # no train_track EST, no protein at evaluation
    tile = CH.build_chains(cache, ctx, d)
    assert tile.chain_mask.sum() == 1 + 2                       # eh (1 intron) + oh (2 introns); no rule 4 split
    rc = TR.rasterise(cache, ectx(ath, ath[6]), d)
    assert np.array_equal(rc, TS.rc_tracks(t))


def test_eval_context_rejections(ath, tmp_path):
    w, cache, ref, panel, wm, win, _rc, bad = ath
    with pytest.raises(CX.ContextError, match="generator"):
        ectx(ath, win, generator="a27-tiles-v1")
    with pytest.raises(CX.ContextError, match="generator"):
        ectx(ath, bad)
    other = tmp_path / "other.fa"
    other.write_text(Path(w.fasta).read_text().replace("A", "C", 1))
    with pytest.raises(CX.ContextError, match="reference"):
        ectx(ath, win, reference=CX.Reference(other))
    with pytest.raises(CX.ContextError, match="not in panel"):
        ectx(ath, CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, False, "gZZ", "centred30720-v1"))
    with pytest.raises(CX.ContextError, match="sha256"):
        CX.Panel(w.panel, "0" * 64, "pilot")
    unlisted = CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, False, "gH", "centred30720-v1", tier=1)
    with pytest.raises(CX.ContextError, match="manifest"):
        ectx(ath, unlisted)


def test_b6_score_molecule_at_evaluation_is_a_hard_error(tmp_path):
    w = FX.make_world(tmp_path)
    rows = [dict(r) for r in w.rows]
    leak = next(r for r in rows if r["molecule_id"] == "oh")
    for r in rows:
        if r["molecule_id"] == "oh":
            r["b6_score"] = True
    cdir = tmp_path / "tampered"
    OC.write_cache(cdir, rows, family="train", libraries=w.libraries, drops=w.drops,
                   bindings={"reference_sha256": {FX.SPECIES: FX._sha(w.fasta)}, "roles_sha256": w.roles.sha256},
                   zero_lists=w.zero)
    cache = OC.ObsCache(cdir)
    win = CX.EvalWindow(FX.SPECIES, FX.CHROM, 765, 31485, False, "gH", "centred30720-v1")
    CX.WindowManifest.write(tmp_path / "wm.tsv", [win])
    ctx = CX.eval_context(win, reference=CX.Reference(w.fasta), panel=CX.Panel(w.panel, w.panel_sha, "pilot"),
                          roles=w.roles, generator="centred30720-v1", cache=cache,
                          window_manifest=CX.WindowManifest(tmp_path / "wm.tsv"))
    o = AG.Owner.from_pairs(FX.SPECIES, cache.owner_libraries("eval_input", FX.SPECIES))
    with pytest.raises(CX.ContextError, match="b6_score"):
        TR.rasterise(cache, ctx, AG.resolve_fixed(1, o, 1.0))
    assert leak["kind"]


def test_zmays_in_a_corpus_context_is_refused(tmp_path):
    w = FX.make_world(tmp_path)
    corpus = CX.B6Corpus(w.corpus, OC.ObsCache(w.cache_dir))
    corpus.row = lambda rn: (rn, "Zmays", FX.CHROM, 0, 600, False, "train", "A" * 600, "wz", "")
    with pytest.raises(CX.ContextError, match="no corpus role"):
        CX.corpus_context(corpus, 1, "train_track")


# ------------------------------------------------------------------------------------------ Z. mays
def test_zmays_eval_c0_to_batch_with_no_corpus(tmp_path):
    z = FX.zmays_eval_c0(tmp_path)
    assert not list(tmp_path.glob("*corpus*.db")) and not list(tmp_path.glob("b5_full*"))
    roles = FX.make_roles([(z.est_dataset, z.est_run, "M-EST", True, False, "tracks+chains")], sha="z" * 64)
    import duckdb
    con = duckdb.connect(str(z.merged), read_only=True)
    rows, drops, libs = OC.extract_rows(con, roles, "eval")
    con.close()
    zq = tmp_path / "zero.qc.json"
    zt = tmp_path / "zero.tsv"
    zt.write_text("species\tchrom\tstart0\tend0\taction\treason\n")
    zq.write_text(json.dumps({"population": "eval_input", "species": "Zmays", "upstream": {"est_c0": z.c0_sha}}))
    cdir = tmp_path / "cache_eval"
    OC.write_cache(cdir, rows, family="eval", libraries=libs, drops=drops,
                   bindings={"reference_sha256": {"Zmays": FX._sha(z.fasta)}, "roles_sha256": roles.sha256,
                             "evidence_sha256": FX._sha(z.merged)},
                   zero_lists={"eval_input": {"Zmays": (zt, zq)}})
    cache = OC.ObsCache(cdir)
    panel_tsv = tmp_path / "zm_panel.tsv"
    panel_tsv.write_text("panel\tspecies_id\tgene_id\tchromosome\tstart\tfin\tstrand\n"
                         "final\tZmays\tgA\tChr1\t10001\t10800\t+\n")
    panel = CX.Panel(panel_tsv, FX._sha(panel_tsv), "final")
    win = CX.EvalWindow("Zmays", "Chr1", 0, 30720, False, "gA", "a27-tiles-v1", tier=30720, offset=0)
    CX.WindowManifest.write(tmp_path / "zm_windows.tsv", [win])
    wm = CX.WindowManifest(tmp_path / "zm_windows.tsv")
    ref = CX.Reference(z.fasta)
    # the eval-only family: bindings pass for evaluation, training consumers refuse it
    assert json.loads(zq.read_text())["upstream"] == {"est_c0": z.c0_sha}      # est_c0-only: no Z. mays protein C0
    ident = BD.verify_inputs(accept={"eval_only"}, evidence=z.merged, roles=roles, cache=cache,
                             references={"Zmays": ref}, panel=panel, window_manifest=wm)
    other = tmp_path / "other.fa"
    other.write_text(Path(z.fasta).read_text().replace("A", "C", 1))
    with pytest.raises(BD.BindingError, match="inputs.reference"):
        BD.verify_inputs(accept={"eval_only"}, evidence=z.merged, roles=roles, cache=cache,
                         references={"Zmays": CX.Reference(other)}, panel=panel, window_manifest=wm)
    assert ident["family"] == "eval_only"
    assert set(ident["provenance"]) == {"evidence.sha256", "c0:Zmays/est.PROVENANCE.txt", "zero_list:eval_input/Zmays",
                                        "zero_qc:eval_input/Zmays", "cache_manifest", "cache_manifest.sha256"}
    assert ident["record_sha256"] == BD.canonical_record_sha256(ident)
    with pytest.raises(BD.BindingError, match="not accepted"):
        BD.verify_inputs(accept={"training"}, evidence=z.merged, roles=roles, cache=cache, freeze=None, corpus=None)
    ctx = CX.eval_context(win, reference=ref, panel=panel, roles=roles, generator="a27-tiles-v1", cache=cache,
                          window_manifest=wm)
    o = AG.Owner.from_pairs("Zmays", cache.owner_libraries("eval_input", "Zmays"))
    d = AG.resolve_fixed(TS.dose_aug_seed(0), o, 1.0)
    t = TR.rasterise(cache, ctx, d)
    tile = CH.build_chains(cache, ctx, d)
    assert t[10100, 0] > 0 and t[:, 7].min() == 1 and tile.chain_mask.any()
    ids = FX.char_encoder(ctx.dna)[0]
    b = DS.b6_collate_fn([sample(ctx, ids, t, tile, d)])
    assert TS.validate_batch(b, training=False) == (1, 30720)
    assert b["meta"]["context_kind"] == ["eval"]
