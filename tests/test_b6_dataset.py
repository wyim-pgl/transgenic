"""B6Dataset / B6EpochSampler / b6_collate_fn: §7.7 contract, worker and resume determinism, arm matrix."""
import hashlib
import multiprocessing as mp

import numpy as np
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_aug as AG
from transgenic.evidence import b6_context as CX
from transgenic.evidence import b6_obs_cache as OC
from transgenic.evidence import b6_track_spec as TS

DS = FX.load_b6_dataset()
AUG = TS.training_aug_seed()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return FX.make_world(tmp_path_factory.mktemp("w"))


def make(world, arm="B6", mode="train_track", **kw):
    cache = OC.ObsCache(world.cache_dir)
    corpus = CX.B6Corpus(world.corpus, cache)
    return DS.B6Dataset(corpus, cache, mode=mode, arm=arm, aug_seed=AUG, encode_dna=FX.char_encoder,
                        tokenize_gsf=FX.gsf_tokens, **kw)


def digest(sample):
    h = hashlib.sha256()
    for k in ("input_ids", "evidence_tracks", "chain_pos", "chain_feat", "chain_mask"):
        h.update(np.ascontiguousarray(sample[k]).tobytes())
    h.update(sample["meta"]["uniforms_sha256"].encode())
    return h.hexdigest()


def test_batch_contract_and_pad_invariant_chain_idx(world):
    ds = make(world)
    s1, s5 = ds[(1, 0)], ds[(5, 0)]
    b = DS.b6_collate_fn([s1, s5])
    assert TS.validate_batch(b, training=True) == (2, 600)
    short = dict(s5)
    for k in ("input_ids", "attention_mask", "evidence_tracks"):
        short[k] = s5[k][:594] if k != "evidence_tracks" else s5[k][:594]
    b2 = DS.b6_collate_fn([s1, short])
    assert not b2["attention_mask"][1, :6].any() and b2["attention_mask"][1, 6:].all()
    m = s1["chain_mask"]
    assert np.array_equal(b["chain_idx"][0][m], s1["chain_pos"][m] // 6)


def test_left_pad_shifts_chain_idx_by_one_sum_then_division():
    s = {"input_ids": np.full(6, 7, np.int64), "attention_mask": np.ones(6, bool),
         "evidence_tracks": np.zeros((6, 31), np.float16), "labels": np.array([1, 2], np.int64),
         "chain_pos": np.zeros((512, 2), np.int32), "chain_feat": np.zeros((512, 6), np.int16),
         "chain_mask": np.zeros(512, bool), "meta": {k: None for k in TS.META_KEYS}}
    s["chain_pos"][1] = (5, 5)
    s["chain_mask"][1] = True
    s["chain_feat"][1] = (0, 0, 3, 0, 0, 1)
    long = dict(s, input_ids=np.full(12, 7, np.int64), attention_mask=np.ones(12, bool),
                evidence_tracks=np.zeros((12, 31), np.float16), chain_mask=np.zeros(512, bool))
    b = DS.b6_collate_fn([s, long])
    assert b["chain_idx"][0, 1].tolist() == [(6 + 5) // 6, (6 + 5) // 6] == [1, 1]
    assert (b["chain_idx"][1] == 0).all() and not b["chain_mask"][:, 0].any()


def _worker(args):
    world_paths, keys = args
    import sys
    sys.path[:0] = world_paths["sys_path"]
    import b6_fixtures as F
    from types import SimpleNamespace
    from transgenic.evidence import b6_context as CX2
    from transgenic.evidence import b6_obs_cache as OC2
    ds_mod = F.load_b6_dataset()
    cache = OC2.ObsCache(world_paths["cache"])
    ds = ds_mod.B6Dataset(CX2.B6Corpus(world_paths["corpus"], cache), cache, mode="train_track", arm="B6",
                          aug_seed=AUG, encode_dna=F.char_encoder, tokenize_gsf=F.gsf_tokens)
    return [digest(ds[k]) for k in keys]


def test_draws_identical_across_worker_counts_and_resume(world):
    import sys
    keys = [(rn, ep) for ep in (0, 1, 2) for rn in (1, 2, 5, 7, 8)]
    ref = [digest(make(world)[k]) for k in keys]
    paths = {"cache": str(world.cache_dir), "corpus": str(world.corpus), "sys_path": list(sys.path)}
    ctx = mp.get_context("spawn")
    with ctx.Pool(3) as pool:
        parts = pool.map(_worker, [(paths, keys[i::3]) for i in range(3)])
    got = [None] * len(keys)
    for i in range(3):
        got[i::3] = parts[i]
    assert got == ref
    order = lambda ep: [1, 2, 5, 7, 8]                     # noqa: E731
    full = list(DS.B6EpochSampler(order, epochs=3))
    resumed = list(DS.B6EpochSampler(order, epochs=3, start_epoch=1, start_index=2))
    assert full[7:] == resumed and len(DS.B6EpochSampler(order, epochs=3, start_epoch=1, start_index=2)) == 8
    assert [digest(make(world)[k]) for k in resumed] == ref[7:]


def test_arm_matrix_uniform_logs_identical_and_arm_semantics(world):
    donors = AG.c3_donor_map([(rn, f"{FX.SPECIES}:{FX.CHROM}:{ws}-{ws + FX.TILE}", FX.TILE, sp, rc)
                              for rn, ws, rc, sp in FX.TILES if sp != "test"], AUG)
    logs = {}
    for arm in TS.ARMS:
        ds = make(world, arm=arm, c3_donors=donors if arm == "C3" else None)
        logs[arm] = [ds[(rn, ep)]["meta"]["uniforms_sha256"] for rn in (1, 5, 7) for ep in (0, 1)]
    assert all(v == logs["B6"] for v in logs.values())
    b6 = make(world)
    key = next((rn, ep) for ep in range(40) for rn in (1,) if not b6[(rn, ep)]["meta"]["resolved"]["dropped"])
    s = {arm: make(world, arm=arm, c3_donors=donors if arm == "C3" else None)[key] for arm in TS.ARMS}
    assert not s["C2"]["evidence_tracks"].any() and not s["C2"]["chain_mask"].any()
    assert not s["C7"]["chain_mask"].any() and np.array_equal(s["C7"]["evidence_tracks"], s["B6"]["evidence_tracks"])
    assert np.array_equal(s["C6"]["chain_mask"], s["B6"]["chain_mask"]) and not s["C6"]["evidence_tracks"][:, 25:].any()
    assert s["C3"]["meta"]["resolved"]["species"] == FX.SPECIES


def test_c3_requires_donor_map(world):
    with pytest.raises(ValueError, match="donor map"):
        make(world, arm="C3")
