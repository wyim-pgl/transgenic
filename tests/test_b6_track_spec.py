"""b6_track_spec: channel layout, RC permutation, rule 6a slot hash, chain-rc-v1, batch contract, spec shas."""
import hashlib

import numpy as np
import pytest

import b6_fixtures as FX
from transgenic.evidence import b6_track_spec as TS


def test_channel_order_is_rule_6_k31():
    assert TS.K == 31 and len(TS.CHANNEL_NAMES) == 31
    assert TS.CHANNEL_NAMES[:8] == tuple(f"est:{c}" for c in
                                         ("cov+", "cov-", "cov?", "donor+", "acceptor+", "donor-", "acceptor-", "mask"))
    assert TS.CHANNEL_NAMES[8] == "ont:cov+" and TS.CHANNEL_NAMES[16] == "pacbio:cov+"
    assert TS.CHANNEL_NAMES[24] == "protein:cds"
    assert TS.CHANNEL_NAMES[25:] == ("phase:0+", "phase:1+", "phase:2+", "phase:0-", "phase:1-", "phase:2-")
    assert TS.MASK_CHANNELS == (7, 15, 23)
    assert TS.phase_channel(2, "-") == 30 and TS.phase_channel(0, "+") == 25


def test_rc_permutation_is_the_frozen_one():
    perm = TS.RC_PERM
    for s in range(3):
        assert perm[8 * s:8 * s + 8] == tuple(8 * s + p for p in (1, 0, 2, 5, 6, 3, 4, 7))
    assert perm[24] == 24
    assert perm[25:] == (28, 29, 30, 25, 26, 27)


def test_rc_tracks_twice_is_identity_and_swaps_strand_keeping_role():
    L = 12
    t = np.zeros((L, TS.K), np.float16)
    t[2, 3] = 1.5            # EST donor+ at base 2
    t[7, 4] = 0.5            # EST acceptor+ at base 7
    t[2:8, 0] = 1.0          # EST cov+
    t[3, TS.phase_channel(1, "+")] = 0.25
    r = TS.rc_tracks(t)
    assert r[L - 1 - 2, 5] == np.float16(1.5)          # donor stays donor, strand becomes -
    assert r[L - 1 - 7, 6] == np.float16(0.5)
    assert (r[L - 8:L - 2, 1] == 1).all()
    assert r[L - 1 - 3, TS.phase_channel(1, "-")] == np.float16(0.25)   # label kept
    assert np.array_equal(TS.rc_tracks(r), t)


def test_slot_hash_equals_a_literal_rule_6a_reference():
    aug, row, epoch = 20261037, "4711", 3
    ref = {s: int.from_bytes(hashlib.blake2b(f"{aug}\t{row}\t{epoch}\t{s}".encode("utf-8"),
                                             digest_size=8).digest(), "big") for s in range(1, 64)}
    assert all(TS.slot_hash(aug, row, epoch, s) == ref[s] for s in range(1, 64))
    pi = TS.slot_permutation(aug, row, epoch)
    assert pi == tuple(sorted(range(1, 64), key=lambda s: (ref[s], s)))
    assert sorted(pi) == list(range(1, 64))
    with pytest.raises(ValueError):
        TS.slot_hash(aug, row, epoch, 0)
    with pytest.raises(ValueError):
        TS.slot_hash(aug, 4711, epoch, 1)               # rows are str keys, never ints


def test_rank_uniform_and_tie_key_literals():
    ref = int.from_bytes(hashlib.blake2b("20261037\tAthaliana\tmolX".encode("utf-8"), digest_size=8).digest(),
                         "big") / 2 ** 64
    assert TS.molecule_rank(20261037, "Athaliana", "molX") == ref
    u = int.from_bytes(hashlib.blake2b("20261037\tdropout\t12\t0".encode("utf-8"), digest_size=8).digest(),
                       "big") / 2 ** 64
    assert TS.uniform(20261037, "dropout", "12", 0) == u
    key = int.from_bytes(hashlib.blake2b("Athaliana\t-\t500-400,300-200".encode("utf-8"), digest_size=8).digest(),
                         "big")
    assert TS.tie_key("Athaliana", "-", [(500, 400), (300, 200)]) == key
    assert TS.training_aug_seed(123) == 20261037 and TS.dose_aug_seed(4) == 20260918
    with pytest.raises(ValueError):
        TS.uniform(1, "slot", "1", 0)


def test_fill_sort_key_orders_long_reads_first_and_pacbio_before_ont():
    keys = [("est", 50, 9, 1), ("ont", 2, 1, 5), ("pacbio", 2, 1, 5), ("ont", 3, 1, 9), ("ont", 2, 2, 9)]
    order = sorted(keys, key=lambda k: TS.fill_sort_key(*k))
    assert order == [("ont", 3, 1, 9), ("ont", 2, 2, 9), ("pacbio", 2, 1, 5), ("ont", 2, 1, 5), ("est", 50, 9, 1)]


def test_chain_rc_v1_twice_is_identity():
    pos = np.zeros((512, 2), np.int32)
    feat = np.zeros((512, 6), np.int16)
    mask = np.zeros(512, bool)
    pos[1], feat[1], mask[1] = (10, 20), (0, 1, 5, 2, 0, 3), True
    pos[2], feat[2], mask[2] = (30, 44), (1, 0, 5, 2, 0, 3), True
    p2, f2, m2 = TS.rc_chain(pos, feat, mask, 100)
    assert tuple(p2[1]) == (89, 79) and f2[1, TS.FEAT_STRAND] == 1 and f2[1, TS.FEAT_ORDER_FWD] == 0
    assert (p2[0] == 0).all() and (f2[0] == 0).all()
    p3, f3, m3 = TS.rc_chain(p2, f2, m2, 100)
    assert np.array_equal(p3, pos) and np.array_equal(f3, feat) and np.array_equal(m3, mask)


def test_chain_idx_one_sum_one_division_and_pad_invariance():
    pos = np.zeros((512, 2), np.int32)
    mask = np.zeros(512, bool)
    pos[1], mask[1] = (5, 11), True
    assert tuple(TS.chain_idx(pos, mask, 0)[1]) == (0, 1)
    assert tuple(TS.chain_idx(pos, mask, 1)[1]) == (1, 2)          # (1+5)//6 = 1, not 0 + 5//6
    assert (TS.chain_idx(pos, mask, 7)[0] == 0).all()


def test_stub_batch_satisfies_the_contract():
    b = FX.stub_batch()
    assert TS.validate_batch(b, training=True) == (2, 60)
    assert b["chain_mask"][1].sum() == 0 and b["chain_mask"][0].sum() == 4
    ev = FX.stub_batch(training=False)
    TS.validate_batch(ev, training=False)
    bad = dict(b)
    bad["chain_mask"] = b["chain_mask"].copy()
    bad["chain_mask"][0, 0] = True
    with pytest.raises(ValueError):
        TS.validate_batch(bad, training=True)


def test_spec_shas_are_stable_hex():
    for v in (TS.TRACK_SPEC_SHA, TS.CHAIN_SPEC_SHA, TS.AUG_SPEC_SHA):
        assert len(v) == 64 and int(v, 16) >= 0
    assert TS.canonical_sha256(TS.TRACK_SPEC) == TS.TRACK_SPEC_SHA
    assert TS.AUG_SPEC["id"] == "b6-aug-uniform-blake2b-v2"
