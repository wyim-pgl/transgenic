"""CP1 (plan v4.1 section 10): the B6 model consumes the WP3 batch contract (plan 7.7) from both context kinds.

Until the WP3 dataset is merged the batches come from ``contract_batch`` below, which builds them from the plan text
(left padding as datasets.py:835-842, labels right-padded -100, chain_idx = (pad_b + 0 + chain_pos) // 6 as one
sum and one division). ``WP3SpecAgreementTests`` checks the model against Writer B's frozen ``b6_track_spec``.
"""
import os
import sys
import unittest

import torch

from transgenic.model import modeling_b6 as b6
from transgenic.training import b6_runtime as rt
from tests.test_b6_model import B6Base, N

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "revision", "scripts", "b6"))


def contract_batch(kind="corpus", lengths=(48, 30), n_real=(40, 0), rc=(False, True), labels=True, seed=0):
    """A plan-7.7 batch: B rows of tile length lengths[b] (multiples of 6), left-padded to Lmax."""
    g = torch.Generator().manual_seed(seed)
    b, lmax = len(lengths), max(lengths)
    ids = torch.full((b, lmax), 0, dtype=torch.int64)             # pad id of the B5 collate
    am = torch.zeros(b, lmax, dtype=torch.bool)
    tracks = torch.zeros(b, lmax, rt.K_TRACKS, dtype=torch.float16)
    idx = torch.zeros(b, N, 2, dtype=torch.int64)
    feat = torch.zeros(b, N, 6, dtype=torch.int16)
    mask = torch.zeros(b, N, dtype=torch.bool)
    for r, (length, n) in enumerate(zip(lengths, n_real)):
        pad = lmax - length
        ids[r, pad:] = torch.randint(1, 16, (length,), generator=g)
        am[r, pad:] = True
        t = torch.rand(length, rt.K_TRACKS, generator=g) * torch.log1p(torch.tensor(64.0))
        t[:, list(rt.MASK_CHANNELS)] = 1.0
        tracks[r, pad:] = t.to(torch.float16)
        if n:
            pos = torch.randint(0, length, (n, 2), generator=g)             # tile-local chain_pos
            idx[r, 1:n + 1] = (pad + 0 + pos) // 6
            feat[r, 1:n + 1] = torch.stack([torch.randint(0, 32, (n,), generator=g),
                                            torch.randint(0, 32, (n,), generator=g),
                                            torch.randint(1, 64, (n,), generator=g),
                                            torch.randint(0, 3, (n,), generator=g),
                                            torch.randint(0, 2, (n,), generator=g),
                                            torch.randint(1, 65, (n,), generator=g)], dim=1).to(torch.int16)
            mask[r, 1:n + 1] = True
    rows = ([str(1000 + r) for r in range(b)] if kind == "corpus" else
            [f"eval:Zmays:chr1:{100 * r}-{100 * r + lengths[r]}:{'rc' if rc[r] else 'fw'}:zm_final" for r in range(b)])
    batch = dict(input_ids=ids, attention_mask=am, evidence_tracks=tracks, chain_idx=idx, chain_feat=feat,
                 chain_mask=mask,
                 labels=None,
                 meta=dict(row=rows, epoch=[0] * b, arm="B6", uniforms_sha256=["0" * 64] * b,
                           resolved=[{"dropped": False, "is_rc": r} for r in rc], context_kind=[kind] * b))
    if labels:
        lab = torch.full((b, 5), -100, dtype=torch.int64)
        lab[0, :5] = torch.tensor([7, 9, 30, 21, 2])
        lab[1, :3] = torch.tensor([8, 22, 2])
        batch["labels"] = lab
    return batch


class BatchContractTests(unittest.TestCase):
    def test_both_context_kinds_validate(self):
        rt.validate_b6_batch(contract_batch("corpus"))
        rt.validate_b6_batch(contract_batch("eval", labels=False), with_labels=False)

    def test_each_violation_is_refused(self):
        def mutated(fn, **kw):
            batch = contract_batch(**kw)
            fn(batch)
            return batch
        cases = {
            "batch keys": lambda b: b.pop("chain_feat"),
            "chain_pos": lambda b: b.update(chain_pos=torch.zeros(1)),
            "column 0": lambda b: b["chain_mask"].__setitem__((0, 0), True),
            "masked chain_idx must be 0": lambda b: b["chain_idx"].__setitem__((1, 7, 0), 1),
            "masked chain_feat must be 0": lambda b: b["chain_feat"].__setitem__((1, 9, 3), 1),
            "attention_mask": lambda b: b.update(attention_mask=b["attention_mask"].long()),
            "Lmax/6": lambda b: b["chain_idx"].__setitem__((0, 1, 0), 8),
            "float16": lambda b: b.update(evidence_tracks=b["evidence_tracks"].float()),
            "left padding": lambda b: b["evidence_tracks"].__setitem__((1, 0, 3), 1.0),
            "int16": lambda b: b.update(chain_feat=b["chain_feat"].long()),
            "slot": lambda b: b["chain_feat"].__setitem__((0, 1, 2), 64),
            "meta": lambda b: b["meta"].pop("uniforms_sha256"),
            "context_kind": lambda b: b["meta"].update(context_kind=["test"] * 2),
        }
        for message, fn in cases.items():
            with self.subTest(case=message), self.assertRaisesRegex(ValueError, message):
                rt.validate_b6_batch(mutated(fn))
        with self.assertRaisesRegex(ValueError, "multiple of 6"):
            rt.validate_b6_batch(contract_batch(lengths=(47, 30), n_real=(0, 0)))


class ArmTransformTests(unittest.TestCase):
    def test_arm_transforms_follow_the_matrix_and_use_no_rng(self):
        batch = contract_batch()
        state = torch.get_rng_state()
        c2 = rt.apply_arm(batch, rt.ARMS["C2"])
        c4 = rt.apply_arm(batch, rt.ARMS["C4"])
        c6 = rt.apply_arm(batch, rt.ARMS["C6"])
        c7 = rt.apply_arm(batch, rt.ARMS["C7"])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for arm in ("B6", "C3"):
            out = rt.apply_arm(batch, rt.ARMS[arm])
            for key in rt.MODEL_KEYS:
                if key in batch:
                    self.assertTrue(torch.equal(out[key], batch[key]), (arm, key))
        self.assertEqual(torch.count_nonzero(c2["evidence_tracks"]).item(), 0)
        self.assertFalse(c2["chain_mask"].any())
        self.assertEqual(torch.count_nonzero(c2["chain_idx"]).item(), 0)
        tr = batch["evidence_tracks"]
        counts = list(rt.COUNT_CHANNELS)
        self.assertTrue(torch.equal(c4["evidence_tracks"][..., counts], (tr[..., counts] > 0).half()))
        self.assertTrue(torch.equal(c4["evidence_tracks"][..., list(rt.MASK_CHANNELS)],
                                    tr[..., list(rt.MASK_CHANNELS)]))
        self.assertTrue(torch.equal(c4["chain_mask"], batch["chain_mask"]))
        self.assertEqual(torch.count_nonzero(c6["evidence_tracks"][..., 25:31]).item(), 0)
        self.assertTrue(torch.equal(c6["evidence_tracks"][..., :25], tr[..., :25]))
        self.assertTrue(torch.equal(c7["evidence_tracks"], tr))
        self.assertFalse(c7["chain_mask"].any())
        for arm, once in (("C4", c4), ("C6", c6), ("C2", c2)):
            twice = rt.apply_arm(once, rt.ARMS[arm])
            self.assertTrue(torch.equal(twice["evidence_tracks"], once["evidence_tracks"]), arm)
        dn = rt.de_novo_view(batch)
        for key in ("evidence_tracks", "chain_idx", "chain_feat", "chain_mask"):
            self.assertTrue(torch.equal(dn[key], c2[key]), key)
        for out in (c2, c4, c6, c7, dn):
            rt.validate_b6_batch(out)
        self.assertEqual(sorted(rt.ARMS), ["B6", "C2", "C3", "C4", "C6", "C7"])
        self.assertEqual([a for a, p in sorted(rt.ARMS.items()) if p.budget == "full"], ["B6", "C2"])


class ModelOnContractTests(B6Base):
    def test_model_trains_on_corpus_batches_and_generates_on_eval_batches(self):
        model = self.trained()
        corpus = contract_batch("corpus", n_real=(40, 3))
        rt.validate_b6_batch(corpus)
        model.train()
        out = model(**rt.model_inputs(corpus))
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        model.eval()
        for arm in rt.ARMS.values():
            with torch.no_grad():
                loss = model(**rt.model_inputs(rt.apply_arm(corpus, arm))).loss
            self.assertTrue(torch.isfinite(loss), arm.name)
        ev = contract_batch("eval", n_real=(12, 0), labels=False)
        rt.validate_b6_batch(ev, with_labels=False)
        inputs = rt.model_inputs(ev)
        inputs.pop("attention_mask")
        seq = model.generate(inputs.pop("input_ids"), max_new_tokens=4, eos_token_id=None, num_beams=2,
                             do_sample=False, **inputs)
        self.assertEqual(seq.shape[0], 2)

    def test_de_novo_view_is_the_sequence_only_path(self):
        model = self.trained()
        corpus = contract_batch("corpus", n_real=(40, 3))
        with torch.no_grad():
            dn = model(**rt.model_inputs(rt.de_novo_view(corpus))).logits
            seq = model(corpus["input_ids"], attention_mask=corpus["attention_mask"], labels=corpus["labels"]).logits
        self.assertTrue(torch.equal(dn, seq))


class WP3StubBatchTests(B6Base):
    """Writer B's own plan-7.7 stub batches (tests/b6_fixtures.py, byte-identical copy) through the model."""

    def test_model_runs_on_wp3_stub_batches_for_every_arm_and_generates(self):
        try:
            from tests.b6_fixtures import stub_batch_torch
        except ImportError as exc:        # since sha 1a7b6b07 the fixtures import WP3's b6_bindings
            self.skipTest(f"needs merged b6_bindings (mandatory after the b6-integration merge): {exc}")
        model = self.trained()
        batch = stub_batch_torch(lengths=(48, 42), label_lengths=(7, 4))
        rt.validate_b6_batch(batch)
        self.assertTrue(batch["chain_mask"][0].any())
        self.assertFalse(batch["chain_mask"][-1].any())                 # B's chain-less row
        for arm in rt.ARMS.values():
            with torch.no_grad():
                loss = model(**rt.model_inputs(rt.apply_arm(batch, arm))).loss
            self.assertTrue(torch.isfinite(loss), arm.name)
        model.train()
        model(**rt.model_inputs(batch)).loss.backward()
        model.eval()
        with torch.no_grad():
            dn = model(**rt.model_inputs(rt.de_novo_view(batch))).logits
            seq = model(batch["input_ids"], labels=batch["labels"]).logits
        self.assertTrue(torch.equal(dn, seq))
        ev = stub_batch_torch(lengths=(48, 42), training=False)
        rt.validate_b6_batch(ev, with_labels=False)
        inputs = rt.model_inputs(ev)
        inputs.pop("attention_mask")
        out = model.generate(inputs.pop("input_ids"), max_new_tokens=4, eos_token_id=None, num_beams=2,
                             do_sample=False, **inputs)
        self.assertEqual(out.shape[0], 2)


class WP3SpecAgreementTests(unittest.TestCase):
    """The model and runtime against Writer B's frozen b6_track_spec (byte-identical copy, sha in progress file)."""

    def test_model_dimensions_are_the_frozen_chain_spec(self):
        from transgenic.evidence import b6_track_spec as spec
        dims = spec.MODEL_DIMS
        memory = b6.ChainMemory(dims["d_encoder_memory"], dims["d_chain"])
        block = b6.ChainCrossAttention(dims["d_encoder_memory"], dims["d_chain"], dims["heads"],
                                       dims["block_dropout"])
        for name, layer in (("W_d", memory.w_d), ("W_a", memory.w_a), ("Q", block.q_proj), ("K", block.k_proj),
                            ("V", block.v_proj), ("out", block.out_proj)):
            self.assertEqual([layer.in_features, layer.out_features], dims[name], name)
        for name, emb in (("E_order", memory.e_order), ("E_order_rev", memory.e_order_rev), ("E_slot", memory.e_slot),
                          ("E_source", memory.e_source), ("E_strand", memory.e_strand)):
            self.assertEqual(emb.num_embeddings, dims[name], name)
        self.assertEqual(memory.w_count.shape, torch.Size([]))                # "scalar * log1p(units_bin)"
        self.assertEqual((block.heads, block.head_dim, block.dropout), (dims["heads"], dims["head_dim"], 0.1))
        self.assertEqual(float(block.alpha), dims["alpha_init"])
        self.assertIs(dims["alpha_weight_decay"], False)
        self.assertEqual((b6.CHAIN_ELEMENTS, b6.CHAIN_FEAT_COLUMNS, b6.DOWNSAMPLE_FACTOR, b6.UNITS_BIN_MAX),
                         (spec.N_TOK, spec.CHAIN_FEAT_FIELDS, spec.DOWNSAMPLE, spec.UNITS_CAP))
        self.assertEqual(b6.CHAIN_EMBED_ROWS, {"order_fwd": dims["E_order"], "order_rev": dims["E_order_rev"],
                                               "slot": dims["E_slot"], "source": dims["E_source"],
                                               "strand": dims["E_strand"]})
        cfg = rt.load_b6_config(os.path.join(ROOT, "configs", "b6_400m_win_v3.json"))
        self.assertEqual((cfg["d_model_decoder"], cfg["chain_d_model"], cfg["decoder_layers"], cfg["chain_heads"]),
                         (dims["d_encoder_memory"], dims["d_chain"], dims["layers"], dims["heads"]))

    def test_chain_index_and_rc_rules_match_the_spec(self):
        import numpy as np
        from transgenic.evidence import b6_track_spec as spec
        batch = contract_batch(n_real=(40, 7))
        self.assertEqual(spec.validate_batch(batch, training=True), (2, 48))
        pos = np.random.default_rng(0).integers(0, 30, size=(512, 2))
        mask = np.zeros(512, dtype=bool)
        mask[1:20] = True
        ref = spec.chain_idx(pos, mask, pad=18)
        ours = torch.where(torch.from_numpy(mask)[:, None], (18 + 0 + torch.from_numpy(pos)) // 6, 0)
        self.assertTrue(np.array_equal(ref, ours.numpy()))
        import wp4_synthetic as syn
        self.assertEqual(syn.rc_permutation(), list(spec.RC_PERM))
        rows = syn.SyntheticB6Dataset(4, seed=4)
        fw, rc = rows[0], rows[1]
        self.assertEqual((fw["kind"], rc["kind"]), ("evidence", "evidence_rc"))
        syn.assert_rc_pair(rows, 0)

    def test_runtime_identity_binds_the_spec_shas(self):
        from transgenic.evidence import b6_track_spec as spec
        import wp4_synthetic as syn
        self.assertEqual(rt.SPEC_SHAS, spec.SPEC_SHAS)
        ident = rt.identity_from_bindings(syn.binding_record())
        self.assertEqual(ident["binding.chain_spec_sha256"], spec.CHAIN_SPEC_SHA)
        with self.assertRaisesRegex(ValueError, "chain_spec_sha256"):
            rt.identity_from_bindings(syn.binding_record(chain_spec_sha256="0" * 64))


if __name__ == "__main__":
    unittest.main()
