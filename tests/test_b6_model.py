"""B6 model acceptance (plan v4.1 section 9), eager fp32 on CPU; only the external Hyena factories are stubbed."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from transgenic.model import modeling_HyenaTransgenic as b5
from transgenic.model import modeling_b6 as b6
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig

L = 48                      # encoder length (multiple of 6); downsampled length 8
N = b6.CHAIN_ELEMENTS


def tiny_kwargs(**kw):
    base = dict(vocab_size=290, d_model=8, decoder_d_model=16, encoder_layers=2, decoder_layers=2,
                encoder_n_layer=1, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                decoder_ffn_dim=32, max_encoder_seqlen=L, max_encoder_position_embeddings=L,
                max_decoder_position_embeddings=64, attention_window=[4, 4], dropout=0.0, decoder_layerdrop=0.0,
                load_pretrained_encoder=False)
    base.update(kw)
    return base


def tiny_b6(**kw):
    kw = tiny_kwargs(**kw)
    kw.setdefault("b6_chain_d_model", 16)
    kw.setdefault("b6_chain_heads", 8)
    kw.setdefault("b6_chain_dropout", 0.0)
    return b6.b6_config(**kw)


class TinyHyena(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(16, config.d_model)

    def forward(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids), hidden_states=None)


def chain_batch(batch, s_len, n_real, seed=0):
    """Contract-shaped chain tensors: column 0 dummy, n_real[b] real elements per row, index 0 when masked."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.zeros(batch, N, 2, dtype=torch.int64)
    feat = torch.zeros(batch, N, 6, dtype=torch.int16)
    mask = torch.zeros(batch, N, dtype=torch.bool)
    for b_, n in enumerate(n_real):
        if n:
            idx[b_, 1:n + 1] = torch.randint(0, s_len, (n, 2), generator=g)
            feat[b_, 1:n + 1, 0] = torch.randint(0, 32, (n,), generator=g).to(torch.int16)
            feat[b_, 1:n + 1, 1] = torch.randint(0, 32, (n,), generator=g).to(torch.int16)
            feat[b_, 1:n + 1, 2] = torch.randint(1, 64, (n,), generator=g).to(torch.int16)
            feat[b_, 1:n + 1, 3] = torch.randint(0, 3, (n,), generator=g).to(torch.int16)
            feat[b_, 1:n + 1, 4] = torch.randint(0, 2, (n,), generator=g).to(torch.int16)
            feat[b_, 1:n + 1, 5] = torch.randint(1, 65, (n,), generator=g).to(torch.int16)
            mask[b_, 1:n + 1] = True
    return dict(chain_idx=idx, chain_feat=feat, chain_mask=mask)


def masked_chains(batch):
    return dict(chain_idx=torch.zeros(batch, N, 2, dtype=torch.int64),
                chain_feat=torch.zeros(batch, N, 6, dtype=torch.int16),
                chain_mask=torch.zeros(batch, N, dtype=torch.bool))


class B6Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        p1 = patch.object(b5.AutoConfig, "from_pretrained", side_effect=lambda *a, **k: SimpleNamespace())
        p2 = patch.object(b5.AutoModel, "from_config", side_effect=lambda config, **k: TinyHyena(config))
        for p in (p1, p2):
            p.start()
            self.addCleanup(p.stop)
        torch.manual_seed(123)
        self.ids = torch.randint(0, 16, (2, L))
        self.prefix = torch.tensor([[2, 7, 9, 30], [2, 8, 22, 40]])
        self.labels = torch.tensor([[7, 9, 30, 2], [8, 22, 40, -100]])
        self.tracks = torch.rand(2, L, 31, dtype=torch.float16)
        self.s_len = L // 6

    def model(self, **kw):
        return b6.B6TransgenicForConditionalGeneration(tiny_b6(**kw)).eval()

    def trained(self, **kw):
        """A model whose evidence projection, chain modules and gates are all non-trivial."""
        m = self.model(**kw)
        with torch.no_grad():
            m.get_encoder().evidence_proj.weight.normal_(0, 0.5)
            for layer in m.transgenic.decoder.layers:
                layer.chain_attn.alpha.fill_(0.7)
        return m


class ConfigAndInitTests(B6Base):
    def test_config_rejects_non_b6(self):
        with self.assertRaisesRegex(ValueError, "not a B6 config"):
            b6.B6TransgenicForConditionalGeneration(HyenaTransgenicConfig(**tiny_kwargs(evidence_channels=31)))
        with self.assertRaisesRegex(ValueError, "evidence_channels=31"):
            tiny_b6(evidence_channels=8)
        with self.assertRaisesRegex(ValueError, "multiple"):
            tiny_b6(b6_chain_d_model=12, b6_chain_heads=8)

    def test_b5_parameters_keep_names_and_initialisation(self):
        torch.manual_seed(7)
        ref = b5.transgenicForConditionalGeneration(HyenaTransgenicConfig(**tiny_kwargs(evidence_channels=31)))
        torch.manual_seed(7)
        model = b6.B6TransgenicForConditionalGeneration(tiny_b6())
        ref_sd, sd = ref.state_dict(), model.state_dict()
        extra = sorted(set(sd) - set(ref_sd))
        self.assertEqual(sorted(set(ref_sd) - set(sd)), [])
        self.assertTrue(all(".chain_attn." in k or ".chain_memory." in k for k in extra), extra)
        for key, value in ref_sd.items():
            self.assertTrue(torch.equal(value, sd[key]), key)
        self.assertIsInstance(model.transgenic, b6.B6TransgenicModel)
        self.assertTrue(all(isinstance(layer, b6.B6DecoderLayer) for layer in model.transgenic.decoder.layers))

    def test_alpha_zero_after_init_and_after_initialize_weights(self):
        model = self.model()
        self.assertEqual(model.chain_alphas(), [0.0, 0.0])
        self.assertEqual(len(b6.alpha_parameter_names(model)), 2)
        with torch.no_grad():
            for layer in model.transgenic.decoder.layers:
                layer.chain_attn.alpha.fill_(1.0)
        model.initialize_weights()
        self.assertEqual(model.chain_alphas(), [0.0, 0.0])
        self.assertEqual(torch.count_nonzero(model.get_encoder().evidence_proj.weight).item(), 0)

    def test_save_pretrained_roundtrip(self):
        model = self.trained()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d, safe_serialization=True)
            loaded = b6.B6TransgenicForConditionalGeneration.from_pretrained(d).eval()
        self.assertEqual(loaded.config.b6_model_version, b6.B6_MODEL_VERSION)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        chains = chain_batch(2, self.s_len, [5, 3])
        with torch.no_grad():
            a = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks, **chains).logits
            b = loaded(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks, **chains).logits
        self.assertTrue(torch.equal(a, b))


class ExactnessTests(B6Base):
    def test_zero_tracks_and_fully_masked_chains_give_sequence_only_bits(self):
        model = self.trained()
        with torch.no_grad():
            seq = model(self.ids, decoder_input_ids=self.prefix, labels=self.labels)
            ev = model(self.ids, decoder_input_ids=self.prefix, labels=self.labels,
                       evidence_tracks=torch.zeros_like(self.tracks), **masked_chains(2))
        self.assertTrue(torch.equal(seq.logits, ev.logits))
        self.assertTrue(torch.equal(seq.loss, ev.loss))

    def test_alpha_zero_with_zero_tracks_gives_b5_bits_even_with_real_chains(self):
        ref = b5.transgenicForConditionalGeneration(HyenaTransgenicConfig(**tiny_kwargs(evidence_channels=31))).eval()
        model = self.model()
        result = model.load_state_dict(ref.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all(".chain_" in k for k in result.missing_keys))
        with torch.no_grad():
            expected = ref(self.ids, decoder_input_ids=self.prefix, labels=self.labels)
            actual = model(self.ids, decoder_input_ids=self.prefix, labels=self.labels,
                           evidence_tracks=torch.zeros_like(self.tracks), **chain_batch(2, self.s_len, [40, 7]))
        self.assertTrue(torch.equal(expected.logits, actual.logits))
        self.assertTrue(torch.equal(expected.loss, actual.loss))

    def test_b6_loss_equals_b5_loss_term_for_term(self):
        ref = b5.transgenicForConditionalGeneration(HyenaTransgenicConfig(**tiny_kwargs(evidence_channels=31))).eval()
        with torch.no_grad():
            out = ref(self.ids, decoder_input_ids=self.prefix, labels=self.labels)
            self.assertTrue(torch.equal(out.loss, b6.b5_gsf_loss(out.logits, self.labels, 290)))

    def test_nonzero_gate_uses_chains_and_masked_row_is_inert(self):
        model = self.trained()
        chains = chain_batch(2, self.s_len, [20, 0])          # row 1 has no chain: m = 0
        with torch.no_grad():
            seq = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks).logits
            ev = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks, **chains).logits
        self.assertFalse(torch.equal(seq[0], ev[0]))
        self.assertTrue(torch.equal(seq[1], ev[1]))

    def test_all_masked_block_is_exact_identity_with_finite_gradients(self):
        torch.manual_seed(1)
        block = b6.ChainCrossAttention(16, 16, 8, 0.0)
        with torch.no_grad():
            block.alpha.fill_(0.9)
        x = torch.randn(2, 5, 16, requires_grad=True)
        memory = torch.randn(2, N, 16, requires_grad=True)
        key_mask = torch.zeros(2, N, dtype=torch.bool)
        out = block(x, memory, key_mask, key_mask.any(-1))
        self.assertTrue(torch.equal(out, x))
        self.assertEqual(torch.count_nonzero(block.attend(x, memory, key_mask) - block.out_proj.bias).item(), 0)
        out.square().sum().backward()
        for name, p in list(block.named_parameters()) + [("x", x), ("memory", memory)]:
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertEqual(block.alpha.grad.item(), 0.0)
        self.assertEqual(torch.count_nonzero(memory.grad).item(), 0)

    def test_attention_dropout_on_weights_and_output_dropout(self):
        torch.manual_seed(3)
        block = b6.ChainCrossAttention(16, 16, 8, dropout=0.0, attn_dropout=0.5).train()
        x, memory = torch.randn(2, 5, 16), torch.randn(2, N, 16)
        key_mask = torch.zeros(2, N, dtype=torch.bool)
        key_mask[:, 1:9] = True
        torch.manual_seed(11)
        got = block.attend(x, memory, key_mask)
        b, t, n = 2, 5, N
        q = block.q_proj(x).view(b, t, 8, 2).transpose(1, 2)
        k = block.k_proj(memory).view(b, n, 8, 2).transpose(1, 2)
        v = block.v_proj(memory).view(b, n, 8, 2).transpose(1, 2)
        w = torch.softmax((q @ k.transpose(-1, -2) * block.scale).masked_fill(~key_mask[:, None, None, :],
                                                                           torch.finfo(torch.float32).min), -1)
        torch.manual_seed(11)
        w = torch.nn.functional.dropout(w, p=0.5, training=True)          # A46 rule 6a: attention dropout
        want = block.out_proj((w @ v).transpose(1, 2).reshape(b, t, 16))
        self.assertTrue(torch.equal(got, want))
        block.eval()
        self.assertFalse(torch.equal(block.attend(x, memory, key_mask), got))
        model = self.model()
        attn = model.transgenic.decoder.layers[0].chain_attn
        self.assertEqual((attn.dropout, attn.attn_dropout), (0.0, 0.0))    # tiny_b6 sets 0.0 for both
        full = b6.ChainCrossAttention(1536, 512, 8, 0.1)
        self.assertEqual((full.dropout, full.attn_dropout), (0.1, 0.1))

    def test_gather_is_pad_invariant(self):
        torch.manual_seed(2)
        memory = b6.ChainMemory(16, 16)
        states = torch.randn(1, 8, 16)
        pad_rows = 3                                            # 18 nt of left padding = 3 downsampled rows
        padded = torch.cat([torch.randn(1, pad_rows, 16), states], dim=1)
        pos = torch.randint(0, 48, (1, N, 2))                   # tile-local chain_pos
        feat = chain_batch(1, 8, [N - 1])["chain_feat"]
        idx_plain = (0 + pos) // 6
        idx_pad = (pad_rows * 6 + pos) // 6                     # one sum, then one division (plan 7.7)
        self.assertTrue(torch.equal(memory(states, idx_plain, feat), memory(padded, idx_pad, feat)))

    def test_static_shape_checks(self):
        model = self.trained()
        bad = chain_batch(2, self.s_len, [1, 1])
        bad["chain_mask"] = bad["chain_mask"].to(torch.uint8)
        with self.assertRaisesRegex(ValueError, "chain_mask"):
            model(self.ids, decoder_input_ids=self.prefix, **bad)
        with self.assertRaisesRegex(ValueError, "go together"):
            model(self.ids, decoder_input_ids=self.prefix, chain_idx=bad["chain_idx"])
        with self.assertRaisesRegex(ValueError, "L_GSF"):
            model(self.ids, labels=self.labels, transcript_counts=torch.ones(2), **chain_batch(2, self.s_len, [1, 1]))


class TrainingPathTests(B6Base):
    def _loss_and_grads(self, model, chains):
        model.zero_grad(set_to_none=True)
        out = model(self.ids, labels=self.labels, evidence_tracks=self.tracks, **chains)
        out.loss.backward()
        return out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    def test_nonreentrant_checkpointing_passes_chain_tensors(self):
        model = self.trained().train()
        chains = chain_batch(2, self.s_len, [30, 4])
        loss0, grads0 = self._loss_and_grads(model, chains)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.assertTrue(model.transgenic.decoder.gradient_checkpointing)
        loss1, grads1 = self._loss_and_grads(model, chains)
        self.assertTrue(torch.equal(loss0, loss1))
        self.assertEqual(set(grads0), set(grads1))
        for name in grads0:
            torch.testing.assert_close(grads0[name], grads1[name], rtol=0, atol=0, msg=name)
        self.assertGreater(grads1["transgenic.chain_memory.w_d.weight"].abs().sum().item(), 0)
        self.assertNotEqual(grads1["transgenic.decoder.layers.0.chain_attn.alpha"].item(), 0.0)

    def test_chainless_batch_gives_zero_chain_gradients_but_same_graph(self):
        model = self.trained().train()
        _, grads = self._loss_and_grads(model, masked_chains(2))
        chain_names = [n for n in grads if ".chain_" in n]
        self.assertTrue(chain_names)                                    # every chain parameter is in the graph
        for name in chain_names:
            self.assertEqual(torch.count_nonzero(grads[name]).item(), 0, name)
            self.assertTrue(torch.isfinite(grads[name]).all(), name)

    def test_alpha_zero_still_receives_gradient(self):
        model = self.model().train()
        with torch.no_grad():
            model.get_encoder().evidence_proj.weight.normal_(0, 0.5)
        _, grads = self._loss_and_grads(model, chain_batch(2, self.s_len, [30, 4]))
        self.assertNotEqual(grads["transgenic.decoder.layers.1.chain_attn.alpha"].item(), 0.0)
        self.assertEqual(torch.count_nonzero(grads["transgenic.decoder.layers.1.chain_attn.q_proj.weight"]).item(), 0)

    def test_layerdrop_select_matches_led_skip_in_train_mode(self):
        model = self.trained(decoder_layerdrop=0.3)
        model.train()
        for seed in range(6):
            torch.manual_seed(seed)
            stock = model(self.ids, labels=self.labels).logits              # LED loop: Python skip
            torch.manual_seed(seed)
            select = model(self.ids, labels=self.labels, evidence_tracks=torch.zeros_like(self.tracks),
                           **masked_chains(2)).logits                          # B6 loop: select, same draws
            self.assertTrue(torch.equal(stock, select), seed)

    def test_compiled_layerdrop_does_not_recompile(self):
        import torch._dynamo as dynamo
        dynamo.reset()
        model = self.trained(decoder_layerdrop=0.5, dropout=0.1).train()
        compiled = torch.compile(model, backend="aot_eager")
        old = (dynamo.config.error_on_recompile, dynamo.config.suppress_errors)
        dynamo.config.error_on_recompile, dynamo.config.suppress_errors = True, False
        try:
            for step in range(8):
                out = compiled(self.ids, labels=self.labels, evidence_tracks=self.tracks,
                               **chain_batch(2, self.s_len, [30, step % 2]))
                out.loss.backward()
        finally:
            dynamo.config.error_on_recompile, dynamo.config.suppress_errors = old
            dynamo.reset()

    def test_no_recompile_at_fixed_shape(self):
        import torch._dynamo as dynamo
        dynamo.reset()
        model = self.trained().train()
        compiled = torch.compile(model, backend="aot_eager")
        old = (dynamo.config.error_on_recompile, dynamo.config.suppress_errors)
        dynamo.config.error_on_recompile, dynamo.config.suppress_errors = True, False
        try:
            cases = [(self.tracks, chain_batch(2, self.s_len, [30, 4])),
                     (torch.zeros_like(self.tracks), masked_chains(2)),
                     (self.tracks, chain_batch(2, self.s_len, [0, N - 1], seed=5)),
                     (torch.zeros_like(self.tracks), chain_batch(2, self.s_len, [N - 1, N - 1], seed=6))]
            for tracks, chains in cases:
                out = compiled(self.ids, labels=self.labels, evidence_tracks=tracks, **chains)
                out.loss.backward()
                self.assertTrue(torch.isfinite(out.loss))
        finally:
            dynamo.config.error_on_recompile, dynamo.config.suppress_errors = old
            dynamo.reset()


class ReEncodingReference(b6.B6TransgenicForConditionalGeneration):
    """Per-step re-encoding: every generation step reruns the encoder, downsampling and chain gather on the raw
    (beam-expanded) inputs with no cache. HF's search loop is shared, so only the caching differs."""

    def generate(self, inputs=None, **kwargs):
        return b5.transgenicForConditionalGeneration.generate(self, inputs, **kwargs)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, reencode_ids=None,
                                      evidence_tracks=None, chain_idx=None, chain_feat=None, chain_mask=None,
                                      **kwargs):
        self.steps = getattr(self, "steps", 0) + 1
        return dict(input_ids=reencode_ids, evidence_tracks=evidence_tracks, chain_idx=chain_idx,
                    chain_feat=chain_feat, chain_mask=chain_mask, decoder_input_ids=decoder_input_ids,
                    use_cache=False)

    def forward(self, input_ids=None, decoder_input_ids=None, evidence_tracks=None, chain_idx=None,
                chain_feat=None, chain_mask=None, reencode_ids=None, **kwargs):
        return super().forward(input_ids, decoder_input_ids=decoder_input_ids, evidence_tracks=evidence_tracks,
                               chain_idx=chain_idx, chain_feat=chain_feat, chain_mask=chain_mask, **kwargs)


class GenerationTests(B6Base):
    def _pair(self):
        model = self.trained()
        ref = ReEncodingReference(model.config).eval()
        ref.load_state_dict(model.state_dict())
        return model, ref

    def _check(self, num_beams):
        model, ref = self._pair()
        chains = chain_batch(2, self.s_len, [25, 0])
        calls = []
        hook = model.get_encoder().register_forward_pre_hook(lambda *a: calls.append(1))
        self.addCleanup(hook.remove)
        common = dict(max_new_tokens=6, eos_token_id=None, num_beams=num_beams, do_sample=False,
                      return_dict_in_generate=True, output_scores=True)
        cached = model.generate(self.ids, evidence_tracks=self.tracks, use_cache=True, **chains, **common)
        self.assertEqual(len(calls), 1)                       # encoder and chain memory once per call
        reference = ref.generate(self.ids, reencode_ids=self.ids, evidence_tracks=self.tracks, use_cache=False,
                                 **chains, **common)
        self.assertEqual(ref.steps, 6)
        self.assertTrue(torch.equal(cached.sequences, reference.sequences))
        for a, b in zip(cached.scores, reference.scores):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        return model, cached

    def test_greedy_with_tracks_and_chains_equals_per_step_reencoding(self):
        self._check(1)

    def test_beam_with_tracks_and_chains_equals_per_step_reencoding(self):
        self._check(2)

    def test_generation_with_masked_chains_equals_sequence_only_generation(self):
        model = self.trained()
        common = dict(max_new_tokens=6, eos_token_id=None, num_beams=2, do_sample=False, use_cache=True)
        seq = model.generate(self.ids, **common)
        ev = model.generate(self.ids, evidence_tracks=torch.zeros_like(self.tracks), **masked_chains(2), **common)
        self.assertTrue(torch.equal(seq, ev))

    def test_generate_rejects_partial_chain_inputs(self):
        model = self.trained()
        with self.assertRaisesRegex(ValueError, "go together"):
            model.generate(self.ids, chain_idx=torch.zeros(2, N, 2, dtype=torch.int64), max_new_tokens=2)


if __name__ == "__main__":
    unittest.main()
