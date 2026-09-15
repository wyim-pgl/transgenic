"""CPU integration checks; only the external Hyena factories are substituted."""
import hashlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_model, load_model
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
from transgenic.model import modeling_HyenaTransgenic as modeling


def tiny_config(**kwargs):
    return HyenaTransgenicConfig(
        vocab_size=288, d_model=8, decoder_d_model=16,
        encoder_layers=1, decoder_layers=1, encoder_n_layer=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_encoder_seqlen=24, max_encoder_position_embeddings=24,
        max_decoder_position_embeddings=32, attention_window=[4],
        dropout=0, decoder_layerdrop=0, load_pretrained_encoder=False,
        **kwargs,
    )


class TinyHyena(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(16, config.d_model)

    def forward(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids), hidden_states=None)


class MVPModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.config_patch = patch.object(modeling.AutoConfig, 'from_pretrained', side_effect=lambda *a, **k: SimpleNamespace())
        self.model_patch = patch.object(modeling.AutoModel, 'from_config', side_effect=lambda config, **k: TinyHyena(config))
        self.config_patch.start()
        self.model_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.model_patch.stop)
        torch.manual_seed(123)
        self.ids = torch.tensor([[3, 4, 5, 6] * 6])
        self.prefix = torch.tensor([[2, 7, 9]])
        self.tracks = torch.arange(192, dtype=torch.float32).reshape(1, 24, 8) / 192

    def model(self, **kwargs):
        return modeling.transgenicForConditionalGeneration(tiny_config(**kwargs)).eval()

    def test_config_validation_roundtrip(self):
        self.assertEqual(HyenaTransgenicConfig().evidence_channels, 0)
        for k in (0, 8):
            with tempfile.TemporaryDirectory() as directory:
                tiny_config(evidence_channels=k).save_pretrained(directory)
                self.assertEqual(HyenaTransgenicConfig.from_pretrained(directory).evidence_channels, k)
        for bad in (-1, True, False, 8.0, '8', None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'evidence_channels'):
                tiny_config(evidence_channels=bad)

    def test_disabled_keys_rng_match_preincrement_baseline(self):
        # Captured from abca40f, torch 2.6.0 / transformers 4.46.3.
        for kwargs in ({}, {'evidence_channels': 0}):
            torch.manual_seed(123)
            model = self.model(**kwargs)
            self.assertFalse(any('evidence' in k for k in model.state_dict()))
            self.assertEqual(hashlib.sha256('\n'.join(model.state_dict()).encode()).hexdigest(),
                             '8ae6c2c2f8ba27620a12be17118d501ce36cebbd304f3f3a6007035a40ca58cb')
            self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                             '5be8a048e9a4c3a4ed4402bf9b1260ca6669810f46c76fda2f587fa167d9dd88')

    def test_projection_zero_after_nested_initialization(self):
        model = self.model(evidence_channels=8)
        projection = model.get_encoder().evidence_proj
        self.assertEqual(projection.weight.shape, (8, 8))
        self.assertIsNone(projection.bias)
        self.assertEqual(torch.count_nonzero(projection.weight).item(), 0)
        model.apply(model._init_weights)
        self.assertEqual(torch.count_nonzero(projection.weight).item(), 0)

    def test_step_zero_bitwise_equal_after_shared_weights_loaded(self):
        baseline = self.model()
        model = self.model(evidence_channels=8)
        result = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(result.missing_keys, ['transgenic.encoder.evidence_proj.weight'])
        self.assertEqual(result.unexpected_keys, [])
        with torch.no_grad():
            expected = baseline(self.ids, decoder_input_ids=self.prefix).logits
            for tracks in (None, torch.zeros_like(self.tracks), self.tracks):
                actual = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=tracks).logits
                self.assertTrue(torch.equal(expected, actual))

    def test_nonzero_fusion_before_position_and_downsampling_and_gradient(self):
        model = self.model(evidence_channels=8)
        encoder = model.get_encoder()
        with torch.no_grad():
            encoder.evidence_proj.weight.copy_(torch.eye(8))
        raw = encoder.hyena(self.ids).last_hidden_state
        expected = raw + self.tracks
        seen = []
        handle = model.transgenic.EncoderOutputPositionalEmbedding.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        self.addCleanup(handle.remove)
        out = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks.double())
        self.assertTrue(torch.equal(out.encoder_last_hidden_state, expected))
        self.assertTrue(torch.equal(seen[0], expected))
        off = model(self.ids, decoder_input_ids=self.prefix)
        self.assertFalse(torch.equal(out.logits, off.logits))
        out.logits.square().mean().backward()
        grad = encoder.evidence_proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        # A learned projection still has no residual on zero evidence.
        with torch.no_grad():
            zero = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=torch.zeros_like(self.tracks))
        self.assertTrue(torch.equal(off.logits, zero.logits))

    def test_zero_projection_receives_training_loss_gradient(self):
        model = self.model(evidence_channels=8)
        out = model(self.ids, labels=self.prefix, evidence_tracks=self.tracks)
        out.loss.backward()
        grad = model.get_encoder().evidence_proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_invalid_tracks_fail_explicitly(self):
        model = self.model(evidence_channels=8)
        for bad in (torch.zeros(24, 8), torch.zeros(2, 24, 8), torch.zeros(1, 23, 8),
                    torch.zeros(1, 24, 7), self.tracks.long(), self.tracks.to(torch.complex64),
                    self.tracks.to('meta'), [1]):
            with self.subTest(shape=getattr(bad, 'shape', None)), self.assertRaisesRegex((ValueError, TypeError), 'evidence_tracks'):
                model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=bad)
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.model()(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks)

    def test_precomputed_states_not_fused_twice(self):
        model = self.model(evidence_channels=8)
        with torch.no_grad():
            model.get_encoder().evidence_proj.weight.copy_(torch.eye(8))
            expected = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks)
            calls = []
            handle = model.get_encoder().register_forward_pre_hook(lambda *args: calls.append(1))
            self.addCleanup(handle.remove)
            actual = model(encoder_outputs=expected.encoder_last_hidden_state,
                           decoder_input_ids=self.prefix, evidence_tracks=self.tracks)
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(expected.logits, actual.logits))

    def test_generation_tracks_once_cache_and_teacher_forcing_agree(self):
        model = self.model(evidence_channels=8)
        with torch.no_grad():
            model.get_encoder().evidence_proj.weight.copy_(torch.eye(8))
        received = []
        handle = model.get_encoder().register_forward_pre_hook(
            lambda module, args, kwargs: received.append(kwargs.get('evidence_tracks')), with_kwargs=True)
        self.addCleanup(handle.remove)
        with torch.no_grad():
            cached = model.generate(self.ids, evidence_tracks=self.tracks, max_new_tokens=4,
                                    eos_token_id=None, use_cache=True, return_dict_in_generate=True,
                                    output_scores=True)
            self.assertEqual(len(received), 1)
            self.assertIs(received[0], self.tracks)
            received.clear()
            uncached = model.generate(self.ids, evidence_tracks=self.tracks, max_new_tokens=4,
                                      eos_token_id=None, use_cache=False)
            self.assertEqual(len(received), 1)
            self.assertIs(received[0], self.tracks)
            self.assertTrue(torch.equal(cached.sequences, uncached))
            for step, score in enumerate(cached.scores):
                forced = model(self.ids, decoder_input_ids=cached.sequences[:, :step + 1],
                               evidence_tracks=self.tracks, use_cache=False).logits[:, -1]
                # CPU float32 incremental/full-prefix LED arithmetic may differ slightly.
                torch.testing.assert_close(score, forced, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.equal(forced.argmax(-1), cached.sequences[:, step + 1]))

    def test_checkpoint_roundtrips_preserve_aliases_and_learned_projection(self):
        for channels in (0, 8):
            with self.subTest(channels=channels), tempfile.TemporaryDirectory() as directory:
                model = self.model(evidence_channels=channels)
                if channels:
                    with torch.no_grad():
                        model.get_encoder().evidence_proj.weight.copy_(torch.eye(8))
                tracks = self.tracks if channels else None
                with torch.no_grad():
                    expected = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=tracks).logits
                save_model(model, directory + '/aliases.safetensors')
                restored = self.model(evidence_channels=channels)
                self.assertEqual(load_model(restored, directory + '/aliases.safetensors'), (set(), []))
                model.save_pretrained(directory, safe_serialization=True)
                hf_restored = modeling.transgenicForConditionalGeneration.from_pretrained(directory).eval()
                for loaded in (restored, hf_restored):
                    self.assertEqual(loaded.config.evidence_channels, channels)
                    self.assertIs(loaded.transgenic.decoder_embed_tokens.weight,
                                  loaded.transgenic.decoder.embed_tokens.weight)
                    self.assertIs(loaded.lm_head.weight, loaded.transgenic.decoder_embed_tokens.weight)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
                    with torch.no_grad():
                        actual = loaded(self.ids, decoder_input_ids=self.prefix, evidence_tracks=tracks).logits
                    self.assertTrue(torch.equal(expected, actual))

    def test_baseline_actual_wrappers(self):
        model = self.model()
        with torch.no_grad():
            out = model(self.ids, decoder_input_ids=self.prefix)
            sequence = model.generate(self.ids, max_new_tokens=3, eos_token_id=None)
        self.assertEqual(out.logits.shape, (1, 3, 288))
        self.assertEqual(out.encoder_last_hidden_state.shape, (1, 24, 8))
        self.assertEqual(sequence.shape, (1, 4))


if __name__ == '__main__':
    unittest.main()
