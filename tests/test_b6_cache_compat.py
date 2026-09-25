"""Run against legacy Transformers and the production SIF's 4.57.6 cache API."""
import torch

from tests.test_b6_model import B6Base, chain_batch


class CacheCompatibilityTests(B6Base):
    def test_cached_chunks_match_uncached_logits_with_and_without_chains(self):
        # Catches absent caches, wrong per-layer cache ownership and wrong prefix length.
        model = self.trained()
        for chains in ({}, chain_batch(2, self.s_len, [5, 0])):
            with self.subTest(chains=bool(chains)), torch.no_grad():
                full = model(self.ids, decoder_input_ids=self.prefix, evidence_tracks=self.tracks,
                             use_cache=False, **chains)
                first = model(self.ids, decoder_input_ids=self.prefix[:, :2], evidence_tracks=self.tracks,
                              use_cache=True, **chains)
                self.assertEqual(len(first.past_key_values), 2)
                for layer in first.past_key_values:
                    self.assertEqual(len(layer), 4)
                    self.assertEqual(layer[0].shape[2], 2)
                    self.assertEqual(layer[2].shape[2], self.s_len)
                second = model(self.ids, decoder_input_ids=self.prefix[:, 2:], evidence_tracks=self.tracks,
                               past_key_values=first.past_key_values, use_cache=True, **chains)
                for layer in second.past_key_values:
                    self.assertEqual(layer[0].shape[2], 4)
                    self.assertEqual(layer[2].shape[2], self.s_len)
                torch.testing.assert_close(second.logits, full.logits[:, 2:], rtol=1e-5, atol=1e-5)

    def test_compiled_checkpointed_backward_matches_eager(self):
        # The GPU bench combines compile, non-reentrant checkpointing and real chains.
        import torch._dynamo as dynamo
        dynamo.reset()
        self.addCleanup(dynamo.reset)
        model = self.trained().train()
        chains = chain_batch(2, self.s_len, [5, 0])
        eager = model(self.ids, labels=self.labels, evidence_tracks=self.tracks, **chains)
        eager.loss.backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        compiled = torch.compile(model, backend="aot_eager")
        with dynamo.config.patch(error_on_recompile=True, suppress_errors=False):
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                result = compiled(self.ids, labels=self.labels, evidence_tracks=self.tracks, **chains)
                result.loss.backward()
                torch.testing.assert_close(result.loss, eager.loss, rtol=1e-5, atol=1e-6)
                actual = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
                self.assertEqual(actual.keys(), expected.keys())
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)

    def test_empty_generation_cache_keeps_entire_decoder_prefix(self):
        # New HF generation supplies an empty Cache before the first call, not None.
        model = self.trained()
        chains = chain_batch(2, self.s_len, [5, 0])
        common = dict(decoder_input_ids=self.prefix, max_new_tokens=3, eos_token_id=None,
                      do_sample=False, return_dict_in_generate=True, output_scores=True)
        with torch.no_grad():
            for beams in (1, 2):
                with self.subTest(beams=beams):
                    cached = model.generate(self.ids, evidence_tracks=self.tracks, **chains,
                                            use_cache=True, num_beams=beams, **common)
                    uncached = model.generate(self.ids, evidence_tracks=self.tracks, **chains,
                                              use_cache=False, num_beams=beams, **common)
                    self.assertTrue(torch.equal(cached.sequences, uncached.sequences))
                    for actual, expected in zip(cached.scores, uncached.scores):
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
