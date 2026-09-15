import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from transgenic.training.mvp_metrics import aggregate_paired_records, paired_metric_record, weighted_ce


class FakeModel(torch.nn.Module):
    def __init__(self, fail_on_call=None):
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append((input_ids, kwargs, torch.is_grad_enabled()))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("forward failed")
        labels = kwargs["labels"]
        tracks = kwargs["evidence_tracks"]
        logits = torch.zeros((*labels.shape, 273), requires_grad=True)
        logits[..., 0] = 0 if not tracks.any() else 1
        return SimpleNamespace(logits=logits, loss=torch.tensor(40.0 + len(self.calls)))


def batch(rn=7, has_evidence=True):
    dna = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(dna, dtype=torch.bool)
    labels = torch.tensor([[0, 2, -100]])
    tracks = torch.ones((1, 3, 8), dtype=torch.float16)
    return (dna, mask, labels, ("gm",), ("Chr1",), (0,), (3,), torch.tensor([1.0]),
            tracks, {"rn": torch.tensor([rn]), "has_evidence": torch.tensor([has_evidence])})


def record(rn, evidence=True, off=2.0, on=1.0, denominator=1.0, composite_off=4.0,
           composite_on=3.0):
    return {
        "rn": rn, "has_evidence": evidence,
        "loss_off": off, "loss_on": on,
        "composite_loss_off": composite_off, "composite_loss_on": composite_on,
        "ce_numerator_off": off * denominator,
        "ce_denominator_off": denominator,
        "ce_numerator_on": on * denominator,
        "ce_denominator_on": denominator,
    }


class WeightedCETests(unittest.TestCase):
    def test_hand_calculated_unequal_weights_and_ignores(self):
        logits = torch.zeros((1, 4, 273), dtype=torch.float64)
        labels = torch.tensor([[2, 17, 0, -100]])
        loss, numerator, denominator = weighted_ce(logits, labels)
        expected = math.log(273)
        torch.testing.assert_close(loss, torch.tensor([expected]), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(numerator, torch.tensor([10 * expected]), rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(denominator, torch.tensor([10.0]))
        self.assertEqual(loss.dtype, torch.float32)

    def test_weight_boundaries(self):
        labels = torch.tensor([[2, 17, 21, 22, 271, 272, 0, -100]])
        logits = torch.zeros((1, labels.shape[1], 273))
        _, _, denominator = weighted_ce(logits, labels)
        self.assertEqual(denominator.item(), 20.0)

    def test_logits_are_already_label_aligned_without_second_shift(self):
        labels = torch.tensor([[2, 17, 21]])
        logits = torch.full((1, 3, 273), -20.0)
        logits[0, torch.arange(3), labels[0]] = 20.0
        loss, _, _ = weighted_ce(logits, labels)
        self.assertLess(loss.item(), 1e-6)

    def test_matches_inherited_weighted_cross_entropy(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 6, 288)
        labels = torch.tensor([[2, 17, 21, 22, 272, -100]])
        weights = torch.ones(288)
        weights[2], weights[17], weights[21], weights[272], weights[22:272] = 5, 4, 3, 3, 2
        expected = F.cross_entropy(logits.view(-1, 288), labels.view(-1), weight=weights,
                                   ignore_index=-100)
        actual, numerator, denominator = weighted_ce(logits, labels)
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[0], numerator[0] / denominator[0])

    def test_invalid_inputs_fail_closed(self):
        valid = torch.zeros((1, 2, 273))
        cases = [
            (valid, torch.tensor([[-100, -100]])),
            (valid.clone().fill_(float("nan")), torch.tensor([[0, 1]])),
            (valid, torch.tensor([[0, 273]])),
            (torch.zeros((1, 2, 272)), torch.tensor([[0, 1]])),
            (torch.zeros((2, 273)), torch.tensor([[0, 1]])),
            (valid, torch.tensor([[0]])),
            (valid, torch.tensor([[0, 1]], dtype=torch.int32)),
            (torch.zeros((0, 2, 273)), torch.empty((0, 2), dtype=torch.long)),
        ]
        for logits, labels in cases:
            with self.subTest(shape=tuple(logits.shape), labels=labels.tolist()), self.assertRaises(ValueError):
                weighted_ce(logits, labels)


class PairedRecordTests(unittest.TestCase):
    def test_forwards_same_inputs_explicit_off_on_without_transcript_counts(self):
        model = FakeModel().train()
        b = batch()
        result = paired_metric_record(model, b)
        self.assertTrue(model.training)
        self.assertEqual(result["rn"], 7)
        self.assertTrue(result["has_evidence"])
        self.assertEqual(result["composite_loss_off"], 41.0)
        self.assertEqual(result["composite_loss_on"], 42.0)
        self.assertEqual(len(model.calls), 2)
        for call in model.calls:
            self.assertIs(call[0], b[0])
            self.assertIs(call[1]["attention_mask"], b[1])
            self.assertIs(call[1]["labels"], b[2])
            self.assertNotIn("transcript_counts", call[1])
            self.assertFalse(call[2])
        self.assertFalse(model.calls[0][1]["evidence_tracks"].any())
        self.assertIs(model.calls[1][1]["evidence_tracks"], b[8])
        self.assertNotAlmostEqual(result["loss_off"], result["composite_loss_off"])

    def test_restores_previous_mode_when_forward_raises(self):
        for previous in (False, True):
            model = FakeModel(fail_on_call=2)
            model.train(previous)
            with self.subTest(previous=previous), self.assertRaisesRegex(RuntimeError, "forward failed"):
                paired_metric_record(model, batch())
            self.assertEqual(model.training, previous)

    def test_rejects_non_batch_one_and_malformed_batches(self):
        b = list(batch())
        b[0] = b[0].repeat(2, 1)
        missing_with_extra = list(batch())
        missing_with_extra[9] = {"rn": torch.tensor([7]), "extra": torch.tensor([1])}
        noninteger_rn = list(batch())
        noninteger_rn[9] = {"rn": torch.tensor([7.5]), "has_evidence": torch.tensor([True])}
        nonboolean_flag = list(batch())
        nonboolean_flag[9] = {"rn": torch.tensor([7]), "has_evidence": torch.tensor([2])}
        for bad in (tuple(b), batch()[:9], tuple(missing_with_extra), tuple(noninteger_rn),
                    tuple(nonboolean_flag)):
            with self.subTest(metadata=bad[-1] if len(bad) == 10 else None), self.assertRaises(ValueError):
                paired_metric_record(FakeModel(), bad)


class AggregateTests(unittest.TestCase):
    def test_equal_tile_mean_not_pooled_and_delta_sign(self):
        a = record(1, off=2, on=1, denominator=1, composite_off=5, composite_on=4)
        b = record(2, off=4, on=1, denominator=100, composite_off=9, composite_on=3)
        result = aggregate_paired_records([a, b], [1, 2])
        self.assertEqual(result["primary_delta"], 2.0)
        self.assertEqual(result["composite_delta"], 3.5)
        self.assertNotAlmostEqual(result["primary_delta"], (1 + 300) / 101)
        self.assertEqual(result["records"], [a, b])

    def test_non_evidence_tiles_are_retained_but_excluded_from_means(self):
        positive = record(1, off=3, on=1)
        negative = record(2, evidence=False, off=100, on=1)
        result = aggregate_paired_records([positive, negative], [1, 2])
        self.assertEqual(result["primary_delta"], 2.0)
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["records"], [positive, negative])

    def test_padding_duplicates_are_deduplicated_with_tight_float_tolerance(self):
        a = record(1)
        duplicate = dict(a)
        duplicate["loss_off"] += 1e-9
        duplicate["ce_numerator_off"] += 1e-9
        result = aggregate_paired_records([a, duplicate], [1])
        self.assertEqual(result["pair_count"], 1)

    def test_missing_extra_and_duplicate_inventory_fail(self):
        for records, inventory in [([record(1)], [1, 2]), ([record(1), record(2)], [1]),
                                   ([record(1)], [1, 1])]:
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                aggregate_paired_records(records, inventory)

    def test_inconsistent_duplicates_flags_denominators_and_values_fail(self):
        a = record(1)
        for field, value in [("has_evidence", False), ("ce_denominator_on", 2.0),
                             ("loss_on", 1.1)]:
            other = dict(a)
            other[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate_paired_records([a, other], [1])

    def test_nonfinite_internally_inconsistent_and_empty_evidence_fail(self):
        bad_nonfinite = record(1)
        bad_nonfinite["loss_on"] = float("nan")
        bad_ratio = record(1)
        bad_ratio["ce_numerator_on"] = 99
        bad_denominator = record(1)
        bad_denominator["ce_denominator_on"] = 0
        missing_with_extra = record(1)
        del missing_with_extra["loss_on"]
        missing_with_extra["extra"] = 1.0
        for records in ([bad_nonfinite], [bad_ratio], [bad_denominator],
                        [record(1, evidence=False)], [missing_with_extra]):
            with self.assertRaises(ValueError):
                aggregate_paired_records(records, [1])


if __name__ == "__main__":
    unittest.main()
