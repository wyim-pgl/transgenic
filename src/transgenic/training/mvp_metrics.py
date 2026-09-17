"""CPU-only paired loss metrics for the B6 MVP evaluation."""

import math
from numbers import Integral, Real

import torch


_NUMERIC_FIELDS = (
    "loss_off", "loss_on", "composite_loss_off", "composite_loss_on",
    "ce_numerator_off", "ce_denominator_off", "ce_numerator_on", "ce_denominator_on",
)
_RTOL = 1e-6
_ATOL = 1e-8


def weighted_ce(logits, labels):
    """Return float32 per-tile weighted CE, numerator, and target-weight denominator."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be a floating [batch, length, vocabulary] tensor")
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2 or
            labels.shape != logits.shape[:2] or labels.shape[0] == 0):
        raise ValueError("labels must match nonempty logits batch and length dimensions")
    if labels.dtype != torch.long:
        raise ValueError("labels must contain int64 token IDs")
    if logits.shape[-1] <= 272:
        raise ValueError("vocabulary must include prescribed token ID 272")
    values = logits.float()
    if not torch.isfinite(values).all():
        raise ValueError("logits must be finite")

    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError("each tile must contain at least one non-ignored target")
    targets = labels.masked_select(valid)
    if (targets < 0).any() or (targets >= logits.shape[-1]).any():
        raise ValueError("target token ID is outside the vocabulary")

    safe_labels = labels.masked_fill(~valid, 0)
    token_loss = -values.log_softmax(dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    weights = torch.ones_like(token_loss)
    weights = torch.where(labels.eq(2), 5.0, weights)
    weights = torch.where(labels.eq(17), 4.0, weights)
    weights = torch.where(labels.eq(21) | labels.eq(272), 3.0, weights)
    weights = torch.where((labels >= 22) & (labels <= 271), 2.0, weights)
    weights = weights * valid
    numerator = (token_loss * weights).sum(dim=1)
    denominator = weights.sum(dim=1)
    loss = numerator / denominator
    if (denominator <= 0).any() or not torch.isfinite(loss).all():
        raise ValueError("weighted CE produced an invalid result")
    return loss, numerator, denominator


def _scalar(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    result = float(value.detach().item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_paired_batch(batch):
    """Validate local paired inputs without a model forward or collectives."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 10:
        raise ValueError("paired evaluation requires a collated 10-tuple batch")
    input_ids, attention_mask, labels, _, _, _, _, _, tracks, metadata = batch
    tensors = (input_ids, attention_mask, labels, tracks)
    if any(not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1
           for value in tensors):
        raise ValueError("paired evaluation requires batch size one tensors")
    if not isinstance(metadata, dict) or not {"rn", "has_evidence"}.issubset(metadata):
        raise ValueError("batch metadata requires rn and has_evidence")
    rn_tensor, evidence_tensor = metadata["rn"], metadata["has_evidence"]
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if (not isinstance(rn_tensor, torch.Tensor) or rn_tensor.numel() != 1 or
            rn_tensor.dtype not in integer_dtypes or
            not isinstance(evidence_tensor, torch.Tensor) or evidence_tensor.numel() != 1 or
            evidence_tensor.dtype != torch.bool):
        raise ValueError("batch metadata must contain one integer rn and boolean evidence flag")
    rn = int(rn_tensor.item())
    has_evidence = bool(evidence_tensor.item())

    return rn, has_evidence


def paired_metric_record(model, batch):
    """Evaluate one collated 10-tuple tile with zero/off and caller-supplied/on tracks."""
    rn, has_evidence = validate_paired_batch(batch)
    input_ids, attention_mask, labels, _, _, _, _, _, tracks, _ = batch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            common = {"attention_mask": attention_mask, "labels": labels}
            off = model(input_ids, evidence_tracks=torch.zeros_like(tracks), **common)
            on = model(input_ids, evidence_tracks=tracks, **common)
            off_loss, off_num, off_den = weighted_ce(off.logits, labels)
            on_loss, on_num, on_den = weighted_ce(on.logits, labels)
            return {
                "rn": rn,
                "has_evidence": has_evidence,
                "loss_off": _scalar(off_loss, "loss_off"),
                "loss_on": _scalar(on_loss, "loss_on"),
                "composite_loss_off": _scalar(off.loss, "composite_loss_off"),
                "composite_loss_on": _scalar(on.loss, "composite_loss_on"),
                "ce_numerator_off": _scalar(off_num, "ce_numerator_off"),
                "ce_denominator_off": _scalar(off_den, "ce_denominator_off"),
                "ce_numerator_on": _scalar(on_num, "ce_numerator_on"),
                "ce_denominator_on": _scalar(on_den, "ce_denominator_on"),
            }
    finally:
        model.train(was_training)


def _validated_record(record):
    required = {"rn", "has_evidence", *_NUMERIC_FIELDS}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("paired record is missing required fields")
    if isinstance(record["rn"], bool) or not isinstance(record["rn"], Integral):
        raise ValueError("record rn must be an integer")
    if not isinstance(record["has_evidence"], bool):
        raise ValueError("record has_evidence must be boolean")
    for field in _NUMERIC_FIELDS:
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"record {field} must be finite numeric")
    for suffix in ("off", "on"):
        denominator = float(record[f"ce_denominator_{suffix}"])
        if denominator <= 0:
            raise ValueError("CE denominators must be positive")
        if not math.isclose(float(record[f"loss_{suffix}"]),
                            float(record[f"ce_numerator_{suffix}"]) / denominator,
                            rel_tol=_RTOL, abs_tol=_ATOL):
            raise ValueError("CE loss disagrees with its numerator and denominator")
    if not math.isclose(float(record["ce_denominator_off"]), float(record["ce_denominator_on"]),
                        rel_tol=_RTOL, abs_tol=_ATOL):
        raise ValueError("off/on CE denominators disagree")
    return record


def aggregate_paired_records(records, expected_rns):
    """Deduplicate gathered records and compute equal-tile means.

    Duplicate floating fields must agree within rtol=1e-6 and atol=1e-8. The
    returned records retain one audited row per expected rn for later bootstrap.
    """
    expected = list(expected_rns)
    if (any(isinstance(rn, bool) or not isinstance(rn, Integral) for rn in expected) or
            len(expected) != len(set(expected))):
        raise ValueError("expected rn inventory must contain unique integers")

    unique = {}
    for candidate in records:
        current = _validated_record(candidate)
        rn = current["rn"]
        if rn not in unique:
            unique[rn] = current
            continue
        previous = unique[rn]
        if previous["has_evidence"] != current["has_evidence"] or any(
            not math.isclose(float(previous[field]), float(current[field]),
                             rel_tol=_RTOL, abs_tol=_ATOL)
            for field in _NUMERIC_FIELDS
        ):
            raise ValueError(f"duplicate records disagree for rn={rn}")

    missing = set(expected) - set(unique)
    extra = set(unique) - set(expected)
    if missing or extra:
        raise ValueError(f"rn inventory mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    ordered = [unique[rn] for rn in expected]
    evidence = [item for item in ordered if item["has_evidence"]]
    if not evidence:
        raise ValueError("evidence-positive paired subset is empty")
    primary = math.fsum(item["loss_off"] - item["loss_on"] for item in evidence) / len(evidence)
    composite = math.fsum(item["composite_loss_off"] - item["composite_loss_on"]
                          for item in evidence) / len(evidence)
    return {
        "primary_delta": primary,
        "composite_delta": composite,
        "pair_count": len(evidence),
        "records": ordered,
    }
