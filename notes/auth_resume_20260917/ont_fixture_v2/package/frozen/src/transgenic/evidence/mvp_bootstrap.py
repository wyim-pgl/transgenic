"""Paired bootstrap and the §5.3 futility decision for the B6 MVP (exploratory).

Pinned statistics contract (B6_MVP_PILOT_v1.md §5.2/§5.3):
* delta_loss = loss_off - loss_on (positive = improvement). NEVER on - off.
* delta_F1 / delta_P / delta_R = metric_on - metric_off.
* Paired TILE bootstrap for the primary: resample tiles with replacement,
  BOOTSTRAP_REPLICATES = 2,000, numpy.random.default_rng(BOOTSTRAP_SEED =
  20260914); a resampled tile carries its three seeds' pairs; one-sided 95%
  bounds are the 5th/95th percentiles of the replicate means.
* Paired LOCUS bootstrap (same replicate count and seed) for delta_F1/P/R.
* delta_loss threshold DELTA_LOSS = 0.005 nats; delta_F1 threshold DELTA_F1 =
  2.0 percentage points. These constants are frozen: no function here accepts
  an override, and mvp_score.py refuses CLI attempts to change them.

The decision function implements §5.3 verbatim: STOP is evaluated FIRST;
GO only if STOP fails; availability safeguards downgrade a prospective STOP to
INCONCLUSIVE, never to GO.

Record validation mirrors src/transgenic/training/mvp_metrics.py exactly
(PAIRED_RECORD fields) without importing torch, so scoring can run on
CPU-only nodes; a unit test asserts the field list here matches mvp_metrics.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

# --- frozen constants (spec §5.2/§5.3); no override path exists ---
DELTA_LOSS = 0.005          # nats
DELTA_F1 = 2.0              # percentage points
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260914
EXPECTED_SEEDS = (1, 2, 3)

# mirror of mvp_metrics._NUMERIC_FIELDS / _RTOL / _ATOL (unit-tested equal)
PAIRED_NUMERIC_FIELDS = (
    "loss_off", "loss_on", "composite_loss_off", "composite_loss_on",
    "ce_numerator_off", "ce_denominator_off", "ce_numerator_on", "ce_denominator_on",
)
_RTOL = 1e-6
_ATOL = 1e-8


class BootstrapError(ValueError):
    """Invalid statistics inputs must stop the decision, not degrade it."""


def _require(condition, message):
    if not condition:
        raise BootstrapError(message)


def validate_paired_record(record):
    """Fail-closed validation of one eval_pairs record (mvp_metrics schema)."""
    required = {"rn", "has_evidence", *PAIRED_NUMERIC_FIELDS}
    _require(isinstance(record, dict) and required.issubset(record),
             "paired record is missing required fields")
    _require(not isinstance(record["rn"], bool) and isinstance(record["rn"], Integral),
             "record rn must be an integer")
    _require(isinstance(record["has_evidence"], bool),
             "record has_evidence must be boolean")
    for field in PAIRED_NUMERIC_FIELDS:
        value = record[field]
        _require(not isinstance(value, bool) and isinstance(value, Real)
                 and math.isfinite(float(value)),
                 f"record {field} must be finite numeric")
    for suffix in ("off", "on"):
        denominator = float(record[f"ce_denominator_{suffix}"])
        _require(denominator > 0, "CE denominators must be positive")
        _require(math.isclose(float(record[f"loss_{suffix}"]),
                              float(record[f"ce_numerator_{suffix}"]) / denominator,
                              rel_tol=_RTOL, abs_tol=_ATOL),
                 "CE loss disagrees with its numerator and denominator")
    _require(math.isclose(float(record["ce_denominator_off"]),
                          float(record["ce_denominator_on"]),
                          rel_tol=_RTOL, abs_tol=_ATOL),
             "off/on CE denominators disagree")
    return record


def build_tiles(records_by_seed, expected_seeds=EXPECTED_SEEDS):
    """Assemble evidence-positive tiles carrying one pair per seed.

    `records_by_seed`: {seed: iterable of validated eval_pairs records}.
    Returns (rns, deltas) with `deltas` a float64 array [n_tiles, n_seeds] of
    loss_off - loss_on, seed order = sorted(expected_seeds). Fail-closed on any
    seed missing a tile, disagreeing evidence flags, or duplicate rns.
    """
    seeds = tuple(sorted(expected_seeds))
    _require(seeds == EXPECTED_SEEDS, "primary requires exactly seeds 1, 2, 3")
    _require(tuple(sorted(records_by_seed)) == seeds,
             f"expected seeds {seeds}, got {tuple(sorted(records_by_seed))}")
    per_seed = {}
    evidence_flags = {}
    for seed in seeds:
        rows = {}
        for record in records_by_seed[seed]:
            validate_paired_record(record)
            rn = int(record["rn"])
            _require(rn not in rows, f"seed {seed}: duplicate rn {rn}")
            rows[rn] = record
            flag = record["has_evidence"]
            _require(evidence_flags.setdefault(rn, flag) == flag,
                     f"rn {rn}: has_evidence disagrees across seeds")
        per_seed[seed] = rows
    inventories = {seed: set(rows) for seed, rows in per_seed.items()}
    first = inventories[seeds[0]]
    for seed in seeds[1:]:
        _require(inventories[seed] == first,
                 f"tile inventory mismatch between seeds {seeds[0]} and {seed}")
    rns = sorted(rn for rn in first if evidence_flags[rn])
    _require(rns, "evidence-positive paired subset is empty")
    deltas = np.array([[float(per_seed[seed][rn]["loss_off"])
                        - float(per_seed[seed][rn]["loss_on"])
                        for seed in seeds] for rn in rns], dtype=np.float64)
    return tuple(rns), deltas


def tile_bootstrap(deltas):
    """Paired tile bootstrap of the primary delta_loss.

    `deltas`: float64 [n_tiles, n_seeds] of loss_off - loss_on. One tile draw
    per replicate; the drawn tile carries ALL its seeds' pairs (the pooled and
    per-seed replicate statistics use the same resampled tile identities).
    Returns a dict with point estimates and one-sided 95% bounds
    (5th/95th percentiles of replicate means).
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    _require(deltas.ndim == 2 and deltas.shape[0] >= 1 and deltas.shape[1] >= 1,
             "deltas must be [n_tiles, n_seeds]")
    _require(np.isfinite(deltas).all(), "deltas must be finite")
    n_tiles = deltas.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, n_tiles, size=(BOOTSTRAP_REPLICATES, n_tiles))
    resampled = deltas[index]                      # [reps, n_tiles, n_seeds]
    pooled_reps = resampled.mean(axis=(1, 2))      # tile carries its seeds
    per_seed_reps = resampled.mean(axis=1)         # [reps, n_seeds]
    return {
        "pooled_point": float(deltas.mean()),
        "per_seed_point": [float(v) for v in deltas.mean(axis=0)],
        "pooled_lower95": float(np.percentile(pooled_reps, 5)),
        "pooled_upper95": float(np.percentile(pooled_reps, 95)),
        "per_seed_lower95": [float(v) for v in np.percentile(per_seed_reps, 5, axis=0)],
        "per_seed_upper95": [float(v) for v in np.percentile(per_seed_reps, 95, axis=0)],
        "n_tiles": int(n_tiles),
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "sign_convention": "delta_loss = loss_off - loss_on",
    }


def locus_bootstrap(per_locus, statistic):
    """Paired locus bootstrap (same replicate count and seed as the tile one).

    `per_locus`: sequence of per-locus paired records (opaque to this
    function). `statistic(records) -> float` computes the pooled paired delta
    (on - off) over a multiset of locus records. Fail-closed if any replicate
    yields a non-finite statistic (e.g. a zero pooled denominator).
    """
    per_locus = list(per_locus)
    _require(per_locus, "locus bootstrap requires at least one locus")
    point = float(statistic(per_locus))
    _require(math.isfinite(point), "statistic is not finite on the full panel")
    n = len(per_locus)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    reps = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        value = float(statistic([per_locus[j] for j in index[i]]))
        _require(math.isfinite(value), f"replicate {i}: statistic not finite")
        reps[i] = value
    return {
        "point": point,
        "lower95": float(np.percentile(reps, 5)),
        "upper95": float(np.percentile(reps, 95)),
        "n_loci": n,
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "sign_convention": "delta_metric = metric_on - metric_off",
    }


@dataclass(frozen=True)
class DecisionInputs:
    """Everything §5.3 needs, gathered fail-closed by the caller.

    primary_lower / primary_upper: one-sided 95% bounds of the pooled tile
        bootstrap (5th/95th percentiles), delta_loss = off - on.
    seed_points: the three per-seed delta_loss point estimates.
    delta_f1_lower: one-sided 95% lower bound on delta_F1 (on - off).
        The secondary readout is computed from the decodes and must exist;
        there is no "secondary unavailable" state.
    structural_available: whether junction P/R against locally aligned
        A-ONT1/A-HiFi reads exists (spec day-5 availability clause is an
        availability FACT recorded by the caller, not a wall-clock trigger).
    delta_r_lower / delta_p_point: junction recall lower bound and precision
        point estimate; required when structural_available.
        INTERPRETIVE NOTE (recorded for author review): the spec's
        contradiction clause "above 0 on delta_R with delta_P not below 0"
        is implemented as delta_R one-sided 95% lower bound > 0 AND the
        delta_P POINT ESTIMATE >= 0; the bound reading applies to delta_R
        (parallel to delta_F1), while "not below 0" on delta_P is read as the
        point estimate since no bound side is named.
    withholding_state: "assessed" or "unassessable" (§5.4; assessed only on
        the provisional-STOP path — pass "unassessable" when no run and no
        substantiated equivalence argument exists).
    withholding_primary_upper: when assessed, the sensitivity re-run's primary
        upper bound; reversal means it is >= DELTA_LOSS.
    """
    primary_lower: float
    primary_upper: float
    seed_points: tuple
    delta_f1_lower: float
    structural_available: bool
    delta_r_lower: float = None
    delta_p_point: float = None
    withholding_state: str = "unassessable"
    withholding_primary_upper: float = None


def decide(inputs):
    """§5.3 decision, STOP-first; returns (decision, trace list of strings)."""
    _require(isinstance(inputs, DecisionInputs), "DecisionInputs required")
    for name in ("primary_lower", "primary_upper", "delta_f1_lower"):
        value = getattr(inputs, name)
        _require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value)),
                 f"{name} must be finite numeric")
    _require(float(inputs.primary_lower) <= float(inputs.primary_upper),
             "primary bounds are inverted")
    seeds = tuple(inputs.seed_points)
    _require(len(seeds) == len(EXPECTED_SEEDS) and all(
        not isinstance(v, bool) and isinstance(v, Real) and math.isfinite(float(v)) for v in seeds),
        "exactly three finite seed-level primary point estimates are required")
    _require(isinstance(inputs.structural_available, bool),
             "structural_available must be an explicit boolean")
    if inputs.structural_available:
        for name in ("delta_r_lower", "delta_p_point"):
            value = getattr(inputs, name)
            _require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value)),
                     f"{name} required when the structural readout is available")
    _require(inputs.withholding_state in ("assessed", "unassessable"),
             "withholding_state must be 'assessed' or 'unassessable'")
    if inputs.withholding_state == "assessed":
        value = inputs.withholding_primary_upper
        _require(not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value)),
                 "assessed withholding requires its primary upper bound")

    trace = []
    prospective_stop = float(inputs.primary_upper) < DELTA_LOSS
    trace.append(f"primary_upper={float(inputs.primary_upper):.6g} "
                 f"{'<' if prospective_stop else '>='} delta_loss={DELTA_LOSS}: "
                 f"prospective STOP {'holds' if prospective_stop else 'fails'}")
    stop_failed = not prospective_stop
    if prospective_stop:
        contradiction = False
        if float(inputs.delta_f1_lower) > 0:
            contradiction = True
            trace.append("contradiction: delta_F1 one-sided 95% lower bound > 0")
        if (inputs.structural_available and float(inputs.delta_r_lower) > 0
                and float(inputs.delta_p_point) >= 0):
            contradiction = True
            trace.append("contradiction: delta_R lower bound > 0 with delta_P "
                         "not below 0")
        reversal = (inputs.withholding_state == "assessed"
                    and float(inputs.withholding_primary_upper) >= DELTA_LOSS)
        if reversal:
            trace.append("withholding reversal: sensitivity primary upper bound "
                         f">= delta_loss={DELTA_LOSS}; STOP fails")
        if contradiction or reversal:
            stop_failed = True
        else:
            if not inputs.structural_available:
                trace.append("availability safeguard: structural readout missing; "
                             "prospective STOP downgraded to INCONCLUSIVE (never GO)")
                return "INCONCLUSIVE", trace
            if inputs.withholding_state == "unassessable":
                trace.append("availability safeguard: §5.4 withholding unassessable; "
                             "prospective STOP downgraded to INCONCLUSIVE (never GO)")
                return "INCONCLUSIVE", trace
            trace.append("STOP: primary upper bound below delta_loss, no available "
                         "readout contradicts, withholding assessed without reversal")
            return "STOP", trace

    if stop_failed:
        go = (float(inputs.primary_lower) > 0
              and all(float(v) > 0 for v in seeds))
        if go:
            trace.append("GO: STOP failed, primary lower bound > 0 and all three "
                         "seed-level delta_loss point estimates positive")
            return "GO", trace
        trace.append("INCONCLUSIVE: STOP failed and the GO criteria do not hold")
        return "INCONCLUSIVE", trace
    raise AssertionError("unreachable")
