"""B6 trainer runtime (plan B6_WP2_WP4_PLAN_v4 section 9; protocol A46 rules 8, 9, 11, 15).

Gate: plan section 10 orders CP2 (129 kb memory bench, <= 100 GB peak reserved per rank) before any B6 run.
The bench passed on 2 ranks (job 3208285, A51) and CP4 passed (jobs 3210279, 3210585); author decision 15
(2026-09-26) set configs/b6_400m_win_v3.json to GATED. A GATED config must carry that evidence
(``check_gate_evidence``); ``--allow-ungated`` stays an integration-test switch only.

Contents
  * batch contract (plan 7.7): ``validate_b6_batch``
  * arm presets (rule 11) and the tensor-level arm transforms: ``ARMS``, ``apply_arm``, ``de_novo_view``
  * optimizer groups with every chain gate alpha outside weight decay: ``build_b6_optimizer``
  * run identity: ``build_run_identity``, ``check_run_config_b6`` (required key set, missing on either side
    refuses), ``write_runtime_identity`` (rule 9 fields plus per-layer alpha per link)
  * compile policy: ``compile_b6`` (suppress_errors off), ``CompileGuard`` (recompile after warm-up raises),
    ``check_inductor_kernels`` (zero Inductor kernels after warm-up raises)
  * ``memory_abort`` (> 110 GB reserved on any rank writes MEMORY_ABORT), budget and step-checkpoint rules
  * WP5 pilot: ``pilot_stop_target`` (hard stop at exactly N_pilot optimizer updates, part of the run identity)
    and ``resolve_wp5_pilot`` (the gated pilot config block: B6/C2, seed 123, aug_seed 20261037, 4 ranks)
Torch is imported lazily so the torch-free parts run in the duckdb-only test environment, as b5_runtime does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from transgenic.training.b5_runtime import (CheckpointLayout, EarlyStopper, accumulation_steps,  # noqa: F401
                                            epoch_batches, load_b5_config, model_kwargs, resolve_checkpointing)

GATE_STATUS = "UNGATED: memory bench not yet run"
GATED_STATUS = "GATED: memory bench passed"

# ---- frozen numbers (rules 8, 9, 11; plan 9) ------------------------------------------------------------------
SEED = 123
AUG_SEED_BASE = 20260914
AUG_SEED = AUG_SEED_BASE + SEED                       # 20261037
# N_pilot = floor(train_rows * PILOT_EPOCHS / EFFECTIVE_BATCH) (A46 rule 11; decision 18 numeric amendment). 4,142 is
# the A46 freeze-checklist value on 198,828 train rows; the binding value is the corpus freeze's ``n_pilot`` record
# (b6_corpus_repair.n_pilot_record), which ``bind_n_pilot`` checks against this constant, the config and the loader.
# Once the decision-18 corpus is applied and its freeze says N != 4142, update together (one A46 rule 11 amendment):
# N_PILOT here, configs/b6_400m_win_v3.json n_pilot, configs/b6_400m_win_v3_wp5_pilot.json n_pilot and
# wp5_pilot.stop_after_updates (and its notes), revision/scripts/b6/wp5_pilot.sbatch N_PILOT=, and the literal 4142
# in tests/test_b6_pilot.py.
N_PILOT = 4142
EFFECTIVE_BATCH = 96
PILOT_EPOCHS = 2
N_PILOT_FORMULA = "floor(train_rows * epochs / effective_batch)"
MAX_EPOCHS = 22
PATIENCE = 3
CHANNEL_DROPOUT_P = 0.5
THINNING_SET = (0.01, 0.1, 0.5, 1.0)
WITHHOLD_SOURCE_RATE = 0.1
WITHHOLD_LIBRARY_RATE = 0.1
MEMORY_ABORT_BYTES = 110 * 10 ** 9                    # rule 9: any rank above 110 GB reserved aborts the link
MEMORY_GATE_BYTES = 100 * 10 ** 9                     # plan 10 CP2: bench gate, peak reserved per rank
MEMORY_ABORT_MARKER = "MEMORY_ABORT"
PRUNE_EXEMPT_MARKER = "PRUNE_EXEMPT"
DATA_ORDER_PREFIX = 1000                              # rule 11: hash of the first 1,000 rn per epoch

# ---- batch contract (plan 7.1, 7.7): Writer B's frozen spec, byte-identical copy ------------------------------
from transgenic.evidence import b6_track_spec as spec  # noqa: E402  (stdlib + numpy only)

K_TRACKS = spec.K
CHAIN_ELEMENTS = spec.N_TOK
MASK_CHANNELS = spec.MASK_CHANNELS
PHASE_CHANNELS = spec.PHASE_CHANNELS
COUNT_CHANNELS = spec.COUNT_CHANNELS
BATCH_KEYS = spec.BATCH_KEYS
META_KEYS = spec.META_KEYS
FEATURE_LIMITS = {"order_fwd": (0, spec.ORDER_CAP), "order_rev": (0, spec.ORDER_CAP),
                  "slot": (spec.CHAIN_SLOTS[0], spec.CHAIN_SLOTS[-1]), "source": (0, len(spec.CHAIN_SOURCES) - 1),
                  "strand": (0, len(spec.CHAIN_STRANDS) - 1), "units_bin": (0, spec.UNITS_CAP)}
CONTEXT_KINDS = ("corpus", "eval")
MODEL_KEYS = ("input_ids", "attention_mask", "evidence_tracks", "labels", "chain_idx", "chain_feat", "chain_mask")
SPEC_SHAS = dict(spec.SPEC_SHAS)
# The frozen numbers above must be the plan's (7.1-7.3, rule 8); a drifted spec copy fails at import.
assert (K_TRACKS, CHAIN_ELEMENTS, MASK_CHANNELS, PHASE_CHANNELS) == (31, 512, (7, 15, 23), (25, 26, 27, 28, 29, 30))
assert tuple(spec.CHAIN_FEAT_FIELDS) == ("order_fwd", "order_rev", "slot", "source", "strand", "units_bin")
assert (spec.FRACTIONS, spec.DROPOUT_P, spec.AUG_SEED_BASE, spec.TRAINING_SEED) == (THINNING_SET, CHANNEL_DROPOUT_P,
                                                                                  AUG_SEED_BASE, SEED)
assert (spec.WITHHOLD_SOURCE_RATE, spec.WITHHOLD_LIBRARY_RATE) == (WITHHOLD_SOURCE_RATE, WITHHOLD_LIBRARY_RATE)
assert tuple(spec.ARMS) == ("B6", "C2", "C3", "C4", "C6", "C7")


def validate_b6_batch(batch: Mapping, *, with_labels: bool = True) -> None:
    """``b6_track_spec.validate_batch`` (the WP3 contract) plus the value checks the model relies on:
    finite non-negative tracks, zero tracks on left padding, feature ranges of real chain elements and a known
    context kind. Eager only (value checks synchronise)."""
    import torch

    spec.validate_batch(batch, training=with_labels)
    ids, am, tr = batch["input_ids"], batch["attention_mask"], batch["evidence_tracks"]
    if not bool(torch.isfinite(tr).all()) or bool((tr < 0).any()):
        raise ValueError("evidence_tracks must be finite and non-negative")
    if bool((tr[~am.bool()] != 0).any()):
        raise ValueError("evidence_tracks must be zero on left padding")
    feat, mask = batch["chain_feat"], batch["chain_mask"]
    real = feat[mask].to(torch.int64)
    for col, (name, (lo, hi)) in enumerate(FEATURE_LIMITS.items()):
        if real.numel() and (bool((real[:, col] < lo).any()) or bool((real[:, col] > hi).any())):
            raise ValueError(f"chain_feat {name} outside [{lo}, {hi}] on a real element")
    kinds = batch["meta"]["context_kind"]
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    if not kinds or any(k not in CONTEXT_KINDS for k in kinds):
        raise ValueError(f"meta context_kind must be in {CONTEXT_KINDS}")


# ---- arms (rule 11; plan 9 arm matrix) ------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmPreset:
    name: str
    tracks: str              # draw | zero | donor | binary | phase_zero
    chains: str              # draw | masked | donor
    budget: str              # full (stop rule) | n_pilot (fixed 4,142 updates, final step)
    dataset_arm: str         # what the WP3 dataset is asked to build (C3 needs donor evidence)
    evidence_channel: str    # logged: on | off


ARMS: Dict[str, ArmPreset] = {
    "B6": ArmPreset("B6", "draw", "draw", "full", "B6", "on"),
    "C2": ArmPreset("C2", "zero", "masked", "full", "C2", "off"),
    "C3": ArmPreset("C3", "donor", "donor", "n_pilot", "C3", "on"),
    "C4": ArmPreset("C4", "binary", "draw", "n_pilot", "C4", "on"),
    "C6": ArmPreset("C6", "phase_zero", "draw", "n_pilot", "C6", "on"),
    "C7": ArmPreset("C7", "draw", "masked", "n_pilot", "C7", "on"),
}


def _mask_chains(batch: Dict) -> None:
    import torch
    batch["chain_mask"] = torch.zeros_like(batch["chain_mask"])
    batch["chain_idx"] = torch.zeros_like(batch["chain_idx"])
    batch["chain_feat"] = torch.zeros_like(batch["chain_feat"])


def apply_arm(batch: Mapping, arm: ArmPreset) -> Dict:
    """Tensor-level arm transform, applied after the dataset and before the model. It uses no RNG, so every
    arm keeps B6's uniform log. Idempotent: C4/C6 transforms applied a second time change nothing, so a
    dataset that already applied them is safe. C3 donor evidence is built by the dataset (``dataset_arm``)."""
    import torch

    out = dict(batch)
    tracks = batch["evidence_tracks"]
    if arm.tracks == "zero":
        out["evidence_tracks"] = torch.zeros_like(tracks)
    elif arm.tracks == "binary":
        binary = tracks.clone()
        counts = list(COUNT_CHANNELS)
        binary[..., counts] = (tracks[..., counts] > 0).to(tracks.dtype)
        out["evidence_tracks"] = binary
    elif arm.tracks == "phase_zero":
        zeroed = tracks.clone()
        zeroed[..., list(PHASE_CHANNELS)] = 0
        out["evidence_tracks"] = zeroed
    elif arm.tracks not in ("draw", "donor"):
        raise ValueError(f"unknown track mode {arm.tracks!r}")
    if arm.chains == "masked":
        _mask_chains(out)
    elif arm.chains not in ("draw", "donor"):
        raise ValueError(f"unknown chain mode {arm.chains!r}")
    return out


def de_novo_view(batch: Mapping) -> Dict:
    """De novo mode: all 31 channels zero and every chain masked (the stop-rule validation input)."""
    import torch

    out = dict(batch)
    out["evidence_tracks"] = torch.zeros_like(batch["evidence_tracks"])
    _mask_chains(out)
    return out


def model_inputs(batch: Mapping) -> Dict:
    return {k: batch[k] for k in MODEL_KEYS if batch.get(k) is not None}


def budget_steps(arm: ArmPreset, n_pilot: int = N_PILOT) -> Optional[int]:
    return n_pilot if arm.budget == "n_pilot" else None


def step_checkpoint_dir(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, f"step_{step:05d}")


# ---- WP5 pilot (work plan v1.2 WP5; A46 rule 11) ----------------------------------------------------------------
# The pilot trains B6 and C2 for exactly N_pilot optimizer updates and archives the step-N checkpoint. The LR
# schedule is the full-run schedule (B5 accounting over max_epochs) truncated at N, NOT a schedule sized to N: rule
# 11 compares B6/C2 at step N with the final step-N checkpoints of C3/C4/C6/C7, whose budget is "exactly the WP5
# pilot's optimizer-update count" on the full-run schedule, so every arm must share one schedule up to step N.

PILOT_FINAL_MARKER = "PILOT_FINAL"
FIXED_STEP_SELECTION = "fixed optimizer-update count; never a best-validation checkpoint"
WP5_PILOT = {"stop_after_updates": N_PILOT, "arms": ["B6", "C2"], "world_size": 4, "seed": SEED,
             "aug_seed": AUG_SEED}


def n_pilot_from_train_rows(train_rows: int) -> int:
    return int(train_rows) * PILOT_EPOCHS // EFFECTIVE_BATCH


def bind_n_pilot(freeze: Mapping, train_rows: int, config_n_pilot: int) -> int:
    """N_pilot bound to the corpus freeze (decision 18): the freeze's ``n_pilot`` record must follow the rule 11
    formula (effective batch 96, 2 epochs) over exactly the loader's train rows (``len(train_rns)``), and the
    config's n_pilot must equal it -- a stale constant or a corpus of another row set refuses the run."""
    rec = freeze.get("n_pilot") if isinstance(freeze, Mapping) else None
    if not isinstance(rec, Mapping):
        raise ValueError("corpus freeze records no n_pilot (pre-decision-18 freeze); rebuild it with b6_build_corpus.py")
    want = {"formula": N_PILOT_FORMULA, "epochs": PILOT_EPOCHS, "effective_batch": EFFECTIVE_BATCH}
    got = {k: rec.get(k) for k in want}
    if got != want:
        raise ValueError(f"corpus freeze n_pilot {got} != the rule 11 formula {want}")
    if rec.get("train_rows") != int(train_rows):
        raise ValueError(f"corpus freeze n_pilot train_rows {rec.get('train_rows')} != the loader's {train_rows} "
                         "train rows")
    value = n_pilot_from_train_rows(train_rows)
    if rec.get("value") != value:
        raise ValueError(f"corpus freeze n_pilot {rec.get('value')} != floor({train_rows} * {PILOT_EPOCHS} / "
                         f"{EFFECTIVE_BATCH}) = {value}")
    if int(config_n_pilot) != value:
        raise ValueError(f"config n_pilot {config_n_pilot} != the corpus freeze N_pilot {value} (A46 rule 11, "
                         "decision 18): update N_PILOT and the configs' n_pilot / wp5_pilot.stop_after_updates")
    return value


def pilot_stop_target(stop_after_updates: Optional[int], n_pilot: int) -> Optional[int]:
    """The hard stop, in optimizer updates. The only admissible target is N_pilot (rule 11 has no other fixed
    budget), so the step checkpoint published at N_pilot is the pilot-final checkpoint."""
    if stop_after_updates is None:
        return None
    if int(stop_after_updates) != int(n_pilot):
        raise ValueError(f"stop_after_updates {stop_after_updates} must equal N_pilot {n_pilot} (A46 rule 11)")
    return int(stop_after_updates)


def check_wp5_pilot_block(cfg: Mapping, path: str) -> None:
    block = cfg.get("wp5_pilot")
    if block is None:
        return
    if cfg.get("gate_status") != GATED_STATUS:
        raise ValueError(f"{path}: a wp5_pilot config must be {GATED_STATUS!r}, not {cfg.get('gate_status')!r}")
    if not isinstance(block, Mapping) or set(block) != set(WP5_PILOT):
        raise ValueError(f"{path}: wp5_pilot must have exactly the keys {sorted(WP5_PILOT)}")
    for key, value in WP5_PILOT.items():
        if block[key] != value:
            raise ValueError(f"{path}: wp5_pilot {key} = {block[key]!r}, frozen value {value!r}")


def resolve_wp5_pilot(cfg: Mapping, arm: str, *, seed: int, stop_after_updates: Optional[int],
                      allow_ungated: bool) -> Optional[Dict[str, int]]:
    """{'stop_after_updates', 'world_size'} for a run, or None for a full-budget run without a stop.

    With a wp5_pilot config the stop target is the block's (an explicit --stop-after-updates must equal it), the
    arm must be B6 or C2 and --allow-ungated is refused. Without one, --stop-after-updates alone still requests the
    N_pilot stop (the config's own gate applies)."""
    block = cfg.get("wp5_pilot")
    if block is None:
        target = pilot_stop_target(stop_after_updates, int(cfg["n_pilot"]))
        return None if target is None else {"stop_after_updates": target, "world_size": None}
    if allow_ungated:
        raise ValueError("REFUSED: the WP5 pilot config is a production run; --allow-ungated is not accepted")
    if arm not in block["arms"]:
        raise ValueError(f"REFUSED: arm {arm} is not a WP5 pilot arm {block['arms']}")
    if seed != block["seed"]:
        raise ValueError(f"REFUSED: seed {seed} != WP5 pilot seed {block['seed']}")
    if stop_after_updates is not None and int(stop_after_updates) != block["stop_after_updates"]:
        raise ValueError(f"REFUSED: --stop-after-updates {stop_after_updates} != the pilot config's "
                         f"{block['stop_after_updates']}")
    target = pilot_stop_target(block["stop_after_updates"], int(cfg["n_pilot"]))
    return {"stop_after_updates": target, "world_size": int(block["world_size"])}


# ---- optimizer ------------------------------------------------------------------------------------------------

ALPHA_SUFFIX = "chain_attn.alpha"


def build_b6_optimizer(model, lr: float, weight_decay: float, expected_alphas: Optional[int] = None):
    """AdamW exactly as B5 (every parameter decayed at ``weight_decay``) except the chain gates alpha_l,
    which are in a separate group with weight_decay 0 (decay would pull the gates back toward 0)."""
    import torch

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len({id(p) for _, p in named}) != len(named):
        raise ValueError("duplicate trainable parameters")
    alphas = [(n, p) for n, p in named if n.endswith(ALPHA_SUFFIX)]
    if expected_alphas is not None and len(alphas) != expected_alphas:
        raise ValueError(f"expected {expected_alphas} chain gates, found {len(alphas)}")
    if not alphas:
        raise ValueError("no chain gate parameters found; is this a B6 model?")
    decay = [p for n, p in named if not n.endswith(ALPHA_SUFFIX)]
    groups = [{"name": "decay", "params": decay, "weight_decay": weight_decay},
              {"name": "alpha_no_decay", "params": [p for _, p in alphas], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


# ---- config ---------------------------------------------------------------------------------------------------

B6_CONFIG_REQUIRED = {"base_recipe": "b5_400m_win_v3", "evidence_channels": K_TRACKS, "chain_elements": CHAIN_ELEMENTS,
                      "chain_d_model": 512, "chain_heads": 8, "chain_dropout": 0.1, "n_pilot": N_PILOT,
                      "aug_seed_base": AUG_SEED_BASE, "channel_dropout_p": CHANNEL_DROPOUT_P,
                      "thinning_set": list(THINNING_SET), "withhold_source_rate": WITHHOLD_SOURCE_RATE,
                      "withhold_library_rate": WITHHOLD_LIBRARY_RATE, "memory_abort_gb": 110,
                      "memory_gate_gb": 100, "effective_batch_size": EFFECTIVE_BATCH, "max_epochs": MAX_EPOCHS,
                      "patience": PATIENCE, "loss": "L_GSF"}


# The frozen 400M architecture (configs/b5_400m_win_v3.json model block, A42; plan 9 chain block). The trainer
# refuses a config that differs, and the CP2 bench certifies only a model whose built architecture digest equals
# FROZEN_ARCHITECTURE_SHA256 (review 2 H3: a 1-layer decoder must not pass the memory gate).
FROZEN_CONFIG_ARCHITECTURE = {
    "encoder_model": "LongSafari/hyenadna-large-1m-seqlen-hf", "d_model_encoder": 768, "d_model_decoder": 1536,
    "encoder_layers": 12, "decoder_layers": 12, "encoder_n_layer": 12, "encoder_attention_heads": 6,
    "decoder_attention_heads": 6, "encoder_ffn_dim": 3072, "decoder_ffn_dim": 3072, "attention_window": [1024] * 12,
    "dropout": 0.1, "vocab_size": 290, "max_encoder_seqlen": 129024, "max_decoder_position_embeddings": 8192,
    "max_label_tokens": 8192, "window_tiers": [30720, 61440, 129024]}
FROZEN_ARCHITECTURE = {
    "encoder_model": "LongSafari/hyenadna-large-1m-seqlen-hf", "encoder_d_model": 768, "decoder_d_model": 1536,
    "encoder_layers": 12, "decoder_layers": 12, "encoder_n_layer": 12, "encoder_attention_heads": 6,
    "decoder_attention_heads": 6, "encoder_ffn_dim": 3072, "decoder_ffn_dim": 3072, "attention_window": [1024] * 12,
    "dropout": 0.1, "decoder_layerdrop": 0.1, "vocab_size": 290, "max_encoder_seqlen": 129024,
    "max_decoder_position_embeddings": 8192, "evidence_channels": 31, "b6_chain_d_model": 512, "b6_chain_heads": 8,
    "b6_chain_dropout": 0.1, "b6_chain_attn_dropout": 0.1, "b6_chain_elements": 512,
    "b6_model_version": "b6-chain-gated-xattn-v1", "built_decoder_layers": 12, "built_chain_blocks": 12,
    "built_evidence_proj": [768, 31], "built_chain_memory": True}


def model_architecture(model) -> Dict:
    """The architecture of a BUILT B6 model: its config fields plus what was actually constructed."""
    base = model
    while hasattr(base, "module") or hasattr(base, "_orig_mod"):
        base = getattr(base, "module", None) or getattr(base, "_orig_mod")
    c = base.config
    arch = {k: getattr(c, k, None) for k in FROZEN_ARCHITECTURE if not k.startswith("built_")}
    arch["attention_window"] = list(arch["attention_window"] or [])
    layers = base.transgenic.decoder.layers
    arch["built_decoder_layers"] = len(layers)
    arch["built_chain_blocks"] = sum(1 for layer in layers if hasattr(layer, "chain_attn"))
    proj = getattr(base.get_encoder(), "evidence_proj", None)
    arch["built_evidence_proj"] = [int(proj.weight.shape[0]), int(proj.weight.shape[1])] if proj is not None else None
    arch["built_chain_memory"] = hasattr(base.transgenic, "chain_memory")
    return arch


def architecture_sha256(arch: Mapping) -> str:
    return canonical_sha256(dict(arch))


def load_b6_config(path: str) -> Dict:
    """The B5 win_v3 recipe checks (``load_b5_config``), the frozen 400M architecture and the frozen B6 block;
    any drift refuses."""
    cfg = load_b5_config(path)
    for key, value in FROZEN_CONFIG_ARCHITECTURE.items():
        if cfg.get(key) != value:
            raise ValueError(f"{path}: {key} = {cfg.get(key)!r}, frozen architecture value {value!r}")
    for key, value in B6_CONFIG_REQUIRED.items():
        if cfg.get(key) != value:
            raise ValueError(f"{path}: {key} = {cfg.get(key)!r}, frozen value {value!r}")
    if cfg.get("gate_status") not in (GATE_STATUS, GATED_STATUS):
        raise ValueError(f"{path}: gate_status {cfg.get('gate_status')!r} is not a recognised state")
    check_gate_evidence(cfg, path)
    check_wp5_pilot_block(cfg, path)
    return cfg


# The CP2 record a GATED config must carry (A51: certified on 2 or 4 ranks; 1 rank is a pre-check only). The bench
# certified the config bytes of its run (certified_config_sha256, UNGATED at the time) and the BUILT architecture;
# the architecture digest is bound here, so a GATED config of a different model is refused.
GATE_EVIDENCE_KEYS = ("memory_bench_job", "memory_bench_world_size", "memory_bench_peak_reserved_gb",
                      "memory_bench_report_sha256", "certified_config_sha256", "architecture_sha256", "cp4_jobs",
                      "decision")


def check_gate_evidence(cfg: Mapping, path: str) -> None:
    if cfg.get("gate_status") != GATED_STATUS:
        return
    ev = cfg.get("gate_evidence")
    if not isinstance(ev, Mapping) or set(ev) != set(GATE_EVIDENCE_KEYS):
        raise ValueError(f"{path}: a GATED config needs gate_evidence with exactly {list(GATE_EVIDENCE_KEYS)}")
    if ev["memory_bench_world_size"] not in (2, 4):
        raise ValueError(f"{path}: gate_evidence memory_bench_world_size must be 2 or 4 (A51)")
    if not float(ev["memory_bench_peak_reserved_gb"]) <= MEMORY_GATE_BYTES / 10 ** 9:
        raise ValueError(f"{path}: gate_evidence peak reserved above the {MEMORY_GATE_BYTES // 10 ** 9} GB gate")
    for key in ("memory_bench_report_sha256", "certified_config_sha256", "architecture_sha256"):
        if not _is_sha(ev[key]):
            raise ValueError(f"{path}: gate_evidence {key} must be a sha256")
    if ev["architecture_sha256"] != FROZEN_ARCHITECTURE_SHA256:
        raise ValueError(f"{path}: gate_evidence architecture_sha256 is not this checkout's frozen architecture")


def b6_model_kwargs(cfg: Mapping) -> Dict:
    kw = model_kwargs(dict(cfg))
    kw.update(evidence_channels=int(cfg["evidence_channels"]), b6_chain_d_model=int(cfg["chain_d_model"]),
              b6_chain_heads=int(cfg["chain_heads"]), b6_chain_dropout=float(cfg["chain_dropout"]),
              b6_chain_attn_dropout=float(cfg.get("chain_attn_dropout", cfg["chain_dropout"])),
              b6_chain_elements=int(cfg["chain_elements"]))
    return kw


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- run identity (rule 9; plan 8 and 9) ----------------------------------------------------------------------

# The WP3 verifier's record (``b6_bindings.verify_inputs`` return value, WP3 of 2026-09-24), training family:
# family, evidence_sha256, roles_sha256, coverage, spec, aug_seed, draws, evlabel, [reference_sha256], exhaustiveness,
# corpus_sha256, base_sha256, interval_manifest_sha256 ({species: sha}), revalidated, zero_lists, provenance
# ({name: sha256}: evidence, c0 PROVENANCE files, freeze, interval manifests, zero lists and QC, cache manifest),
# cache_manifest_sha256, record_sha256 (canonical JSON of the rest, ``b6_bindings.canonical_record_sha256``).
# The nested ``spec`` dict is flattened. Every field is carried into the identity, so any change refuses a resume.
BINDING_KEYS = ("family", "evidence_sha256", "roles_sha256", "coverage", "aug_seed", "evlabel", "exhaustiveness",
                "corpus_sha256", "base_sha256", "interval_manifest_sha256", "zero_lists", "provenance",
                "cache_manifest_sha256", "record_sha256",
                "track_spec_sha256", "chain_spec_sha256", "aug_spec_sha256", "aug_spec_id")
PRESENT_KEYS = ("draws", "revalidated")          # must be present; the value may be None / empty
OPTIONAL_KEYS = ("reference_sha256",)            # present only when the consumer passed references
HEX64 = frozenset("0123456789abcdef")


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


FROZEN_ARCHITECTURE_SHA256 = hashlib.sha256(json.dumps(FROZEN_ARCHITECTURE, sort_keys=True, separators=(",", ":"),
                                                     default=str).encode("utf-8")).hexdigest()


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= HEX64


def identity_from_bindings(record: Mapping, record_sha256_fn=None) -> Dict[str, object]:
    """The binding part of the identity. Training accepts only the ``training`` family (plan 8).

    ``record_sha256_fn`` is WP3's ``b6_bindings.canonical_record_sha256``; when given (always, in ``main``), the
    record's own record_sha256 must equal it, so a record edited after verification is refused."""
    flat = dict(record)
    flat.update(flat.pop("spec", {}) or {})
    missing = [k for k in BINDING_KEYS if flat.get(k) in (None, "", {}, [])]
    missing += [k for k in PRESENT_KEYS if k not in flat]
    if missing:
        raise ValueError(f"binding record lacks {missing}")
    if flat["family"] != "training":
        raise ValueError(f"training refuses binding family {flat['family']!r} (plan 8: eval_only is rejected)")
    if flat["evlabel"] != "absent":
        raise ValueError("evlabel must be absent from every B6 training input (A50.2)")
    for key in ("track_spec_sha256", "chain_spec_sha256", "aug_spec_sha256", "aug_spec_id"):
        if flat[key] != SPEC_SHAS[key]:
            raise ValueError(f"binding {key} {flat[key]} != this checkout's b6_track_spec {SPEC_SHAS[key]}")
    if int(flat["aug_seed"]) != AUG_SEED:
        raise ValueError(f"binding aug_seed {flat['aug_seed']} != {AUG_SEED} (rule 8: 20260914 + seed 123)")
    prov = flat["provenance"]
    if not isinstance(prov, Mapping) or not prov or not all(_is_sha(v) for v in prov.values()):
        raise ValueError("binding provenance must map each PROVENANCE/sidecar name to its sha256")
    if not _is_sha(flat["record_sha256"]):
        raise ValueError("binding record_sha256 must be a sha256")
    if record_sha256_fn is not None and record_sha256_fn(record) != record["record_sha256"]:
        raise ValueError("binding record_sha256 does not match the record: it changed after verification")
    out = {f"binding.{k}": flat[k] for k in BINDING_KEYS + PRESENT_KEYS}
    out.update({f"binding.{k}": flat.get(k) for k in OPTIONAL_KEYS})
    out["draw_spec"] = {"aug_spec_id": flat["aug_spec_id"], "aug_spec_sha256": flat["aug_spec_sha256"],
                        "aug_seed": int(flat["aug_seed"]), "draws": flat["draws"],
                        "fractions": list(THINNING_SET), "channel_dropout_p": CHANNEL_DROPOUT_P,
                        "withhold_source_rate": WITHHOLD_SOURCE_RATE, "withhold_library_rate": WITHHOLD_LIBRARY_RATE}
    return out


RUN_KEYS = ("recipe_sha256", "arm", "seed", "aug_seed", "init_policy", "init_weights_sha256", "data_order_policy",
            "batch_size", "accumulation_steps", "world_size", "effective_batch", "max_epochs", "patience", "budget",
            "n_pilot", "stop_after_updates", "channel_dropout_p", "thinning_set", "withhold_source_rate", "withhold_library_rate",
            "compile_policy", "checkpointing", "mixed_precision", "memory_abort_bytes", "deterministic", "loss",
            "model_version", "torch", "transformers", "accelerate", "duckdb")

IDENTITY_REQUIRED_KEYS = (RUN_KEYS + tuple(f"binding.{k}" for k in BINDING_KEYS + PRESENT_KEYS + OPTIONAL_KEYS)
                          + ("draw_spec",))
CHECKPOINT_IDENTITY = "identity.json"


def state_digest(model) -> str:
    """sha256 of the actual weights (tied aliases once, sorted names, dtype, shape, bytes): the initialisation
    identity. Two arms that load different pretrained bytes of identical shapes get different digests."""
    import torch
    tensors, _ = dedup_state_dict(model)
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        h.update(f"{name}\t{t.dtype}\t{tuple(t.shape)}\n".encode("utf-8"))
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_checkpoint_identity(ckpt_dir: str, identity: Mapping) -> None:
    with open(os.path.join(ckpt_dir, CHECKPOINT_IDENTITY), "w") as fh:
        json.dump(dict(identity), fh, indent=1, sort_keys=True, default=str)


def check_checkpoint_identity(ckpt_dir: str, identity: Mapping) -> None:
    """Refuse a checkpoint that does not carry, or does not match, the current verified identity. Called before
    any state is loaded, so a checkpoint of another arm, seed, corpus or initialisation never enters the run."""
    path = os.path.join(ckpt_dir, CHECKPOINT_IDENTITY)
    if not os.path.exists(path):
        raise RuntimeError(f"resume refused: {path} is missing; the checkpoint's training identity is unknown")
    with open(path) as fh:
        stored = json.load(fh)
    try:
        check_run_config_b6(stored, json.loads(json.dumps(dict(identity), default=str)))
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} (checkpoint {ckpt_dir})") from None


def build_run_identity(run: Mapping, bindings: Mapping, record_sha256_fn=None) -> Dict:
    missing = [k for k in RUN_KEYS if k not in run]
    if missing:
        raise ValueError(f"run identity lacks {missing}")
    identity = {k: run[k] for k in RUN_KEYS}
    identity.update(identity_from_bindings(bindings, record_sha256_fn))
    # Canonical JSON form: the live WP3 record holds tuples (e.g. coverage), the stored identity holds lists; without
    # this a legitimate resume is refused (found by the CP4 WP3-fixture mode).
    return json.loads(json.dumps(identity, sort_keys=True, default=str))


def check_run_config_b6(first: Mapping, current: Mapping) -> None:
    """Refuse a resume unless every required key is present on BOTH sides and equal (no 'if key exists')."""
    for key in IDENTITY_REQUIRED_KEYS:
        if key not in first:
            raise RuntimeError(f"resume refused: {key} missing from the run's first identity")
        if key not in current:
            raise RuntimeError(f"resume refused: {key} missing from the current identity")
        if first[key] != current[key]:
            raise RuntimeError(f"resume refused: {key} differs ({first[key]!r} != {current[key]!r})")


def identity_paths(run_dir: str) -> Tuple[str, str]:
    return os.path.join(run_dir, "run_config_b6.json"), os.path.join(run_dir, "runtime_identity.txt")


def record_identity(run_dir: str, identity: Mapping, alphas: Sequence[float], link: Mapping) -> None:
    """First link: write run_config_b6.json. Every link: verify it, then append this link's section, with
    per-layer alpha at link start, to runtime_identity.txt (rule 9)."""
    cfg_path, txt_path = identity_paths(run_dir)
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            check_run_config_b6(json.load(fh), identity)
    else:
        with open(cfg_path + ".tmp", "w") as fh:
            json.dump(dict(identity), fh, indent=1, sort_keys=True)
        os.replace(cfg_path + ".tmp", cfg_path)
    lines = [f"# link {link.get('link', '?')} {time.strftime('%Y-%m-%dT%H:%M:%S')}"]
    lines += [f"{k}={json.dumps(identity[k], sort_keys=True)}" for k in sorted(identity)]
    lines += [f"link.{k}={json.dumps(v, sort_keys=True)}" for k, v in sorted(link.items())]
    lines += [f"alpha.layer{i:02d}={a!r}" for i, a in enumerate(alphas)]
    with open(txt_path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


# ---- compile policy (rule 9: a Dynamo fallback is a failure) ---------------------------------------------------

def compile_b6(model, backend="inductor"):
    """Whole-model torch.compile with suppress_errors off. B5 turns it on (train_HyenaTransgenic.py:320)."""
    import torch

    torch._dynamo.config.suppress_errors = False
    # DDP bucket graph partitioning changed stochastic gradients across a cold resume in CP4.
    # Keep Inductor and DDP, but compile without that partitioning (2-rank diagnosis 3207863).
    torch._dynamo.config.optimize_ddp = False
    return torch.compile(model, backend=backend)


def compiled_graph_count() -> int:
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def inductor_kernel_count() -> int:
    from torch._inductor import metrics
    return int(metrics.generated_kernel_count)


class CompileGuard:
    """Recompiles are allowed during warm-up and on the first sight of a new batch shape (a new tile tier);
    any later compile raises. ``suppress_errors`` switched back on also raises."""

    def __init__(self, warmup_steps: int):
        self.warmup_steps = int(warmup_steps)
        self.seen = set()
        self.graphs = None

    def observe(self, optimizer_step: int, shape_key, graphs_now: int, suppress_errors: bool) -> None:
        if suppress_errors:
            raise RuntimeError("torch._dynamo.config.suppress_errors is on: an eager fallback would be silent")
        new = self.graphs is not None and graphs_now > self.graphs
        if new and optimizer_step > self.warmup_steps and shape_key in self.seen:
            raise RuntimeError(f"recompile after warm-up at optimizer step {optimizer_step} for an already "
                               f"compiled shape {shape_key}; graphs {self.graphs} -> {graphs_now}")
        self.graphs = graphs_now
        self.seen.add(shape_key)


def inductor_activity() -> int:
    """Kernels generated plus FX-graph cache hits. A resumed link on a warm Inductor cache loads its kernels
    instead of generating them, so generated_kernel_count alone would flag it as eager."""
    from torch._dynamo.utils import counters
    return inductor_kernel_count() + int(counters["inductor"]["fxgraph_cache_hit"])


def check_inductor_kernels(backend, before: int, after: int) -> None:
    if backend == "inductor" and after <= before:
        raise RuntimeError(f"no Inductor kernel generated or loaded after warm-up ({before} -> {after}): "
                           "the model ran eager")


def check_compiled(graphs_before: int, graphs_after: int) -> None:
    """Any backend: at least one Dynamo graph must have been compiled on this link by the end of warm-up."""
    if graphs_after <= graphs_before:
        raise RuntimeError(f"no Dynamo graph compiled on this link ({graphs_before} -> {graphs_after}): "
                           "the model ran eager")


# ---- memory abort (rule 9) ------------------------------------------------------------------------------------

def memory_abort(reserved_bytes: Sequence[int], run_dir: Optional[str], *, threshold: int = MEMORY_ABORT_BYTES,
                 step: Optional[int] = None, write: bool = True) -> bool:
    """True if any rank's reserved bytes exceed the threshold; the main process writes MEMORY_ABORT."""
    peak = max(int(x) for x in reserved_bytes)
    if peak <= threshold:
        return False
    if write and run_dir:
        with open(os.path.join(run_dir, MEMORY_ABORT_MARKER), "w") as fh:
            json.dump({"reserved_bytes_per_rank": [int(x) for x in reserved_bytes], "threshold_bytes": threshold,
                       "optimizer_step": step, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
    return True


# ---- data order (rule 11) -------------------------------------------------------------------------------------

def data_order_hash(rows: Iterable) -> str:
    """sha256 over the first 1,000 row keys of an epoch, one decimal per line."""
    head = []
    for r in rows:
        head.append(str(int(r)) if not isinstance(r, str) else r)
        if len(head) == DATA_ORDER_PREFIX:
            break
    return hashlib.sha256("\n".join(head).encode("utf-8")).hexdigest()


def b5_data_order(train_rns: Sequence[int], seed: int, epoch: int) -> List[int]:
    """The B5 per-epoch order (rule 11), exactly: accelerate ``SeedableRandomSampler`` (B5 prepares its loader
    with use_seedable_sampler=True after torch.manual_seed(seed), so initial_seed = seed) seeds a CPU generator
    with seed + epoch and RandomSampler yields torch.randperm(n). ``train_rns`` is the B5 train list in dataset
    index order: ``split = 'train' AND gff IS NOT NULL AND COALESCE(train_weight, 1.0) > 0 ORDER BY rn``
    (datasets.py isoformDataHyena). Under DDP accelerate then deals batch i to rank i % world, as for B5."""
    import torch
    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch))
    return [int(train_rns[i]) for i in torch.randperm(len(train_rns), generator=g).tolist()]


# ---- pending gradients at an epoch checkpoint (B5 A28 rule, rank-local) -----------------------------------------

@contextmanager
def checkpoint_io(accelerator):
    """Finish rank-local checkpoint I/O, then propagate any failure before the next collective.

    Never place collectives in this context. Accelerate's replicated DDP save_state is local I/O;
    sharded backends and automatic checkpoint naming can contain barriers and are not supported here.
    """
    from accelerate.utils import DistributedType
    from transgenic.training.mvp_finalize import _collective_preflight

    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU):
        raise RuntimeError("B6 coordinated checkpoint I/O requires replicated DDP or a single process")
    if accelerator.project_configuration.automatic_checkpoint_naming:
        raise RuntimeError("B6 coordinated checkpoint I/O forbids automatic checkpoint naming")
    failure = None
    try:
        yield
    except Exception as exc:
        failure = exc

    def check():
        if failure is not None:
            raise RuntimeError(f"B6 checkpoint I/O failed: {failure}") from failure

    _collective_preflight(accelerator, check)


def pending_grads_path(ckpt_dir: str, rank: int) -> str:
    return os.path.join(ckpt_dir, f"pending_grads_rank{rank}.pt")


def save_pending_grads(model, ckpt_dir: str, rank: int) -> int:
    import torch
    grads = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None}
    torch.save(grads, pending_grads_path(ckpt_dir, rank))
    return len(grads)


def load_pending_grads(model, ckpt_dir: str, rank: int) -> int:
    import torch
    path = pending_grads_path(ckpt_dir, rank)
    if not os.path.exists(path):
        return 0
    grads = torch.load(path, map_location="cpu")
    for n, p in model.named_parameters():
        if n in grads:
            p.grad = grads[n].to(device=p.device, dtype=p.dtype)
    return len(grads)


# ---- deterministic weight files ------------------------------------------------------------------------------

def dedup_state_dict(model) -> Tuple[Dict, Dict[str, str]]:
    """State dict with tied aliases removed: per shared storage the sorted-first name is kept. Returns
    (tensors, {alias: kept})."""
    sd = model.state_dict()
    groups: Dict[Tuple, List[str]] = {}
    for name, t in sd.items():
        key = (t.device.type, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype)
        groups.setdefault(key, []).append(name)
    tensors, aliases = {}, {}
    for names in groups.values():
        names = sorted(names)
        tensors[names[0]] = sd[names[0]].detach().contiguous().cpu()
        for alias in names[1:]:
            aliases[alias] = names[0]
    return tensors, dict(sorted(aliases.items()))


def save_weights_deterministic(model, path: str) -> Dict[str, str]:
    """safetensors file whose bytes depend only on the weights. ``safetensors.torch.save_model`` writes the
    alias map into the header in a varying key order when three names share one tensor (decoder embeddings,
    decoder.embed_tokens and lm_head here), so two saves of identical weights can differ byte-wise."""
    from safetensors.torch import save_file
    tensors, aliases = dedup_state_dict(model)
    save_file(tensors, path, metadata={"format": "pt"})
    return aliases


def load_weights(model, path: str) -> None:
    """Inverse of ``save_weights_deterministic``: every missing key must be a tied alias of a loaded one."""
    from safetensors.torch import load_file
    tensors = load_file(path)
    _, aliases = dedup_state_dict(model)
    result = model.load_state_dict(tensors, strict=False)
    if result.unexpected_keys or set(result.missing_keys) - set(aliases):
        raise RuntimeError(f"{path}: unexpected {result.unexpected_keys}, "
                           f"missing {sorted(set(result.missing_keys) - set(aliases))}")


# ---- CLI ------------------------------------------------------------------------------------------------------

def parse_args_b6(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train the B6 evidence-mode model ({GATE_STATUS})")
    p.add_argument("--config", required=True, help="configs/b6_400m_win_v3.json")
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--bindings", required=True, help="WP3 binding manifest verified by b6_bindings.verify_inputs")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None, help="'auto' or a checkpoint directory")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every-n-steps", type=int, default=200)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--mixed-precision", default="bf16", choices=("bf16", "no"))
    p.add_argument("--compile-backend", default="inductor", help="'none' only for local debugging, never a run")
    p.add_argument("--compile-warmup-steps", type=int, default=3)
    p.add_argument("--deterministic", action="store_true",
                   help="torch.use_deterministic_algorithms (needs CUBLAS_WORKSPACE_CONFIG=:4096:8); recorded")
    p.add_argument("--allow-ungated", action="store_true",
                   help="acknowledge that the memory gate has not passed (integration tests only)")
    p.add_argument("--stop-after-updates", type=int, default=None,
                   help="WP5 pilot: stop at exactly this many optimizer updates (must equal N_pilot = 4,142) and "
                        "mark step_04142 as the pilot-final checkpoint; the wp5_pilot config sets it itself")
    a = p.parse_args(argv)
    if a.seed != SEED:
        p.error(f"rule 11: every B6 arm uses seed {SEED}")
    return a


def link_record(**kw) -> Dict:
    return dict(kw)


def asdict_arm(arm: ArmPreset) -> Dict:
    return asdict(arm)
