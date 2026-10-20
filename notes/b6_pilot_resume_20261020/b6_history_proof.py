"""Fresh B6 0→410 / snapshot400 / new-process410 diagnostic proof.

Bundle unchanged history_proof.py and probe.py in B6_C2_HELPER_DIR on deployment.
Locally the exact repository-relative dependency is used; CWD is never searched.
All heavy hashing and trainer calls belong inside the allocated NGC container.
"""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
C2_DIR = Path(
    os.environ.get(
        "B6_C2_HELPER_DIR",
        str(HERE.parent / "checkpoint_archive_20261009/c2_failure_3345919/resume_determinism"),
    )
).resolve()
_spec = importlib.util.spec_from_file_location(
    "b6_reviewed_c2_history", C2_DIR / "history_proof.py"
)
_contract_spec = importlib.util.spec_from_file_location(
    "b6_receipt_contract", HERE / "b6_receipt_contract.py"
)
contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(contract)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
_state_spec = importlib.util.spec_from_file_location(
    "b6_reviewed_c2_state", C2_DIR / "production_link.py"
)
reviewed_state = importlib.util.module_from_spec(_state_spec)
_state_spec.loader.exec_module(reviewed_state)
probe = base.probe
sha, require = probe.sha, base.require
SOURCE_COMMIT = "198ce14ce53266f53b7bda5d0c4cdfd94deb43b4"
PROFILE = "b6_fresh400"
TIERS = {30720, 61440, 129024}
POSITION = dict(epoch=0, step=9840, global_step=410, pending_micro=0)
SNAPSHOT = dict(epoch=0, step=9600, global_step=400, pending_micro=0)


def read(path):
    return json.loads(Path(path).read_text())


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_hash(value, name):
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "invalid digest: " + name,
    )


def start_position(run, *, fresh=False):
    run = Path(run)
    if fresh:
        require(run.name == "continuous", "fresh is only valid for continuous B6")
        require(
            run.is_dir() and not run.is_symlink() and not any(run.iterdir()),
            "fresh continuous must be an empty reserved directory",
        )
        return 0
    require(run.name == "resumed", "resume is only valid for resumed B6")
    meta = read(run / "latest_state/meta.json")
    require(
        all(type(meta.get(k)) is int and meta[k] == v for k, v in SNAPSHOT.items()),
        "resume must start from snapshot400",
    )
    return 9600


def init_fresh(root):
    root = Path(root)
    verified = verified_record(root, current_allocation=True)
    (root / "continuous").mkdir()  # exclusive reservation, once per allocation
    start_position(root / "continuous", fresh=True)
    probe.write_json(
        root / "fresh_start.json",
        dict(
            job_id=verified["job_id"],
            verified_record_sha256=sha(root / "verified_record.json"),
            kind="b6_fresh_continuous",
            step=0,
        ),
    )


def fresh_start(run):
    run = Path(run)
    require(
        run.name == "continuous" and run.is_dir() and not run.is_symlink(),
        "wrong fresh run",
    )
    verified = verified_record(run.parent, current_allocation=True)
    ticket = read(run.parent / "fresh_start.json")
    require(
        ticket
        == dict(
            job_id=verified["job_id"],
            verified_record_sha256=sha(run.parent / "verified_record.json"),
            kind="b6_fresh_continuous",
            step=0,
        ),
        "wrong fresh launch ticket",
    )
    # The ticket reserved an empty directory before launch. Other ranks may
    # now have opened exclusive traces or begun collective trainer setup.
    require(
        not os.path.lexists(run / "latest_state"),
        "fresh continuous cannot load prior checkpoint",
    )
    return 0


def verify_manifest(path, required):
    path = Path(path)
    members = set()
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, "invalid manifest row")
        digest, name = match.groups()
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts and name not in members,
            "unsafe/duplicate manifest member",
        )
        member = path.parent / relative
        require(
            member.is_file() and not member.is_symlink() and sha(member) == digest,
            "manifest member bytes changed: " + name,
        )
        members.add(name)
    require(set(required) <= members, "manifest omits executed helper/source members")
    return members


def require_operating_fingerprint(value):
    # Keep the reviewed fingerprint policy, substituting only the B6 namespace.
    adapted = dict(value, cache_env=dict(value.get("cache_env", {})))
    env = value.get("cache_env", {})
    require(
        env.get("TORCHINDUCTOR_CACHE_DIR") == "/work/compiler_cache/b6_history_v1/inductor"
        and env.get("TRITON_CACHE_DIR") == "/work/compiler_cache/b6_history_v1/triton",
        "wrong B6 cache identity",
    )
    adapted["cache_env"].update(
        TORCHINDUCTOR_CACHE_DIR="/work/compiler_cache/c2_history_v1/inductor",
        TRITON_CACHE_DIR="/work/compiler_cache/c2_history_v1/triton",
    )
    base.require_operating_fingerprint(adapted)


def checkpoint_binding(run, position, verified):
    from transgenic.training.b6_compile_history import CompileHistory

    checkpoint = run / "latest_state"
    for name in ("meta.json", "identity.json", "compile_history.json"):
        require(
            (checkpoint / name).is_file() and not (checkpoint / name).is_symlink(),
            "missing/linked checkpoint metadata: " + name,
        )
    meta, identity = read(checkpoint / "meta.json"), read(checkpoint / "identity.json")
    require(
        all(type(meta.get(k)) is int and meta[k] == v for k, v in position.items()),
        "wrong checkpoint position: " + str(run),
    )
    expected = dict(
        arm="B6",
        seed=123,
        aug_seed=20261037,
        world_size=4,
        n_pilot=4099,
        stop_after_updates=4099,
        mixed_precision="bf16",
        checkpointing="nonreentrant",
        deterministic=True,
        recipe_sha256=verified["recipe_sha256"],
    )
    require(
        all(type(identity.get(k)) is type(v) and identity[k] == v for k, v in expected.items()),
        "wrong B6 checkpoint identity",
    )
    require_hash(identity.get("init_weights_sha256"), "initial weights")
    require(
        all(identity.get(k) == v for k, v in verified["identity_from_bindings"].items()),
        "checkpoint differs from verified inputs",
    )
    require(
        meta.get("identity_sha256") == digest_json(identity),
        "unbound checkpoint identity digest",
    )
    payload = read(checkpoint / "compile_history.json")
    fingerprint = payload["fingerprint"]
    require_operating_fingerprint(fingerprint)
    require(
        fingerprint.get("recipe") == verified["recipe"]
        and verified["recipe"].get("_sha256") == verified["recipe_sha256"],
        "history recipe fingerprint differs",
    )
    history = CompileHistory.load(
        checkpoint / "compile_history.json",
        meta["compile_history_sha256"],
        identity,
        fingerprint,
        SimpleNamespace(num_processes=4, process_index=0),
        checkpoint=position,
    )
    require(history.bootstrap is None, "fresh B6 cannot have a bootstrap")
    require(
        history.entries and all(e["origin"] == "observed" for e in history.entries),
        "B6 requires nonempty observed compiler history",
    )
    events = sum(r["delta"] + r["activity_delta"] for e in history.entries for r in e["ranks"])
    require(events > 0, "missing real compiler events")
    # Every history event must have occurred before the committed train cursor.
    require(
        all(
            e["position"]["phase"] == "train"
            and e["position"]["epoch"] == 0
            and e["position"]["batch_index"] < position["step"]
            and e["position"]["optimizer_step"] == e["position"]["batch_index"] // 24
            for e in history.entries
        ),
        "history event outside diagnostic train cursor",
    )
    state = checkpoint / "accelerate_state"
    expected_files = base.STATE_KEYS | {"model.safetensors"}
    require(
        {p.name for p in state.iterdir()} == expected_files,
        "unexpected/missing complete state file inventory",
    )
    require(
        all(p.is_file() and not p.is_symlink() for p in state.iterdir()),
        "linked state files",
    )
    file_hashes = {p.name: sha(p) for p in state.iterdir()}
    return dict(
        position=dict(position),
        state_file_sha256=file_hashes,
        meta_sha256=sha(checkpoint / "meta.json"),
        identity_sha256=sha(checkpoint / "identity.json"),
        history_sha256=sha(checkpoint / "compile_history.json"),
        identity=identity,
        fingerprint=fingerprint,
        compiler_events=events,
        history_entries=len(history.entries),
    )


def tensor_descriptor(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype))


def capture_prepared_schema(model_state, optimizer, scheduler):
    """Metadata only from actual prepared objects, before checkpoint load/train.

    Accelerate.save_state uses get_state_dict(_models[0], unwrap=False), then
    retains the first key per id_tensor_storage group during safetensors save.
    Apply that exact alias policy without contiguous copies or weight hashing.
    """
    from accelerate.utils.other import id_tensor_storage
    import torch

    model_state = dict(model_state)
    require(
        model_state and all(isinstance(v, torch.Tensor) for v in model_state.values()),
        "invalid prepared model state",
    )
    reviewed_state.compatibility_model(model_state)
    full_model = {k: tensor_descriptor(v) for k, v in model_state.items()}
    model, aliases, storage = {}, {}, {}
    for name, value in model_state.items():
        token = id_tensor_storage(value)
        if token in storage:
            aliases[name] = storage[token]
        else:
            storage[token] = name
            model[name] = full_model[name]
    groups = optimizer.state_dict()["param_groups"]
    parameters = {}
    require(len(groups) == len(optimizer.param_groups), "prepared optimizer group disagreement")
    for serialized, actual in zip(groups, optimizer.param_groups):
        require(
            len(serialized["params"]) == len(actual["params"]),
            "prepared optimizer parameter disagreement",
        )
        for key, value in zip(serialized["params"], actual["params"]):
            require(
                type(key) is int and str(key) not in parameters,
                "duplicate prepared optimizer parameter",
            )
            parameters[str(key)] = tensor_descriptor(value)
    return dict(
        version=1,
        kind="b6_prepared_checkpoint_schema",
        model=model,
        full_model=full_model,
        tied_aliases=aliases,
        optimizer_groups=[g["params"] for g in groups],
        optimizer_group_keys=[sorted(g) for g in groups],
        optimizer_parameters=parameters,
        scheduler_keys=sorted(scheduler.state_dict()),
    )


def prepared_schema(root):
    """Each rank's pre-training schema is independently tied to its trace."""
    schemas, hashes = {}, {}
    for phase in ("continuous", "resumed"):
        for rank in range(4):
            path = root / phase / ("prepared_schema_rank%d.json" % rank)
            require(path.is_file() and not path.is_symlink(), "missing prepared schema")
            schema = read(path)
            with (root / phase / ("trace_rank%d.jsonl" % rank)).open() as stream:
                first = json.loads(stream.readline())
            require(
                first.get("event") == "prepared"
                and first.get("rank") == rank
                and first.get("micro") == (0 if phase == "continuous" else 9600)
                and first.get("prepared_schema_sha256") == sha(path),
                "prepared schema is not trace-bound",
            )
            schemas[phase + "/rank%d" % rank] = schema
            hashes[phase + "/rank%d" % rank] = sha(path)
    require(
        len({digest_json(value) for value in schemas.values()}) == 1,
        "prepared schema differs across ranks/processes",
    )
    schema = schemas["continuous/rank0"]
    contract.validate_prepared_schema(schema)
    return schema, hashes


def complete_checkpoint(run, schema, position):
    """Validate payload contents before exact canonical comparison (C2 policies)."""
    import torch
    from transgenic.training.b6_compile_history import encode_rng

    directory = run / "latest_state/accelerate_state"
    weights = reviewed_state.load_weights(directory / "model.safetensors")
    require(
        {k: tensor_descriptor(v) for k, v in weights.items()} == schema["model"],
        "model keys/shape/dtype differ from independently prepared schema",
    )
    state = {}
    for name in sorted(base.STATE_KEYS):
        value = torch.load(directory / name, map_location="cpu", weights_only=False)
        require(type(value) is dict and bool(value), "empty checkpoint state: " + name)
        reviewed_state.finite(value)
        state[name] = value
    optimizer = state["optimizer.bin"]
    require(
        type(optimizer.get("state")) is dict
        and bool(optimizer["state"])
        and type(optimizer.get("param_groups")) is list
        and bool(optimizer["param_groups"]),
        "partial AdamW optimizer",
    )
    groups = optimizer["param_groups"]
    require(
        len(groups) == len(schema["optimizer_groups"]) and all(type(g) is dict for g in groups),
        "optimizer groups missing",
    )
    require(
        [g.get("params") for g in groups] == schema["optimizer_groups"]
        and [sorted(g) for g in groups] == schema["optimizer_group_keys"],
        "optimizer group/parameter fields changed",
    )
    parameter_ids = [key for group in groups for key in group["params"]]
    require(
        parameter_ids
        and len(parameter_ids) == len(set(parameter_ids))
        and all(type(key) is int for key in parameter_ids)
        and set(optimizer["state"]) <= set(parameter_ids),
        "invalid optimizer parameter groups",
    )
    for group in groups:
        require(
            type(group["lr"]) in (int, float)
            and group["lr"] >= 0
            and type(group["eps"]) in (int, float)
            and group["eps"] > 0
            and type(group["weight_decay"]) in (int, float)
            and group["weight_decay"] >= 0
            and type(group["amsgrad"]) is bool,
            "invalid AdamW group values",
        )
        betas = group["betas"]
        require(
            type(betas) in (tuple, list)
            and len(betas) == 2
            and all(type(beta) in (int, float) and 0 <= beta < 1 for beta in betas),
            "invalid AdamW betas",
        )
    slots = {}
    for key, value in optimizer["state"].items():
        require(
            type(key) is int
            and type(value) is dict
            and {"step", "exp_avg", "exp_avg_sq"} <= set(value),
            "partial AdamW moments",
        )
        expected = schema["optimizer_parameters"][str(key)]
        for name in ("exp_avg", "exp_avg_sq"):
            require(
                isinstance(value[name], torch.Tensor)
                and tensor_descriptor(value[name]) == expected,
                "AdamW moment shape/dtype changed",
            )
        step = value["step"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and float(step) > 0
            and float(step).is_integer()
            and float(step) <= position["global_step"],
            "invalid AdamW step",
        )
        if any(key in group["params"] and group.get("amsgrad") is True for group in groups):
            require(
                isinstance(value.get("max_exp_avg_sq"), torch.Tensor)
                and tensor_descriptor(value["max_exp_avg_sq"]) == expected,
                "partial AMSGrad moment",
            )
        slots[str(key)] = {
            name: tensor_descriptor(value[name]) for name in ("exp_avg", "exp_avg_sq")
        }
    scheduler = state["scheduler.bin"]
    require(
        set(scheduler) == set(schema["scheduler_keys"])
        and {"base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"} <= set(scheduler),
        "partial scheduler",
    )
    require(
        type(scheduler["last_epoch"]) is int
        and scheduler["last_epoch"] >= 0
        and type(scheduler["_step_count"]) is int
        and scheduler["_step_count"] > 0,
        "invalid scheduler counters",
    )
    for name in ("base_lrs", "_last_lr", "lr_lambdas"):
        require(
            type(scheduler[name]) is list and len(scheduler[name]) == len(groups),
            "partial scheduler groups",
        )
    for name in ("base_lrs", "_last_lr"):
        require(
            all(type(value) in (int, float) and value >= 0 for value in scheduler[name]),
            "invalid scheduler LR values",
        )
    require(
        scheduler["_last_lr"] == [g["lr"] for g in groups], "scheduler/optimizer LR disagreement"
    )
    require(
        all(value is None or type(value) is dict for value in scheduler["lr_lambdas"]),
        "invalid scheduler lambda state",
    )
    rng_keys = {}
    cuda_devices = {}
    for rank in range(4):
        rng = state["random_states_%d.pkl" % rank]
        require(
            {
                "step",
                "random_state",
                "numpy_random_seed",
                "torch_manual_seed",
                "torch_cuda_manual_seed",
            }
            <= set(rng),
            "partial rank RNG",
        )
        require(type(rng["step"]) is int and rng["step"] >= 0, "invalid RNG step")
        cuda = rng["torch_cuda_manual_seed"]
        require(
            type(cuda) is list
            and len(cuda) == 4
            and all(
                isinstance(v, torch.Tensor) and v.layout == torch.strided and v.is_contiguous()
                for v in cuda
            ),
            "exactly four CUDA RNG states required",
        )
        encode_rng(
            dict(
                python=rng["random_state"],
                numpy=rng["numpy_random_seed"],
                torch_cpu=rng["torch_manual_seed"],
                torch_cuda=cuda,
            )
        )
        rng_keys[str(rank)] = sorted(rng)
        cuda_devices[str(rank)] = len(cuda)
    structure = dict(
        model={k: tensor_descriptor(v) for k, v in weights.items()},
        optimizer_groups=[g["params"] for g in groups],
        optimizer_state=slots,
        scheduler_keys=sorted(scheduler),
        rng_keys=rng_keys,
        cuda_rng_devices=cuda_devices,
    )
    tree = dict(
        model={k: probe.tree(v) for k, v in weights.items()},
        state={k: probe.tree(v) for k, v in state.items()},
    )
    return tree, structure


class TrainingTrace(base.TrainingTrace):
    def pre(self, module, positional, inputs):
        super().pre(module, positional, inputs)
        if self.depth:
            return
        import torch

        tracks, mask = inputs["evidence_tracks"], inputs["chain_mask"]
        idx, feat = inputs["chain_idx"], inputs["chain_feat"]
        # The collator already validates the full batch. Check the actual model
        # inputs as well, so masked dummy/padding cannot count as chain evidence.
        require(
            mask.ndim == 2 and mask.dtype == torch.bool and not bool(mask[:, 0].any()),
            "chain dummy must be masked",
        )
        valid = idx[mask]
        require(
            not valid.numel()
            or bool(((valid >= 0) & (valid < inputs["input_ids"].shape[-1] // 6)).all()),
            "invalid chain element indices",
        )
        require(
            not bool((idx[~mask] != 0).any()) and not bool((feat[~mask] != 0).any()),
            "masked chain elements must be zero",
        )
        self.before["observed"] = dict(
            dna_tier=int(inputs["input_ids"].shape[-1]),
            tracks_nonzero=int(torch.count_nonzero(tracks).item()),
            valid_chain_elements=int(mask.sum().item()),
            tracks_sha256=probe.digest(tracks),
            chain_sha256=probe.digest(dict(idx=idx, feat=feat, mask=mask)),
        )


def verified_record(root, *, current_allocation=False):
    value = read(root / "verified_record.json")
    require(
        value.get("version") == 1 and value.get("kind") == "b6_fresh_allocation_verification",
        "wrong verification record kind",
    )
    require(value.get("source_commit") == SOURCE_COMMIT, "verification source mismatch")
    job = value.get("job_id")
    require(
        type(job) is str and job.isascii() and job.isdecimal(),
        "invalid verification job",
    )
    require(root.name == job, "proof root must equal allocation job ID")
    if current_allocation:
        require(job == os.environ.get("SLURM_JOB_ID"), "stale verification job")
    require_hash(value.get("recipe_sha256"), "recipe")
    require_hash(value.get("bindings_sha256"), "bindings")
    require(
        type(value.get("identity_from_bindings")) is dict,
        "missing verified input identity",
    )
    require(type(value.get("recipe")) is dict, "missing verified recipe payload")
    require(
        type(value.get("mmap_dir")) is str and bool(value["mmap_dir"]),
        "missing frozen mmap path",
    )
    require(
        set(value.get("provenance", {}))
        >= {"image_sha256", "source_manifest_sha256", "helper_manifest_sha256"},
        "missing runtime provenance",
    )
    for k, v in value["provenance"].items():
        require_hash(v, k)
    return value


def validate_rng_trace(value):
    require(
        type(value) is dict and set(value) == {"cpu", "python", "numpy", "cuda"},
        "incomplete trace RNG payload",
    )
    for name, digest in value.items():
        require_hash(digest, "trace RNG " + name)


def observed_summary(rows):
    tiers, tracks, chains = set(), 0, 0
    witnesses = {}
    for micro, row in rows.items():
        validate_rng_trace(row.get("rng_before"))
        validate_rng_trace(row.get("rng_after"))
        observation = row.get("observed", {})
        tier = observation.get("dna_tier")
        require(
            type(tier) is int
            and tier in TIERS
            and row["shapes"].get("input_ids", {}).get("shape") == [1, tier],
            "invalid DNA tier observation",
        )
        for key in ("tracks_nonzero", "valid_chain_elements"):
            require(
                type(observation.get(key)) is int and observation[key] >= 0,
                "invalid evidence observation",
            )
        for key in ("tracks_sha256", "chain_sha256"):
            require_hash(observation.get(key), key)
        require_hash(row["inputs"], "trace input")
        tiers.add(tier)
        tracks += observation["tracks_nonzero"]
        chains += observation["valid_chain_elements"]
        keys = ["tier" + str(tier)]
        if observation["tracks_nonzero"]:
            keys.append("nonzero_tracks")
        if observation["valid_chain_elements"]:
            keys.append("valid_chain")
        for key in keys:
            if key not in witnesses:
                witnesses[key] = dict(micro=micro, inputs_sha256=row["inputs"], **observation)
    return dict(
        dna_tiers=sorted(tiers),
        tracks_nonzero=tracks,
        valid_chain_elements=chains,
        witnesses=witnesses,
    )


def compare_pair(root):
    root = Path(root)
    verified = verified_record(root)
    require(
        read(root / "fresh_start.json")
        == dict(
            job_id=verified["job_id"],
            verified_record_sha256=sha(root / "verified_record.json"),
            kind="b6_fresh_continuous",
            step=0,
        ),
        "unbound fresh start",
    )
    bindings = {
        name: checkpoint_binding(root / name, SNAPSHOT if name == "at400" else POSITION, verified)
        for name in ("continuous", "at400", "resumed")
    }
    require(
        bindings["continuous"]["identity"]
        == bindings["at400"]["identity"]
        == bindings["resumed"]["identity"],
        "cross-run identity mismatch",
    )
    require(
        bindings["continuous"]["fingerprint"]
        == bindings["at400"]["fingerprint"]
        == bindings["resumed"]["fingerprint"],
        "cross-run fingerprint mismatch",
    )
    schema, schema_hashes = prepared_schema(root)
    snapshot, snapshot_structure = complete_checkpoint(root / "at400", schema, SNAPSHOT)
    a, left_structure = complete_checkpoint(root / "continuous", schema, POSITION)
    b, right_structure = complete_checkpoint(root / "resumed", schema, POSITION)
    structures = dict(at400=snapshot_structure, continuous=left_structure, resumed=right_structure)
    for structure in (left_structure, right_structure):
        for key, value in snapshot_structure["optimizer_state"].items():
            require(
                structure["optimizer_state"].get(key) == value,
                "snapshot-initialized optimizer state disappeared",
            )
        require(
            structure["rng_keys"] == snapshot_structure["rng_keys"],
            "rank RNG fields changed since snapshot",
        )
    require(bool(a["model"]) and bool(b["model"]), "empty model state")
    require(
        set(a["state"]) == set(b["state"]) == base.STATE_KEYS,
        "incomplete optimizer/scheduler/rank RNG state",
    )
    different = sorted(
        k for k in set(a["model"]) | set(b["model"]) if a["model"].get(k) != b["model"].get(k)
    )
    states = {k: a["state"][k] == b["state"][k] for k in sorted(base.STATE_KEYS)}
    traces, observations, trace_hashes, row_counts = {}, {}, {}, {}
    for rank in range(4):
        x, y = (
            base._trace(root / name, rank, start_step=0, snapshot_step=400, stop_step=410)
            for name in ("continuous", "resumed")
        )
        for name, value, start in [("continuous", x, 0), ("resumed", y, 9600)]:
            require(
                set(value["forward"]) == set(range(start + 1, 9841)),
                "incomplete forward trace " + name + " rank" + str(rank),
            )
            require(
                set(value["accumulation_boundary"]) == set(range(start + 24, 9841, 24)),
                "incomplete gradient trace " + name + " rank" + str(rank),
            )
            require(
                all(
                    type(row.get("loss")) in (int, float) and math.isfinite(row["loss"])
                    for row in value["forward"].values()
                ),
                "nonfinite trace loss",
            )
            for boundary in value["accumulation_boundary"].values():
                require_hash(boundary.get("gradients"), "boundary gradients")
                validate_rng_trace(boundary.get("rng"))
            row_counts[name + "/rank" + str(rank)] = {
                key: len(value[key]) for key in ("forward", "accumulation_boundary")
            }
            observations[name + "/rank" + str(rank)] = observed_summary(value["forward"])
            trace_hashes[name + "/rank" + str(rank)] = sha(
                root / name / ("trace_rank%d.jsonl" % rank)
            )
        traces[str(rank)] = {
            key: [m for m in range(9601, 9841) if x["forward"][m][key] != y["forward"][m][key]]
            for key in (
                "inputs",
                "shapes",
                "rng_before",
                "rng_after",
                "loss",
                "context",
                "observed",
            )
        }
        for key, field in [("gradients", "gradients"), ("boundary_rng", "rng")]:
            traces[str(rank)][key] = [
                m
                for m in range(9624, 9841, 24)
                if x["accumulation_boundary"][m][field] != y["accumulation_boundary"][m][field]
            ]
    require(
        set(tier for item in observations.values() for tier in item["dna_tiers"]) == TIERS
        and sum(item["tracks_nonzero"] for item in observations.values()) > 0
        and sum(item["valid_chain_elements"] for item in observations.values()) > 0,
        "missing global nonzero evidence, valid chain or DNA tiers",
    )
    histories = {name: bindings[name]["history_sha256"] for name in ("continuous", "resumed")}
    report = dict(
        version=1,
        kind="b6_history_comparison",
        profile=PROFILE,
        arm="B6",
        source_commit=SOURCE_COMMIT,
        validation_job_id=verified["job_id"],
        verified_record_sha256=sha(root / "verified_record.json"),
        fresh_start_sha256=sha(root / "fresh_start.json"),
        checkpoint_digests=dict(continuous=probe.digest(a), resumed=probe.digest(b)),
        prepared_schema=schema,
        prepared_schema_sha256=digest_json(schema),
        prepared_schema_file_sha256=schema_hashes,
        checkpoint_state_contracts=structures,
        checkpoint_history_sha256=histories,
        checkpoint_bindings=bindings,
        model_equal=not different,
        different_model_keys=different,
        states_equal=states,
        history_equal=histories["continuous"] == histories["resumed"],
        traces=traces,
        observed_evidence=observations,
        trace_sha256=trace_hashes,
        trace_row_counts=row_counts,
        finite_losses=True,
        runtime_fingerprint_sha256=digest_json(bindings["resumed"]["fingerprint"]),
    )
    report["passed"] = (
        not different
        and all(states.values())
        and report["history_equal"]
        and all(not v for r in traces.values() for v in r.values())
    )
    return report


def accepted_report(root, report):
    verified = verified_record(root)
    contract.validate_report(report, SOURCE_COMMIT, verified["job_id"])
    # Recompute everything, including full CPU tensor bytes and trace inventories.
    # A rewritten comparison with recomputed declared hashes is never authoritative.
    require(
        digest_json(read(root / "comparison.json")) == digest_json(report),
        "comparison bytes or report changed",
    )
    require(
        digest_json(compare_pair(root)) == digest_json(report),
        "current proof differs from comparison",
    )


def receipt_value(root, report):
    verified = verified_record(root)
    binding = report["checkpoint_bindings"]["resumed"]
    return dict(
        version=1,
        kind="b6_history_validation_ready",
        source_commit=SOURCE_COMMIT,
        validation_job_id=verified["job_id"],
        comparison_sha256=sha(root / "comparison.json"),
        verified_record_sha256=sha(root / "verified_record.json"),
        provenance=verified["provenance"],
        runtime_fingerprint_sha256=report["runtime_fingerprint_sha256"],
        candidate=dict(
            relative_run="resumed",
            relative_checkpoint="latest_state",
            **POSITION,
            meta_sha256=binding["meta_sha256"],
            identity_sha256=binding["identity_sha256"],
            history_sha256=binding["history_sha256"],
            canonical_checkpoint_sha256=report["checkpoint_digests"]["resumed"]
        ),
    )


def write_validation_receipt(root, report, *, source_commit_file="/work/repo/SOURCE_COMMIT"):
    root = Path(root)
    require(
        Path(source_commit_file).read_text().strip() == SOURCE_COMMIT,
        "allocated source commit mismatch",
    )
    require(
        os.environ.get("SLURM_JOB_ID") == verified_record(root)["job_id"],
        "receipt requires current allocation",
    )
    accepted_report(root, report)
    value = receipt_value(root, report)
    contract.validate_ready(value, report, value)
    with (root / "validation_receipt.json").open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def validate_receipt(root):
    root = Path(root)
    report = read(root / "comparison.json")
    accepted_report(root, report)
    receipt = read(root / "validation_receipt.json")
    contract.validate_ready(receipt, report, receipt_value(root, report))
    return receipt


def effective_recipe(config):
    from transgenic.training.b6_runtime import load_b6_config

    return dict(load_b6_config(str(config)), _sha256=sha(config))


def verify(args):
    """Fresh allocated verification; no predecessor/checkpoint receipt exists."""
    trainer = probe.load_trainer()
    require(
        Path("/work/repo/SOURCE_COMMIT").read_text().strip() == SOURCE_COMMIT,
        "wrong allocated source",
    )
    verify_manifest(
        args.helper_manifest,
        {
            "b6_history_proof.py",
            "b6_history_proof.sbatch",
            "test_b6_history_proof.py",
            "b6_receipt_contract.py",
            "test_b6_receipt_contract.py",
            "c2/history_proof.py",
            "c2/probe.py",
            "c2/production_link.py",
        },
    )
    verify_manifest(
        args.source_manifest,
        {
            "configs/b6_400m_win_v3_wp5_pilot.json",
            "train/train_HyenaTransgenic_B6.py",
            "src/transgenic/training/b6_runtime.py",
            "src/transgenic/training/b6_compile_history.py",
            "src/transgenic/training/b6_compile_replay.py",
            "src/transgenic/training/b6_replay_state.py",
        },
    )
    paths = read(args.bindings)
    _, _, bd, _, oc, roles = trainer._wp3()
    cache = oc.ObsCache(paths["cache"])
    require(
        str(cache.mmap_dir) == args.mmap_dir and Path(cache.mmap_dir).is_dir(),
        "frozen mmap path mismatch",
    )
    require(os.environ.get("B6_OBS_CACHE_REQUIRE_PREBUILT") == "1", "prebuilt mmap required")
    record = trainer.verify_training_inputs(
        paths,
        roles=roles.load_roles(paths["roles"]),
        cache=cache,
        b6_bindings=bd,
        aug_seed=20261037,
    )
    stats = {}
    for value in list(paths.values()) + [str(cache.mmap_dir)]:
        for val in value.values() if isinstance(value, dict) else [value]:
            if not isinstance(val, str):
                continue
            path = Path(val)
            for p in path.rglob("*") if path.is_dir() else [path]:
                if p.is_file():
                    stat = p.stat()
                    stats[str(p)] = [stat.st_size, stat.st_mtime_ns]
    require(stats, "empty verified input inventory")
    value = dict(
        version=1,
        kind="b6_fresh_allocation_verification",
        source_commit=SOURCE_COMMIT,
        job_id=os.environ["SLURM_JOB_ID"],
        verified_unix=time.time(),
        record=record,
        paths=paths,
        input_stats=stats,
        mmap_dir=str(cache.mmap_dir),
        bindings_sha256=sha(args.bindings),
        recipe_sha256=sha(args.config),
        identity_from_bindings=json.loads(
            json.dumps(trainer.rt.identity_from_bindings(record, bd.canonical_record_sha256))
        ),
        recipe=effective_recipe(args.config),
        provenance=dict(
            image_sha256=sha(args.image),
            source_manifest_sha256=sha(args.source_manifest),
            helper_manifest_sha256=sha(args.helper_manifest),
        ),
    )
    probe.write_json(args.record, value)
    print("B6_INPUT_VERIFIED", record["record_sha256"], flush=True)


def run(args):
    """B6 adapter of the reviewed C2 tracing/snapshot flow; no bootstrap/reset."""
    import accelerate
    import torch
    from transgenic.training import b6_compile_replay as replay

    run_dir = Path(args.run)
    start = fresh_start(run_dir) if args.fresh else start_position(run_dir)
    require(args.stop == 410, "diagnostic stop must be410; pilot target remains4099")
    require(args.fresh == bool(args.snapshot), "only fresh continuous publishes snapshot400")
    if args.snapshot:
        require(Path(args.snapshot) == run_dir.parent / "at400", "wrong snapshot path")
    payload = verified_record(run_dir.parent, current_allocation=True)
    require(
        sha(args.bindings) == payload["bindings_sha256"]
        and sha(args.config) == payload["recipe_sha256"],
        "bindings/config changed since verification",
    )
    for path, expected in payload["input_stats"].items():
        stat = Path(path).stat()
        require(
            [stat.st_size, stat.st_mtime_ns] == expected,
            "verified input changed: " + path,
        )
    trainer = probe.load_trainer()

    def verified(paths, *, roles, cache, b6_bindings, aug_seed):
        require(paths == payload["paths"] and aug_seed == 20261037, "input identity changed")
        require(str(cache.mmap_dir) == payload["mmap_dir"], "mmap path changed")
        trainer.rt.identity_from_bindings(payload["record"], b6_bindings.canonical_record_sha256)
        return dict(payload["record"])

    trainer.verify_training_inputs = verified
    _, _, _, context, _, _ = trainer._wp3()
    original_corpus_init = context.B6Corpus.__init__

    def corpus_init(self, corpus_path, cache, expected_sha256=None, **kwargs):
        require(
            str(corpus_path) == payload["paths"]["corpus"]
            and expected_sha256 == payload["record"]["corpus_sha256"],
            "corpus changed",
        )
        return original_corpus_init(self, corpus_path, cache, expected_sha256=None, **kwargs)

    context.B6Corpus.__init__ = corpus_init
    rank = int(os.environ["RANK"])
    trace = TrainingTrace(run_dir / ("trace_rank%d.jsonl" % rank), rank, start)
    original_execute = replay.execute_replay

    @functools.wraps(original_execute)
    def execute(*values, **kwargs):
        with trace.priming():
            return original_execute(*values, **kwargs)

    replay.execute_replay = execute
    original_fingerprint = replay.runtime_fingerprint

    def fingerprint(*values, **kwargs):
        result = original_fingerprint(*values, **kwargs)
        require_operating_fingerprint(result)
        return result

    replay.runtime_fingerprint = fingerprint
    prepared = {}
    original_prepare = accelerate.Accelerator.prepare

    def prepare(self, *objects, **kwargs):
        result = original_prepare(self, *objects, **kwargs)
        model = result[0]
        require(
            self.num_processes == 4
            and isinstance(model, torch.nn.parallel.DistributedDataParallel),
            "four-rank DDP required",
        )
        trace.attach(model)
        prepared["model"] = model
        require(len(self._models) == 1, "exactly one prepared checkpoint model required")
        schema = capture_prepared_schema(
            self.get_state_dict(self._models[0], unwrap=False), result[1], result[4]
        )
        schema_path = run_dir / ("prepared_schema_rank%d.json" % rank)
        probe.write_json(schema_path, schema)
        trace.emit(
            "prepared",
            reducer=model._get_ddp_logging_data(),
            prepared_schema_sha256=sha(schema_path),
        )
        return result

    accelerate.Accelerator.prepare = prepare
    original_backward = accelerate.Accelerator.backward

    def backward(self, *values, **kwargs):
        result = original_backward(self, *values, **kwargs)
        trace.after_backward(prepared["model"])
        return result

    accelerate.Accelerator.backward = backward
    original_run = trainer.run_b6

    def bounded(spec, *values, **kwargs):
        require(
            spec.arm.name == "B6"
            and spec.stop_after_updates == 4099
            and spec.legacy_compile_bootstrap is None,
            "wrong B6 recipe",
        )
        return original_run(
            dataclasses.replace(spec, stop_at_step=410, save_every_n_steps=10), *values, **kwargs
        )

    trainer.run_b6 = bounded
    if args.snapshot:
        original_rename = os.rename

        def rename(source, destination, *values, **kwargs):
            result = original_rename(source, destination, *values, **kwargs)
            if (
                rank == 0
                and str(source) == str(run_dir / "latest_state.tmp")
                and str(destination) == str(run_dir / "latest_state")
            ):
                meta = read(Path(destination) / "meta.json")
                if meta["global_step"] == 400:
                    probe.clone(args.run, args.snapshot)
                    trace.emit(
                        "snapshot_published",
                        path=args.snapshot,
                        checkpoint={k: meta[k] for k in SNAPSHOT},
                        receipt_sha256=sha(Path(args.snapshot) / "DIAGNOSTIC_SOURCE.json"),
                    )
            return result

        os.rename = rename
    argv = [
        "--config",
        args.config,
        "--arm",
        "B6",
        "--seed",
        "123",
        "--bindings",
        args.bindings,
        "--output-dir",
        args.run,
        "--stop-after-updates",
        "4099",
        "--save-every-n-steps",
        "10",
        "--mixed-precision",
        "bf16",
        "--deterministic",
    ]
    if not args.fresh:
        argv += ["--resume", "auto"]
    try:
        result = trainer.main(argv)
        require(
            result.get("preempted") and result["global_step"] == 410 and not result.get("done"),
            "wrong diagnostic termination",
        )
        trace.emit("finished", result=result, rng=base.rng_digest())
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("verify")
    for name in (
        "bindings",
        "record",
        "mmap-dir",
        "image",
        "source-manifest",
        "helper-manifest",
        "config",
    ):
        p.add_argument("--" + name, required=True)
    p = subs.add_parser("run")
    for name in ("bindings", "record", "run", "config"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--stop", type=int, default=410)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--snapshot")
    p = subs.add_parser("clone")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    for name in ("compare", "validate-receipt", "init-fresh"):
        p = subs.add_parser(name)
        p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args)
    elif args.command == "run":
        require(
            Path(args.record) == Path(args.run).parent / "verified_record.json",
            "wrong verification record path",
        )
        run(args)
    elif args.command == "init-fresh":
        init_fresh(args.root)
    elif args.command == "clone":
        probe.clone(args.source, args.target)
    elif args.command == "compare":
        report = compare_pair(args.root)
        probe.write_json(Path(args.root) / "comparison.json", report)
        write_validation_receipt(args.root, report)
        print("B6_HISTORY_VALIDATION_READY", flush=True)
    else:
        validate_receipt(args.root)


if __name__ == "__main__":
    main()
