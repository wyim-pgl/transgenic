from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader, Dataset

from transgenic.training import mvp_finalize as finalize

ROOT = Path(__file__).parents[1]
PYTHON = "/home/wyim/miniconda3/envs/pdb/bin/python"
TORCHRUN = str(Path(PYTHON).with_name("torchrun"))
DEPS = ROOT / ".testdeps"
DUCKDB_DEPS = Path("/tmp/b6-dataset-verify.D2Is9a/.testdeps")
SHA = "1" * 64


def initialization_audit():
    snapshot = {"sha256": SHA, "size": 1, "device": 1, "inode": 1, "mtime_ns": 1}
    weights = {"path": "/tmp/source/epoch/model.safetensors",
               "before": dict(snapshot), "after": dict(snapshot)}
    run_config = {"path": "/tmp/source/run_config.json",
                  "before": dict(snapshot), "after": dict(snapshot)}
    source_fields = {name: {"value": None, "provenance": "config-default"}
                     for name in finalize._AUDIT_SEMANTIC_FIELDS}
    target_fields = {name: {"value": None, "provenance": "target-config"}
                     for name in finalize._AUDIT_SEMANTIC_FIELDS}
    source_fields["evidence_channels"]["value"] = 0
    target_fields["evidence_channels"]["value"] = 8
    digest = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit = {
        "schema_version": "b6-weights-init-v1", "mode": "weights-only",
        "requested_path": "/tmp/source/epoch", "resolved_checkpoint_dir": "/tmp/source/epoch",
        "resolved_weights_path": "/tmp/source/epoch/model.safetensors",
        "files": {"model.safetensors": weights, "run_config.json": run_config,
                  "runtime_identity.txt": None, "config.json": None,
                  "meta.json": None, "runtime.json": None},
        "recipe": {"canonical_sha256": SHA},
        "source_semantics": {"fields": source_fields, "sha256": digest(source_fields)},
        "target_semantics": {"fields": target_fields, "sha256": digest(target_fields)},
        "implementation": {
            "b5_runtime.py": {"path": "/tmp/source/b5_runtime.py", "sha256": SHA},
            "configuration_transgenic.py": {"path": "/tmp/source/configuration_transgenic.py", "sha256": SHA}},
        "compatibility": {"transition": "evidence_channels:0->8",
            "exception": "exactly missing transgenic.encoder.evidence_proj.weight; initialized to verified zeros",
            "same8_supported": False},
        "load": {"missing_keys": ["transgenic.encoder.evidence_proj.weight"],
                 "unexpected_keys": [], "safetensors_metadata": None},
        "provenance_caveat": "recorded loader output; source bytes are not re-certified here",
    }
    return {"audit": audit, "audit_sha256": hashlib.sha256(finalize._canonical_json(audit)).hexdigest(),
            "verification": "recorded"}


def recipe(scope="test", world=1):
    if scope == "production":
        return {"max_updates": 250, "warmup_updates": 12, "batch_size": 1,
                "accumulation_steps": 24, "required_world_size": 4,
                "backbone_lr": 2e-5, "projection_lr": 2e-4,
                "weight_decay": 0.02, "clip_grad_norm": 1.0}
    return {"max_updates": 2, "warmup_updates": 1, "batch_size": 1,
            "accumulation_steps": 2 if world == 1 else 1, "required_world_size": world,
            "backbone_lr": 2e-5, "projection_lr": 2e-4,
            "weight_decay": 0.02, "clip_grad_norm": 1.0}


def metadata(scope="test", world=1):
    seed = 1
    return {
        "identity_scope": scope,
        "initialization_audit": initialization_audit(),
        "data_identities": {
            name: {"sha256": SHA, "verification": "recorded"}
            for name in ("corpus", "evidence", "panel")
        },
        "sif_code_identity": {
            "sif": {"sha256": SHA, "verification": "recorded"},
            "code": {"sha256": SHA, "verification": "recorded"},
        },
        "runtime_packages": {name: "test-version" for name in
                             ("python", "torch", "transformers", "accelerate", "safetensors")},
        "normalized_recipe": recipe(scope, world),
        "seed_augmentation": {"seed": seed, "aug_seed": 20260914 + seed},
    }


def summary(scope="test"):
    if scope == "production":
        return {"successful_updates": 250, "effective_batch_size": 96}
    return {"successful_updates": 2, "effective_batch_size": 2}


class Config:
    def to_dict(self):
        return {"model_type": "toy", "hidden_size": 2}


class SaveToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(2, 2, bias=False)
        self.alias = self.shared
        self.config = Config()
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None, evidence_tracks=None):
        self.forward_calls += 1
        signal = evidence_tracks.float().mean(dim=(1, 2)) * self.shared.weight[0, 0]
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 273))
        logits[..., 2] = signal[:, None]
        loss = -logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).mean()
        return SimpleNamespace(logits=logits, loss=loss)


class OneDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (torch.tensor([0]), torch.ones(1, dtype=torch.bool), torch.tensor([2]),
                index, "chr", 0, 1, torch.tensor(1.0), torch.ones((1, 8)),
                {"rn": 11, "has_evidence": True})


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


class LocalAccelerator:
    num_processes = 1
    process_index = 0
    is_main_process = True
    device = torch.device("cpu")
    distributed_type = SimpleNamespace(name="NO")

    def unwrap_model(self, model, keep_fp32_wrapper=True):
        assert keep_fp32_wrapper is True
        return model

    def wait_for_everyone(self):
        pass


def publish(tmp_path, *, final_name="final", md=None, train=None):
    model = SaveToy()
    loader = DataLoader(EmptyDataset(), batch_size=1)
    paired = {"primary_delta": 1.0, "composite_delta": 2.0, "pair_count": 1,
              "records": [{"rn": 1, "has_evidence": True, "loss_off": 2.0,
                           "loss_on": 1.0, "composite_loss_off": 3.0,
                           "composite_loss_on": 1.0, "ce_numerator_off": 2.0,
                           "ce_denominator_off": 1.0, "ce_numerator_on": 1.0,
                           "ce_denominator_on": 1.0}]}
    return finalize._publish_final(LocalAccelerator(), model, paired, tmp_path / final_name,
                                   metadata=md or metadata(), training_summary=train or summary())


def test_single_rank_ordinary_loader_preserves_mode_and_publishes(tmp_path):
    model = SaveToy().eval()
    result = finalize.evaluate_and_publish_final(
        LocalAccelerator(), model, DataLoader(OneDataset(), batch_size=1), [11], tmp_path / "final",
        metadata=metadata(), training_summary=summary())
    assert [row["rn"] for row in result["records"]] == [11]
    assert result["pair_count"] == 1 and model.forward_calls == 2 and not model.training
    assert (tmp_path / "final" / "TRAINING_DONE").is_file()


def test_prepublication_check_runs_after_aggregation_before_filesystem(tmp_path):
    final = tmp_path / "final"
    observed = []

    def check():
        observed.append(not final.exists() and not list(tmp_path.glob(".final.staging-*")))

    result = finalize.evaluate_and_publish_final(
        LocalAccelerator(), SaveToy(), DataLoader(OneDataset(), batch_size=1), [11], final,
        metadata=metadata(), training_summary=summary(), prepublication_check=check)
    assert observed == [True]
    assert result["pair_count"] == 1 and final.is_dir()


def test_prepublication_check_failure_leaves_no_artifact(tmp_path):
    final = tmp_path / "final"

    def reject():
        raise ValueError("identity changed during evaluation")

    with pytest.raises(ValueError, match="identity changed during evaluation"):
        finalize.evaluate_and_publish_final(
            LocalAccelerator(), SaveToy(), DataLoader(OneDataset(), batch_size=1), [11], final,
            metadata=metadata(), training_summary=summary(), prepublication_check=reject)
    assert not final.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("distributed_type", ["FSDP", "DEEPSPEED"])
def test_sharded_distributed_types_are_rejected(distributed_type):
    accelerator = LocalAccelerator()
    accelerator.distributed_type = SimpleNamespace(name=distributed_type)
    with pytest.raises(ValueError, match="replicated DDP"):
        finalize._validate_evaluation_inputs(
            accelerator, SaveToy(), DataLoader(OneDataset(), batch_size=1), [11], metadata(), summary())


def test_metadata_schema_and_production_budget_fail_closed(tmp_path):
    bad = metadata()
    del bad["data_identities"]["panel"]
    with pytest.raises(ValueError, match="panel"):
        publish(tmp_path, md=bad)
    bad = metadata()
    bad["data_identities"]["corpus"]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA256"):
        publish(tmp_path, md=bad)
    bad = metadata()
    bad["runtime_packages"] = {"x": "1"}
    with pytest.raises(ValueError, match="runtime_packages"):
        publish(tmp_path, md=bad)
    bad = metadata()
    bad["normalized_recipe"] = {"x": 1}
    with pytest.raises(ValueError, match="normalized_recipe"):
        publish(tmp_path, md=bad)
    bad = metadata()
    bad["initialization_audit"]["audit"]["load"]["missing_keys"] = []
    bad["initialization_audit"]["audit_sha256"] = hashlib.sha256(
        finalize._canonical_json(bad["initialization_audit"]["audit"])).hexdigest()
    with pytest.raises(ValueError, match="initialization_audit load"):
        publish(tmp_path, md=bad)
    with pytest.raises(ValueError, match="250.*96"):
        publish(tmp_path, md=metadata("production"), train=summary("test"))
    finalize._validate_metadata(metadata("production"), summary("production"))
    assert not any(tmp_path.iterdir())


def test_expected_inventory_rejects_arbitrary_iterables_without_consuming(tmp_path):
    class ThrowingIterable:
        def __iter__(self):
            raise AssertionError("must not consume")
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("list subclass must not be consumed")
    candidates = [object(), ThrowingIterable(), (value for value in itertools.count()), HostileList([11])]
    for index, candidate in enumerate(candidates):
        model = SaveToy()
        with pytest.raises(ValueError, match="list, tuple, or range"):
            finalize.evaluate_and_publish_final(
                LocalAccelerator(), model, DataLoader(OneDataset(), batch_size=1), candidate,
                tmp_path / f"final-{index}", metadata=metadata(), training_summary=summary())
        assert model.forward_calls == 0


def test_initialization_audit_requires_full_snapshots_and_exact_semantics():
    for mutation, match in (("snapshot", "snapshot"), ("semantics", "semantic field set")):
        md = metadata()
        audit = md["initialization_audit"]["audit"]
        if mutation == "snapshot":
            audit["files"]["model.safetensors"]["before"].pop("size")
            audit["files"]["model.safetensors"]["after"].pop("size")
        else:
            for side in ("source_semantics", "target_semantics"):
                audit[side]["fields"].pop("dropout")
                audit[side]["sha256"] = hashlib.sha256(json.dumps(
                    audit[side]["fields"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        md["initialization_audit"]["audit_sha256"] = hashlib.sha256(
            finalize._canonical_json(audit)).hexdigest()
        with pytest.raises(ValueError, match=match):
            finalize._validate_metadata(md, summary())


def test_real_initialization_audit_rejects_nested_key_drift_and_bad_metadata():
    from tests.test_weights_init import WeightsInitTests
    case = WeightsInitTests("test_success_from_best_is_alias_safe_zeroes_projection_and_audits_hashes")
    case.setUp()
    try:
        original = case._load()
        mutations = [
            ("recipe missing", lambda audit: audit["recipe"].pop("canonical_sha256")),
            ("recipe extra", lambda audit: audit["recipe"].update(extra="x")),
            *[(f"compatibility missing {key}", lambda audit, key=key: audit["compatibility"].pop(key))
              for key in ("transition", "exception", "same8_supported")],
            ("compatibility extra", lambda audit: audit["compatibility"].update(extra="x")),
            *[(f"load missing {key}", lambda audit, key=key: audit["load"].pop(key))
              for key in ("missing_keys", "unexpected_keys", "safetensors_metadata")],
            ("load extra", lambda audit: audit["load"].update(extra="x")),
            ("load bad metadata", lambda audit: audit["load"].update(safetensors_metadata={"key": 1})),
            *[(f"implementation missing {key}",
               lambda audit, key=key: audit["implementation"]["b5_runtime.py"].pop(key))
              for key in ("path", "sha256")],
            ("implementation record extra", lambda audit:
                audit["implementation"]["b5_runtime.py"].update(extra="x")),
            ("implementation filename missing", lambda audit:
                audit["implementation"].pop("b5_runtime.py")),
            ("implementation filename extra", lambda audit:
                audit["implementation"].update({"extra.py": {"path": "/tmp/x", "sha256": SHA}})),
        ]
        for _, mutate in mutations:
            audit = copy.deepcopy(original)
            mutate(audit)
            md = metadata()
            md["initialization_audit"] = {
                "audit": audit,
                "audit_sha256": hashlib.sha256(finalize._canonical_json(audit)).hexdigest(),
                "verification": "recorded"}
            with pytest.raises(ValueError, match="initialization_audit"):
                finalize._validate_metadata(md, summary())
    finally:
        case.doCleanups()


def test_real_load_weights_only_audit_is_accepted_by_finalize(tmp_path):
    from tests.test_weights_init import WeightsInitTests
    case = WeightsInitTests("test_success_from_best_is_alias_safe_zeroes_projection_and_audits_hashes")
    case.setUp()
    try:
        audit = case._load()
        md = metadata()
        md["initialization_audit"] = {
            "audit": audit, "audit_sha256": hashlib.sha256(finalize._canonical_json(audit)).hexdigest(),
            "verification": "recorded"}
        result = finalize.evaluate_and_publish_final(
            LocalAccelerator(), SaveToy(), DataLoader(OneDataset(), batch_size=1), [11],
            tmp_path / "final", metadata=md, training_summary=summary())
        assert result["pair_count"] == 1 and (tmp_path / "final" / "TRAINING_DONE").is_file()
    finally:
        case.doCleanups()


def test_status_schema_marks_utf8_diagnostic_truncation():
    status = finalize._status(ValueError("é" * 10_000))
    assert status["truncated"] is True
    assert status["message"].endswith("...[diagnostic truncated]")
    assert len(status["message"].encode()) <= finalize.MVP_STATUS_MESSAGE_MAX_BYTES
    assert finalize._decode_status(finalize._status_bytes(status)) == status


def test_transport_bounds_check_sum_and_padded_size_at_exact_boundary(monkeypatch):
    monkeypatch.setattr(finalize, "MVP_MAX_SERIALIZED_BYTES", 6)
    assert finalize._validate_transport_lengths([3, 3]) == 3
    with pytest.raises(ValueError, match="aggregate"):
        finalize._validate_transport_lengths([4, 3])
    monkeypatch.setattr(finalize, "MVP_MAX_SERIALIZED_BYTES", 5)
    with pytest.raises(ValueError, match="padded"):
        finalize._validate_transport_lengths([3, 2])


def test_nonfinite_json_and_bounds_are_rejected_before_staging(tmp_path, monkeypatch):
    bad = metadata()
    bad["normalized_recipe"]["lr"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        publish(tmp_path, md=bad)
    monkeypatch.setattr(finalize, "MVP_MAX_RECORDS", 0)
    with pytest.raises(ValueError, match="MVP_MAX_RECORDS"):
        publish(tmp_path)
    assert not any(tmp_path.iterdir())


def test_atomic_payload_manifest_marker_and_tied_reload(tmp_path):
    result = publish(tmp_path)
    final = tmp_path / "final"
    assert result["primary_delta"] == 1.0
    assert (final / "TRAINING_DONE").is_file()
    names = {"metadata.json", "paired_records.jsonl", "paired_summary.json",
             "model.safetensors", "config.json"}
    manifest = json.loads((final / "manifest.json").read_text())
    assert set(manifest["files"]) == names
    for name in names:
        entry = manifest["files"][name]
        assert entry["size"] == (final / name).stat().st_size
        assert len(entry["sha256"]) == 64
    assert json.loads((final / "metadata.json").read_text())["training_summary"] == summary()
    restored = SaveToy()
    missing, unexpected = load_model(restored, final / "model.safetensors")
    assert not missing and not unexpected
    assert restored.shared is restored.alias
    assert not list(tmp_path.glob(".final.staging-*"))


@pytest.mark.parametrize("existing", ["file", "empty", "nonempty", "symlink"])
def test_existing_destinations_are_unchanged(tmp_path, existing):
    final = tmp_path / "final"
    if existing == "file":
        final.write_text("keep")
    elif existing == "empty":
        final.mkdir()
    elif existing == "nonempty":
        final.mkdir(); (final / "keep").write_text("keep")
    else:
        target = tmp_path / "target"; target.write_text("keep"); final.symlink_to(target)
    before = (final.lstat().st_mode, final.read_text() if final.is_file() else None)
    with pytest.raises(FileExistsError):
        publish(tmp_path)
    assert (final.lstat().st_mode, final.read_text() if final.is_file() else None) == before


@pytest.mark.parametrize("point", ["payload", "manifest", "marker", "rename"])
def test_prerename_faults_leave_no_final_or_staging(tmp_path, monkeypatch, point):
    target = {"payload": "_write_payload_files", "manifest": "_write_manifest",
              "marker": "_write_done_marker", "rename": "_rename_noreplace"}[point]
    original = getattr(finalize, target)
    def fail(*args, **kwargs):
        raise OSError(f"injected {point}")
    monkeypatch.setattr(finalize, target, fail)
    with pytest.raises(OSError, match=point):
        publish(tmp_path)
    assert not (tmp_path / "final").exists()
    assert not list(tmp_path.glob(".final.staging-*"))
    monkeypatch.setattr(finalize, target, original)


def test_race_at_rename_never_overwrites_winner(tmp_path, monkeypatch):
    def race(source, destination):
        destination.mkdir()
        (destination / "winner").write_text("keep")
        raise FileExistsError(destination)
    monkeypatch.setattr(finalize, "_rename_noreplace", race)
    with pytest.raises(FileExistsError):
        publish(tmp_path)
    assert (tmp_path / "final" / "winner").read_text() == "keep"
    assert not list(tmp_path.glob(".final.staging-*"))


def test_concurrent_publishers_have_exactly_one_winner(tmp_path):
    outcomes = []
    def worker():
        try:
            publish(tmp_path)
            outcomes.append("won")
        except FileExistsError:
            outcomes.append("lost")
    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sorted(outcomes) == ["lost", "won"]
    assert (tmp_path / "final" / "TRAINING_DONE").is_file()


def test_parent_fsync_failure_is_committed_and_never_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize, "_fsync_parent", lambda path: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(finalize.PublicationDurabilityError) as caught:
        publish(tmp_path)
    assert caught.value.committed is True and caught.value.final_dir == tmp_path / "final"
    assert (tmp_path / "final" / "TRAINING_DONE").is_file()
    with pytest.raises(FileExistsError):
        publish(tmp_path)


def test_unknown_save_wrapper_is_rejected(tmp_path):
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.module = SaveToy()
    paired = {"primary_delta": 0.0, "composite_delta": 0.0, "pair_count": 0, "records": []}
    with pytest.raises(ValueError, match="wrapper"):
        finalize._publish_final(LocalAccelerator(), Wrapper(), paired, tmp_path / "final",
                                metadata=metadata(), training_summary=summary())


def probe(tmp_path, case, ranks=2, timeout=120):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(DEPS), str(DUCKDB_DEPS)))
    env["MVP_FINALIZE_TEST_DIR"] = str(tmp_path)
    command = [TORCHRUN, "--standalone", f"--nproc-per-node={ranks}",
               str(ROOT / "tests" / "mvp_finalize_probe.py"), case]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                            timeout=timeout)
    assert result.returncode == 0, f"{command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert records
    return records[-1]


@pytest.mark.parametrize("size", [3, 1])
def test_real_two_rank_gloo_padded_paired_evaluation_and_compiled_path(tmp_path, size):
    result = probe(tmp_path, f"success-{size}")
    assert result["world_size"] == 2
    assert result["rns"] == list(range(size))
    assert result["pair_count"] == size
    expected = sum(math.log(273) - (math.log(math.exp(0.05 * (i + 1)) + 272) - 0.05 * (i + 1))
                   for i in range(size)) / size
    assert result["primary_delta"] == pytest.approx(expected, rel=2e-5)
    assert result["composite_delta"] == pytest.approx(expected, rel=2e-5)
    assert result["record_fields"] == ["ce_denominator_off", "ce_denominator_on",
        "ce_numerator_off", "ce_numerator_on", "composite_loss_off", "composite_loss_on",
        "has_evidence", "loss_off", "loss_on", "rn"]
    assert result["same_summary"] and result["backend_calls"] > 0
    assert result["compiled_forwards"] == 2 * ((size + 1) // 2)
    assert result["saved_keys_clean"] and result["tied_reload"] and result["marker"]


@pytest.mark.parametrize("case", ["missing", "extra", "inconsistent", "bad_sampler"])
def test_collective_inventory_and_loader_failures_prevent_publication(tmp_path, case):
    result = probe(tmp_path, case)
    assert result == {"all_failed": True, "case": case, "final_exists": False}


@pytest.mark.parametrize("case", ["inventory-noniterable", "inventory-throwing",
                                  "inventory-infinite", "inventory-subclass"])
def test_one_rank_arbitrary_inventory_rejects_promptly_on_all_ranks(tmp_path, case):
    result = probe(tmp_path, case, timeout=30)
    assert result["all_failed"] and result["same_error"] and not result["final_exists"]
    assert result["error"][0] == "ValueError" and "list, tuple, or range" in result["error"][1]


def test_huge_status_is_bounded_identically_without_object_collectives(tmp_path):
    result = probe(tmp_path, "huge-status", timeout=30)
    assert result["all_failed"] and result["same_error"] and not result["final_exists"]
    assert result["object_calls"] == 0 and result["error"][0] == "ValueError"
    assert result["error"][1].endswith("...[diagnostic truncated]")
    assert len(result["error"][1].encode()) <= finalize.MVP_STATUS_MESSAGE_MAX_BYTES


def test_collective_byte_cap_rejects_before_payload_collective(tmp_path):
    result = probe(tmp_path, "byte-cap")
    assert result == {"all_failed": True, "case": "byte-cap", "payload_calls": 0,
                      "final_exists": False}


@pytest.mark.parametrize("case", ["publish-write-fail", "publish-rename-fail"])
def test_collective_prerename_publication_errors_are_identical_and_leave_no_final(tmp_path, case):
    result = probe(tmp_path, case)
    assert result["case"] == case and result["all_failed"] and result["same_error"]
    assert result["error"][0] == "OSError" and case.split("-")[1] in result["error"][1]
    assert not result["error"][2] and not result["final_exists"]


def test_collective_postrename_durability_error_has_committed_status_on_all_ranks(tmp_path):
    result = probe(tmp_path, "durability")
    assert result == {"all_committed": True, "case": "durability", "marker": True}
