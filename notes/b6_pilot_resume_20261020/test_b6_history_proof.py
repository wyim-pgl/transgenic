"""CPU artifacts exercise the B6 fresh proof; GPU/Slurm execution is separate."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transgenic.training.b6_compile_history import (
    CompileHistory,
    input_signature,
    encode_rng,
)
from transgenic.training.b6_replay_state import capture_rng

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b6_proof", HERE / "b6_history_proof.py")
proof = importlib.util.module_from_spec(spec) if spec else None
if spec and Path(spec.origin).exists():
    spec.loader.exec_module(proof)
else:
    proof = None


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


class B6ProofTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(proof, "B6 fresh proof implementation is missing")
        allocation = patch.dict(os.environ, SLURM_JOB_ID="1234")
        allocation.start()
        self.addCleanup(allocation.stop)

    def fixture(self, root):
        fingerprint = dict(
            world_size=4,
            backend="inductor",
            mixed_precision="bf16",
            checkpointing="nonreentrant",
            deterministic=True,
            effective_caches={"fx_local": True, "aot_local": True},
            inductor={"force_disable_caches": False},
            recipe={"_sha256": "a" * 64},
            cache_env={
                "TORCHINDUCTOR_CACHE_DIR": "/work/compiler_cache/b6_history_v1/inductor",
                "TRITON_CACHE_DIR": "/work/compiler_cache/b6_history_v1/triton",
            },
        )
        identity = dict(
            arm="B6",
            seed=123,
            aug_seed=20261037,
            world_size=4,
            stop_after_updates=4099,
            n_pilot=4099,
            recipe_sha256="a" * 64,
            init_weights_sha256="b" * 64,
            mixed_precision="bf16",
            checkpointing="nonreentrant",
            deterministic=True,
        )
        for name, step in [("continuous", 410), ("at400", 400), ("resumed", 410)]:
            state = root / name / "latest_state/accelerate_state"
            state.mkdir(parents=True)
            history = CompileHistory(
                identity,
                fingerprint,
                types.SimpleNamespace(num_processes=4, process_index=0),
            )
            history.entries = [
                dict(
                    ordinal=0,
                    origin="observed",
                    position=dict(
                        phase="train",
                        epoch=0,
                        batch_index=0,
                        view="arm",
                        training=True,
                        grad_enabled=True,
                        optimizer_step=0,
                    ),
                    ranks=[
                        dict(
                            rank=r,
                            inputs=input_signature({"labels": torch.tensor([[1, 2]])}),
                            meta={"row": [str(r)]},
                            rng_before=encode_rng(capture_rng()),
                            delta=1,
                            activity_delta=1,
                        )
                        for r in range(4)
                    ],
                )
            ]
            position = dict(epoch=0, step=step * 24, global_step=step, pending_micro=0)
            meta = dict(
                position,
                identity_sha256=proof.digest_json(identity),
                compile_history_sha256=history.save(
                    state.parent / "compile_history.json", checkpoint=position
                ),
            )
            put(state.parent / "meta.json", meta)
            put(state.parent / "identity.json", identity)
            save_file(
                {"weight": torch.tensor([1.0, 2.0]), "buffer": torch.tensor([3.0])},
                state / "model.safetensors",
            )
            for file in ["optimizer.bin", "scheduler.bin"] + [
                f"random_states_{r}.pkl" for r in range(4)
            ]:
                torch.save({"state": torch.tensor([3.0, 4.0])}, state / file)
        snapshot = root / "at400"
        put(
            snapshot / "DIAGNOSTIC_SOURCE.json",
            dict(
                sha256={
                    str(p.relative_to(snapshot)): proof.sha(p)
                    for p in (snapshot / "latest_state").rglob("*")
                    if p.is_file()
                }
            ),
        )
        for name, start in [("continuous", 0), ("resumed", 9600)]:
            for rank in range(4):
                with (root / name / f"trace_rank{rank}.jsonl").open("w") as f:

                    def emit(row):
                        f.write(json.dumps(dict(rank=rank, **row)) + "\n")

                    emit(
                        dict(
                            event="prepared",
                            micro=start,
                            reducer=dict(rank=rank, world_size=4, backend_name="nccl"),
                        )
                    )
                    for micro in range(start + 1, 9841):
                        tier = [30720, 61440, 129024][(micro - 1) % 3]
                        emit(
                            dict(
                                event="forward",
                                micro=micro,
                                inputs="c" * 64,
                                shapes={"input_ids": {"shape": [1, tier]}},
                                rng_before={
                                    k: "d" * 64 for k in ("cpu", "python", "numpy", "cuda")
                                },
                                rng_after={k: "e" * 64 for k in ("cpu", "python", "numpy", "cuda")},
                                loss=2.0,
                                context=dict(training=True, grad_enabled=True, view="arm"),
                                observed=dict(
                                    dna_tier=tier,
                                    tracks_nonzero=1,
                                    valid_chain_elements=1,
                                    tracks_sha256="f" * 64,
                                    chain_sha256="0" * 64,
                                ),
                            )
                        )
                        if micro % 24 == 0:
                            emit(
                                dict(
                                    event="accumulation_boundary",
                                    micro=micro,
                                    gradients="1" * 64,
                                    rng={k: "2" * 64 for k in ("cpu", "python", "numpy", "cuda")},
                                )
                            )
                        if name == "continuous" and rank == 0 and micro == 9600:
                            emit(
                                dict(
                                    event="snapshot_published",
                                    micro=micro,
                                    path=str(snapshot),
                                    checkpoint=dict(
                                        epoch=0,
                                        step=9600,
                                        global_step=400,
                                        pending_micro=0,
                                    ),
                                    receipt_sha256=proof.sha(snapshot / "DIAGNOSTIC_SOURCE.json"),
                                )
                            )
                    emit(
                        dict(
                            event="finished",
                            micro=9840,
                            result=dict(preempted=True, global_step=410, history=[]),
                            rng={},
                        )
                    )
        put(
            root / "verified_record.json",
            dict(
                version=1,
                kind="b6_fresh_allocation_verification",
                job_id="1234",
                source_commit="198ce14ce53266f53b7bda5d0c4cdfd94deb43b4",
                recipe_sha256="a" * 64,
                identity_from_bindings={},
                bindings_sha256="3" * 64,
                mmap_dir="/hdd_prep/obs_cache.arrow",
                recipe=fingerprint["recipe"],
                provenance=dict(
                    image_sha256="4" * 64,
                    source_manifest_sha256="5" * 64,
                    helper_manifest_sha256="6" * 64,
                ),
                paths={},
                input_stats={},
            ),
        )
        put(
            root / "fresh_start.json",
            dict(
                job_id="1234",
                verified_record_sha256=proof.sha(root / "verified_record.json"),
                kind="b6_fresh_continuous",
                step=0,
            ),
        )
        (root / "SOURCE_COMMIT").write_text("198ce14ce53266f53b7bda5d0c4cdfd94deb43b4\n")

    def publish(self, root):
        report = proof.compare_pair(root)
        put(root / "comparison.json", report)
        return proof.write_validation_receipt(
            root, report, source_commit_file=root / "SOURCE_COMMIT"
        )

    def test_fresh_without_checkpoint_is_only_explicit_continuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            run = root / "continuous"
            run.mkdir()
            self.assertEqual(proof.start_position(run, fresh=True), 0)
            for name, fresh in [
                ("continuous", False),
                ("resumed", True),
                ("resumed", False),
            ]:
                with self.subTest(name=name, fresh=fresh), self.assertRaises(
                    (ValueError, FileNotFoundError)
                ):
                    proof.start_position(root / name, fresh=fresh)
            (run / "leftover").write_text("x")
            with self.assertRaises(ValueError):
                proof.start_position(run, fresh=True)

    def test_complete_cpu_fixture_publishes_b6_without_c2_witnesses(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SLURM_JOB_ID="1234"):
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            receipt = self.publish(root)
            self.assertEqual(receipt["kind"], "b6_history_validation_ready")
            self.assertEqual(receipt["candidate"]["global_step"], 410)
            self.assertEqual(receipt["candidate"]["step"], 9840)
            self.assertEqual(receipt["comparison_sha256"], proof.sha(root / "comparison.json"))
            self.assertEqual(
                receipt["candidate"]["identity_sha256"],
                proof.sha(root / "resumed/latest_state/identity.json"),
            )
            self.assertNotIn("predecessor_job_id", receipt)
            self.assertEqual(proof.validate_receipt(root), receipt)

    def test_checkpoint_identity_history_and_cache_rejections(self):
        mutations = [
            ("identity", "arm", "C2"),
            ("identity", "stop_after_updates", 410),
            ("identity", "world_size", 1),
            (
                "history",
                "bootstrap",
                {"kind": "reconstructed_legacy_compile_bootstrap"},
            ),
            ("history", "entries", []),
            ("fingerprint", "cache_env", {}),
            ("fingerprint", "recipe", {"_sha256": "9" * 64}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            for target, key, value in mutations:
                with self.subTest(target=target, key=key):
                    checkpoint = root / "resumed/latest_state"
                    files = {p: p.read_bytes() for p in checkpoint.glob("*.json")}
                    path = checkpoint / (
                        "identity.json" if target == "identity" else "compile_history.json"
                    )
                    obj = json.loads(path.read_text())
                    (obj["fingerprint"] if target == "fingerprint" else obj)[key] = value
                    put(path, obj)
                    if target != "identity":
                        meta = json.loads((checkpoint / "meta.json").read_text())
                        meta["compile_history_sha256"] = proof.sha(path)
                        put(checkpoint / "meta.json", meta)
                    with self.assertRaises(ValueError):
                        proof.compare_pair(root)
                    for p, raw in files.items():
                        p.write_bytes(raw)

    def test_model_optimizer_rng_and_history_mismatch_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SLURM_JOB_ID="1234"):
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            for name in ["model.safetensors", "optimizer.bin", "random_states_3.pkl"]:
                path = root / "resumed/latest_state/accelerate_state" / name
                original = path.read_bytes()
                if name.endswith("safetensors"):
                    save_file({"weight": torch.tensor([1.0, 9.0])}, path)
                else:
                    torch.save({"state": torch.tensor([99.0])}, path)
                with self.subTest(name=name):
                    report = proof.compare_pair(root)
                    self.assertFalse(report["passed"])
                    put(root / "comparison.json", report)
                    with self.assertRaises(ValueError):
                        proof.write_validation_receipt(
                            root, report, source_commit_file=root / "SOURCE_COMMIT"
                        )
                    self.assertFalse((root / "validation_receipt.json").exists())
                path.write_bytes(original)
            path = root / "resumed/latest_state/compile_history.json"
            obj = json.loads(path.read_text())
            obj["entries"][0]["ranks"][0]["delta"] = 2
            put(path, obj)
            meta = json.loads(path.with_name("meta.json").read_text())
            meta["compile_history_sha256"] = proof.sha(path)
            put(path.with_name("meta.json"), meta)
            self.assertFalse(proof.compare_pair(root)["passed"])

    def test_missing_trace_nonfinite_and_absent_evidence_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            path = root / "continuous/trace_rank2.jsonl"
            original = path.read_text()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                proof.compare_pair(root)
            path.write_text(original)
            rows = [json.loads(x) for x in original.splitlines()]
            for mode in ["missing_row", "nonfinite", "tier"]:
                bad = copy.deepcopy(rows)
                if mode == "missing_row":
                    bad.pop(2)
                elif mode == "nonfinite":
                    bad[1]["loss"] = float("nan")
                else:
                    for row in bad:
                        if row["event"] == "forward":
                            if mode == "tier":
                                row["observed"]["dna_tier"] = 30720
                            else:
                                row["observed"][
                                    "tracks_nonzero" if mode == "tracks" else "valid_chain_elements"
                                ] = 0
                path.write_text("".join(json.dumps(r) + "\n" for r in bad))
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    proof.compare_pair(root)
            path.write_text(original)

    def test_receipt_rechecks_current_bytes_and_does_not_trust_rehashed_lies(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SLURM_JOB_ID="1234"):
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            receipt = self.publish(root)
            for name in [
                "comparison.json",
                "resumed/latest_state/meta.json",
                "resumed/latest_state/identity.json",
                "resumed/latest_state/compile_history.json",
                "resumed/latest_state/accelerate_state/optimizer.bin",
            ]:
                p = root / name
                original = p.read_bytes()
                p.write_bytes(original + b" ")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    proof.validate_receipt(root)
                p.write_bytes(original)
            report = json.loads((root / "comparison.json").read_text())
            report["model_equal"] = False
            put(root / "comparison.json", report)
            receipt["comparison_sha256"] = proof.sha(root / "comparison.json")
            put(root / "validation_receipt.json", receipt)
            with self.assertRaises(ValueError):
                proof.validate_receipt(root)

    def test_priming_does_not_consume_real_counters_and_observations_use_real_inputs(
        self,
    ):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, **inputs):
                return types.SimpleNamespace(loss=self.weight * inputs["evidence_tracks"].sum())

        inputs = dict(
            input_ids=torch.zeros((1, 30720), dtype=torch.long),
            evidence_tracks=torch.ones((1, 3, 31)),
            chain_mask=torch.tensor([[False, True]]),
            chain_idx=torch.tensor([[[0, 0], [1, 2]]]),
            chain_feat=torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int16),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            model = Model()
            trace = proof.TrainingTrace(path, 0, 23)
            trace.attach(model)
            with trace.priming():
                model(**inputs).loss.backward()
                trace.after_backward(model)
            model.zero_grad()
            model(**inputs).loss.backward()
            trace.after_backward(model)
            trace.close()
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual([r["micro"] for r in rows], [24, 24])
            self.assertEqual(rows[0]["observed"]["tracks_nonzero"], 93)
            self.assertEqual(rows[0]["observed"]["valid_chain_elements"], 1)

    def test_globally_absent_tracks_or_chain_refuses_ready(self):
        for field in ("tracks_nonzero", "valid_chain_elements"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "1234"
                root.mkdir()
                self.fixture(root)
                for phase in ("continuous", "resumed"):
                    for rank in range(4):
                        path = root / phase / f"trace_rank{rank}.jsonl"
                        rows = [json.loads(x) for x in path.read_text().splitlines()]
                        for row in rows:
                            if row["event"] == "forward":
                                row["observed"][field] = 0
                        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    proof.compare_pair(root)
                self.assertFalse((root / "validation_receipt.json").exists())

    def test_rank_with_no_biological_evidence_does_not_invent_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            for phase in ("continuous", "resumed"):
                path = root / phase / "trace_rank2.jsonl"
                rows = [json.loads(x) for x in path.read_text().splitlines()]
                for row in rows:
                    if row["event"] == "forward":
                        row["observed"]["tracks_nonzero"] = 0
                        row["observed"]["valid_chain_elements"] = 0
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertTrue(proof.compare_pair(root)["passed"])

    def test_fresh_ticket_reserves_once_and_all_four_ranks_can_join(self):
        self.assertTrue(
            callable(getattr(proof, "init_fresh", None)),
            "fresh launch ticket is missing",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SLURM_JOB_ID="1234"):
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            # The actual launch starts without continuous artifacts.
            import shutil

            shutil.rmtree(root / "continuous")
            (root / "fresh_start.json").unlink()
            proof.init_fresh(root)
            (root / "continuous/trace_rank0.jsonl").write_text("")
            self.assertEqual(proof.fresh_start(root / "continuous"), 0)
            with self.assertRaises(FileExistsError):
                proof.init_fresh(root)
            (root / "continuous/latest_state").mkdir()
            with self.assertRaises(ValueError):
                proof.fresh_start(root / "continuous")

    def test_manifest_requires_executed_members_and_checks_bytes(self):
        self.assertTrue(
            callable(getattr(proof, "verify_manifest", None)),
            "manifest coverage verification is missing",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            (root / "helper.py").write_text("pass\n")
            manifest = root / "manifest.sha256"
            manifest.write_text(proof.sha(root / "helper.py") + "  helper.py\n")
            proof.verify_manifest(manifest, {"helper.py"})
            with self.assertRaises(ValueError):
                proof.verify_manifest(manifest, {"helper.py", "batch.sbatch"})
            (root / "helper.py").write_text("changed")
            with self.assertRaises(ValueError):
                proof.verify_manifest(manifest, {"helper.py"})

    def test_recipe_payload_and_embedded_identity_digest_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            path = root / "verified_record.json"
            original = path.read_bytes()
            value = json.loads(original)
            value["recipe"]["lr"] = 999
            put(path, value)
            with self.assertRaises(ValueError):
                proof.compare_pair(root)
            path.write_bytes(original)
            path = root / "resumed/latest_state/meta.json"
            value = json.loads(path.read_text())
            value["identity_sha256"] = "0" * 64
            put(path, value)
            with self.assertRaises(ValueError):
                proof.compare_pair(root)

    def test_empty_rng_trace_payload_cannot_pass_by_equal_omission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            for phase in ("continuous", "resumed"):
                path = root / phase / "trace_rank0.jsonl"
                rows = [json.loads(x) for x in path.read_text().splitlines()]
                for row in rows:
                    if row["event"] == "forward":
                        row["rng_before"] = {}
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                proof.compare_pair(root)

    def test_independent_later_allocation_can_recheck_but_cannot_republish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "1234"
            root.mkdir()
            self.fixture(root)
            receipt = self.publish(root)
            with patch.dict(os.environ, SLURM_JOB_ID="5678"):
                self.assertEqual(proof.validate_receipt(root), receipt)
                with self.assertRaises(ValueError):
                    proof.write_validation_receipt(
                        root,
                        proof.read(root / "comparison.json"),
                        source_commit_file=root / "SOURCE_COMMIT",
                    )

    def test_host_copy_peak_gate_survives_optimized_python(self):
        code = (
            (HERE / "b6_history_proof.sbatch")
            .read_text()
            .split("<<'PY'\n", 1)[1]
            .split("\nPY", 1)[0]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            state = reference / "accelerate_state"
            state.mkdir(parents=True)
            for name in ["model.safetensors", "optimizer.bin", "scheduler.bin"] + [
                f"random_states_{r}.pkl" for r in range(4)
            ]:
                (state / name).write_bytes(b"x")
            (reference / "meta.json").write_text("{}")
            (reference / "identity.json").write_text("{}")
            storage = dict(
                project_id=22168,
                pool="ddn_ssd",
                quota_remaining_bytes=10 * 1024 ** 3,
                filesystem_available_bytes=10 * 1024 ** 3,
            )
            put(root / "storage_preflight.json", storage)
            result = subprocess.run(
                [os.sys.executable, "-O", "-c", code, str(root), str(reference)],
                capture_output=True,
            )
            self.assertNotEqual(
                result.returncode, 0, "wrong project must fail even with PYTHONOPTIMIZE"
            )
            self.assertFalse((root / "proof_copy_peak.json").exists())
            storage["project_id"] = 22169
            put(root / "storage_preflight.json", storage)
            subprocess.run(
                [os.sys.executable, "-O", "-c", code, str(root), str(reference)], check=True
            )
            peak = json.loads((root / "proof_copy_peak.json").read_text())
            self.assertEqual(peak["checkpoint_bytes"], 11)
            self.assertEqual(peak["proof_copy_peak_bytes"], 2147483725)
            (root / "proof_copy_peak.json").unlink()
            storage["quota_remaining_bytes"] = 100
            put(root / "storage_preflight.json", storage)
            result = subprocess.run(
                [os.sys.executable, "-O", "-c", code, str(root), str(reference)],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "proof_copy_peak.json").exists())

    def test_verified_recipe_keeps_source_loader_defaults_and_full_schedule(self):
        self.assertTrue(
            callable(getattr(proof, "effective_recipe", None)),
            "effective recipe binding is missing",
        )
        from transgenic.training import b6_runtime

        path = (
            Path(b6_runtime.__file__).resolve().parents[3] / "configs/b6_400m_win_v3_wp5_pilot.json"
        )
        recipe = proof.effective_recipe(path)
        self.assertEqual(recipe["lr_warmup_fraction"], 0.05)
        self.assertEqual(recipe["n_pilot"], 4099)
        self.assertEqual(recipe["wp5_pilot"]["stop_after_updates"], 4099)
        self.assertEqual(recipe["_sha256"], proof.sha(path))

    def test_cli_help_and_batch_parse_without_cwd_dependency(self):
        subprocess.run(["bash", "-n", str(HERE / "b6_history_proof.sbatch")], check=True)
        subprocess.run(
            [os.sys.executable, str(HERE / "b6_history_proof.py"), "--help"],
            cwd="/tmp",
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    unittest.main()
