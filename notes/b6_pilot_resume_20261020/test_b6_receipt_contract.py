"""Small acceptance predicates are usable by the Python3.6 host controller."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
PATH = HERE / "b6_receipt_contract.py"
contract = None
if PATH.exists():
    spec = importlib.util.spec_from_file_location("contract_under_test", str(PATH))
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
SOURCE = "198ce14ce53266f53b7bda5d0c4cdfd94deb43b4"
H = "a" * 64


def fixture():
    identity = dict(
        arm="B6",
        seed=123,
        aug_seed=20261037,
        world_size=4,
        n_pilot=4099,
        stop_after_updates=4099,
        mixed_precision="bf16",
        checkpointing="nonreentrant",
        deterministic=True,
        recipe_sha256=H,
        init_weights_sha256=H,
    )
    fingerprint = dict(
        world_size=4,
        backend="inductor",
        mixed_precision="bf16",
        checkpointing="nonreentrant",
        deterministic=True,
        effective_caches={"fx_local": True, "aot_local": True},
        inductor={"force_disable_caches": False},
        recipe={"_sha256": H},
        cache_env={
            "TORCHINDUCTOR_CACHE_DIR": "/work/compiler_cache/b6_history_v1/inductor",
            "TRITON_CACHE_DIR": "/work/compiler_cache/b6_history_v1/triton",
        },
    )
    bindings = {}
    for name, step in [("continuous", 410), ("at400", 400), ("resumed", 410)]:
        bindings[name] = dict(
            meta_sha256=H,
            identity_sha256=H,
            history_sha256=H,
            identity=identity,
            fingerprint=fingerprint,
            compiler_events=1,
            history_entries=1,
            position=dict(epoch=0, step=step * 24, global_step=step, pending_micro=0),
            state_file_sha256={
                key: H
                for key in ["model.safetensors", "optimizer.bin", "scheduler.bin"]
                + ["random_states_%d.pkl" % r for r in range(4)]
            },
        )
    report = dict(
        version=1,
        kind="b6_history_comparison",
        profile="b6_fresh400",
        arm="B6",
        source_commit=SOURCE,
        validation_job_id="1234",
        verified_record_sha256=H,
        fresh_start_sha256=H,
        checkpoint_digests={"continuous": H, "resumed": H},
        checkpoint_history_sha256={"continuous": H, "resumed": H},
        checkpoint_bindings=bindings,
        model_equal=True,
        different_model_keys=[],
        history_equal=True,
        states_equal={
            key: True
            for key in ["optimizer.bin", "scheduler.bin"]
            + ["random_states_%d.pkl" % r for r in range(4)]
        },
        traces={
            str(rank): {
                key: []
                for key in [
                    "inputs",
                    "shapes",
                    "rng_before",
                    "rng_after",
                    "loss",
                    "context",
                    "observed",
                    "gradients",
                    "boundary_rng",
                ]
            }
            for rank in range(4)
        },
        observed_evidence={
            phase
            + "/rank%d"
            % rank: dict(
                dna_tiers=[30720, 61440, 129024],
                tracks_nonzero=1,
                valid_chain_elements=1,
                witnesses={},
            )
            for phase in ("continuous", "resumed")
            for rank in range(4)
        },
        trace_sha256={
            phase + "/rank%d" % rank: H for phase in ("continuous", "resumed") for rank in range(4)
        },
        runtime_fingerprint_sha256=H,
        passed=True,
        finite_losses=True,
        trace_row_counts={
            phase
            + "/rank%d"
            % rank: dict(
                forward=9840 if phase == "continuous" else 240,
                accumulation_boundary=410 if phase == "continuous" else 10,
            )
            for phase in ("continuous", "resumed")
            for rank in range(4)
        },
    )
    if contract is not None:
        report["runtime_fingerprint_sha256"] = contract.canonical_sha(fingerprint)
    receipt = dict(
        version=1,
        kind="b6_history_validation_ready",
        source_commit=SOURCE,
        validation_job_id="1234",
        comparison_sha256=H,
        verified_record_sha256=H,
        runtime_fingerprint_sha256=report["runtime_fingerprint_sha256"],
        provenance=dict(image_sha256=H, source_manifest_sha256=H, helper_manifest_sha256=H),
        candidate=dict(
            relative_run="resumed",
            relative_checkpoint="latest_state",
            epoch=0,
            step=9840,
            global_step=410,
            pending_micro=0,
            meta_sha256=H,
            identity_sha256=H,
            history_sha256=H,
            canonical_checkpoint_sha256=H,
        ),
    )
    return report, receipt


class ReceiptContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(contract, "host receipt contract implementation is missing")

    def test_accepts_complete_report_and_exact_independent_ready_bindings(self):
        report, receipt = fixture()
        contract.validate_report(report, SOURCE, "1234")
        contract.validate_ready(receipt, report, copy.deepcopy(receipt))

    def test_false_comparison_is_rejected_even_when_claimed_hashes_are_recomputed(self):
        report, receipt = fixture()
        for field in ("passed", "model_equal", "history_equal"):
            bad = copy.deepcopy(report)
            bad[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_ready(receipt, bad, copy.deepcopy(receipt))
        bad = copy.deepcopy(report)
        bad["states_equal"]["random_states_3.pkl"] = False
        with self.assertRaises(ValueError):
            contract.validate_report(bad, SOURCE, "1234")
        bad = copy.deepcopy(report)
        bad["traces"]["2"]["loss"] = [9624]
        with self.assertRaises(ValueError):
            contract.validate_report(bad, SOURCE, "1234")

    def test_wrong_profile_identity_missing_rank_and_zero_evidence_refuse(self):
        report, _ = fixture()
        for mutation in (
            "arm",
            "profile",
            "job",
            "rank",
            "state",
            "evidence",
            "tier",
            "recipe",
            "position",
        ):
            bad = copy.deepcopy(report)
            if mutation == "arm":
                bad["arm"] = "C2"
            elif mutation == "profile":
                bad["profile"] = "validate400"
            elif mutation == "job":
                bad["validation_job_id"] = "9999"
            elif mutation == "rank":
                del bad["traces"]["3"]
            elif mutation == "state":
                del bad["states_equal"]["random_states_3.pkl"]
            elif mutation == "evidence":
                for row in bad["observed_evidence"].values():
                    row["valid_chain_elements"] = 0
            elif mutation == "tier":
                for row in bad["observed_evidence"].values():
                    row["dna_tiers"] = [30720]
            elif mutation == "recipe":
                bad["checkpoint_bindings"]["resumed"]["identity"]["stop_after_updates"] = 410
            else:
                bad["checkpoint_bindings"]["resumed"]["position"]["global_step"] = 400
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contract.validate_report(bad, SOURCE, "1234")

    def test_candidate_and_strict_types_must_match_independent_values(self):
        report, receipt = fixture()
        for field, value in [
            ("meta_sha256", "b" * 64),
            ("identity_sha256", "b" * 64),
            ("history_sha256", "b" * 64),
            ("canonical_checkpoint_sha256", "b" * 64),
            ("relative_run", "continuous"),
            ("global_step", 4099),
        ]:
            bad = copy.deepcopy(receipt)
            bad["candidate"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_ready(bad, report, receipt)
        bad = copy.deepcopy(receipt)
        bad["version"] = True
        with self.assertRaises(ValueError):
            contract.validate_ready(bad, report, receipt)
        bad = copy.deepcopy(report)
        bad["states_equal"]["optimizer.bin"] = 1
        with self.assertRaises(ValueError):
            contract.validate_report(bad, SOURCE, "1234")

    def test_python36_grammar_and_import_have_no_heavy_dependencies(self):
        if sys.version_info >= (3, 8):
            ast.parse(PATH.read_text(), feature_version=(3, 6))
        script = "import runpy,sys;runpy.run_path(sys.argv[1]);assert not any(k.split('.')[0] in {'torch','numpy','accelerate','safetensors'} for k in sys.modules)"
        subprocess.run([sys.executable, "-S", "-c", script, str(PATH)], check=True, cwd="/tmp")


if __name__ == "__main__":
    unittest.main()
