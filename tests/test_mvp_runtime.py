import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transgenic.training.mvp_runtime import MVPRecipe, build_optimizer, lr_scale


ROOT = Path(__file__).parents[1]
PYTHON = "/home/wyim/miniconda3/envs/pdb/bin/python"
TORCHRUN = "/home/wyim/miniconda3/envs/pdb/bin/torchrun"
DEPS = ROOT / ".testdeps"
DUCKDB_DEPS = Path("/tmp/b6-dataset-verify.D2Is9a/.testdeps")


class OptimizerToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 1)
        self.transgenic = torch.nn.Module()
        self.transgenic.encoder = torch.nn.Module()
        self.transgenic.encoder.evidence_proj = torch.nn.Linear(2, 1, bias=False)
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


def test_recipe_validation_and_schedule_points():
    recipe = MVPRecipe()
    assert (recipe.max_updates, recipe.warmup_updates, recipe.batch_size,
            recipe.accumulation_steps, recipe.required_world_size) == (250, 12, 1, 24, 4)
    expected = {0: 0, 1: 1 / 12, 11: 11 / 12, 12: 1, 13: 237 / 238,
                249: 1 / 238, 250: 0}
    for completed, value in expected.items():
        assert lr_scale(completed, recipe) == pytest.approx(value)
    for kwargs in ({"max_updates": True}, {"warmup_updates": 250},
                   {"batch_size": 0}, {"accumulation_steps": 1.5},
                   {"backbone_lr": float("nan")}, {"projection_lr": 0},
                   {"weight_decay": -1}, {"clip_grad_norm": 0}):
        with pytest.raises((TypeError, ValueError)):
            MVPRecipe(**kwargs)


def test_optimizer_groups_are_named_disjoint_exhaustive_and_uniform_decay():
    model = OptimizerToy()
    recipe = MVPRecipe(max_updates=3, warmup_updates=1, accumulation_steps=1,
                       required_world_size=1)
    optimizer = build_optimizer(model, recipe)
    assert [group["name"] for group in optimizer.param_groups] == ["backbone", "projection"]
    assert [group["lr"] for group in optimizer.param_groups] == [recipe.backbone_lr, recipe.projection_lr]
    assert {group["weight_decay"] for group in optimizer.param_groups} == {recipe.weight_decay}
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(trainable)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    projection = dict(model.named_parameters())["transgenic.encoder.evidence_proj.weight"]
    assert optimizer.param_groups[1]["params"] == [projection]

    del model.transgenic.encoder.evidence_proj
    with pytest.raises(ValueError, match="exactly one"):
        build_optimizer(model, recipe)


def test_all_250_updates_match_post_step_lambdalr_parameters_and_state():
    torch.manual_seed(91)
    actual_model = OptimizerToy().double()
    reference_model = OptimizerToy().double()
    reference_model.load_state_dict(actual_model.state_dict())
    recipe = MVPRecipe(required_world_size=1)
    actual = build_optimizer(actual_model, recipe)
    reference = build_optimizer(reference_model, recipe)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        reference, [lambda k: lr_scale(k, recipe)] * 2
    )
    for group, peak in zip(actual.param_groups, (recipe.backbone_lr, recipe.projection_lr)):
        group["lr"] = peak * lr_scale(0, recipe)

    for completed in range(recipe.max_updates):
        assert [group["lr"] for group in actual.param_groups] == pytest.approx(
            [group["lr"] for group in reference.param_groups])
        x = torch.tensor([[0.25, -0.75]], dtype=torch.float64)
        for model, optimizer in ((actual_model, actual), (reference_model, reference)):
            optimizer.zero_grad()
            loss = sum(parameter.square().sum() for parameter in model.parameters()) + model.backbone(x).sum()
            loss.backward()
            optimizer.step()
        scheduler.step()
        for group, peak in zip(actual.param_groups, (recipe.backbone_lr, recipe.projection_lr)):
            group["lr"] = peak * lr_scale(completed + 1, recipe)

    for actual_parameter, reference_parameter in zip(actual_model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual_parameter, reference_parameter, rtol=0, atol=0)
    assert actual.state_dict()["state"].keys() == reference.state_dict()["state"].keys()
    for key in actual.state_dict()["state"]:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual.state_dict()["state"][key][field],
                                       reference.state_dict()["state"][key][field], rtol=0, atol=0)
    assert [group["lr"] for group in actual.param_groups] == [0.0, 0.0]


def _probe(ranks, case, timeout=120):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(DEPS), str(DUCKDB_DEPS)))
    command = ([PYTHON] if ranks == 1 else [TORCHRUN, "--standalone", f"--nproc-per-node={ranks}"])
    command += [str(ROOT / "tests" / "mvp_runtime_probe.py"), case]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                            timeout=timeout)
    assert result.returncode == 0, f"{command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert records, result.stdout
    return records[-1]


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_actual_accelerate_cpu_gloo_core(ranks):
    record = _probe(ranks, "core")
    assert record["world_size"] == ranks
    assert record["distributed_type"] == ("NO" if ranks == 1 else "MULTI_CPU")
    assert record["successful_updates"] == 3
    assert record["microbatches_per_rank"] == [12] * ranks
    assert record["epochs_crossed"] == 2
    assert record["effective_batch_size"] == 4 * ranks
    assert record["sampler_epochs"] == [0, 1, 2]
    assert record["optimizer_steps"] == 3
    assert record["clip_calls"] == 3
    assert record["analytic_match"] is True
    assert record["parameters_synchronized"] is True


def test_literal_four_rank_250_by_24_budget():
    record = _probe(4, "literal", timeout=180)
    assert record["successful_updates"] == 250
    assert record["microbatches_per_rank"] == [6000] * 4
    assert record["effective_batch_size"] == 96
    assert record["optimizer_steps"] == 250
    assert record["forward_calls"] == 6000
    assert record["parameters_synchronized"] is True
    assert record["lr_trace"][0]["update"] == 1
    assert record["lr_trace"][0]["backbone"] == 0
    assert record["lr_trace"][11]["backbone"] == pytest.approx(11 / 12 * MVPRecipe().backbone_lr)
    assert record["lr_trace"][12]["backbone"] == pytest.approx(MVPRecipe().backbone_lr)
    assert record["lr_trace"][-1]["backbone"] == pytest.approx(1 / 238 * MVPRecipe().backbone_lr)
    assert record["final_lrs"] == [0, 0]


@pytest.mark.parametrize("case", ["nan_loss", "nan_gradient", "skipped", "empty",
                                  "unequal", "invalid_settings", "invalid_batch",
                                  "prepared_loader", "unprepared_components",
                                  "sampler_mismatch"])
def test_collective_failures_and_contract_rejections(case):
    ranks = 2 if case in {"nan_loss", "nan_gradient", "empty", "unequal",
                          "sampler_mismatch"} else 1
    record = _probe(ranks, case)
    assert record["case"] == case
    assert record["all_ranks_rejected"] is True
