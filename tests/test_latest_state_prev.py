"""latest_state is published by renames (2026-09-10): the previous checkpoint moves to latest_state.prev before the
new one is renamed into place, and resume_dir('auto') accepts .prev while that window is open."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transgenic.training.b5_runtime import CheckpointLayout  # noqa: E402


def _state(d: str, global_step: int):
    os.makedirs(os.path.join(d, "accelerate_state"), exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump({"epoch": 0, "step": 1, "global_step": global_step}, fh)


def test_resume_uses_prev_when_publication_was_interrupted(tmp_path):
    layout = CheckpointLayout(str(tmp_path))
    _state(layout.latest_state_dir() + ".prev", 400)
    os.makedirs(layout.latest_state_dir() + ".tmp")           # the interrupted replacement, never valid
    assert layout.resume_dir("auto") == layout.latest_state_dir() + ".prev"


def test_resume_prefers_the_published_latest_state(tmp_path):
    layout = CheckpointLayout(str(tmp_path))
    _state(layout.latest_state_dir() + ".prev", 400)
    _state(layout.latest_state_dir(), 600)
    assert layout.resume_dir("auto") == layout.latest_state_dir()
