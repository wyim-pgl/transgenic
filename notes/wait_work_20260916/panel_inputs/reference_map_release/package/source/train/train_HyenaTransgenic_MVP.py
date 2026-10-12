#!/usr/bin/env python3
"""Exploratory fixed-recipe MVP entrypoint: EXPLORATORY_MVP_NOT_FOR_PRODUCTION."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transgenic.training.mvp_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
