#!/usr/bin/env python3
"""Run `33_run_polishing_benchmark.py`'s pipeline for an arm that is not in its TOOLS list.

`33` gates `--tool` on a hard-coded TOOLS tuple, and `run_tool_pipeline` re-checks it. That
gate is worth keeping — it is what stops a typo from silently producing `typo_completed.gff3`
— so this runner registers the extra arm with the module at import time instead of editing
the list. Everything downstream (chunking, resume, checksum preflight, loss gates, merge,
provenance) is `33`'s own code, unmodified and unforked.

The arm must already be staged as `polishing_benchmark/inputs/{tool}_Athaliana.gff3` locally
AND on the GPU host; `33`'s `verify_host_input` will refuse the run otherwise, which is the
intended behaviour and the reason staging is a separate, deliberate step.

    python3 revision/scripts/54_run_extra_arm.py --tool tair10cdshelixerutr \
        --max-unexplained-missing-loci 40 --acknowledge-high-loss-threshold
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
RUNNER = SCRIPTS / "33_run_polishing_benchmark.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_polishing_benchmark", RUNNER)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `33` defines frozen dataclasses at import time, and
    # `dataclasses._is_type` resolves annotations through `sys.modules[cls.__module__]`.
    # Without this the import dies on the first `@dataclass` with an opaque AttributeError.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    mod = load_runner()

    # Read the requested tool names off the argv this process was given, and register any
    # that `33` does not already know about. Registration happens before `main()` builds its
    # parser, which is what makes them acceptable `--tool` values.
    requested = []
    for i, arg in enumerate(argv):
        if arg == "--tool":
            for value in argv[i + 1:]:
                if value.startswith("-"):
                    break
                requested.append(value)
    extra = tuple(t for t in requested if t not in mod.TOOLS)
    if not extra:
        print("nothing to register — every requested tool is already in 33's TOOLS; "
              "call 33 directly", file=sys.stderr)
    mod.TOOLS = mod.TOOLS + extra

    for tool in extra:
        staged = mod.LOCAL_INPUTS / f"{tool}_Athaliana.gff3"
        if not staged.exists():
            raise SystemExit(f"{tool}: no staged input at {staged}")
        print(f"registered extra arm {tool!r} (input {staged})", file=sys.stderr)

    return mod.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
