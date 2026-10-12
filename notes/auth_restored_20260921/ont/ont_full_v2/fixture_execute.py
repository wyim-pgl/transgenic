#!/usr/bin/env python3
"""ONT real-tool fixture driver that drives the production entry point.

The v4 fixture (`notes/auth_resume_20260917/ont_fixture_v4/fixture.py`) imported
the wrapper but reimplemented its orchestration, calling only `child`, `identity`,
`save` and `check_source_closure`. `execute`, `slurm`, `pinned`, `fresh` and
`check_boundary` therefore never ran in any real allocation, including job
6165965. This driver removes that duplication: it prepares the request and hands
control to `wrapper.execute()`, which performs the closure preflight, scheduler
binding, budget enforcement, workload supervision and publication itself.

Preparation only. This module never submits, cancels or modifies a Slurm job, and
it grants no scientific, CLEAN, F4 or full-run acceptance. The unresolved
`.extern OUT_OF_MEMORY 0:125` operational gate is untouched.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_wrapper(release, wrapper_sha, gate_sha, pins_sha):
    """Pin the release's own identities before importing anything from it.

    The frozen-tree closure check is deliberately NOT repeated here: it belongs
    to `execute()`, which runs it before any gate or driver import and again
    after postflight.
    """
    release = Path(release).resolve(strict=True)
    for name, sha in (('gate.py', gate_sha), ('source_pins.json', pins_sha), ('wrapper.py', wrapper_sha)):
        if digest(release/name) != sha:
            raise ValueError('release pin mismatch: '+name)
    spec = importlib.util.spec_from_file_location('ont_fixture_wrapper', release/'wrapper.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper


def build_request(wrapper, release, control, run, budget, tools, scontrol, cgroup,
                  seams, substitutions=None, fixture=None):
    """Assemble the execute() request, with every substitution declared up front."""
    if not seams:
        raise ValueError('a fixture request must declare its seams')
    request = dict(scope=wrapper.SCOPE, runtime_reviewed=True,
                   release=str(release), control=str(control), run=run,
                   budget=dict(budget), tools=dict(tools), scontrol=dict(scontrol),
                   cgroup=dict(cgroup), seams=list(seams),
                   substitutions=None if substitutions is None else dict(substitutions))
    if fixture is not None:
        request['fixture'] = dict(fixture)
    return request


def run(wrapper, request, request_path):
    """Write the request and hand the whole orchestration to execute()."""
    request_path = Path(request_path)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True)+'\n')
    return wrapper.execute(str(request_path), digest(request_path),
                           digest(Path(wrapper.__file__).resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--request', required=True,
                        help='JSON file holding the prepared request body')
    parser.add_argument('--wrapper-sha256', required=True)
    parser.add_argument('--gate-sha256', required=True)
    parser.add_argument('--pins-sha256', required=True)
    options = parser.parse_args(argv)
    wrapper = load_wrapper(options.release, options.wrapper_sha256,
                           options.gate_sha256, options.pins_sha256)
    request = json.loads(Path(options.request).read_text())
    receipt = run(wrapper, request, options.request)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
