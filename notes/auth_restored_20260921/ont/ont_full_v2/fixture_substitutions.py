#!/usr/bin/env python3
"""Declared, hashed substitution module for the ONT real-tool fixture.

The fixture drives the production entry point `wrapper.execute()`. Everything it
cannot supply outside a real allocation is replaced here, in one pinned file, so
the substitution is an auditable artifact with a SHA256 rather than a monkeypatch
buried in a driver. `execute()` pins this file, records its identity in the
receipt, and independently records whether each binding still resolves to its
pinned source.

This substitutes bindings only. It never relaxes a wrapper guard, never touches
the scheduler, cgroup, file-limit, scratch or closure checks, and grants no
scientific acceptance. A production request sets `substitutions` to null.
"""
from pathlib import Path
from types import SimpleNamespace


def apply(gate, request):
    """Replace the gate's data/science bindings. Returns what was replaced."""
    fixture = request['fixture']
    out = Path(fixture['output_directory'])
    roles = fixture['roles']
    replaced = []

    driver = SimpleNamespace(
        output_directory=lambda root, run: out,
        w=SimpleNamespace(load_panel=lambda panels: list(range(50))))

    def load_driver(source):
        return driver

    def read_binding(*args):
        return {'roles': {'path': roles}}

    def reference_binding(*args):
        return {'files': {'fasta': {'md5': '0'*32}, 'gff': {'md5': '0'*32}}}

    def verify_alignment(args, receipt):
        return {'declared_science_substitution': True}

    for name, replacement in (('load_driver', load_driver), ('read_binding', read_binding),
                              ('reference_binding', reference_binding),
                              ('verify_alignment', verify_alignment)):
        setattr(gate, name, replacement)
        replaced.append(name)

    return dict(replaced=replaced,
                reason='no approved runtime tooling on this host; canonical authority, '
                       'panel and reference bindings are fixture stand-ins',
                scientific_acceptance=False)
