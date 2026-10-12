#!/usr/bin/env python3
"""Caller-owned request pin gate for the reviewed CPU panel verifier.

Supply the reviewed request digest and size independently, never by hashing the
current request at launch. Success checks the caller binding only; independent
receipt review/pinning and source-closure packaging remain separate decisions.
Late failures retain all helper output and diagnostics. No retry or cleanup.
"""
import argparse
import re

import mvp_panel_compatibility as compatibility


def verify_pinned_panel(input_path, expected_sha256, expected_size, out):
    """Invoke the existing verifier once, gated by an independent request pin."""
    d = compatibility.d
    d.require(isinstance(expected_sha256, str)
              and re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None,
              'expected SHA256 must be 64 lowercase hexadecimal characters')
    d.require(type(expected_size) is int and expected_size >= 0,
              'expected size must be a nonnegative integer')
    request = compatibility._canonical_path(input_path)
    expected = dict(path=str(request), sha256=expected_sha256, size=expected_size)
    compatibility._recheck([expected])

    report = compatibility.verify_and_publish(request, out)
    output_id = compatibility.p.file_identity(compatibility._canonical_path(out))
    serialized = d.read_json(out)
    d.require(isinstance(serialized, dict)
              and serialized.get('verification_request') == expected,
              'serialized verification_request differs from independently pinned request')
    d.require(report == serialized, 'returned report differs from serialized output')
    compatibility._recheck([expected, output_id])
    return serialized


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='canonical absolute reviewed request path')
    parser.add_argument('--expected-sha256', required=True, help='independently reviewed request SHA256')
    parser.add_argument('--expected-size', required=True, type=int, help='independently reviewed request byte size')
    parser.add_argument('--out', required=True, help='fresh external receipt path')
    args = parser.parse_args(argv)
    verify_pinned_panel(args.input, args.expected_sha256, args.expected_size, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
