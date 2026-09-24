"""WP2-roles (plan v4.1 §3): the B6 dataset-role manifest ``DATASET_ROLES.b6_v1.tsv`` and its sha256 sidecar.

``load_roles(path)`` verifies the sidecar (``<path>.sha256``, first token = sha256 of the file) and fails closed on:

* a header that is not exactly ``COLUMNS``; a duplicate (dataset, run); a non-``true``/``false`` boolean;
* a row that is both ``b6_input`` and ``b6_score``, or neither;
* a scoring set (``SCORING_SETS``: A-ONT1, A-CCS23, M-FLNC, M-ONT) row not marked ``b6_score`` (and not input);
* an input row whose ``b6_input_scope`` is outside ``INPUT_SCOPES``, or a non-input row whose scope is not ``-``;
* an M-HQ18 row that is not ``tracks_only`` (rule 6a + A47.3);
* A-ONT2 lacking any of its three PRJNA594286 runs.

Selection is by (dataset, run) only, never by source class or species: ``Roles.get`` raises ``RolesError`` for a
pair the manifest does not name.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("dataset", "run", "species", "roles_file", "role", "set_label", "b6_input", "b6_score", "b6_input_scope",
           "basis")
SCORING_SETS = ("A-ONT1", "A-CCS23", "M-FLNC", "M-ONT")
INPUT_SCOPES = ("tracks+chains", "tracks_only")
NO_SCOPE = "-"
A_ONT2_BIOPROJECT = "PRJNA594286"
A_ONT2_RUNS = ("SRR10611193", "SRR10611194", "SRR10611195")


class RolesError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise RolesError(message)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bool(value, where):
    _require(value in ("true", "false"), f"{where}: boolean field {value!r} is not 'true' or 'false'")
    return value == "true"


@dataclass(frozen=True)
class RoleRow:
    dataset: str
    run: str
    species: str
    roles_file: str
    role: str
    set_label: str
    b6_input: bool
    b6_score: bool
    b6_input_scope: str
    basis: str


@dataclass(frozen=True)
class Roles:
    path: str
    sha256: str
    rows: tuple          # RoleRow, sorted by (dataset, run)

    def get(self, dataset, run):
        for r in self.rows:
            if r.dataset == dataset and r.run == run:
                return r
        raise RolesError(f"({dataset!r}, {run!r}) is not in {self.path}; roles are selected by dataset/run only")

    def is_input(self, dataset, run):
        return self.get(dataset, run).b6_input

    def is_score(self, dataset, run):
        return self.get(dataset, run).b6_score

    def scope(self, dataset, run):
        return self.get(dataset, run).b6_input_scope

    def inputs(self, species=None):
        return tuple(r for r in self.rows if r.b6_input and (species is None or r.species == species))

    def scores(self, species=None):
        return tuple(r for r in self.rows if r.b6_score and (species is None or r.species == species))


def verify_sidecar(path, sidecar=None):
    """sha256 of ``path`` after checking it equals the first token of its sidecar (default ``<path>.sha256``)."""
    sidecar = Path(sidecar) if sidecar else Path(str(path) + ".sha256")
    _require(sidecar.is_file(), f"roles sidecar {sidecar} missing")
    tokens = sidecar.read_text().split()
    _require(tokens and len(tokens[0]) == 64, f"roles sidecar {sidecar} has no sha256")
    got = _sha256(path)
    _require(got == tokens[0], f"{path}: sha256 {got} != sidecar {tokens[0]}")
    return got


def load_roles(path, *, sidecar=None):
    path = str(path)
    sha = verify_sidecar(path, sidecar)
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        _require(header is not None and tuple(header) == COLUMNS, f"{path}: header {header} != {list(COLUMNS)}")
        rows, seen = [], set()
        for n, cells in enumerate(reader, start=2):
            where = f"{path}:{n}"
            _require(len(cells) == len(COLUMNS), f"{where}: {len(cells)} fields, expected {len(COLUMNS)}")
            d = dict(zip(COLUMNS, cells))
            key = (d["dataset"], d["run"])
            _require(key not in seen, f"{where}: duplicate (dataset, run) {key}")
            seen.add(key)
            row = RoleRow(d["dataset"], d["run"], d["species"], d["roles_file"], d["role"], d["set_label"],
                          _bool(d["b6_input"], where), _bool(d["b6_score"], where), d["b6_input_scope"], d["basis"])
            _require(not (row.b6_input and row.b6_score), f"{where}: {key} is both b6_input and b6_score")
            _require(row.b6_input or row.b6_score, f"{where}: {key} is neither b6_input nor b6_score")
            if row.set_label in SCORING_SETS:
                _require(row.b6_score and not row.b6_input,
                         f"{where}: {row.set_label} is a scoring set and must be b6_score only")
            if row.b6_input:
                _require(row.b6_input_scope in INPUT_SCOPES,
                         f"{where}: input scope {row.b6_input_scope!r} not in {INPUT_SCOPES}")
            else:
                _require(row.b6_input_scope == NO_SCOPE, f"{where}: non-input row has scope {row.b6_input_scope!r}")
            if row.set_label == "M-HQ18":
                _require(row.b6_input_scope == "tracks_only", f"{where}: M-HQ18 must be tracks_only")
            rows.append(row)
    a_ont2 = {r.run for r in rows if r.set_label == "A-ONT2" and A_ONT2_BIOPROJECT in r.dataset}
    missing = sorted(set(A_ONT2_RUNS) - a_ont2)
    _require(not missing, f"{path}: A-ONT2 lacks {A_ONT2_BIOPROJECT} run(s) {missing}")
    return Roles(path, sha, tuple(sorted(rows, key=lambda r: (r.dataset, r.run))))
