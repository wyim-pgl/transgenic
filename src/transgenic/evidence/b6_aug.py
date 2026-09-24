"""B6 augmentation draws (plan §7.3; A46 rule 8; D7).

A ``DrawSpec`` is only the recipient's six uniforms, keyed by (aug_seed, purpose, row, epoch) with blake2b
(``b6_track_spec.uniform``).  ``resolve(draw, owner)`` turns it into a ``ResolvedDraw`` against the evidence owner's
admitted sources and libraries (the recipient, or the donor in C3).  One resolved draw feeds both
``b6_tracks.rasterise`` and ``b6_chains.build_chains``, so tracks and chains always see the same molecule subset.
No ``torch.Generator`` and no process or worker state: the same (aug_seed, row, epoch) gives the same draw in any
worker, after any resume.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

from transgenic.evidence import b6_track_spec as TS


@dataclass(frozen=True)
class DrawSpec:
    aug_seed: int
    row: str
    epoch: int
    uniforms: tuple            # ((purpose, u), ...) in TS.PURPOSES order

    def u(self, purpose):
        return dict(self.uniforms)[purpose]

    def log_record(self):
        """The uniform log entry: identical across arms by construction (rule 11)."""
        return {"aug_seed": self.aug_seed, "row": self.row, "epoch": self.epoch,
                "uniforms": {p: repr(u) for p, u in self.uniforms}}

    def sha256(self):
        return hashlib.sha256(json.dumps(self.log_record(), sort_keys=True, separators=(",", ":"))
                              .encode("utf-8")).hexdigest()


def make_draw(aug_seed, row, epoch):
    if isinstance(row, bool) or not isinstance(row, str):
        raise ValueError(f"row must be a str key, got {row!r}")
    return DrawSpec(int(aug_seed), row, int(epoch),
                    tuple((p, TS.uniform(aug_seed, p, row, epoch)) for p in TS.PURPOSES))


@dataclass(frozen=True)
class Owner:
    """The evidence owner's admitted libraries in the population: {track source -> sorted library tuple}."""
    species: str
    libraries: tuple           # ((source, library), ...) sorted; sources in est/ont/pacbio only

    @property
    def sources(self):
        return tuple(sorted({s for s, _ in self.libraries}))

    @classmethod
    def from_pairs(cls, species, pairs):
        pairs = tuple(sorted({(str(s), str(lib)) for s, lib in pairs}))
        bad = [s for s, _ in pairs if s not in TS.TRACK_SOURCES]
        if bad:
            raise ValueError(f"owner libraries must be track sources {TS.TRACK_SOURCES}, got {sorted(set(bad))}")
        return cls(species, pairs)


@dataclass(frozen=True)
class ResolvedDraw:
    aug_seed: int              # rank stream (training aug_seed, or 20260914 + d in dose-response)
    species: str               # evidence owner's species (rank key)
    dropped: bool
    fraction: float
    masks: tuple               # ((source, 0/1), ...) for est/ont/pacbio
    withheld_source: str | None
    withheld_library: tuple | None   # (source, library)
    admitted: tuple            # the owner's (source, library) pairs still admitted after withholding
    kind: str                  # 'train' | 'fixed'
    qc: dict = field(default_factory=dict, compare=False, hash=False)
    epoch: int = 0             # slot-hash epoch (rule 6a); 0 at evaluation

    def mask(self, source):
        return dict(self.masks)[source]

    def library_admitted(self, source, library):
        if self.dropped:
            return False
        if source == "protein":                  # D7: protein is never withheld
            return True
        return (source, library) in self.admitted_set

    @property
    def admitted_set(self):
        return frozenset(self.admitted)

    @property
    def phase_enabled(self):
        return not self.dropped and self.mask("est") == 1

    def keeps(self, species, molecule_id):
        """Rule 8 thinning: keep iff rank < fraction (fraction 1 keeps all, 0 keeps none)."""
        if self.dropped or self.fraction <= 0:
            return False
        if self.fraction >= 1:
            return True
        return _rank(self.aug_seed, species, molecule_id) < self.fraction

    def record(self):
        return {"aug_seed": self.aug_seed, "species": self.species, "dropped": self.dropped,
                "fraction": self.fraction, "masks": dict(self.masks), "withheld_source": self.withheld_source,
                "withheld_library": list(self.withheld_library) if self.withheld_library else None,
                "kind": self.kind, "epoch": self.epoch, "qc": dict(self.qc)}


@lru_cache(maxsize=1 << 20)
def _rank(aug_seed, species, molecule_id):
    return TS.molecule_rank(aug_seed, species, molecule_id)


def _pick(seq, u):
    return seq[min(int(math.floor(u * len(seq))), len(seq) - 1)]


def resolve(draw, owner):
    """Resolve a recipient's uniforms against the evidence owner (plan §7.3)."""
    qc = {}
    dropped = draw.u("dropout") < TS.DROPOUT_P
    fraction = TS.FRACTIONS[min(int(math.floor(4 * draw.u("fraction"))), 3)]
    admitted = set(owner.libraries)
    withheld_source = withheld_library = None
    if not dropped:
        if draw.u("withhold_source") < TS.WITHHOLD_SOURCE_RATE:
            present = [s for s in TS.TRACK_SOURCES if s in owner.sources]
            if present:
                withheld_source = _pick(sorted(present), draw.u("withhold_source_pick"))
                admitted = {p for p in admitted if p[0] != withheld_source}
            else:
                qc["withhold_source_noop"] = True
        if draw.u("withhold_library") < TS.WITHHOLD_LIBRARY_RATE:
            libs = sorted(owner.libraries)
            if libs:
                withheld_library = _pick(libs, draw.u("withhold_library_pick"))
                admitted.discard(withheld_library)
            else:
                qc["withhold_library_noop"] = True
    masks = tuple((s, 0 if dropped else int(any(p[0] == s for p in admitted))) for s in TS.TRACK_SOURCES)
    return ResolvedDraw(draw.aug_seed, owner.species, dropped, fraction, masks, withheld_source, withheld_library,
                        tuple(sorted(admitted)) if not dropped else (), "train", qc, draw.epoch)


def resolve_fixed(aug_seed, owner, fraction):
    """Evaluation / dose-response: no dropout and no withholding; ``fraction`` in {1, 0.5, 0.1, 0.01, 0}.

    Fraction 0 keeps the masks (evidence present, no molecule retained).  De novo mode is ``resolve_off``.
    """
    if fraction not in TS.DOSE_FRACTIONS:
        raise ValueError(f"fraction {fraction!r} not in {TS.DOSE_FRACTIONS}")
    masks = tuple((s, int(s in owner.sources)) for s in TS.TRACK_SOURCES)
    return ResolvedDraw(int(aug_seed), owner.species, False, float(fraction), masks, None, None,
                        tuple(owner.libraries), "fixed", {})


def resolve_off(owner, aug_seed=0):
    """De novo mode: evidence off (all channels zero, chains fully masked)."""
    return ResolvedDraw(int(aug_seed), owner.species, True, 0.0, tuple((s, 0) for s in TS.TRACK_SOURCES), None,
                        None, (), "fixed", {})


# ------------------------------------------------------------------------------------------ arm transforms
def apply_arm_tracks(tracks, arm):
    """Arm transform of a [L, 31] float16 raster (in place on a copy).  C3 is a donor choice, not a transform."""
    import numpy as np

    if arm not in TS.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    t = np.array(tracks, dtype=np.float16, copy=True)
    if arm == "C2":
        t[:] = 0
    elif arm == "C4":
        cols = list(TS.COUNT_CHANNELS)
        t[:, cols] = (t[:, cols] > 0).astype(np.float16)
    elif arm == "C6":
        t[:, list(TS.PHASE_CHANNELS)] = 0
    return t


def arm_masks_chains(arm):
    """True iff the arm executes the chain branch with every element masked (C2, C7)."""
    if arm not in TS.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return arm in ("C2", "C7")


# ------------------------------------------------------------------------------------------ C3 donors (D6)
def c3_donor_map(rows, aug_seed):
    """Fixed RC-matched derangement donors within (tier, split).

    ``rows``: iterable of (rn, wid, tier, split, is_rc).  Forward rows of each (tier, split) are sorted by
    blake2b(f"{aug_seed}\\tc3\\t{tier}\\t{split}\\t{wid}") and each is paired with the next, cyclically; an RC twin
    (same wid) takes its forward row's donor wid.  Returns {rn: donor_wid}.  A group with < 2 forward rows is a
    build error, reported with counts.
    """
    rows = list(rows)
    groups = {}
    for rn, wid, tier, split, is_rc in rows:
        if not is_rc:
            groups.setdefault((str(tier), str(split)), []).append(str(wid))
    small = {g: len(v) for g, v in groups.items() if len(set(v)) < 2}
    if small:
        raise ValueError(f"C3: (tier, split) groups with < 2 forward rows: {sorted(small.items())}")
    donor_of_wid = {}
    for (tier, split), wids in groups.items():
        if len(set(wids)) != len(wids):
            raise ValueError(f"C3: duplicate forward wid in ({tier}, {split})")
        order = sorted(wids, key=lambda w: (int.from_bytes(hashlib.blake2b(
            f"{int(aug_seed)}\tc3\t{tier}\t{split}\t{w}".encode("utf-8"), digest_size=8).digest(), "big"), w))
        for i, w in enumerate(order):
            donor_of_wid[(tier, split, w)] = order[(i + 1) % len(order)]
    out = {}
    for rn, wid, tier, split, is_rc in rows:
        key = (str(tier), str(split), str(wid))
        if key not in donor_of_wid:
            raise ValueError(f"C3: RC row rn={rn} has no forward twin wid={wid} in ({tier}, {split})")
        out[rn] = donor_of_wid[key]
    return out
