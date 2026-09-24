"""Synthetic plan-7.7 rows for the WP4 trainer tests and the rule 15 four-rank integration driver.

Not WP3 data and not ``tests/b6_fixtures.py`` (Writer B's): rows come from the pair number alone, so every rank,
worker and resume sees identical content. Rows come in forward/RC twin pairs (rn = first_rn + 2p, 2p + 1; one
``gene_id`` per pair). The RC twin is derived from its forward row with every transformation applied together:

  * DNA: reverse complement (A<->T, C<->G, N kept) of the forward bases, then the Hyena char ids;
  * label: ``gsf_contract.reverse_complement_v3`` of the forward GSF at the tile length, then the v3 GSF tokens;
  * tracks: ``b6_track_spec.rc_tracks`` (RC_PERM, then reverse L);
  * chains: ``b6_track_spec.rc_chain`` (chain-rc-v1: p -> L-1-p, strand swapped).
``assert_rc_pair`` re-derives all four from the forward twin and fails on any disagreement. Every third pair is a
dropped tile (all-zero tracks, no chain), so batches mix zero-track and non-zero-track rows.

Keys are (index, epoch), as WP3's B6EpochSampler yields (rn, epoch); ``uniforms_sha256`` is a deterministic hash of
(seed, row, epoch), standing in for WP3's DrawSpec hash, so the trainer's draw log can be compared across arms.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np
import torch

from transgenic.evidence import b6_track_spec as TS
from transgenic.utils import gsf_contract as gc

K = TS.K
N = TS.N_TOK
CHAR_IDS = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def rc_permutation() -> List[int]:
    """WP3's frozen RC permutation, cross-checked against the plan 7.1 literal."""
    literal = []
    for s in range(3):
        literal += [8 * s + c for c in (1, 0, 2, 5, 6, 3, 4, 7)]
    literal += [24] + [28, 29, 30, 25, 26, 27]
    if list(TS.RC_PERM) != literal:
        raise AssertionError("b6_track_spec.RC_PERM differs from plan 7.1")
    return list(TS.RC_PERM)


@lru_cache(maxsize=1)
def _gsf_tokenizer():
    from transgenic.model.tokenization_transgenic import GFFTokenizer
    return GFFTokenizer(vocab_version="v3")


def gsf_token_ids(gsf: str) -> np.ndarray:
    tok = _gsf_tokenizer()
    return np.asarray([tok._convert_token_to_id(t) for t in tok._tokenize(gsf)], np.int64)


def dna_ids(seq: str) -> np.ndarray:
    return np.asarray([CHAR_IDS[c] for c in seq], np.int64)


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def forward_row(seed: int, pair: int, L: int) -> Dict:
    """The forward twin: bases, one gene's GSF, dense tracks and a few chains (none on a dropped tile)."""
    rng = np.random.default_rng(seed * 100003 + pair)
    seq = "".join(rng.choice(list("ACGT"), size=L))
    s = int(rng.integers(1, L // 4))
    cds_end = s + 3 + int(rng.integers(3, L // 3))
    end = min(L, cds_end + int(rng.integers(1, L // 4)))
    strand = "+" if pair % 2 == 0 else "-"
    gene = gc.Gene(f"g{pair}", "chr1", strand, s, end,
                   {"t1": [gc.Feature("five_prime_UTR", s + 1, s + 3, "."), gc.Feature("CDS", s + 4, cds_end, "0"),
                           gc.Feature("three_prime_UTR", cds_end + 1, end, ".")]})
    gsf = gc.window_to_gsf_v3([gene], 0)
    tracks = (rng.random((L, K)) * TS.COV_CLIP).astype(np.float16)
    tracks[:, list(TS.MASK_CHANNELS)] = 1.0
    pos = np.zeros((N, 2), np.int32)
    feat = np.zeros((N, TS.N_FEAT), np.int16)
    mask = np.zeros(N, bool)
    dropped = pair % 3 == 1
    if dropped:
        tracks[:] = 0
    else:
        n = int(rng.integers(1, 60))
        pos[1:n + 1] = rng.integers(0, L, size=(n, 2))
        feat[1:n + 1] = np.stack([rng.integers(0, 32, n), rng.integers(0, 32, n), rng.integers(1, 64, n),
                                  rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(1, 65, n)], 1)
        mask[1:n + 1] = True
    return dict(seq=seq, gsf=gsf, tracks=tracks, chain_pos=pos, chain_feat=feat, chain_mask=mask,
                kind="zero" if dropped else "evidence")


def rc_row(fw: Dict, L: int) -> Dict:
    pos, feat, mask = TS.rc_chain(fw["chain_pos"], fw["chain_feat"], fw["chain_mask"], L)
    return dict(seq=revcomp(fw["seq"]), gsf=gc.reverse_complement_v3(fw["gsf"], L), tracks=TS.rc_tracks(fw["tracks"]),
                chain_pos=pos, chain_feat=feat, chain_mask=mask, kind=fw["kind"] + "_rc")


class SyntheticB6Dataset(torch.utils.data.Dataset):
    def __init__(self, n_rows: int, length: int = 48, seed: int = 0, kind: str = "corpus", first_rn: int = 1000):
        if length % 6 or n_rows % 2:
            raise ValueError("tile length must be a multiple of 6 and rows come in forward/RC pairs")
        self.n_rows, self.length, self.seed, self.kind, self.first_rn = n_rows, length, seed, kind, first_rn

    def __len__(self):
        return self.n_rows

    def row_key(self, i: int) -> str:
        rn = self.first_rn + i
        if self.kind == "corpus":
            return str(rn)
        return f"eval:Athaliana:chr1:{rn}-{rn + self.length}:{'rc' if i % 2 else 'fw'}:pilot"

    def raw(self, i: int) -> Dict:
        fw = forward_row(self.seed, i // 2, self.length)
        return rc_row(fw, self.length) if i % 2 else fw

    def __getitem__(self, key) -> Dict:
        i, epoch = key if isinstance(key, tuple) else (key, 0)
        r = self.raw(int(i))
        row = self.row_key(int(i))
        uniforms = hashlib.sha256(f"{self.seed}\t{row}\t{int(epoch)}".encode("utf-8")).hexdigest()
        return dict(ids=torch.from_numpy(dna_ids(r["seq"])), tracks=torch.from_numpy(np.ascontiguousarray(r["tracks"])),
                    chain_pos=torch.from_numpy(r["chain_pos"]), chain_feat=torch.from_numpy(r["chain_feat"]),
                    chain_mask=torch.from_numpy(r["chain_mask"]), labels=torch.from_numpy(gsf_token_ids(r["gsf"])),
                    row=row, epoch=int(epoch), uniforms_sha256=uniforms, kind=r["kind"])


def assert_rc_pair(ds: SyntheticB6Dataset, pair: int) -> None:
    """Re-derive the RC twin from the forward twin (DNA, label, tracks, chains) and check involution."""
    L = ds.length
    fw, rc = ds.raw(2 * pair), ds.raw(2 * pair + 1)
    if rc["seq"] != revcomp(fw["seq"]) or revcomp(rc["seq"]) != fw["seq"]:
        raise AssertionError(f"pair {pair}: RC DNA is not the reverse complement")
    if rc["gsf"] != gc.reverse_complement_v3(fw["gsf"], L) or gc.reverse_complement_v3(rc["gsf"], L) != fw["gsf"]:
        raise AssertionError(f"pair {pair}: RC label is not reverse_complement_v3 of the forward label")
    if not np.array_equal(rc["tracks"], TS.rc_tracks(fw["tracks"])) or \
            not np.array_equal(TS.rc_tracks(rc["tracks"]), fw["tracks"]):
        raise AssertionError(f"pair {pair}: RC tracks disagree with rc_tracks")
    back = TS.rc_chain(rc["chain_pos"], rc["chain_feat"], rc["chain_mask"], L)
    if not (np.array_equal(back[0], fw["chain_pos"]) and np.array_equal(back[1], fw["chain_feat"])
            and np.array_equal(back[2], fw["chain_mask"])):
        raise AssertionError(f"pair {pair}: RC chains are not chain-rc-v1 of the forward chains")
    m = fw["chain_mask"]
    if m.any() and not np.array_equal(rc["chain_pos"][m], (L - 1) - fw["chain_pos"][m]):
        raise AssertionError(f"pair {pair}: RC chain positions are not L-1-p")


def synthetic_collate(samples: Sequence[Dict]) -> Dict:
    """Plan 7.7: left padding, labels right-padded -100, chain_idx = (pad_b + 0 + chain_pos) // 6."""
    b = len(samples)
    lmax = max(s["ids"].shape[0] for s in samples)
    tmax = max(s["labels"].shape[0] for s in samples)
    ids = torch.zeros(b, lmax, dtype=torch.int64)
    am = torch.zeros(b, lmax, dtype=torch.bool)
    tracks = torch.zeros(b, lmax, K, dtype=torch.float16)
    labels = torch.full((b, tmax), -100, dtype=torch.int64)
    idx = torch.zeros(b, N, 2, dtype=torch.int64)
    feat = torch.zeros(b, N, TS.N_FEAT, dtype=torch.int16)
    mask = torch.zeros(b, N, dtype=torch.bool)
    for r, s in enumerate(samples):
        L = s["ids"].shape[0]
        pad = lmax - L
        ids[r, pad:], am[r, pad:], tracks[r, pad:] = s["ids"], True, s["tracks"]
        labels[r, :s["labels"].shape[0]] = s["labels"]
        m = s["chain_mask"]
        idx[r] = torch.from_numpy(TS.chain_idx(s["chain_pos"].numpy(), m.numpy(), pad))
        feat[r], mask[r] = s["chain_feat"], m
    meta = dict(row=[s["row"] for s in samples], epoch=[s["epoch"] for s in samples], arm="B6",
                uniforms_sha256=[s["uniforms_sha256"] for s in samples],
                resolved=[{"kind": s["kind"]} for s in samples],
                context_kind=["eval" if s["row"].startswith("eval:") else "corpus" for s in samples])
    return dict(input_ids=ids, attention_mask=am, evidence_tracks=tracks, labels=labels, chain_idx=idx,
                chain_feat=feat, chain_mask=mask, meta=meta)


class EpochSampler(torch.utils.data.Sampler):
    """Deterministic per-epoch permutation keyed by (seed, epoch) only; yields (index, epoch) like WP3's
    B6EpochSampler and exposes ``epoch_rows(epoch)``, the canonical row order the trainer hashes."""

    def __init__(self, n: int, seed: int = 123, rows: Sequence[str] = None):
        self.n, self.seed, self.epoch = n, seed, 0
        self.rows = list(rows) if rows is not None else [str(i) for i in range(n)]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def permutation(self, epoch: int) -> List[int]:
        g = torch.Generator().manual_seed(self.seed * 1000 + int(epoch))
        return torch.randperm(self.n, generator=g).tolist()

    def epoch_rows(self, epoch: int) -> List[str]:
        return [self.rows[i] for i in self.permutation(epoch)]

    def __iter__(self):
        return iter([(i, self.epoch) for i in self.permutation(self.epoch)])

    def __len__(self):
        return self.n


def sampler_for(ds: SyntheticB6Dataset, seed: int = 123) -> EpochSampler:
    return EpochSampler(len(ds), seed, rows=[ds.row_key(i) for i in range(len(ds))])


def eval_keys(ds: SyntheticB6Dataset) -> List:
    return [(i, 0) for i in range(len(ds))]


def record_sha256(record: Dict) -> str:
    """WP3's ``b6_bindings.canonical_record_sha256`` when importable; otherwise the documented formula (sha256 of
    canonical JSON, sort_keys, separators (",", ":"), without record_sha256 and "_" keys), for tests only."""
    try:
        from transgenic.evidence.b6_bindings import canonical_record_sha256
        return canonical_record_sha256(record)
    except ImportError:
        import json
        body = {k: v for k, v in record.items() if k != "record_sha256" and not k.startswith("_")}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=list)
                              .encode("utf-8")).hexdigest()


def binding_record(**overrides) -> Dict:
    """A training-family record shaped like ``b6_bindings.verify_inputs``' (WP3 of 2026-09-24; placeholder shas,
    tests only), sealed with record_sha256 after the overrides are applied."""
    from transgenic.training.b6_runtime import AUG_SEED, SPEC_SHAS
    sha = lambda tag: hashlib.sha256(tag.encode()).hexdigest()  # noqa: E731
    record = dict(family="training", evidence_sha256=sha("evidence"), roles_sha256=sha("roles"),
                  coverage=[["Athaliana", "est"], ["Athaliana", "protein"]], spec=dict(SPEC_SHAS), aug_seed=AUG_SEED,
                  draws=None, evlabel="absent", exhaustiveness={"Athaliana": {"est": {"C": 0, "M": 100}}},
                  corpus_sha256=sha("corpus"), base_sha256=sha("base"),
                  interval_manifest_sha256={"Athaliana": sha("manifest")}, revalidated={},
                  zero_lists={"train_track": {"Athaliana": sha("zero")}},
                  provenance={"cache_manifest": sha("cm"), "evidence.sha256": sha("ev"), "freeze": sha("fz")},
                  cache_manifest_sha256=sha("cache"))
    spec_overrides = {k: overrides.pop(k) for k in list(overrides) if k in SPEC_SHAS}
    record["spec"].update(spec_overrides)
    seal = overrides.pop("_seal", True)
    record.update(overrides)
    if seal:
        record["record_sha256"] = record_sha256(record)
    return record
