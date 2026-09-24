"""B6 evidence-mode input contract, version 1 (A46 rules 6, 6a, 8, 9; plan v4.1 §7.1-7.3, §7.7).

Single source of truth for everything WP3 produces and WP4 consumes:

* the K = 31 per-nucleotide track channels, their order and their RC permutation (rule 6, P:992);
* the chain-memory element layout, the rule 6a slot hash, the fill order and ``chain-rc-v1`` (rule 6a, P:993);
* the rule 8 molecule rank and the recipient-keyed augmentation uniforms (plan §7.3);
* the §7.7 batch contract (exact keys, dtypes, shapes) and a dtype-name based validator that accepts numpy
  arrays or torch tensors without importing torch;
* ``TRACK_SPEC_SHA``, ``CHAIN_SPEC_SHA`` and ``AUG_SPEC_SHA``: sha256 of the canonical JSON of each spec dict
  (``sort_keys=True``, separators ``(",", ":")``, UTF-8).  They go into ``runtime_identity.txt`` (rule 9) and every
  cache/track manifest; a consumer compares them, never re-derives them.

This module imports only the standard library and numpy.  It is imported by full path (plan §0: new file, not
part of the MVP byte pin).  Changing any value below changes a spec sha and therefore refuses every resume and
every cache built under the old value; that is intended.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

# ------------------------------------------------------------------------------------------------ tracks (rule 6)
TRACK_SPEC_VERSION = "b6-tracks-k31-v1"
TRACK_SOURCES = ("est", "ont", "pacbio")            # block order EST, ONT, PacBio (P:992)
SOURCE_BLOCK_CHANNELS = ("cov+", "cov-", "cov?", "donor+", "acceptor+", "donor-", "acceptor-", "mask")
BLOCK = len(SOURCE_BLOCK_CHANNELS)                   # 8
PROTEIN_CHANNEL = 24                                 # protein CDS log1p(distinct organisms), unstranded
PHASE_CHANNELS = (25, 26, 27, 28, 29, 30)            # (0,+) (1,+) (2,+) (0,-) (1,-) (2,-)
K = 31

OFF_COV_PLUS, OFF_COV_MINUS, OFF_COV_UNKNOWN = 0, 1, 2
OFF_DONOR_PLUS, OFF_ACCEPTOR_PLUS, OFF_DONOR_MINUS, OFF_ACCEPTOR_MINUS = 3, 4, 5, 6
OFF_MASK = 7


def source_base(source):
    """First channel of a track source's block (EST 0, ONT 8, PacBio 16)."""
    return BLOCK * TRACK_SOURCES.index(source)


def channel_names():
    names = [f"{s}:{c}" for s in TRACK_SOURCES for c in SOURCE_BLOCK_CHANNELS]
    names.append("protein:cds")
    names += [f"phase:{p}{st}" for st in ("+", "-") for p in (0, 1, 2)]
    assert len(names) == K
    return tuple(names)


CHANNEL_NAMES = channel_names()
MASK_CHANNELS = tuple(source_base(s) + OFF_MASK for s in TRACK_SOURCES)          # 7, 15, 23
COUNT_CHANNELS = tuple(c for c in range(K) if c not in MASK_CHANNELS)            # everything C4 binarises


def phase_channel(phase, strand):
    """Channel of codon position ``phase`` (0/1/2) on transcript strand ``strand`` ('+'/'-')."""
    if phase not in (0, 1, 2) or strand not in ("+", "-"):
        raise ValueError(f"bad phase/strand {phase!r}/{strand!r}")
    return PHASE_CHANNELS[0] + (0 if strand == "+" else 3) + phase


SOURCE_BLOCK_RC = (1, 0, 2, 5, 6, 3, 4, 7)          # frozen (P:992): strand swapped, donor/acceptor role kept


def _rc_permutation():
    perm = []
    for s in range(len(TRACK_SOURCES)):
        perm += [BLOCK * s + p for p in SOURCE_BLOCK_RC]
    perm.append(PROTEIN_CHANNEL)                     # identity: only coordinate-reversed
    perm += [28, 29, 30, 25, 26, 27]                 # strand swapped, codon-position label kept (P:992)
    assert sorted(perm) == list(range(K))
    assert all(perm[perm[c]] == c for c in range(K)), "the RC permutation must be an involution"
    return tuple(perm)


RC_PERM = _rc_permutation()
TRACK_DTYPE = "float16"
COV_CLIP_MOLECULES = 64                              # coverage clipped at log1p(64) (rule 6)
COV_CLIP = float(np.log1p(COV_CLIP_MOLECULES))


def rc_tracks(tracks):
    """RC a [L, K] track array: ``new[:, c] = old[:, RC_PERM[c]]`` then reverse L.  Twice = identity."""
    arr = np.asarray(tracks)
    if arr.ndim != 2 or arr.shape[1] != K:
        raise ValueError(f"tracks must be [L, {K}], got {arr.shape}")
    return np.ascontiguousarray(arr[::-1][:, list(RC_PERM)])


TRACK_SPEC = {
    "version": TRACK_SPEC_VERSION,
    "K": K,
    "channel_names": list(CHANNEL_NAMES),
    "source_block_order": list(TRACK_SOURCES),
    "source_block_channels": list(SOURCE_BLOCK_CHANNELS),
    "protein_channel": PROTEIN_CHANNEL,
    "phase_channels": list(PHASE_CHANNELS),
    "rc": "new[c] = old[RC_PERM[c]] then reverse L; source block [1,0,2,5,6,3,4,7]+8s; protein identity; "
          "phase [28,29,30,25,26,27] (strand swapped, label kept)",
    "rc_perm": list(RC_PERM),
    "dtype": TRACK_DTYPE,
    "values": {
        "coverage": "log1p(min(distinct molecules covering the base, 64)); a molecule's clipped blocks are unioned "
                    "per molecule (one count per molecule per base); strand = C0 transcript strand (+/-/?)",
        "donor_acceptor": "log1p(distinct molecules with a junction whose donor0/acceptor0 is that base); C0 "
                          "donor0/acceptor0 (0-based inclusive, strand-aware); '?' junctions never counted; "
                          "not clipped",
        "protein": "log1p(distinct OrthoDB organisms among retained protein queries whose accepted primary CDS "
                   "block covers the base); unstranded",
        "phase": "log1p(distinct EST molecules whose route-B est_protein_hsp interval labels the base with (codon "
                 "position, strand)); codon position of base g = (phase + (g-start0)) % 3 on '+', "
                 "(phase - (g-start0)) % 3 on '-' (c0_schema.PHASE_CONVENTION = "
                 "codon_position_at_genomic_start0_v1)",
    },
    "mask": "1 iff the evidence owner's species has an admitted library of that source in the population and "
            "the tile is kept; stays 1 at fraction 0.01..1 with zero molecules and at N bases; 0 when the tile "
            "is dropped, the source is withheld, or its last admitted library is withheld",
    "zero_list": "all six phase channels are zeroed over the population's A48 zero list intervals",
    "n_bases": "DNA N bases are not zeroed (no reference-derived feature, no tile-wise normalisation)",
    "rc_order": "rasterise in genomic orientation, apply zero list, then rc_tracks iff the tile is RC",
    "c4": "count channels (every channel except the three masks) become 1.0 where > 0; masks unchanged",
    "c6": "channels 25-30 zeroed",
    "c2": "all 31 channels zero",
}

# ------------------------------------------------------------------------------------------ chains (rule 6a)
CHAIN_SPEC_VERSION = "b6-chain-memory-v1"
CHAIN_RC_RULE = "chain-rc-v1"
N_TOK = 512                                          # elements per tile, element 0 = dummy
N_CHAIN_SLOTS = 64                                   # E_slot rows; slot 0 = dummy
CHAIN_SLOTS = tuple(range(1, N_CHAIN_SLOTS))         # chains use slots 1..63 -> at most 63 chains per tile
MAX_CHAINS = len(CHAIN_SLOTS)
ORDER_CAP = 31
UNITS_CAP = 64
DOWNSAMPLE = 6                                       # encoder memory stride
TOKEN_OFFSET = 0                                     # HyenaDNA char tokenizer: no leading special token
CHAIN_SOURCES = ("est", "ont", "pacbio")             # source code 0 / 1 / 2
CHAIN_STRANDS = ("+", "-")                           # strand code 0 / 1; '?' chains are not used
CHAIN_FEAT_FIELDS = ("order_fwd", "order_rev", "slot", "source", "strand", "units_bin")
FEAT_ORDER_FWD, FEAT_ORDER_REV, FEAT_SLOT, FEAT_SOURCE, FEAT_STRAND, FEAT_UNITS = range(6)
N_FEAT = len(CHAIN_FEAT_FIELDS)
POS_DONOR, POS_ACCEPTOR = 0, 1
LONG_READ_SOURCES = ("ont", "pacbio")
# Final tie-break of the fill order (D8): PacBio before ONT.  EST chains are filled after every long-read chain.
FILL_SOURCE_RANK = {"pacbio": 0, "ont": 1, "est": 2}

MODEL_DIMS = {                                       # plan §9 projections, frozen with the chain spec (rule 9)
    "d_encoder_memory": 1536, "d_chain": 512, "layers": 12, "heads": 8, "head_dim": 64,
    "W_d": [1536, 512], "W_a": [1536, 512], "Q": [1536, 512], "K": [512, 512], "V": [512, 512],
    "out": [512, 1536], "E_order": ORDER_CAP + 1, "E_order_rev": ORDER_CAP + 1, "E_slot": N_CHAIN_SLOTS,
    "E_source": len(CHAIN_SOURCES), "E_strand": len(CHAIN_STRANDS), "w_count": "scalar * log1p(units_bin)",
    "block_dropout": 0.1, "alpha_init": 0.0, "alpha_weight_decay": False,
    "lr_group": "B5 recipe learning rate and schedule; alpha excluded from weight decay",
    "gate": "x <- x + tanh(alpha_l) * m * out, m = chain_mask.any(-1); after encoder_attn_layer_norm",
    "kv_cache": "none",
}


def _blake2b_u64(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


SLOT_HASH_ENCODING = 'blake2b(f"{aug_seed}\\t{row}\\t{epoch}\\t{s}".encode("utf-8"), digest_size=8), big-endian'
TIE_KEY_ENCODING = 'blake2b(f"{species}\\t{strand}\\t{introns}".encode("utf-8"), digest_size=8), big-endian; ' \
                   'introns = ",".join(f"{donor0}-{acceptor0}") over genomic junctions in transcription order'
RANK_ENCODING = 'blake2b(f"{aug_seed}\\t{species}\\t{molecule_id}".encode("utf-8"), digest_size=8), ' \
                'big-endian / 2**64'
UNIFORM_ENCODING = 'blake2b(f"{aug_seed}\\t{p}\\t{row}\\t{epoch}".encode("utf-8"), digest_size=8), ' \
                   'big-endian / 2**64'


def _check_row(row):
    if not isinstance(row, str) or not row:
        raise ValueError(f"row key must be a nonempty str (decimal rn or eval window key), got {row!r}")


def slot_hash(aug_seed, row, epoch, s):
    """Rule 6a slot hash h(s) for s in 1..63."""
    _check_row(row)
    if s not in CHAIN_SLOTS:
        raise ValueError(f"slot {s} outside 1..{N_CHAIN_SLOTS - 1}")
    return _blake2b_u64(f"{int(aug_seed)}\t{row}\t{int(epoch)}\t{int(s)}")


def slot_permutation(aug_seed, row, epoch):
    """pi = slots 1..63 sorted by (h(s), s); the chain at fill rank r (1-based) gets pi[r-1]."""
    return tuple(sorted(CHAIN_SLOTS, key=lambda s: (slot_hash(aug_seed, row, epoch, s), s)))


def introns_text(introns):
    """``introns`` = iterable of (donor0, acceptor0) in transcription order -> the tie-key intron list text."""
    return ",".join(f"{int(d)}-{int(a)}" for d, a in introns)


def tie_key(species, strand, introns):
    if strand not in CHAIN_STRANDS:
        raise ValueError(f"chain strand must be + or -, got {strand!r}")
    return _blake2b_u64(f"{species}\t{strand}\t{introns_text(introns)}")


def fill_sort_key(source, units, n_introns, key):
    """Sort key of rule 6a fill: long reads first, then units desc, intron count desc, tie key asc, PacBio < ONT."""
    return (0 if source in LONG_READ_SOURCES else 1, -int(units), -int(n_introns), int(key), FILL_SOURCE_RANK[source])


def chain_idx(chain_pos, chain_mask, pad):
    """Batch ``chain_idx`` = (pad + TOKEN_OFFSET + chain_pos) // 6, one sum then one division; 0 where masked."""
    pos = np.asarray(chain_pos, dtype=np.int64)
    idx = (int(pad) + TOKEN_OFFSET + pos) // DOWNSAMPLE
    return np.where(np.asarray(chain_mask, dtype=bool)[..., None], idx, 0).astype(np.int64)


def rc_chain(chain_pos, chain_feat, chain_mask, L):
    """``chain-rc-v1``: p -> L-1-p for real elements, donor/acceptor roles kept, strand swapped, order, slot,
    source and units unchanged; masked elements stay all-zero.  Twice = identity."""
    pos = np.asarray(chain_pos, dtype=np.int32)
    feat = np.asarray(chain_feat, dtype=np.int16)
    mask = np.asarray(chain_mask, dtype=bool)
    if pos.shape != (N_TOK, 2) or feat.shape != (N_TOK, N_FEAT) or mask.shape != (N_TOK,):
        raise ValueError("chain arrays must be [512,2], [512,6], [512]")
    if mask[0]:
        raise ValueError("element 0 is the dummy and must be masked")
    new_pos = np.where(mask[:, None], (int(L) - 1) - pos, 0).astype(np.int32)
    new_feat = feat.copy()
    new_feat[mask, FEAT_STRAND] = 1 - feat[mask, FEAT_STRAND]
    return new_pos, new_feat, mask.copy()


CHAIN_SPEC = {
    "version": CHAIN_SPEC_VERSION,
    "rc_rule": CHAIN_RC_RULE,
    "rc": "p -> L-1-p for donor and acceptor positions of real elements; donor/acceptor roles kept; strand "
          "swapped (0<->1); order_fwd, order_rev, slot, source, units_bin unchanged; masked elements all zero; "
          "twice = identity",
    "n_tok": N_TOK, "n_chain_slots": N_CHAIN_SLOTS, "chain_slots": [CHAIN_SLOTS[0], CHAIN_SLOTS[-1]],
    "max_chains": MAX_CHAINS,
    "dummy": "element 0: mask False, positions 0, features 0, always present so the branch executes",
    "element": "one per (chain, intron); chain_pos[:,0] = donor0, chain_pos[:,1] = acceptor0 (C0 junction "
               "boundary bases, 0-based inclusive, strand-aware), tile-local, in output orientation",
    "feature_fields": list(CHAIN_FEAT_FIELDS),
    "order": "order_fwd = min(i, 31), order_rev = min(n-1-i, 31), i = intron index in transcription direction",
    "order_cap": ORDER_CAP,
    "units": "units_bin = min(distinct independent molecule units of the merged chain, 64); model uses "
             "log1p(units_bin)",
    "units_cap": UNITS_CAP,
    "source_codes": {"est": 0, "ont": 1, "pacbio": 2},
    "strand_codes": {"+": 0, "-": 1},
    "chain_unit": "ordered accepted-intron list of one accepted primary placement (C0 partial_chain + "
                  "partial_chain_intron); '?' strand and mono-exonic excluded; audit placements never used",
    "merge": "identical (species, source, strand, intron list) chains merged within a source only (D8); "
             "units = distinct molecule_id among retained molecules",
    "split": "training/validation tiles: split at every rule 4(c) excluded junction, at every non-accepted "
             "junction, at junctions touching a tile rule 4(a) interval (edge_partial included) and at tile "
             "edges; evaluation windows: split at window edges only; a fragment keeps >= 1 intron",
    "fill": "long-read (ONT, PacBio) chains first, then EST; within each: units desc, intron count desc, tie "
            "key asc, then PacBio < ONT; a chain that does not fit whole in the remaining elements or slots is "
            "dropped whole; truncations counted per source",
    "fill_source_rank": FILL_SOURCE_RANK,
    "tie_key_encoding": TIE_KEY_ENCODING,
    "slot_hash_encoding": SLOT_HASH_ENCODING,
    "slot_permutation": "pi = slots 1..63 sorted by (h(s), s); chain at fill rank r (1-based) gets pi[r-1]; "
                        "row = decimal rn (training/validation) or the eval window key; epoch = 0 at evaluation; "
                        "never consumes an augmentation stream",
    "populations": {"train_track": "train tiles; train_tiles_excluded molecules dropped",
                    "valid_track": "valid tiles; train_tiles_excluded molecules admitted",
                    "eval_input": "evaluation windows; b6_input datasets only, selected by dataset/run; "
                                  "b6_score molecule is a hard error; tracks_only datasets never feed chains"},
    "chain_idx": "(pad_b + TOKEN_OFFSET + chain_pos) // 6, one sum then one division; 0 where masked; "
                 "TOKEN_OFFSET = 0; downsample 6",
    "token_offset": TOKEN_OFFSET, "downsample": DOWNSAMPLE,
    "model": MODEL_DIMS,
}

# ---------------------------------------------------------------------------------- augmentation (rule 8, §7.3)
AUG_SPEC_ID = "b6-aug-uniform-blake2b-v2"
AUG_SEED_BASE = 20260914
TRAINING_SEED = 123
FRACTIONS = (0.01, 0.1, 0.5, 1.0)
DOSE_FRACTIONS = (1.0, 0.5, 0.1, 0.01, 0.0)
DOSE_DRAWS = (0, 1, 2, 3, 4)
DROPOUT_P = 0.5
WITHHOLD_SOURCE_RATE = 0.1
WITHHOLD_LIBRARY_RATE = 0.1
PURPOSES = ("dropout", "fraction", "withhold_source", "withhold_source_pick", "withhold_library",
            "withhold_library_pick")
ARMS = ("B6", "C2", "C3", "C4", "C6", "C7")


def training_aug_seed(seed=TRAINING_SEED):
    """``aug_seed`` = 20260914 + seed (123 -> 20261037)."""
    return AUG_SEED_BASE + int(seed)


def dose_aug_seed(draw):
    if draw not in DOSE_DRAWS:
        raise ValueError(f"dose-response draw must be 0..4, got {draw!r}")
    return AUG_SEED_BASE + int(draw)


def molecule_rank(aug_seed, species, molecule_id):
    """Rule 8 rank in [0, 1); a draw at fraction f keeps rank < f (nested by construction)."""
    return _blake2b_u64(f"{int(aug_seed)}\t{species}\t{molecule_id}") / 2 ** 64


def uniform(aug_seed, purpose, row, epoch):
    """Recipient-keyed uniform u(p) in [0, 1) (plan §7.3).  Purposes are non-numeric, so they never collide with
    the slot hash key, whose last field is a slot number."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}")
    _check_row(row)
    return _blake2b_u64(f"{int(aug_seed)}\t{purpose}\t{row}\t{int(epoch)}") / 2 ** 64


AUG_SPEC = {
    "id": AUG_SPEC_ID,
    "aug_seed": "20260914 + seed for training (123 -> 20261037); 20260914 + d for dose-response draw d in 0..4",
    "rank_encoding": RANK_ENCODING,
    "uniform_encoding": UNIFORM_ENCODING,
    "purposes": list(PURPOSES),
    "draw_spec": "DrawSpec = the recipient's six uniforms keyed by (aug_seed, purpose, row, epoch); no "
                 "torch.Generator; resolved identities are QC only",
    "resolve": {
        "dropped": "u_dropout < 0.5",
        "fraction": "{0.01, 0.1, 0.5, 1}[floor(4 * u_fraction)]",
        "withhold_source": "fires iff u_withhold_source < 0.1; picks sorted(owner.sources)[floor(u_pick * n)] "
                           "among est/ont/pacbio present; none present -> recorded no-op",
        "withhold_library": "fires iff u_withhold_library < 0.1; picks sorted(owner.libraries)[floor(u_pick * n)] "
                            "over (source, library) pairs of est/ont/pacbio; withholding a source's last library "
                            "zeroes its mask",
        "est_withheld": "removes phase channels too (phase is EST-derived)",
        "owner": "the recipient in every arm except C3, where it is the donor",
    },
    "protein_d7": "thinning is query-level: rule 8 rank with molecule_id := the protein query_id (the miniprot "
                  "query / OrthoDB protein id as aligned, e.g. '29760:4'), never the C0 molecule_id; the C0 "
                  "molecule_id is used only for exclusion and QC; channel 24 = "
                  "log1p(distinct OrthoDB organisms among retained queries covering the base); protein is never "
                  "withheld and drops only with channel dropout",
    "fractions": list(FRACTIONS), "dose_fractions": list(DOSE_FRACTIONS), "dose_draws": list(DOSE_DRAWS),
    "dropout_p": DROPOUT_P, "withhold_source_rate": WITHHOLD_SOURCE_RATE,
    "withhold_library_rate": WITHHOLD_LIBRARY_RATE,
    "dose_fraction_zero": "fraction 0 keeps masks at 1 with no molecules (evidence present, none retained)",
    "arms": {"B6": "tracks and chains from one resolved draw", "C2": "all 31 channels zero, chains fully masked",
             "C3": "donor tile's evidence resolved with the recipient's uniforms against the donor's admitted "
                   "sources/libraries, RC-transformed iff the recipient is RC; donor = next forward row, "
                   "cyclically, in (tier, split) ordered by blake2b(f\"{aug_seed}\\tc3\\t{tier}\\t{split}\\t{wid}\")",
             "C4": "count channels binarised, masks unchanged, chains as B6",
             "C6": "channels 25-30 zeroed, chains as B6", "C7": "tracks as B6, chain elements all masked"},
    "sampler": "B6EpochSampler yields (rn, epoch); workers hold no RNG state",
}

# ------------------------------------------------------------------------------------ batch contract (§7.7)
BATCH_KEYS = ("input_ids", "attention_mask", "evidence_tracks", "labels", "chain_idx", "chain_feat", "chain_mask",
              "meta")
BATCH_CONTRACT = {
    "input_ids": ("int64", ("B", "Lmax"), "left-padded"),
    "attention_mask": ("bool", ("B", "Lmax"), "left-padded, True = real (B5 collate dtype)"),
    "evidence_tracks": ("float16", ("B", "Lmax", K), "left-padded zeros"),
    "labels": ("int64", ("B", "T"), "right-padded -100; training only (None otherwise)"),
    "chain_idx": ("int64", ("B", N_TOK, 2), "(pad_b + 0 + chain_pos) // 6; 0 where masked; < Lmax/6"),
    "chain_feat": ("int16", ("B", N_TOK, N_FEAT), "cast to int64 before embedding"),
    "chain_mask": ("bool", ("B", N_TOK), "True = real; column 0 always False"),
    "meta": ("dict", (), "row, epoch, arm, uniforms_sha256, resolved draw, context kind"),
}
META_KEYS = ("row", "epoch", "arm", "uniforms_sha256", "resolved", "context_kind")
LABEL_PAD = -100


def _dtype_name(x):
    name = str(getattr(x, "dtype", type(x).__name__))
    return name.replace("torch.", "")


def validate_batch(batch, *, training):
    """Check a collated batch against §7.7 (numpy arrays or torch tensors).  Raises ValueError."""
    if set(batch) != set(BATCH_KEYS):
        raise ValueError(f"batch keys {sorted(batch)} != {sorted(BATCH_KEYS)}")
    ids = batch["input_ids"]
    if _dtype_name(ids) != "int64" or len(ids.shape) != 2:
        raise ValueError("input_ids must be int64 [B, Lmax]")
    B, Lmax = (int(v) for v in ids.shape)
    if Lmax % DOWNSAMPLE:
        raise ValueError(f"Lmax {Lmax} is not a multiple of {DOWNSAMPLE}")
    shapes = {"attention_mask": (B, Lmax), "evidence_tracks": (B, Lmax, K), "chain_idx": (B, N_TOK, 2),
              "chain_feat": (B, N_TOK, N_FEAT), "chain_mask": (B, N_TOK)}
    for key, shape in shapes.items():
        x = batch[key]
        want = BATCH_CONTRACT[key][0]
        if _dtype_name(x) != want or tuple(int(v) for v in x.shape) != shape:
            raise ValueError(f"{key}: want {want}{list(shape)}, got {_dtype_name(x)}{list(x.shape)}")
    labels = batch["labels"]
    if training:
        if labels is None or _dtype_name(labels) != "int64" or len(labels.shape) != 2 or int(labels.shape[0]) != B:
            raise ValueError("labels must be int64 [B, T] in training")
    elif labels is not None:
        raise ValueError("labels must be None outside training")
    mask = np.asarray(batch["chain_mask"])
    if mask[:, 0].any():
        raise ValueError("chain_mask column 0 must be False")
    idx = np.asarray(batch["chain_idx"])
    if (idx[~mask] != 0).any():
        raise ValueError("masked chain_idx must be 0")
    if (idx < 0).any() or (idx >= Lmax // DOWNSAMPLE).any():
        raise ValueError("chain_idx out of [0, Lmax/6)")
    feat = np.asarray(batch["chain_feat"])
    if (feat[~mask] != 0).any():
        raise ValueError("masked chain_feat must be 0")
    meta = batch["meta"]
    if not isinstance(meta, dict) or set(meta) != set(META_KEYS):
        raise ValueError(f"meta keys must be {META_KEYS}")
    return B, Lmax


# ----------------------------------------------------------------------------------------------- spec shas
def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


TRACK_SPEC_SHA = canonical_sha256(TRACK_SPEC)
CHAIN_SPEC_SHA = canonical_sha256(CHAIN_SPEC)
AUG_SPEC_SHA = canonical_sha256(AUG_SPEC)
SPEC_SHAS = {"track_spec_sha256": TRACK_SPEC_SHA, "chain_spec_sha256": CHAIN_SPEC_SHA,
             "aug_spec_sha256": AUG_SPEC_SHA, "aug_spec_id": AUG_SPEC_ID}
