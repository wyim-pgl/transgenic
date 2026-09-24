"""B6 contexts (plan §7.4): two constructors, one type.

* ``corpus_context(corpus, rn, mode)`` serves train and valid tiles of the B6 corpus (row key = decimal rn).
* ``eval_context(window, *, reference, panel, roles, generator, cache)`` serves evaluation windows with no corpus
  membership (row key = ``eval:{species}:{chrom}:{ws}-{we}:{fw|rc}:{panel_id}``).

A ``Context`` also owns row admission (``admit``): which cached observation rows a raster/chain build may use for
this tile, per population.  Admission is the single place where rule 4 / rule 10 policy is applied at load time;
``b6_tracks`` and ``b6_chains`` call it and never re-derive it.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transgenic.evidence import b6_track_spec as TS

GENERATORS = ("centred30720-v1", "a27-tiles-v1")
CORPUS_MODES = {"train_track": "train", "valid_track": "valid"}
EVAL_ONLY_SPECIES = ("Zmays",)          # no B5/B6 corpus rows (data/manifests/b5_excluded_species_v1.tsv:2)


class ContextError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise ContextError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq):
    return seq.translate(_COMP)[::-1]


@dataclass(frozen=True)
class Context:
    row: str
    species: str
    chrom: str
    ws: int
    we: int
    L: int
    is_rc: bool
    kind: str                           # corpus | eval
    split: str                          # train | valid | eval
    population: str                     # train_track | valid_track | eval_input
    tile_exclusions: tuple              # ((start0, end0, cause, scope), ...) genomic, rule 4(a), corpus only
    excluded_junctions: frozenset       # {(strand, 'donor'|'acceptor', coordinate0)}, rule 4(c), corpus only
    dna: str                            # output-oriented (RC iff is_rc)
    dna_n_mask: np.ndarray = field(compare=False, repr=False)
    zero_intervals: tuple = ()          # A48 phase zero list, genomic, clipped to [ws, we)
    roles_scope: str = "training"       # training | eval_input
    panel_locus: tuple | None = None    # (locus_id, chrom, start0, end0) at evaluation
    family: str = "train"               # cache family the context reads
    meta: dict = field(default_factory=dict, compare=False)

    def admit(self, cols, idx, *, chains=False):
        """Boolean mask over cache row indices ``idx`` admitted for this context's population."""
        state = cols["molecule_state"][idx]
        tte = cols["train_tiles_excluded"][idx].astype(bool)
        b_in = cols["b6_input"][idx].astype(bool)
        b_sc = cols["b6_score"][idx].astype(bool)
        _require(not b_sc.any(), f"{self.row}: a b6_score observation reached admission (cache or role breach)")
        if self.population == "train_track":
            ok = (state == "train_track") & ~tte
        elif self.population == "valid_track":
            ok = state == "train_track"
        else:
            ok = b_in.copy()
            if chains:
                ok &= cols["b6_input_scope"][idx] == "tracks+chains"
            if self.family == "train":
                # A. thaliana: excluded_heldout molecules whose primary placement overlaps the panel locus
                ok &= state == "excluded_heldout"
                mids = cols["molecule_id"][idx]
                allowed = self.meta.get("locus_molecules")
                _require(allowed is not None, f"{self.row}: eval context without panel-locus molecule set")
                ok &= np.fromiter((m in allowed for m in mids), bool, len(mids))
        return ok


# ------------------------------------------------------------------------------------------- reference FASTA
class Reference:
    """Indexed FASTA reader (faidx-style, own index); sha256 of the file bytes is the identity."""

    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = sha256_file(self.path)
        self._index = {}
        self._fh = None
        self._pid = None
        with open(self.path, "rb") as fh:
            name, seq_off, line_bases, line_bytes, length = None, 0, 0, 0, 0
            pos = 0
            for line in fh:
                if line.startswith(b">"):
                    if name is not None:
                        self._index[name] = (length, seq_off, line_bases, line_bytes)
                    name = line[1:].split()[0].decode()
                    seq_off, line_bases, line_bytes, length = pos + len(line), 0, 0, 0
                else:
                    n = len(line.rstrip(b"\r\n"))
                    if line_bases == 0:
                        line_bases, line_bytes = n, len(line)
                    length += n
                pos += len(line)
            if name is not None:
                self._index[name] = (length, seq_off, line_bases, line_bytes)

    def length(self, chrom):
        _require(chrom in self._index, f"{chrom} not in {self.path}")
        return self._index[chrom][0]

    def fetch(self, chrom, start0, end0):
        length, off, lb, lbytes = self._index.get(chrom, (None, 0, 0, 0))
        _require(length is not None, f"{chrom} not in {self.path}")
        _require(0 <= start0 <= end0 <= length, f"{chrom}:{start0}-{end0} outside [0, {length}]")
        if self._fh is None or self._pid != os.getpid():
            self._fh, self._pid = open(self.path, "rb"), os.getpid()
        if start0 == end0:
            return ""
        a = off + (start0 // lb) * lbytes + start0 % lb
        b = off + ((end0 - 1) // lb) * lbytes + (end0 - 1) % lb + 1
        self._fh.seek(a)
        raw = self._fh.read(b - a).replace(b"\n", b"").replace(b"\r", b"")
        _require(len(raw) == end0 - start0, f"{self.path}: irregular line lengths at {chrom}:{start0}")
        return raw.decode("ascii").upper()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_fh"] = d["_pid"] = None
        return d


# --------------------------------------------------------------------------------------------------- panels
@dataclass(frozen=True)
class PanelLocus:
    panel: str
    locus_id: str
    species: str
    chrom: str
    start0: int
    end0: int
    strand: str


class Panel:
    """Frozen locus panel (b6_panels_v1.tsv format).  ``expected_sha256`` is mandatory: A. thaliana =
    mvp_windows.PANEL_SHA256 (P rule 12); the Z. mays §9 list sha is recorded before first use."""

    def __init__(self, path, expected_sha256, panel_id):
        _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64, "panel sha256 is required")
        self.path = Path(path)
        self.sha256 = sha256_file(self.path)
        _require(self.sha256 == expected_sha256, f"panel {path} sha256 {self.sha256} != frozen {expected_sha256}")
        self.panel_id = panel_id
        lines = self.path.read_text().splitlines()
        header = lines[0].split("\t")
        need = ("panel", "species_id", "gene_id", "chromosome", "start", "fin", "strand")
        _require(all(c in header for c in need), f"{path}: header lacks {need}")
        col = {c: header.index(c) for c in header}
        self.loci = {}
        for line in lines[1:]:
            f = line.split("\t")
            if f[col["panel"]] != panel_id:
                continue
            loc = PanelLocus(panel_id, f[col["gene_id"]], f[col["species_id"]], f[col["chromosome"]],
                             int(f[col["start"]]) - 1, int(f[col["fin"]]), f[col["strand"]])
            _require(loc.locus_id not in self.loci, f"{path}: duplicate locus {loc.locus_id}")
            self.loci[loc.locus_id] = loc
        _require(bool(self.loci), f"{path}: no locus in panel {panel_id!r}")


@dataclass(frozen=True)
class EvalWindow:
    species: str
    chrom: str
    ws: int
    we: int
    is_rc: bool
    locus_id: str
    generator: str
    tier: int | None = None
    offset: int | None = None


class WindowManifest:
    """The sha-bound list of evaluation windows (TSV: species chrom ws we is_rc locus_id generator tier offset)."""

    COLUMNS = ("species", "chrom", "ws", "we", "is_rc", "locus_id", "generator", "tier", "offset")

    def __init__(self, path, *, sidecar=None):
        self.path = Path(path)
        side = Path(sidecar) if sidecar else Path(str(path) + ".sha256")
        _require(side.is_file(), f"window manifest sidecar {side} missing")
        self.sha256 = sha256_file(self.path)
        _require(side.read_text().split()[0] == self.sha256, f"{path}: sha256 != sidecar")
        lines = self.path.read_text().splitlines()
        _require(tuple(lines[0].split("\t")) == self.COLUMNS, f"{path}: header != {self.COLUMNS}")
        self.windows = []
        for line in lines[1:]:
            f = line.split("\t")
            self.windows.append(EvalWindow(f[0], f[1], int(f[2]), int(f[3]), f[4] == "true", f[5], f[6],
                                           int(f[7]) if f[7] else None, int(f[8]) if f[8] else None))

    @staticmethod
    def write(path, windows):
        with open(path, "w") as fh:
            fh.write("\t".join(WindowManifest.COLUMNS) + "\n")
            for w in windows:
                fh.write("\t".join([w.species, w.chrom, str(w.ws), str(w.we), "true" if w.is_rc else "false",
                                    w.locus_id, w.generator, "" if w.tier is None else str(w.tier),
                                    "" if w.offset is None else str(w.offset)]) + "\n")
        Path(str(path) + ".sha256").write_text(f"{sha256_file(path)}  {Path(path).name}\n")


def window_key(species, chrom, ws, we, is_rc, panel_id):
    return f"eval:{species}:{chrom}:{ws}-{we}:{'rc' if is_rc else 'fw'}:{panel_id}"


def a27_tiling():
    """(WINDOW_TIERS, tile_windows) of ``gsf_contract`` (A27).  gsf_contract imports torch through postprocess, so a
    torch-free process loads the two definitions from the pinned source file itself (no copy, no drift)."""
    try:
        from transgenic.utils import gsf_contract as gc

        return gc.WINDOW_TIERS, gc.tile_windows
    except ImportError:
        import ast

        src = Path(__file__).resolve().parents[1] / "utils" / "gsf_contract.py"
        tree = ast.parse(src.read_text())
        keep = [n for n in tree.body
                if (isinstance(n, ast.FunctionDef) and n.name == "tile_windows")
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "WINDOW_TIERS" for t in n.targets))]
        _require(len(keep) == 2, f"{src}: tile_windows / WINDOW_TIERS not found")
        ns = {"List": list, "Tuple": tuple}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(src), "exec"), ns)  # noqa: S102 (pinned source)
        return ns["WINDOW_TIERS"], ns["tile_windows"]


def _generate(window, locus, chrom_len):
    """Reproduce (ws, we) from the named generator; fail closed."""
    if window.generator == "centred30720-v1":
        from transgenic.evidence import mvp_windows as MW

        pl = MW.PanelLocus(locus.panel, locus.locus_id, locus.chrom, locus.start0 + 1, locus.end0, locus.strand,
                           locus.end0 - locus.start0)
        return MW.evaluation_window(pl, chrom_len)
    if window.generator == "a27-tiles-v1":
        tiers, tile_windows = a27_tiling()
        _require(window.tier in tiers, f"a27-tiles-v1 needs a recorded tier, got {window.tier}")
        _require(window.offset is not None and 0 <= window.offset < window.tier, "a27-tiles-v1 needs an offset")
        tiles = tile_windows(chrom_len, window.tier, window.offset)
        _require((window.ws, window.we) in tiles, f"({window.ws}, {window.we}) is not an a27 tile")
        _require(window.ws < locus.end0 and window.we > locus.start0, "a27 tile does not overlap its locus")
        return window.ws, window.we
    raise ContextError(f"unknown window generator {window.generator!r}; allowed {GENERATORS}")


def _n_mask(dna):
    b = np.frombuffer(dna.encode("ascii"), np.uint8)
    return (b == ord("N")) | (b == ord("n"))


def locus_molecules(cache, species, chrom, start0, end0):
    """Molecule ids with an accepted primary placement overlapping [start0, end0) on chrom (eval admission)."""
    sh, idx = cache.track_rows(species, chrom, start0, end0)
    kinds = sh.cols["kind"][idx]
    return frozenset(sh.cols["molecule_id"][idx[kinds == "alignment"]].tolist())


def eval_context(window, *, reference, panel, roles, generator, cache, window_manifest):
    """Evaluation window context; needs no corpus.  All identities are verified before any row is admitted."""
    _require(window.generator == generator, f"window generator {window.generator!r} != requested {generator!r}")
    _require(generator in GENERATORS, f"unknown generator {generator!r}")
    _require(window in window_manifest.windows, "window is not in the sha-bound window manifest")
    refs = cache.manifest["bindings"].get("reference_sha256")
    _require(isinstance(refs, dict) and window.species in refs,
             f"cache binds no reference_sha256 for {window.species} (b6_build_obs_cache verifies it against C0 "
             "inputs.reference)")
    ref_sha = refs[window.species]
    _require(reference.sha256 == ref_sha, f"reference sha256 {reference.sha256} != evidence inputs.reference {ref_sha}")
    _require(cache.manifest["bindings"].get("roles_sha256") == roles.sha256,
             "roles sha256 differs from the one the cache was built with")
    _require(window.locus_id in panel.loci, f"locus {window.locus_id!r} not in panel {panel.panel_id}")
    locus = panel.loci[window.locus_id]
    _require(locus.species == window.species and locus.chrom == window.chrom, "window/locus species or chrom differ")
    ws, we = _generate(window, locus, reference.length(window.chrom))
    _require((ws, we) == (window.ws, window.we), f"window ({window.ws}, {window.we}) != generator ({ws}, {we})")
    family = cache.family
    if window.species in EVAL_ONLY_SPECIES:
        _require(family == "eval", f"{window.species} evaluation needs the eval-only cache family")
    else:
        _require(family == "train", f"{window.species} evaluation reads its training-family cache")
    dna = reference.fetch(window.chrom, ws, we)
    if window.is_rc:
        dna = revcomp(dna)
    row = window_key(window.species, window.chrom, ws, we, window.is_rc, panel.panel_id)
    meta = {"reference_sha256": reference.sha256, "panel_sha256": panel.sha256,
            "window_manifest_sha256": window_manifest.sha256, "roles_sha256": roles.sha256, "generator": generator}
    if family == "train":
        meta["locus_molecules"] = locus_molecules(cache, window.species, window.chrom, locus.start0, locus.end0)
    return Context(row=row, species=window.species, chrom=window.chrom, ws=ws, we=we, L=we - ws, is_rc=window.is_rc,
                   kind="eval", split="eval", population="eval_input", tile_exclusions=(),
                   excluded_junctions=frozenset(), dna=dna, dna_n_mask=_n_mask(dna),
                   zero_intervals=tuple(cache.zero_intervals("eval_input", window.species, window.chrom, ws, we)),
                   roles_scope="eval_input", panel_locus=(locus.locus_id, locus.chrom, locus.start0, locus.end0),
                   family=family, meta=meta)


# --------------------------------------------------------------------------------------------------- corpus
class B6Corpus:
    """Read-only B6 corpus (``b5_full_b6_v1.db``) + the training-family cache it is consumed with."""

    def __init__(self, db_path, cache, *, expected_sha256=None):
        import duckdb

        self.path = Path(db_path)
        _require(not Path(str(db_path) + ".wal").exists(), f"{db_path}.wal exists; refusing an unsealed snapshot")
        self.sha256 = sha256_file(self.path) if expected_sha256 is not None else None
        if expected_sha256 is not None:
            _require(self.sha256 == expected_sha256, f"corpus sha256 {self.sha256} != {expected_sha256}")
        self.cache = cache
        _require(cache.family == "train", "a corpus context needs the training-family cache")
        self._con = None
        self._pid = None
        self._duckdb = duckdb

    def con(self):
        if self._con is None or self._pid != os.getpid():
            self._con = self._duckdb.connect(str(self.path), read_only=True)
            self._pid = os.getpid()
        return self._con

    def row(self, rn):
        r = self.con().execute("SELECT rn, species_id, chromosome, start, fin, is_rc, split, sequence, gene_id, gff "
                               "FROM geneList WHERE rn = ?", [int(rn)]).fetchall()
        _require(len(r) == 1, f"rn {rn} not in corpus")
        return r[0]

    def twin_rns(self, wid):
        return {int(x[0]) for x in self.con().execute("SELECT rn FROM geneList WHERE gene_id = ?", [wid]).fetchall()}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_con"] = d["_pid"] = None
        d.pop("_duckdb", None)
        return d

    def __setstate__(self, d):
        import duckdb

        self.__dict__.update(d)
        self._duckdb = duckdb


def corpus_context(corpus, rn, mode):
    """Train/valid tile context.  ``mode`` = population ('train_track' | 'valid_track'), which must match the
    tile's split; test tiles and eval-only species are refused."""
    _require(mode in CORPUS_MODES, f"mode {mode!r} not in {tuple(CORPUS_MODES)}")
    rn_, species, chrom, ws, we, is_rc, split, dna, wid, _gff = corpus.row(rn)
    _require(species not in EVAL_ONLY_SPECIES, f"{species} has no corpus role; refusing a corpus context")
    _require(split != "test", f"rn {rn} is a test tile; refusing (rule 4)")
    _require(split == CORPUS_MODES[mode], f"rn {rn} split {split!r} does not match mode {mode!r}")
    ws, we = int(ws), int(we)
    _require(len(dna) == we - ws, f"rn {rn}: sequence length {len(dna)} != {we - ws}")
    cache = corpus.cache
    twins = corpus.twin_rns(wid)
    excl = []
    for s0, e0, cause, scope, tile_rn in cache.excluded_intervals(species, chrom):
        if not (s0 < we and e0 > ws):
            continue
        if scope == "train_tiles" and split != "train":
            continue
        if tile_rn >= 0 and tile_rn not in twins:
            continue
        excl.append((s0, e0, cause, scope))
    return Context(row=str(int(rn_)), species=species, chrom=chrom, ws=ws, we=we, L=we - ws, is_rc=bool(is_rc),
                   kind="corpus", split=split, population=mode, tile_exclusions=tuple(sorted(excl)),
                   excluded_junctions=cache.excluded_junctions(species, chrom), dna=dna, dna_n_mask=_n_mask(dna),
                   zero_intervals=tuple(cache.zero_intervals(mode, species, chrom, ws, we)), roles_scope="training",
                   panel_locus=None, family="train", meta={"wid": wid, "tier": we - ws})


def donor_context(corpus, recipient, donor_wid):
    """C3: the donor forward tile's context under the recipient's population, re-keyed to the recipient row.

    Evidence comes from the donor's window and owner; orientation follows the recipient (RC iff recipient RC)."""
    rns = [r for r in corpus.twin_rns(donor_wid)]
    fw = None
    for r in sorted(rns):
        row = corpus.row(r)
        if not row[5]:
            fw = r
    _require(fw is not None, f"C3 donor {donor_wid} has no forward row")
    d = corpus_context(corpus, fw, recipient.population)
    _require(d.L == recipient.L, f"C3 donor length {d.L} != recipient {recipient.L} (tier mismatch)")
    return Context(row=recipient.row, species=d.species, chrom=d.chrom, ws=d.ws, we=d.we, L=d.L,
                   is_rc=recipient.is_rc, kind="corpus", split=d.split, population=d.population,
                   tile_exclusions=d.tile_exclusions, excluded_junctions=d.excluded_junctions,
                   dna=recipient.dna, dna_n_mask=recipient.dna_n_mask, zero_intervals=d.zero_intervals,
                   roles_scope="training", panel_locus=None, family="train",
                   meta={"wid": d.meta["wid"], "tier": d.meta["tier"], "c3_donor_rn": fw})
