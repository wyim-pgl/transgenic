#!/usr/bin/env python3
"""Swiss-Prot reviewed Viridiplantae as a separately labelled sensitivity/audit set (protocol A30).

Input: the UniProtKB/Swiss-Prot flat file of the plants taxonomic division
(`knowledgebase/taxonomic_divisions/uniprot_sprot_plants.dat.gz`; release, date and md5 recorded in
`evidence/DATASET_ROLES.tsv` at download). Plain or gzip-compressed.

Outputs (`--out-dir`):
  swissprot_viridiplantae.tsv     one row per Viridiplantae entry: accession, entry_name, taxid, pe, nterm_experimental,
                                  caution_types, caution_sequences, gene_xrefs, length, sha256  (the sensitivity-set table)
  swissprot_viridiplantae.fa      the same entries as protein FASTA (miniprot input of the evaluation-only alignment, A30.4)
  <species>.swissprot_flags.tsv   A22-schema flag file per training species, consumed together with the GeenuFF file by
                                  `scripts/build_b5_database.py --qc-flags <geenuff.tsv> <swissprot.tsv>`
  swissprot_summary.json          counts per stratum and per species, proteome mapping counts, input sha256

The set is never a label source (A19 unchanged). It feeds (a) loss masking through the A22 mechanism and
(b) the A30 sensitivity outcomes on start/stop/phase.

Hard flags (`swissprot_caution_<type>`, train_weight 0 through `build_b5.loss_mask_decision`) are emitted only when
  1. the entry carries a SEQUENCE CAUTION of a structural type: erroneous initiation, erroneous termination,
     frameshift, erroneous gene model prediction;
  2. its gene cross-references (TAIR, Araport, EnsemblPlants, Gramene, KEGG, ...) map to exactly one gene of the
     species (same NCBI taxid); ambiguous or unmapped entries produce no flag and are counted (unmapped per xref DB);
  3. the gene has at least one protein in the current reference proteome (`--proteome <species>=<fasta>`) and none of
     them equals the curated sequence. If one does, the caution was fixed in the annotation and only
     `swissprot_note_caution_resolved_in_reference` is recorded; a gene without any reference protein, or a species
     without a proteome, gets `swissprot_note_unverified_no_proteome` (soft).
Every other caution type (erroneous translation, miscellaneous discrepancy) is a soft `swissprot_note_<type>`.

Gene mapping: transcript and protein identifiers are resolved through the transcript->gene map of the reference GFF3
(`--gff <species>=<gff3>`: transcript-level features and their Parent), after stripping a Phytozome `.p` protein suffix;
without a map, one trailing `.suffix` is stripped as a fallback. A species whose reference proteome maps to fewer than
`--min-proteome-mapped` (default 0.9) of its records aborts the run (fail closed) so that a header-format mismatch can
never turn every caution into a hard flag.

N-terminal experimental evidence (stratum `pe1_nterm_experimental`): PE level 1 and an INIT_MET feature or a CHAIN
feature starting at residue 1 or 2 with an experimental evidence code (ECO:0000269).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

STRUCTURE_CAUTIONS = {"erroneous_initiation", "erroneous_termination", "frameshift", "erroneous_gene_model_prediction"}
GENE_XREF_DBS = {"TAIR", "Araport", "EnsemblPlants", "Gramene", "KEGG", "Phytozome", "MaizeGDB", "SGN", "SoyBase", "GeneID"}
TRANSCRIPT_TYPES = {"mRNA", "transcript", "ncRNA", "lnc_RNA", "lncRNA", "rRNA", "tRNA", "snRNA", "snoRNA", "miRNA",
                    "pseudogenic_transcript", "primary_transcript", "antisense_RNA", "mRNA_TE_gene"}
NTERM_KEYS = {"INIT_MET", "CHAIN"}
EXPERIMENTAL_ECO = "ECO:0000269"
DEFAULT_MIN_PROTEOME_MAPPED = 0.9

_OX_RE = re.compile(r"NCBI_TaxID=(\d+)|,\s*(\d+)\s*[{;]")
_PE_RE = re.compile(r"^PE\s+(\d)")
_CAUTION_RE = re.compile(r"Sequence=([^;]+);\s*Type=([^;]+);")
_FT_KEY_RE = re.compile(r"^FT\s{3}(\S+)\s+(\S+)")
_KEGG_PREFIX_RE = re.compile(r"^[a-z]{2,5}:")


@dataclass
class Entry:
    entry_name: str = ""
    accession: str = ""
    taxid: int = 0
    taxids: Set[int] = field(default_factory=set)
    viridiplantae: bool = False
    pe: int = 0
    nterm_experimental: bool = False
    cautions: List[Tuple[str, str]] = field(default_factory=list)
    caution_blocks: int = 0
    xrefs_by_db: Dict[str, Set[str]] = field(default_factory=dict)
    sequence: str = ""
    truncated: bool = False

    @property
    def gene_xrefs(self) -> Set[str]:
        return {x for ids in self.xrefs_by_db.values() for x in ids}

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode()).hexdigest()

    @property
    def caution_types(self) -> List[str]:
        return [t for _, t in self.cautions]

    @property
    def structural_caution(self) -> bool:
        return any(slug(t) in STRUCTURE_CAUTIONS for t in self.caution_types)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def flag_name(caution_type: str) -> str:
    s = slug(caution_type)
    return ("swissprot_caution_" if s in STRUCTURE_CAUTIONS else "swissprot_note_") + s


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _xref_ids(line: str) -> Tuple[str, List[str]]:
    body = line[5:].rstrip().rstrip(".")
    parts = [p.strip() for p in body.split(";")]
    db, ids = parts[0], []
    for p in parts[1:]:
        if not p or p == "-" or p.startswith("locus:"):
            continue
        p = p.split(" ")[0]
        if db == "KEGG":
            p = _KEGG_PREFIX_RE.sub("", p)
        ids.append(p)
    return db, ids


def parse_dat(path: str) -> Iterator[Entry]:
    """Yield one Entry per Swiss-Prot flat-file record. A record without its closing `//` (truncated file) is still
    yielded, marked `truncated=True`. SEQUENCE CAUTION records are parsed only inside `-!- SEQUENCE CAUTION:` blocks."""
    e = Entry()
    oc: List[str] = []
    seq: List[str] = []
    in_seq = in_caution = started = False
    ft_key = ft_loc = ""

    def finalize(entry: Entry) -> Entry:
        entry.sequence = "".join(seq).replace(" ", "").upper()
        entry.viridiplantae = "Viridiplantae" in " ".join(oc)
        return entry

    with _open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("//"):
                yield finalize(e)
                e, oc, seq, in_seq, in_caution, started, ft_key, ft_loc = Entry(), [], [], False, False, False, "", ""
                continue
            if in_seq:
                seq.append(line.strip())
                continue
            tag = line[:2]
            if tag == "ID":
                e.entry_name = line[5:].split()[0]
                started = True
            elif tag == "AC" and not e.accession:
                e.accession = line[5:].split(";")[0].strip()
            elif tag == "OX":
                ids = [int(a or b) for a, b in _OX_RE.findall(line)]
                if ids:
                    e.taxid, e.taxids = ids[0], set(ids)
            elif tag == "OC":
                oc.append(line[5:])
            elif tag == "PE":
                m = _PE_RE.match(line)
                if m:
                    e.pe = int(m.group(1))
            elif tag == "CC":
                body = line[5:]
                if body.startswith("-!-"):
                    in_caution = body.startswith("-!- SEQUENCE CAUTION")
                    if in_caution:
                        e.caution_blocks += 1
                if in_caution:
                    m = _CAUTION_RE.search(line)
                    if m:
                        e.cautions.append((m.group(1).strip(), m.group(2).strip()))
            elif tag == "DR":
                db, ids = _xref_ids(line)
                if db in GENE_XREF_DBS:
                    e.xrefs_by_db.setdefault(db, set()).update(ids)
            elif tag == "FT":
                m = _FT_KEY_RE.match(line)
                if m:
                    ft_key, ft_loc = m.group(1), m.group(2)
                elif "/evidence=" in line and EXPERIMENTAL_ECO in line and ft_key in NTERM_KEYS:
                    start = ft_loc.split("..")[0].lstrip("<?")
                    if ft_key == "INIT_MET" or start in ("1", "2"):
                        e.nterm_experimental = True
            elif tag == "SQ":
                in_seq = True
    if started:
        e.truncated = True
        yield finalize(e)


def gene_of(xref: str, genes: Set[str], tx2gene: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve a gene, transcript or protein identifier to a gene id of the reference annotation: exact gene id, then the
    transcript->gene map of the GFF3, after stripping a Phytozome `.p` protein suffix; without a map, one trailing
    `.suffix` is stripped as a fallback (AT1G01010.1 -> AT1G01010)."""
    tx2gene = tx2gene or {}
    cands = [xref]
    if xref.endswith(".p"):
        cands.append(xref[:-2])
    for c in list(cands):
        if "." in c:
            cands.append(c.rsplit(".", 1)[0])
    for c in cands:
        if c in genes:
            return c
        g = tx2gene.get(c)
        if g in genes:
            return g
    return None


_gene_of = gene_of  # backwards-compatible alias


def _gff_attrs(col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in col.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_gff_gene_map(path: str) -> Tuple[Set[str], Dict[str, str]]:
    """Gene ids and the transcript->gene map (transcript-level features only) of a reference GFF3."""
    genes: Set[str] = set()
    parents: Dict[str, str] = {}
    with _open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            attrs = _gff_attrs(c[8])
            if c[2] == "gene" and "ID" in attrs:
                genes.add(attrs["ID"])
            elif c[2] in TRANSCRIPT_TYPES and "ID" in attrs and "Parent" in attrs:
                parents[attrs["ID"]] = attrs["Parent"].split(",")[0]
    tx2gene = {t: g for t, g in parents.items() if g in genes}
    return genes, tx2gene


def _proteome_by_gene(proteome: Dict[str, str], genes: Set[str], tx2gene: Dict[str, str]) -> Tuple[Dict[str, Set[str]], int, int]:
    out: Dict[str, Set[str]] = defaultdict(set)
    mapped = 0
    for pid, s in proteome.items():
        g = gene_of(pid, genes, tx2gene)
        if g:
            mapped += 1
            out[g].add(s.rstrip("*").upper())
    return out, len(proteome), mapped


def species_flags(entries: Iterable[Entry], gene_ids: Dict[str, Set[str]], taxids: Dict[str, Set[int]],
                  proteomes: Dict[str, Dict[str, str]], tx2gene: Optional[Dict[str, Dict[str, str]]] = None,
                  min_proteome_mapped: float = DEFAULT_MIN_PROTEOME_MAPPED) -> Tuple[List[Dict], Dict[str, Dict]]:
    """A22-schema flag rows per training species and a per-species count summary (A30.2). Fails closed when a
    reference proteome maps to fewer than `min_proteome_mapped` of its records."""
    entries = [e for e in entries if e.viridiplantae]
    rows: List[Dict] = []
    seen = set()
    summary: Dict[str, Dict] = {}
    for sp, genes in gene_ids.items():
        st: Dict = defaultdict(int)
        unmapped_by_db: Dict[str, int] = defaultdict(int)
        txmap = (tx2gene or {}).get(sp, {})
        prot: Dict[str, Set[str]] = {}
        if sp in proteomes:
            prot, n_rec, n_map = _proteome_by_gene(proteomes[sp], genes, txmap)
            st["proteome_records"], st["proteome_mapped"] = n_rec, n_map
            frac = n_map / n_rec if n_rec else 0.0
            if frac < min_proteome_mapped:
                raise SystemExit(f"{sp}: only {n_map}/{n_rec} reference proteins map to a gene id (fraction {frac:.3f} < "
                                 f"{min_proteome_mapped}); check --gff/--gene-ids and the FASTA headers, or lower --min-proteome-mapped")
        want = taxids.get(sp, set())
        for e in entries:
            if not (e.taxids & want):
                continue
            st["entries"] += 1
            targets = {g for g in (gene_of(x, genes, txmap) for x in e.gene_xrefs) if g}
            if not targets:
                st["unmapped"] += 1
                for db in e.xrefs_by_db:
                    unmapped_by_db[db] += 1
                continue
            if len(targets) > 1:
                st["ambiguous"] += 1
                continue
            st["mapped"] += 1
            gene = next(iter(targets))
            names: List[str] = []
            for ctype in e.caution_types:
                name = flag_name(ctype)
                if name.startswith("swissprot_caution_"):
                    if not prot.get(gene):
                        name = "swissprot_note_unverified_no_proteome"
                        st["unverified"] += 1
                    elif e.sequence in prot[gene]:
                        name = "swissprot_note_caution_resolved_in_reference"
                        st["resolved_in_reference"] += 1
                    else:
                        st["hard_flags"] += 1
                else:
                    st["soft_notes"] += 1
                names.append(name)
            for name in names:
                key = (sp, gene, name)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"species_id": sp, "gene_id": gene, "transcript_id": "", "flag": name, "start": 0, "end": 0})
        for k in ("entries", "mapped", "ambiguous", "unmapped", "hard_flags", "soft_notes", "resolved_in_reference", "unverified",
                  "proteome_records", "proteome_mapped"):
            st.setdefault(k, 0)
        st["unmapped_by_db"] = dict(sorted(unmapped_by_db.items()))
        summary[sp] = dict(st)
    return rows, summary


def strata(entries: Iterable[Entry]) -> Dict[str, int]:
    v = [e for e in entries if e.viridiplantae]
    return {"pe1": sum(e.pe == 1 for e in v),
            "pe1_nterm_experimental": sum(e.pe == 1 and e.nterm_experimental for e in v),
            "caution_structure": sum(e.structural_caution for e in v),
            "caution_any": sum(bool(e.cautions) for e in v),
            "caution_blocks_total": sum(e.caution_blocks for e in v),
            "caution_blocks_unparsed": sum(1 for e in v if e.caution_blocks and not e.cautions),
            "truncated_entries": sum(e.truncated for e in v)}


def read_gene_ids(path: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2:
                out[c[idx.get("species_id", 0)]].add(c[idx.get("gene_id", 1)])
    return out


def read_fasta(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    name, buf = None, []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def _parse_kv(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected SPECIES=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dat", required=True, help="uniprot_sprot_plants.dat[.gz]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gene-ids", help="TSV species_id<TAB>gene_id of the reference annotations (alternative to --gff)")
    ap.add_argument("--gff", action="append", default=[], help="SPECIES=<reference GFF3>: gene ids and transcript->gene map (repeatable)")
    ap.add_argument("--species-taxid", action="append", default=[], help="SPECIES=taxid[,taxid...] (repeatable)")
    ap.add_argument("--proteome", action="append", default=[], help="SPECIES=<reference protein FASTA> (repeatable; required for hard flags)")
    ap.add_argument("--min-proteome-mapped", type=float, default=DEFAULT_MIN_PROTEOME_MAPPED,
                    help="abort when a reference proteome maps to fewer than this fraction of its records (fail closed)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    entries = list(parse_dat(a.dat))
    kept = [e for e in entries if e.viridiplantae]
    with open(os.path.join(a.out_dir, "swissprot_viridiplantae.tsv"), "w") as fh:
        fh.write("accession\tentry_name\ttaxid\tpe\tnterm_experimental\tcaution_types\tcaution_sequences\tgene_xrefs\tlength\tsha256\n")
        for e in kept:
            fh.write("\t".join([e.accession, e.entry_name, str(e.taxid), str(e.pe), str(int(e.nterm_experimental)),
                                ";".join(e.caution_types), ";".join(s for s, _ in e.cautions), ";".join(sorted(e.gene_xrefs)),
                                str(e.length), e.sha256]) + "\n")
    with open(os.path.join(a.out_dir, "swissprot_viridiplantae.fa"), "w") as fh:
        for e in kept:
            fh.write(f">{e.accession} {e.entry_name} taxid={e.taxid} pe={e.pe}\n")
            for i in range(0, e.length, 60):
                fh.write(e.sequence[i:i + 60] + "\n")
    taxids = {k: {int(t) for t in v.split(",") if t} for k, v in _parse_kv(a.species_taxid).items()}
    proteomes = {k: read_fasta(v) for k, v in _parse_kv(a.proteome).items()}
    gene_ids = read_gene_ids(a.gene_ids) if a.gene_ids else {}
    tx2gene: Dict[str, Dict[str, str]] = {}
    for sp, path in _parse_kv(a.gff).items():
        genes, txmap = read_gff_gene_map(path)
        gene_ids.setdefault(sp, set()).update(genes)
        tx2gene[sp] = txmap
    for sp in taxids:
        gene_ids.setdefault(sp, set())
    rows, per_species = species_flags(kept, gene_ids, taxids, proteomes, tx2gene, a.min_proteome_mapped)
    by_sp: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_sp[r["species_id"]].append(r)
    for sp in gene_ids:
        with open(os.path.join(a.out_dir, f"{sp}.swissprot_flags.tsv"), "w") as fh:
            fh.write("species_id\tgene_id\ttranscript_id\tflag\tstart\tend\n")
            for r in sorted(by_sp.get(sp, []), key=lambda r: (r["gene_id"], r["flag"])):
                fh.write(f"{r['species_id']}\t{r['gene_id']}\t{r['transcript_id']}\t{r['flag']}\t{r['start']}\t{r['end']}\n")
    summary = {"dat": os.path.abspath(a.dat), "dat_sha256": file_sha256(a.dat), "entries_total": len(entries),
               "entries_viridiplantae": len(kept), "strata": strata(kept), "species": per_species,
               "taxids": {k: sorted(v) for k, v in taxids.items()},
               "proteomes": {k: os.path.abspath(v) for k, v in _parse_kv(a.proteome).items()},
               "gff": {k: os.path.abspath(v) for k, v in _parse_kv(a.gff).items()},
               "min_proteome_mapped": a.min_proteome_mapped}
    with open(os.path.join(a.out_dir, "swissprot_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in ("entries_total", "entries_viridiplantae", "strata", "species")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
