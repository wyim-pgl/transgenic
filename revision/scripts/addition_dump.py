"""Read-only export of frozen B1 structures; no evidence or corpus consumption."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


def load_filter():
    spec = importlib.util.spec_from_file_location(
        "frozen_filter36", Path(__file__).with_name("36_filter_additions_structurally.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def introns(structure):
    # Script 48's chain uses flanking CDS bases; C0 uses inclusive intronic bases.
    return tuple((left[1] + 1, right[0] - 1)
                 for left, right in zip(structure, structure[1:]))


class IndexedContig:
    """String-slice interface for script 36 without loading a 2.2 GB genome."""
    def __init__(self, handle, length, offset, bases, width):
        self.handle, self.length, self.offset = handle, length, offset
        self.bases, self.width = bases, width

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        start, end, step = key.indices(self.length)
        if step != 1:
            raise ValueError("only contiguous genome slices are supported")
        if start >= end:
            return ""
        first = self.offset + start // self.bases * self.width + start % self.bases
        last = self.offset + (end - 1) // self.bases * self.width + (end - 1) % self.bases
        self.handle.seek(first)
        sequence = self.handle.read(last - first + 1).replace(b"\n", b"").replace(b"\r", b"")
        if len(sequence) != end - start or b">" in sequence:
            raise ValueError("FASTA index does not match sequence layout")
        return sequence.decode("ascii")


def indexed_genome(path, handle, aliases):
    genome = {}
    for line in Path(str(path) + ".fai").read_text().splitlines():
        name, length, offset, bases, width, *_ = line.split("\t")
        genome[name] = IndexedContig(handle, *map(int, (length, offset, bases, width)))
        if name in aliases:
            genome[aliases[name]] = genome[name]
    return genome


def structure_rows(pred, ref, primary, metadata, genome, species, filters):
    rows = []
    for locus in sorted(pred):
        reference = ref.get(locus, {})
        supplied = reference.get(primary.get(locus))
        alternatives = set(reference.values()) - {supplied}
        reference_chains = {introns(s) for s in reference.values()}
        alternative_chains = {introns(s) for s in alternatives if len(s) > 1}
        reference_introns = {j for c in reference_chains for j in c}
        for structure in sorted(set(pred[locus].values()) - {supplied}):
            tids = sorted(t for t, s in pred[locus].items() if s == structure)
            locations = {metadata[t] for t in tids}
            if len(locations) != 1:
                raise ValueError(f"{locus}: collapsed emissions disagree on contig/strand")
            contig, strand = locations.pop()
            if strand not in ("+", "-") or contig not in genome:
                raise ValueError(f"{locus}: missing genome contig or strand")
            chromosome = genome[contig]
            if any(not 1 <= a <= b <= len(chromosome) for a, b in structure):
                raise ValueError(f"{locus}: CDS outside genome")
            chain = introns(structure)
            novel = tuple(j for j in chain if j not in reference_introns)
            exact = structure in alternatives
            chain_match = bool(chain) and chain in alternative_chains
            ambiguity = None
            if supplied is None:
                novelty, ambiguity = None, "missing supplied primary CDS"
            elif exact or chain_match:
                novelty = "reference-alt"
            elif novel:
                novelty = "junction-novel"
            elif chain not in reference_chains:
                novelty = "combination-novel"
            else:
                novelty, ambiguity = None, "chain matches only primary, or unmatched monoexonic addition"
            orf = filters.has_complete_orf(filters.spliced_cds(chromosome, structure, strand))
            canonical = filters.has_canonical_introns(chromosome, structure, strand)
            identity = [species, locus, contig, strand, structure]
            stable_id = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
            rows.append(dict(addition_id=f"{species}:{stable_id}", locus_id=locus,
                             species=species, contig=contig, strand=strand,
                             supplied_primary_id=primary.get(locus), supplied_primary_cds=supplied,
                             prediction_transcript_ids=tids, cds_intervals=structure,
                             introns=chain, novel_introns=novel, novelty=novelty,
                             classification_ambiguity=ambiguity, exact_cds_match=exact,
                             alternative_intron_chain_match=chain_match,
                             complete_orf=orf, canonical_introns=canonical,
                             filter_pass=orf and canonical))
    return rows


def write_frozen_dump(destination, scorer):
    """Export into an exclusive directory, with completion manifest written last."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    filters = load_filter()
    inputs = [scorer.PRED, scorer.REF, scorer.AT_PRED, scorer.AT_DATA / "TAIR10.gtf",
              scorer.AT_DATA / "primary_transcript_ids.txt",
              scorer.ROOT / "genomes/Zmays_493_APGv4.fa", scorer.AT_DATA / "TAIR10_genome.fa"]
    expected_md5 = ["08b45a51e9116f2901b447b9787c7544", "f62364903053e0fb433f6b0b80ed4df9",
                    "670d419b887271e0a9b8a4d82b9bf700", "c6ccc302afda2a40f729b655288413ef",
                    "8442eed7f20e879ebd1928a615fe0747", "a8d3069cd9554885670848cc3df185cb",
                    "987ca803466e79f98b7f06af8ca94557"]
    provenance = []
    for path, expected in zip(inputs, expected_md5):
        actual = digest(path, "md5")
        if actual != expected:
            raise ValueError(f"frozen input MD5 mismatch: {path}")
        provenance.append(dict(path=str(path), md5=actual))
    for fasta in inputs[5:]:
        index = Path(str(fasta) + ".fai")
        provenance.append(dict(path=str(index), sha256=digest(index)))
    summaries = {}
    for species, prediction, fasta, expected_count in [
            ("Zmays", scorer.PRED, inputs[5], 3363),
            ("Athaliana", scorer.AT_PRED, inputs[6], 1103)]:
        if species == "Zmays":
            pred, duplicate_audit = scorer.read_prediction(prediction)
            ref, primary, _ = scorer.read_reference(scorer.REF)
        else:
            pred, ref, primary = scorer.read_athaliana()
            duplicate_audit = None
        _, _, metadata = filters.read_cds_by_transcript(prediction)
        with fasta.open("rb") as handle:
            genome = indexed_genome(fasta, handle, filters.SEQID_ALIASES)
            rows = structure_rows(pred, ref, primary, metadata, genome, species, filters)
        aggregate = scorer.score(pred, ref, primary, sorted(pred), species)
        if len(rows) != expected_count or len(rows) != aggregate["added_transcripts"]:
            raise ValueError(f"{species}: frozen addition count mismatch")
        if sum(r["exact_cds_match"] for r in rows) != aggregate["added_matching_RefGenV4_alternative_exact_CDS"]:
            raise ValueError(f"{species}: exact match reconciliation failed")
        if sum(r["alternative_intron_chain_match"] for r in rows) != aggregate["added_matching_RefGenV4_alternative_intron_chain"]:
            raise ValueError(f"{species}: chain match reconciliation failed")
        output = destination / f"{species}.additions.jsonl"
        with output.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summaries[species] = dict(rows=len(rows), filter_pass=sum(r["filter_pass"] for r in rows),
                                  novelty_counts=dict(Counter(r["novelty"] or "UNRESOLVED" for r in rows)),
                                  aggregate=aggregate, duplicate_audit=duplicate_audit,
                                  output_sha256=digest(output))
    code = [Path(scorer.__file__), Path(__file__), Path(filters.__file__),
            Path(__file__).with_name("28_score_added_isoforms.py")]
    manifest = dict(schema="frozen-addition-dump-v1", coordinates="1-based inclusive; genomic order",
                    status="export complete; unresolved novelty rows are not scorer-ready",
                    inputs=provenance, code_sha256={str(p.resolve()): digest(p) for p in code},
                    species=summaries)
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
