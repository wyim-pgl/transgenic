#!/usr/bin/env python3
"""Frozen B1 evidence support, issue #24.

Coordinates throughout are 1-based inclusive genomic intervals, including introns
on the minus strand. Chains are stored in genomic order; strand is a separate key.
SAM fixtures are supported without third-party packages; BAMs are streamed through
samtools. A PAF alone cannot establish the SAM supplementary/chimera exclusions.
See EVIDENCE_SCORER_READINGS.md for the input contract and unresolved rule gates.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


class EvidenceError(ValueError):
    """Missing or inconsistent evidence must not become an unsupported prediction."""


class RuleUnresolved(EvidenceError):
    """A frozen rule needs an author decision, not an implementation default."""


SOURCES = ("ONT", "PacBio", "EST", "FL-cDNA")
THRESHOLDS = {"ONT": 3, "PacBio": 2, "EST": 2, "FL-cDNA": 2}
CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")
CS = re.compile(r"(:\d+|=[A-Za-z]+|\*[A-Za-z]{2}|[+-][A-Za-z]+|~[A-Za-z]{2}\d+[A-Za-z]{2})")


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def sam_lines(path, samtools="samtools"):
    if str(path).endswith(".bam"):
        # A failed decoder is not an empty evidence source.
        p = subprocess.Popen([samtools, "view", "-h", str(path)], stdout=subprocess.PIPE,
                             text=True)
        try:
            yield from p.stdout
        finally:
            p.stdout.close()
            rc = p.wait()
        require(rc == 0, f"samtools view failed ({rc}): {path}")
    else:
        require(not str(path).endswith((".paf", ".paf.gz")),
                "PAF alone lacks SAM flags; supply original BAM/SAM with cs and ts tags")
        with open_text(path) as f:
            yield from f


@dataclass(frozen=True)
class Alignment:
    read: str
    flag: int
    contig: str
    start: int
    mapq: int
    cigar: str
    tags: dict
    end: int
    query_length: int
    aligned_query: int
    blocks: tuple
    introns: tuple

    @property
    def read_strand(self):
        return "-" if self.flag & 16 else "+"


def parse_sam(line):
    if line.startswith("@") or not line.strip():
        return None
    f = line.rstrip("\n").split("\t")
    require(len(f) >= 11, "SAM row has fewer than 11 fields")
    flag, start, mapq = int(f[1]), int(f[3]), int(f[4])
    tags = {}
    for tag in f[11:]:
        p = tag.split(":", 2)
        require(len(p) == 3 and p[0] not in tags, f"{f[0]}: malformed/duplicate SAM tag")
        tags[p[0]] = p[2]
    if flag & 4:
        return Alignment(f[0], flag, f[2], start, mapq, f[5], tags, start, 0, 0, (), ())
    ops = CIGAR.findall(f[5])
    require(ops and "".join(n + op for n, op in ops) == f[5], f"{f[0]}: invalid CIGAR")
    pos, block_start, qlen, aligned = start, start, 0, 0
    blocks, introns = [], []
    for n, op in ops:
        n = int(n)
        require(n > 0, f"{f[0]}: zero CIGAR operation")
        if op in "MIS=XH":
            qlen += n
        if op in "MI=X":
            aligned += n
        if op == "N":
            require(pos > block_start, f"{f[0]}: empty exon")
            blocks.append((block_start, pos - 1))
            introns.append((pos, pos + n - 1))
            pos += n
            block_start = pos
        elif op in "MD=X":
            pos += n
    require(pos > block_start and qlen > 0, f"{f[0]}: empty alignment")
    blocks.append((block_start, pos - 1))
    return Alignment(f[0], flag, f[2], start, mapq, f[5], tags,
                     pos - 1, qlen, aligned, tuple(blocks), tuple(introns))


def transcript_strand(aln, uf=False, oriented_strand=None):
    ts = aln.tags.get("ts")
    require(ts in (None, "+", "-", "?"), f"{aln.read}: invalid ts tag")
    if uf:
        require(ts in (None, "+", "?"), f"{aln.read}: ts contradicts audited -uf")
        strand = aln.read_strand
    elif ts in ("+", "-"):
        # ts is relative to the original read, including in reverse SAM records (A39).
        strand = aln.read_strand if ts == "+" else ("+" if aln.read_strand == "-" else "-")
    else:
        strand = None
    require(oriented_strand in (None, "+", "-"), "invalid oriented genomic strand")
    if oriented_strand:
        require(strand in (None, oriented_strand), f"{aln.read}: conflicting strand evidence")
        strand = oriented_strand
    return strand


@dataclass(frozen=True)
class Edit:
    op: str
    length: int
    ref_start: int


def parse_cs(aln):
    value = aln.tags.get("cs")
    require(value is not None, f"{aln.read}: missing cs tag (converted PAF is not a substitute)")
    tokens = CS.findall(value)
    require(tokens and "".join(tokens) == value, f"{aln.read}: malformed cs")
    pos, qlen, edits, introns = aln.start, 0, [], []
    for token in tokens:
        op = token[0]
        n = (int(token[1:]) if op == ":" else int(token[3:-2]) if op == "~"
             else 1 if op == "*" else len(token) - 1)
        require(n > 0, f"{aln.read}: empty cs operation")
        edits.append(Edit(op, n, pos))
        if op == "~":
            introns.append((pos, pos + n - 1))
        if op != "+":
            pos += n
        if op in ":=*+":
            qlen += n
    require(pos - 1 == aln.end and tuple(introns) == aln.introns and qlen == aln.aligned_query,
            f"{aln.read}: cs/CIGAR disagreement")
    return edits


def anchor_ok(edits, index, direction, width, max_mismatch, exclusion):
    """Nearest exonic reference bases; clipping and another intron cannot supply anchors."""
    bases, mismatches = 0, 0
    j = index + direction
    while 0 <= j < len(edits):
        edit = edits[j]
        if edit.op == "~":
            break
        if edit.op in "+-":
            if bases < exclusion:
                return False
            # Beyond the exclusion zone an indel is permitted; deleted bases
            # still cannot supply aligned anchor length.
        else:
            take = min(width - bases, edit.length)
            bases += take
            if edit.op == "*":
                mismatches += take
        if bases >= width:
            return mismatches <= max_mismatch
        j += direction
    return False


def canonical(genome, contig, intron, strand):
    require(contig in genome, f"genome missing contig {contig}")
    start, end = intron
    seq = genome[contig]
    if start < 1 or end > len(seq) or end - start < 3:
        return False
    motifs = (seq[start - 1:start + 1].upper(), seq[end - 2:end].upper())
    return motifs == (("GT", "AG") if strand == "+" else ("CT", "AC"))


def correct_junction(genome, contig, raw, strand, source):
    radius = {"ONT": 3, "PacBio": 1, "EST": 0, "FL-cDNA": 0}[source]
    if radius == 0:
        return raw, "uncorrected"
    candidates = [(raw[0] + d, raw[1] + a)
                  for d in range(-radius, radius + 1) for a in range(-radius, radius + 1)
                  if canonical(genome, contig, (raw[0] + d, raw[1] + a), strand)]
    if len(candidates) > 1:
        return None, "ambiguous_correction"
    if candidates:
        return candidates[0], "canonical" if candidates[0] == raw else "corrected"
    return raw, "noncanonical"


@dataclass(frozen=True)
class Junction:
    raw: tuple
    corrected: tuple | None
    status: str
    accepted: bool
    canonical: bool


def call_junctions(aln, genome, strand, source):
    require(source in SOURCES, f"unknown source: {source}")
    edits = parse_cs(aln)
    width, mismatches, zone = (15, 1, 8) if source in ("EST", "FL-cDNA") else (20, 2, 6)
    result = []
    for i, edit in enumerate(edits):
        if edit.op != "~":
            continue
        raw = (edit.ref_start, edit.ref_start + edit.length - 1)
        if strand is None:
            result.append(Junction(raw, None, "strand_ambiguous", False, False))
            continue
        corrected, status = correct_junction(genome, aln.contig, raw, strand, source)
        anchored = all(anchor_ok(edits, i, direction, width, mismatches, zone)
                       for direction in (-1, 1))
        if not anchored:
            status = "anchor_failed"
        result.append(Junction(raw, corrected, status, anchored and corrected is not None,
                               corrected is not None and canonical(genome, aln.contig, corrected, strand)))
    return tuple(result)


def alignment_rejection(records, mapq=20):
    """Inspect the whole read group before accepting its primary, irrespective of record order."""
    mapped = [r for r in records if not r.flag & 4]
    if any(r.flag & 2048 or r.tags.get("SA") for r in mapped):
        return "supplementary_or_chimeric"
    primary = [r for r in mapped if not r.flag & 0x900]
    if len({(r.contig, r.read_strand) for r in primary}) > 1:
        return "cross_contig_or_strand"
    if len(primary) != 1:
        return "unmapped" if not mapped else "ambiguous_primary"
    a = primary[0]
    if a.mapq < mapq:
        return "mapq"
    if any(r is not a and r.tags.get("AS") is not None and
           r.tags.get("AS") == a.tags.get("AS") for r in mapped):
        return "equal_best_placement"
    return None


def wilson(successes, total):
    require(0 <= successes <= total, "invalid binomial counts")
    if total == 0:
        return None, None, None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return p, max(0., centre - radius), min(1., centre + radius)


@dataclass(frozen=True)
class Addition:
    addition_id: str
    locus_id: str
    species: str
    assembly: str
    contig: str
    strand: str
    cds_start: int
    cds_end: int
    locus_start: int
    locus_end: int
    introns: tuple
    novel_introns: tuple
    novelty: str
    filter_pass: bool
    exact_cds_match: bool
    control: str = "addition"

    def __post_init__(self):
        require(self.strand in ("+", "-"), "addition requires genomic strand")
        require(self.locus_start <= self.cds_start <= self.cds_end <= self.locus_end,
                f"{self.addition_id}: invalid locus/CDS span")
        require(tuple(sorted(set(self.introns))) == self.introns,
                f"{self.addition_id}: introns must be unique, genomic-ordered")
        require(all(self.cds_start < d <= a < self.cds_end for d, a in self.introns),
                f"{self.addition_id}: intron outside CDS")
        require(all(a < d for (_, a), (d, _) in zip(self.introns, self.introns[1:])),
                f"{self.addition_id}: overlapping introns")
        require(set(self.novel_introns) <= set(self.introns), "novel introns not in chain")
        require(self.novelty in ("junction-novel", "combination-novel", "reference-alt"),
                "invalid novelty class")
        require(bool(self.novel_introns) == (self.novelty == "junction-novel"),
                "novelty class disagrees with novel introns")
        require(type(self.filter_pass) is bool and type(self.exact_cds_match) is bool,
                "filter_pass/exact_cds_match must be frozen booleans")
        require(self.control in ("addition", "positive", "negative"), "unknown control class")


@dataclass
class Observation:
    alignment: Alignment
    species: str
    assembly: str
    source: str
    dataset: str
    run: str
    library: str
    bioproject: str
    genotype_stratum: str
    annotation_independence: str
    model_independent: bool
    strand: str | None
    junctions: tuple
    molecule: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def chain(self):
        # Keeping failed calls in the structure prevents deletion of a bad extra intron
        # from manufacturing an otherwise exact chain.
        return tuple(j.corrected if j.corrected is not None else j.raw for j in self.junctions)


def overlaps(obs, addition):
    a = obs.alignment
    return (obs.species == addition.species and obs.assembly == addition.assembly and
            a.contig == addition.contig and a.start <= addition.locus_end and
            a.end >= addition.locus_start)


def spanning(obs, start, end):
    return obs.alignment.start <= start and obs.alignment.end >= end


def callable_dimensions(addition, observations):
    same = [o for o in observations if overlaps(o, addition) and o.strand == addition.strand]
    chain = bool(addition.introns) and any(spanning(o, addition.introns[0][0] - 20,
                                                 addition.introns[-1][1] + 20) for o in same)
    junction = bool(addition.novel_introns) and all(
        any(spanning(o, d - 20, a + 20) for o in same) for d, a in addition.novel_introns)
    covered = bool(same)
    return {"chain_callable": chain, "junction_callable": junction,
            "locus_covered": covered, "uncovered": not covered,
            "strand_ambiguous_coverage": any(overlaps(o, addition) and o.strand is None
                                             for o in observations),
            "callability": "chain-callable" if chain else "junction-callable" if junction
                           else "locus-covered" if covered else "uncovered"}


def chain_witness(addition, obs):
    if not addition.introns or obs.strand != addition.strand or not overlaps(obs, addition):
        return False
    calls = [j for j in obs.junctions if j.raw[0] <= addition.cds_end and j.raw[1] >= addition.cds_start]
    if tuple(j.corrected for j in calls) != addition.introns or not all(j.accepted for j in calls):
        return False
    first, last = addition.introns[0][0], addition.introns[-1][1]
    # Span alone could count a flank hidden inside a different intron. Chain anchors
    # have to be in the terminal aligned blocks, as required by §6/A16.
    raw_first, raw_last = calls[0].raw[0], calls[-1].raw[1]
    return (any(s <= first - 20 and e >= raw_first - 1 for s, e in obs.alignment.blocks) and
            any(s <= raw_last + 1 and e >= last + 20 for s, e in obs.alignment.blocks))


def normalize_clone(value):
    value = re.sub(r"(?:[\s_.:/-]+(?:5['′]?|3['′]?|forward|reverse|fwd|rev))$", "",
                   value.strip().lower())
    return re.sub(r"[^a-z0-9]", "", value)


def assign_molecules(observations):
    """A11/A16 units; refuse a non-transitive PCR component instead of picking a linkage."""
    newest = {}
    for o in observations:
        if o.source in ("EST", "FL-cDNA"):
            accession = o.metadata.get("accession", o.alignment.read)
            match = re.fullmatch(r"(.+?)(?:\.(\d+))?", accession)
            stem, version = match[1], int(match[2] or 0)
            key = (o.species, o.source, o.library, o.metadata.get("arm", "primary"), stem)
            newest[key] = max(newest.get(key, -1), version)
    retained = []
    for o in observations:
        if o.source in ("EST", "FL-cDNA"):
            accession = o.metadata.get("accession", o.alignment.read)
            match = re.fullmatch(r"(.+?)(?:\.(\d+))?", accession)
            key = (o.species, o.source, o.library, o.metadata.get("arm", "primary"), match[1])
            if int(match[2] or 0) < newest[key]:
                continue
        retained.append(o)
    observations = retained
    pcr = defaultdict(list)
    for o in observations:
        m, a = o.metadata, o.alignment
        base = (o.species, o.assembly, o.source, o.library)
        require(o.library and o.library != "unknown", f"{a.read}: missing biological library identity")
        if o.source in ("EST", "FL-cDNA"):
            clone = normalize_clone(m.get("clone", ""))
            if clone:
                key = base + ("clone", clone)
            else:
                key = base + ("signature", a.contig, o.strand, a.blocks)
        elif o.source == "PacBio":
            require(m.get("source_molecule"), f"{a.read}: missing source FLNC/ZMW molecule; clusters are not units")
            key = base + ("source_molecule", m["source_molecule"])
        elif m.get("umi_status") == "present":
            require(all(m.get(k) for k in ("sample", "umi", "locus")), f"{a.read}: incomplete UMI metadata")
            key = base + ("umi", m["sample"], m["umi"], m["locus"], o.strand)
        elif m.get("protocol") == "direct_RNA":
            key = base + ("read", o.run, a.read)
        else:
            require(m.get("protocol") == "cDNA" and m.get("umi_status") == "absent",
                    f"{a.read}: unresolved ONT protocol/UMI status")
            pcr[base + (a.contig, o.strand, o.chain)].append(o)
            continue
        o.molecule = hashlib.sha256(repr(key).encode()).hexdigest()
    for key, group in pcr.items():
        group.sort(key=lambda o: (o.alignment.start, o.alignment.end, o.run, o.alignment.read))
        neighbours = [set([i]) for i in range(len(group))]
        for i, o in enumerate(group):
            for j in range(i + 1, len(group)):
                p = group[j]
                if p.alignment.start - o.alignment.start > 10:
                    break
                if abs(p.alignment.end - o.alignment.end) <= 10:
                    neighbours[i].add(j)
                    neighbours[j].add(i)
        visited = set()
        for i in range(len(group)):
            if i in visited:
                continue
            component, pending = set(), [i]
            while pending:
                j = pending.pop()
                if j not in component:
                    component.add(j)
                    pending.extend(neighbours[j] - component)
            if any(neighbours[j] != component for j in component):
                raise RuleUnresolved("A16 non-transitive 10-nt PCR equivalence: " +
                                     ", ".join(group[j].alignment.read for j in sorted(component)))
            visited.update(component)
            ends = sorted((group[j].alignment.start, group[j].alignment.end) for j in component)
            unit = hashlib.sha256(repr((key, ends[0])).encode()).hexdigest()
            for j in component:
                group[j].molecule = unit
    return observations


def support_counts(observations):
    by_library = defaultdict(set)
    for o in observations:
        require(o.molecule, f"{o.alignment.read}: molecule units have not been assigned")
        by_library[o.library].add(o.molecule)
    source = observations[0].source if observations else None
    require(len({o.source for o in observations}) <= 1, "thresholds must be applied per source")
    units = sum(min(10, len(v)) if source in ("EST", "FL-cDNA") else len(v)
                for v in by_library.values())
    runs = len({(o.dataset, o.run) for o in observations})
    threshold = THRESHOLDS.get(source, math.inf)
    return {"raw_reads": len({(o.dataset, o.run, o.alignment.read) for o in observations}),
            "units": units, "libraries": len(by_library), "runs": runs,
            "supported": units >= threshold,
            "high_confidence": units >= threshold and (
                source in ("EST", "FL-cDNA") or runs >= 2 or
                units >= (5 if source == "ONT" else 3))}


def score_addition(addition, observations):
    observations = [o for o in observations if overlaps(o, addition)]
    row = asdict(addition)
    row.update(callable_dimensions(addition, observations))
    same = [o for o in observations if o.strand == addition.strand]
    per_source = {}
    for source in SOURCES:
        obs = [o for o in same if o.source == source]
        junctions = {}
        for intron in addition.introns:
            carrying = [o for o in obs if any(j.accepted and j.corrected == intron for j in o.junctions)]
            junctions[intron] = support_counts(carrying)
        witnesses = [o for o in obs if chain_witness(addition, o)]
        count = support_counts(witnesses)
        constituent_rule = bool(witnesses) and all(v["supported"] for v in junctions.values())
        whole_chain_rule = count["supported"]
        if constituent_rule != whole_chain_rule:
            raise RuleUnresolved(
                f"{addition.addition_id}/{source}: §§5–6/A16 chain threshold unresolved: "
                f"{count['units']} complete-chain units; threshold-supported constituents="
                f"{constituent_rule}. See EVIDENCE_SCORER_READINGS.md")
        per_source[source] = {"chain": count, "junctions": junctions}
    supported = {j for j in addition.introns if any(
        per_source[s]["junctions"][j]["supported"] for s in SOURCES)}
    complete_sources = [s for s in SOURCES if per_source[s]["chain"]["supported"]]
    novel_n = len(set(addition.novel_introns) & supported)
    novel_status = ("N/A" if not addition.novel_introns else "all" if novel_n == len(addition.novel_introns)
                    else "some" if novel_n else "none")
    chain_status = "complete" if complete_sources else "partial" if supported else "none"
    if any(s in ("ONT", "PacBio") for s in complete_sources):
        tier = "T1"
    elif complete_sources:
        tier = "T2"
    elif novel_status == "all":
        tier = "T3"
    elif supported:
        tier = "T4"
    elif row["chain_callable"] or row["junction_callable"]:
        tier = "T5"
    else:
        tier = "T6"
    row.update(chain_support=chain_status, novel_junction_support=novel_status, tier=tier,
               constituent_junction_support=bool(addition.introns) and len(supported) == len(addition.introns),
               supported_junctions=len(supported), novel_supported_junctions=novel_n,
               chain_witness_reads=sum(v["chain"]["raw_reads"] for v in per_source.values()),
               chain_units_by_source={s: per_source[s]["chain"]["units"] for s in SOURCES},
               single_record_support=any(v["chain"]["raw_reads"] == 1 for v in per_source.values()),
               high_confidence_sources=[s for s in complete_sources if per_source[s]["chain"]["high_confidence"]],
               annotation_independence=sorted({o.annotation_independence for o in same}),
               model_independent=all(o.model_independent for o in same) if same else None,
               chain_negative=chain_status != "complete" if row["chain_callable"] else None,
               junction_negative=novel_status != "all" if row["junction_callable"] else None,
               intron_count=len(addition.introns), gene_length=addition.locus_end - addition.locus_start + 1,
               chain_applicable=bool(addition.introns),
               junction_counts={s: {f"{d}-{a}": v for (d, a), v in per_source[s]["junctions"].items()}
                                for s in SOURCES})
    return row


def table_inputs(rows):
    """Numerators and denominators are exported, including empty strata (undefined rate)."""
    result = []
    dimensions = ("species", "assembly", "arm", "genotype", "scope", "control", "length_bin")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in dimensions)].append(row)
    for key, group in sorted(grouped.items()):
        for filtered in (False, True):
            selected = [r for r in group if not filtered or r["filter_pass"]]
            for metric, eligible, callable_key, success in (
                ("chain", selected, "chain_callable", lambda r: r["chain_support"] == "complete"),
                ("novel_junction", [r for r in selected if r["novelty"] == "junction-novel"],
                 "junction_callable", lambda r: r["novel_junction_support"] == "all"),
                ("exact_CDS", selected, None, lambda r: r["exact_cds_match"]),
            ):
                for denominator in (("all", "callable") if callable_key else ("all",)):
                    cohort = [r for r in eligible if denominator == "all" or r[callable_key]]
                    n, k = len(cohort), sum(success(r) for r in cohort)
                    rate, lo, hi = wilson(k, n)
                    result.append(dict(zip(dimensions, key), filtered=filtered, metric=metric,
                                       denominator=denominator, numerator=k, n=n, rate=rate,
                                       wilson95_low=lo, wilson95_high=hi,
                                       callable_unsupported=sum(not success(r) for r in eligible
                                                                if callable_key and r[callable_key]),
                                       noncallable=sum(not r[callable_key] for r in eligible) if callable_key else None))
    return result


def score_scopes(additions, observations, run_specs):
    rows = []
    for addition in additions:
        specs = [s for s in run_specs if s["species"] == addition.species]
        sources = sorted({s["source"] for s in specs})
        has_est = any(s in ("EST", "FL-cDNA") for s in sources)
        for arm in (("primary", "min121") if has_est else ("primary",)):
            base = [o for o in observations if o.species == addition.species and
                    (o.source not in ("EST", "FL-cDNA") or o.metadata["arm"] == arm)]
            scopes = [(s, lambda o, s=s: o.source == s) for s in sources]
            scopes += [("tier1_union", lambda o: o.source in ("ONT", "PacBio")), ("union", lambda o: True)]
            scopes += [(f"run:{s['dataset']}:{s['run']}",
                        lambda o, s=s: o.dataset == s["dataset"] and o.run == s["run"])
                       for s in specs if s.get("arm", "primary") == arm or s["source"] in ("ONT", "PacBio")]
            genotypes = sorted({s["genotype_stratum"] for s in specs} | {"reference", "species_union"})
            for genotype in genotypes:
                stratum = [o for o in base if genotype == "species_union" or o.genotype_stratum == genotype]
                for scope, predicate in scopes:
                    row = score_addition(addition, [o for o in stratum if predicate(o)])
                    declared = [s for s in specs if
                                (scope == "union" or scope == s["source"] or
                                 scope == f"run:{s['dataset']}:{s['run']}" or
                                 (scope == "tier1_union" and s["source"] in ("ONT", "PacBio")))]
                    row.update(arm=arm, genotype=genotype, scope=scope,
                               declared_annotation_independence=sorted({s["annotation_independence"] for s in declared}),
                               declared_model_independence=sorted({s["model_independent"] for s in declared}),
                               length_bin="all")
                    rows.append(row)
    return rows


def load_genome(path):
    result, contig, parts = {}, None, []
    with open_text(path) as f:
        for line in f:
            if line.startswith(">"):
                if contig is not None:
                    result[contig] = "".join(parts).upper()
                contig = line[1:].split()[0]
                require(contig not in result, f"duplicate FASTA contig: {contig}")
                parts = []
            else:
                require(contig is not None, "sequence before FASTA header")
                parts.append(line.strip())
    if contig is not None:
        result[contig] = "".join(parts).upper()
    require(result and all(result.values()), "empty reference sequence")
    return result


def read_roles(paths):
    roles = {}
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row["dataset"], row["run"])
                require(key not in roles, f"duplicate dataset/run role: {key}")
                require(row["role"] in ("b1_validation_only", "c2_training_eligible", "excluded", "sensitivity_set"),
                        f"unknown role: {key}")
                roles[key] = row
    return roles


def observation_from_records(records, spec, metadata, genome, role):
    rejection = alignment_rejection(records)
    if rejection:
        return None, rejection
    aln = next(r for r in records if not r.flag & (4 | 0x900))
    require(aln.read in metadata, f"{spec['run']}/{aln.read}: missing C0 read metadata")
    meta = dict(metadata[aln.read])
    required = ("library", "bioproject", "genotype_stratum", "ingestion_qc_pass", "mapping_ambiguous")
    require(all(k in meta for k in required), f"{aln.read}: missing metadata: {', '.join(k for k in required if k not in meta)}")
    require(type(meta["ingestion_qc_pass"]) is bool and type(meta["mapping_ambiguous"]) is bool,
            f"{aln.read}: QC states must be booleans, not missing/unknown")
    if not meta["ingestion_qc_pass"]:
        require(meta.get("ingestion_qc_reason"), f"{aln.read}: QC failure needs a reason")
        return None, "ingestion:" + meta["ingestion_qc_reason"]
    if meta["mapping_ambiguous"]:
        return None, "mapping_ambiguous_family_only"
    edits = parse_cs(aln)
    matches = sum(e.length for e in edits if e.op in ":=")
    errors = sum(e.length for e in edits if e.op in "*+-")
    identity = matches / (matches + errors)
    aligned_fraction = aln.aligned_query / aln.query_length
    source = spec["source"]
    training = role["role"] == "c2_training_eligible"
    if training or source in ("EST", "FL-cDNA"):
        if aligned_fraction < .80:
            return None, "aligned_fraction"
        floor = (.98 if source == "PacBio" else .95) if training else .90
        if identity < floor:
            return None, "identity"
    if source in ("EST", "FL-cDNA"):
        require(type(meta.get("post_trim_length")) is int, f"{aln.read}: missing post_trim_length")
        if meta["post_trim_length"] < (121 if spec["arm"] == "min121" else 100):
            return None, "length_floor"
        if errors / (matches + errors) > .08:
            return None, "est_divergence"
        if any(not 20 <= a - d + 1 <= 200000 for d, a in aln.introns):
            return None, "est_intron_length"
    require(meta["genotype_stratum"] in ("reference", "known_nonreference", "hybrid_pooled", "unknown"),
            f"{aln.read}: invalid genotype stratum")
    if spec["species"] == "Zmays" and meta["genotype_stratum"] != "reference":
        if errors / (matches + errors) > .08:
            return None, "maize_nonreference_divergence"
        require(type(meta.get("paralog_competes")) is bool, f"{aln.read}: missing paralog competition audit")
        if meta["paralog_competes"]:
            require(type(meta.get("allele_discriminating_matches_100nt")) is int,
                    f"{aln.read}: missing allele-discriminating match audit")
            if meta["allele_discriminating_matches_100nt"] < 2:
                return None, "maize_allele_ambiguity"
    strand = transcript_strand(aln, spec["uf"], meta.get("oriented_genomic_strand"))
    calls = call_junctions(aln, genome, strand, source)
    if training:
        calls = tuple(Junction(j.raw, j.corrected, j.status if j.canonical else "training_noncanonical",
                               j.accepted and j.canonical, j.canonical) for j in calls)
    meta.update(arm=spec.get("arm", "primary"), aligned_fraction=aligned_fraction, identity=identity)
    return Observation(aln, spec["species"], spec["assembly"], source, spec["dataset"], spec["run"],
                       meta["library"], meta["bioproject"], meta["genotype_stratum"],
                       spec["annotation_independence"], spec["model_independent"], strand, calls,
                       metadata=meta), None


def load_run(spec, metadata, genome, role, samtools="samtools"):
    # Name grouping must precede the primary-only filter, or a later supplementary
    # record cannot invalidate a primary already counted. No SAM is written to disk.
    groups = defaultdict(list)
    for line in sam_lines(spec["alignment"], samtools):
        aln = parse_sam(line)
        if aln is not None:
            groups[aln.read].append(aln)
    observations, rejected = [], Counter()
    for read in sorted(groups):
        observation, reason = observation_from_records(groups[read], spec, metadata, genome, role)
        if reason:
            rejected[reason] += 1
        else:
            observations.append(observation)
    return observations, {"raw_reads": len(groups), "accepted_reads": len(observations),
                          "rejections": dict(sorted(rejected.items()))}


def write_csv(path, rows):
    require(rows, f"refusing empty table without schema: {path}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, separators=(",", ":"))
                             if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samtools", default="samtools")
    args = parser.parse_args(argv)
    try:
        execute(args.config, args.out, args.samtools)
    except (EvidenceError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"evidence scorer stopped: {exc}", file=sys.stderr)
        return 2
    return 0


def execute(config_path, output, samtools="samtools"):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    require(config.get("schema") == "b1-evidence-v1", "config requires schema b1-evidence-v1")
    root = config_path.parent
    output = Path(output).resolve()
    live_tree = Path("/data/gpfs/assoc/pgl/data/Transgenic/evidence").resolve()
    require(output != live_tree and live_tree not in output.parents,
            "output must not be under the live evidence tree")
    require(not output.exists(), f"output already exists: {output}; use a new output directory")
    hashes = {str(config_path): digest(config_path), str(Path(__file__).resolve()): digest(__file__)}

    def checked(entry):
        require(isinstance(entry, dict) and set(entry) >= {"path", "sha256"},
                "each input requires path and frozen sha256")
        path = (root / entry["path"]).resolve()
        actual = digest(path)
        require(actual == entry["sha256"], f"input hash mismatch: {path}")
        hashes[str(path)] = actual
        return path

    roles = read_roles([checked(e) for e in config["role_manifests"]])
    genome_path = checked(config["genome"])
    genome = load_genome(genome_path)
    additions_path = checked(config["additions"])
    additions = []
    for row in json.loads(additions_path.read_text()):
        row["introns"] = tuple(tuple(v) for v in row["introns"])
        row["novel_introns"] = tuple(tuple(v) for v in row["novel_introns"])
        additions.append(Addition(**row))
    require(additions, "no frozen additions/controls")
    require(len({a.addition_id for a in additions}) == len(additions), "duplicate addition/control ID")
    purpose = config.get("purpose")
    require(purpose in ("synthetic", "b1", "c0_diagnostic"), "explicit purpose required")
    if purpose == "b1":
        raise RuleUnresolved("B1 primary scoring is blocked pending the chain-unit and PCR-linkage author decisions in EVIDENCE_SCORER_READINGS.md")
    if purpose == "b1":
        gate = json.loads(checked(config["qc_gate"]).read_text())
        require(gate.get("passed") is True, "A4 pre-addition QC gate is not passed")
        require(gate.get("seed") == 123 and gate.get("loci") == 2000,
                "A4 QC gate requires 2,000 loci, seed 123")
        require(gate.get("agreement", 0) >= .95, "A4 canonical EST agreement below 95%")
        require(sum(a.control == "positive" for a in additions) >= 500,
                "P4 requires at least 500 frozen reference alternatives")
        require(config.get("positive_control_seed") == 123, "P4 sampling seed must be 123")
        require(sum(a.control == "negative" for a in additions) == sum(a.control == "addition" for a in additions),
                "P5 requires one frozen +9-nt decoy per addition")
    specs, observations, qc = [], [], []
    seen = set()
    for original in config["runs"]:
        spec = dict(original)
        key = (spec["dataset"], spec["run"])
        require(key in roles, f"missing dataset role: {key}")
        role = roles[key]
        require(role["role"] != "excluded", f"excluded run requested: {key}")
        if purpose == "b1":
            require(role["role"] == "b1_validation_only", f"not validation-only: {key}")
        require(role["species"] == spec["species"], f"manifest species mismatch: {key}")
        require(spec["source"] in SOURCES, f"unknown source: {key}")
        require(spec["assembly"] == config["assembly"], f"assembly mismatch: {key}")
        require(type(spec["uf"]) is bool, f"uf must be an explicit boolean: {key}")
        require(spec["annotation_independence"] in ("independent", "historically used by the reference", "unknown"),
                f"missing annotation independence: {key}")
        require(type(spec["model_independent"]) is bool, f"missing model independence: {key}")
        spec.setdefault("arm", "primary")
        require(spec["arm"] in ("primary", "min121"), f"invalid arm: {key}")
        armkey = key + (spec["arm"],)
        require(armkey not in seen, f"duplicate run/arm: {armkey}")
        seen.add(armkey)
        if spec.get("status") != "complete":
            raise EvidenceError(f"missing/incomplete run: {key}, status={spec.get('status', 'missing')}")
        spec["alignment"] = str(checked(spec["alignment"]))
        if purpose != "synthetic":
            done_path = checked(spec["done"])
            checked(spec["provenance"])
            done = dict(line.split("=", 1) for line in done_path.read_text().splitlines() if "=" in line)
            require(done.get("bam_md5") == digest(spec["alignment"], "md5"), f"DONE BAM hash mismatch: {key}")
            if spec["source"] == "ONT":
                audit = json.loads(checked(spec["orientation_audit"]).read_text())
                require(done.get("uf") == ("-uf" if spec["uf"] else "none"), f"DONE uf mismatch: {key}")
                require(audit.get("status") == done.get("audit_status"), f"audit/DONE disagreement: {key}")
                require(spec["uf"] == (audit.get("status") == "PASS"), f"uf/audit disagreement: {key}")
                if spec["uf"]:
                    require(audit.get("sense_read_fraction", 0) >= .95, f"A39 orientation gate failed: {key}")
        meta_path = checked(spec["metadata"])
        meta = json.loads(meta_path.read_text())
        got, stats = load_run(spec, meta, genome, role, samtools)
        observations.extend(got)
        stats.update(dataset=key[0], run=key[1], species=spec["species"], source=spec["source"],
                     arm=spec["arm"], annotation_independence=spec["annotation_independence"],
                     model_independent=spec["model_independent"], uf=spec["uf"])
        qc.append(stats)
        specs.append(spec)
    require(specs, "no runs declared")
    for spec in specs:
        if spec["source"] in ("EST", "FL-cDNA"):
            key = (spec["dataset"], spec["run"])
            require(all(key + (arm,) in seen for arm in ("primary", "min121")),
                    f"A37 missing paired EST arm: {key}")
    require(all(a.assembly == config["assembly"] for a in additions), "addition assembly mismatch")
    require(all(any(s["species"] == a.species for s in specs) for a in additions),
            "species has additions but no declared evidence runs")
    observations = assign_molecules(observations)
    # Controls are completed before any addition is overlapped with evidence (§10).
    controls = [a for a in additions if a.control != "addition"]
    rows = score_scopes(controls, observations, specs)
    rows.extend(score_scopes([a for a in additions if a.control == "addition"], observations, specs))
    tables = table_inputs(rows)
    # Commit output only after every run and every rule gate passes. Failed runs
    # leave no seemingly complete table with a smaller denominator.
    output.mkdir(parents=True)
    write_csv(output / "per_addition.csv", rows)
    write_csv(output / "S12_inputs.csv", tables)
    write_csv(output / "S12c_runs.csv", qc)
    with open(output / "C0_observations.jsonl", "w") as f:
        for o in observations:
            f.write(json.dumps(asdict(o), sort_keys=True) + "\n")
    report = {"schema": config["schema"], "purpose": purpose, "inputs_sha256": hashes,
              "source_thresholds": THRESHOLDS,
              "missing_outputs": ["P2 reference-alternative recall requires frozen reference alternative IDs/denominator",
                                  "A3 TES/TSS, A6 rarefaction and S12d terminal completeness are separate C0 analyses",
                                  "A7 MAPQ10/placement sensitivities are not primary outcomes"],
              "outputs_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "PROVENANCE.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (output / "DONE").write_text("provenance_sha256=" + digest(output / "PROVENANCE.json") + "\n")


if __name__ == "__main__":
    sys.exit(main())
