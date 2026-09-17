"""Bounded, pure normalized EST membership/leakage diagnostics, NOT production C0.

Caller supplies complete inventories, report joins, biological-library aliases and
all-member audit placements (including rejected support). Completeness booleans
and receipt/hash strings are caller statements, NOT independent source attestation.
No source scanning, sequence verification, aligner, scorer, chain construction or
publication occurs here. All outputs are exploratory and training_eligible=False.
Original intervals and all genomic intervals are zero-based half-open; upstream
must convert UniVec's one-based inclusive report ranges. `accession` is the original
versioned accession, never a guessed `_partN` QNAME. uid is a caller-provided unique
inventory key (namespace it by dataset/arm/name); original names remain in inventory.

Support acceptance is upstream, not re-scored here. Primary/min121 are separate
support views; only explicit parent relations link arms. Component IDs are sorted
member tuples, an implementation detail, NOT a newly frozen protocol hash formula.
Library cap 10 remains the existing scorer's per-feature cap, not a molecule,
phase or raster cap. Unknown policies and contradictory metadata raise ValueError.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Member:
    uid: str
    species: str
    assembly: str
    dataset: str
    arm: str
    accession: str
    library: str
    clone: str
    clone_status: str  # known / clone_unavailable / uninspected
    original_accession: str
    original_length: int
    original_interval: tuple
    screened_sha256: str
    provenance: str
    bioproject: str
    genotype: str
    role: str  # train_track / excluded_heldout / never (caller declaration)
    support_accepted: bool
    report_id: str  # authoritative normalized UniVec report-row identity


@dataclass(frozen=True)
class Placement:
    uid: str
    member: str
    contig: str
    strand: str  # transcript strand, not inferred from FLAG
    start: int
    end: int
    blocks: tuple  # ordered CIGAR-derived genomic blocks, never literal CIGAR
    kind: str  # primary / secondary / supplementary / ambiguous
    mapq: int


@dataclass(frozen=True)
class Relation:
    left: str
    right: str
    kind: str  # mate / parent / split / version
    provenance: str


@dataclass(frozen=True)
class Audit:
    member: str
    total_placements: int  # exact number of supplied complete audit placements
    secondary_placements: int
    complete: bool
    provenance: str


@dataclass(frozen=True)
class Exclusion:
    uid: str
    kind: str
    scope: str  # global / train_context
    context: str | None
    contig: str
    start: int
    end: int


# Exact helper body from 59_evidence_support.py:407-410 (verified against 9603e20).
# Local copy avoids importing the scorer's unrelated CLI/parser dependencies.
def normalize_clone(value):
    value = re.sub(r"(?:[\s_.:/-]+(?:5['′]?|3['′]?|forward|reverse|fwd|rev))$", "",
                   value.strip().lower())
    return re.sub(r"[^a-z0-9]", "", value)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def interval(start, end):
    require(type(start) is int and type(end) is int and 0 <= start < end,
            'invalid half-open interval')


def active_exclusions(exclusions, context):
    global_kinds = {'heldout', 'test_block', 'test_orthogroup', 'hard_flag',
                    'overlap_component', 'decoy_masked', 'shared_junction'}
    contextual = {'valid_orthogroup', 'partial'}
    seen, active = set(), []
    for e in exclusions:
        require(e.uid and e.uid not in seen and e.contig, 'duplicate/invalid exclusion')
        seen.add(e.uid)
        interval(e.start, e.end)
        require(e.kind in global_kinds | contextual, 'unknown exclusion policy')
        require((e.kind in contextual and e.scope == 'train_context' and bool(e.context)) or
                (e.kind in global_kinds and e.scope == 'global' and e.context is None),
                'invalid exclusion scope')
        if e.scope == 'global' or e.context == context:
            active.append(e)
    return sorted(active, key=lambda e: e.uid)


def close_molecules(*, species, assembly, members, placements, relations, audits,
                    exclusions, library_aliases, inventory_complete, identity_complete,
                    exclusions_complete, context, policy):
    """Return auditable components, edges, all originals and per-arm support members.

    `diagnostic_ready` only means the supplied tables pass bounded checks; it never
    authorizes training. Any unknown identity or missing audit blocks every ready
    component, not just the unknown row. `training_support_members` is empty until
    that diagnostic passes; it is still not a training authorization.
    """
    require(policy == 'A11_A21_A46_normalized_v1', 'unknown scientific policy')
    require(bool(species) and bool(assembly) and bool(context), 'explicit scope required')
    flags = (inventory_complete, identity_complete, exclusions_complete)
    require(all(type(x) is bool for x in flags), 'completeness must be explicit booleans')
    blockers = {name for name, flag in zip(('inventory_incomplete','identity_incomplete',
                                          'exclusions_incomplete'), flags) if not flag}
    for alias, canonical in library_aliases.items():
        require(alias and canonical and canonical != 'unknown' and
                library_aliases.get(canonical) == canonical, 'conflicting library aliases')
    members = tuple(sorted(members, key=lambda m: m.uid))
    placements = tuple(sorted(placements, key=lambda p: p.uid))
    by_id = {m.uid: m for m in members}
    require(len(by_id) == len(members), 'duplicate member')
    libs, clones, accessions = {}, {}, {}
    report_groups = defaultdict(list)
    accession_reports = defaultdict(set)
    accession_libraries = defaultdict(set)
    for m in members:
        require(m.uid and m.dataset and m.provenance and m.report_id and m.bioproject and
                m.genotype, 'missing identity/provenance fields')
        require((m.species,m.assembly) == (species,assembly), 'mixed species/assembly')
        require(m.arm in ('primary','min121') and m.role in
                ('train_track','excluded_heldout','never'), 'unknown arm/role policy')
        require(type(m.support_accepted) is bool, 'explicit support decision required')
        require(m.clone_status in ('known','clone_unavailable','uninspected'), 'unknown clone policy')
        require(m.accession == m.original_accession, 'original accession mismatch')
        match = re.fullmatch(r'([A-Za-z0-9_]+)(?:\.(\d+))?', m.accession)
        require(match is not None, 'invalid original accession')
        accessions[m.uid] = (match[1], int(match[2] or 0))
        require(type(m.original_length) is int and m.original_length > 0, 'invalid original length')
        require(len(m.original_interval) == 2, 'invalid original interval')
        interval(*m.original_interval)
        require(m.original_interval[1] <= m.original_length, 'original interval out of bounds')
        require(re.fullmatch(r'[a-f0-9]{64}',m.screened_sha256) is not None, 'invalid sequence hash')
        libs[m.uid] = library_aliases.get(m.library)
        clones[m.uid] = normalize_clone(m.clone)
        if libs[m.uid] is None or m.clone_status == 'uninspected':
            blockers.add('unresolved_identity:'+m.uid)
        if m.clone_status == 'known':
            require(bool(clones[m.uid]), 'known clone has empty identity')
        elif m.clone_status == 'clone_unavailable':
            require(not m.clone, 'unavailable clone has contradictory identity')
        report_groups[(m.dataset,m.arm,m.report_id)].append(m)
        accession_reports[(m.dataset,m.arm,m.accession)].add(m.report_id)
        if libs[m.uid] is not None:
            accession_libraries[(m.species,m.assembly,m.arm,accessions[m.uid][0])].add(libs[m.uid])
    # Versions remain one accession across download datasets; library-scoped
    # union/latest keys must not hide contradictory resolved preparation IDs.
    require(all(len(libraries) <= 1 for libraries in accession_libraries.values()),
            'accession stem has conflicting canonical libraries')
    require(all(len(ids) == 1 for ids in accession_reports.values()),
            'same original accession has contradictory report identity')
    for group in report_groups.values():
        require(len({(m.accession,m.original_length) for m in group}) == 1,
                'contradictory report accession/length')
        ranges = sorted(m.original_interval for m in group)
        require(all(a[1] <= b[0] for a,b in zip(ranges,ranges[1:])), 'overlapping report pieces')
    by_member = defaultdict(list)
    require(len({p.uid for p in placements}) == len(placements), 'duplicate placement')
    for p in placements:
        require(p.uid and p.member in by_id and p.contig, 'orphan/invalid placement')
        require(p.kind in ('primary','secondary','supplementary','ambiguous') and
                p.strand in ('+','-','unknown'), 'unknown placement policy')
        require(type(p.mapq) is int and 0 <= p.mapq <= 255, 'invalid MAPQ')
        interval(p.start,p.end)
        require(bool(p.blocks), 'missing aligned blocks')
        for a,b in p.blocks:
            interval(a,b)
            require(p.start <= a < b <= p.end, 'block outside placement span')
        require(all(a[1] <= b[0] for a,b in zip(p.blocks,p.blocks[1:])), 'unordered blocks')
        by_member[p.member].append(p)
    for m in members:
        require(not m.support_accepted or bool(by_member[m.uid]), 'accepted support has no placement')
    audit_map = {a.member:a for a in audits}
    require(len(audit_map) == len(audits) and set(audit_map) <= set(by_id), 'duplicate/orphan audit')
    missing_audit = set()
    for m in members:
        a = audit_map.get(m.uid)
        if a is None:
            missing_audit.add(m.uid)
            continue
        require(type(a.complete) is bool and a.provenance and
                type(a.total_placements) is int and type(a.secondary_placements) is int and
                0 <= a.secondary_placements <= a.total_placements, 'invalid audit receipt')
        require(a.total_placements == len(by_member[m.uid]) and a.secondary_placements ==
                sum(p.kind == 'secondary' for p in by_member[m.uid]), 'audit inventory mismatch')
        if not a.complete:
            missing_audit.add(m.uid)
    if missing_audit:
        blockers.add('missing_member_audit')
    parent = {m.uid:m.uid for m in members}
    edges = set()

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a,b,reason,provenance):
        if a == b:
            return
        left,right = sorted((a,b))
        edges.add((left,right,reason,provenance))
        x,y = sorted((root(a),root(b)))
        parent[y] = x

    keys = {}
    for m in members:
        if libs[m.uid] is None or m.clone_status == 'uninspected':
            continue
        base = (libs[m.uid],m.arm)
        units = [('version',base+(accessions[m.uid][0],))]
        if m.clone_status == 'known':
            units.append(('clone',base+(clones[m.uid],)))
        else:
            primary = [p for p in by_member[m.uid] if p.kind == 'primary']
            if len(primary) == 1 and primary[0].strand != 'unknown':
                p = primary[0]
                units.append(('exact_blocks',base+(p.contig,p.strand,p.blocks)))
            else:
                blockers.add('unresolved_fallback:'+m.uid)
        for reason,key in units:
            if (reason,key) in keys:
                union(keys[reason,key],m.uid,reason,'normalized_inventory')
            else:
                keys[reason,key] = m.uid
    for group in report_groups.values():
        for m in group[1:]:
            union(group[0].uid,m.uid,'split',m.report_id)
    for r in relations:
        require(r.left in by_id and r.right in by_id and r.left != r.right and r.provenance,
                'orphan/invalid relation')
        a,b = by_id[r.left],by_id[r.right]
        require(r.kind in ('mate','parent','split','version'), 'unknown relation policy')
        require(libs[a.uid] is not None and libs[a.uid] == libs[b.uid], 'relation library contradiction')
        require(r.kind == 'parent' or a.arm == b.arm, 'cross-arm link requires explicit parent')
        if r.kind == 'version':
            require(accessions[a.uid][0] == accessions[b.uid][0], 'version accession contradiction')
        if r.kind == 'split':
            require((a.dataset,a.arm,a.report_id) == (b.dataset,b.arm,b.report_id), 'split needs report proof')
        if r.kind == 'mate':
            require(a.clone_status == b.clone_status == 'known' and clones[a.uid] == clones[b.uid],
                    'mate needs matching documented clone identity')
        union(a.uid,b.uid,r.kind,r.provenance)
    groups = defaultdict(list)
    for m in members:
        groups[root(m.uid)].append(m)
    active = active_exclusions(exclusions,context)
    newest = {}
    for m in members:
        key = (libs[m.uid],m.arm,accessions[m.uid][0])
        newest[key] = max(newest.get(key,-1),accessions[m.uid][1])
    components = []
    lt50 = 0
    for group in groups.values():
        ids = tuple(m.uid for m in group)
        require(len({libs[m.uid] for m in group}) == 1, 'component library contradiction')
        require(len({clones[m.uid] for m in group if m.clone_status == 'known'}) <= 1,
                'component clone contradiction')
        ps = [p for m in group for p in by_member[m.uid]]
        # Counts describe supplied placements even if the independent completeness
        # receipt is missing; missing receipts separately block the entire view.
        total = len(ps)
        secondary = sum(p.kind == 'secondary' for p in ps)
        lt50 += int(total < 50)
        causes = {('missing_audit',uid) for uid in ids if uid in missing_audit}
        if secondary >= 50:
            causes.add(('secondary_saturation',str(secondary)))
        # ponytail: bounded O(placements * exclusions); indexed production joins out of scope.
        for p in ps:
            overlaps = [e for e in active if p.contig == e.contig and p.start < e.end and e.start < p.end]
            causes.update(('overlap',p.member,p.uid,e.uid,e.kind) for e in overlaps)
            if p.mapq < 5 and not overlaps:
                causes.add(('low_mapq_outside',p.member,p.uid))
        support = defaultdict(list)
        for m in group:
            key = (libs[m.uid],m.arm,accessions[m.uid][0])
            if m.support_accepted and accessions[m.uid][1] == newest[key]:
                require(m.original_interval[1]-m.original_interval[0] >=
                        (100 if m.arm == 'primary' else 121), 'support below arm length minimum')
                support[m.arm].append(m.uid)
        primary_members = [m for m in group if m.arm == 'primary']
        disposition = 'excluded' if causes else ('never' if not primary_members or
                      any(m.role != 'train_track' for m in primary_members) else 'retained')
        components.append(dict(id=ids,members=ids,support_members={k:tuple(v) for k,v in sorted(support.items())},
                               disposition=disposition,causes=tuple(sorted(causes)),
                               total_placements=total,secondary_placements=secondary))
    n = len(components)
    fraction = lt50/n if n else None
    ready = not blockers and n > 0 and lt50*1000 >= n*999
    for c in components:
        c['ready'] = ready and c['disposition'] == 'retained'
        c['training_eligible'] = False
        c['training_support_members'] = c['support_members'].get('primary',()) if c['ready'] else ()
    result = dict(species=species,assembly=assembly,context=context,members=members,placements=placements,
                  audits=tuple(sorted(audits,key=lambda a:a.member)),
                  relations=tuple(sorted(relations,key=lambda r:(r.left,r.right,r.kind,r.provenance))),
                  exclusions=tuple(sorted(exclusions,key=lambda e:e.uid)),
                  library_aliases=tuple(sorted(library_aliases.items())),
                  edges=tuple(sorted(edges)),components=components,blockers=tuple(sorted(blockers)),
                  diagnostic_ready=ready,training_eligible=False,exploratory=True,
                  exhaustiveness=dict(molecules_lt50=lt50,molecules_all=n,fraction=fraction))
    recheck_retained(result,exclusions=exclusions,context=context)
    return result


def recheck_retained(result, *, exclusions, context):
    """Independent all-member/span rejoin; never trusts stored overlap causes."""
    require(context == result['context'], 'recheck context mismatch')
    exclusions = tuple(sorted(exclusions,key=lambda e:e.uid))
    require(exclusions == result['exclusions'], 'recheck exclusion manifest mismatch')
    active = active_exclusions(exclusions,context)
    retained = {uid for c in result['components'] if c['disposition'] == 'retained' for uid in c['members']}
    for p in result['placements']:
        if p.member in retained:
            for e in active:
                require(not (p.contig == e.contig and max(p.start,e.start) < min(p.end,e.end)),
                        'retained molecule overlaps exclusion: '+p.member+'/'+e.uid)
    return True
