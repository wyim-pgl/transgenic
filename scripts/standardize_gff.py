import os
import re
import sys
from collections import defaultdict

# Maximum reasonable coordinate values for validation
# Largest plant chromosome is ~5Gb (wheat), use 500M as generous max
MAX_COORDINATE = 500_000_000
# Maximum reasonable gene length (500kb - largest known plant genes)
MAX_GENE_LENGTH = 500_000
# Maximum reasonable mRNA/transcript length (100kb)
MAX_MRNA_LENGTH = 100_000
# Maximum reasonable intron length (~20kb covers nearly all plant introns;
# prevents megabase-spanning introns from prompt-mode hallucinations)
MAX_INTRON_LENGTH = 20_000

# Map species prefix -> genome FASTA index file (for chromosome length validation)
SPECIES_TO_FAI = {
    "A_thaliana": "Athaliana_167_TAIR10.fa.fai",
    "B_distachyon": "Bdistachyon_314_v3.0.fa.fai",
    "B_rapa": "BrapaO_302V_711_v1.0.fa.fai",
    "G_max": "Gmax_880_v6.0.fa.fai",
    "L_sativa": "Lsativa_467_v8.fa.fai",
    "O_sativa": "Osativa_323_v7.0.fa.fai",
    "P_patens": "Ppatens_318_v3.fa.fai",
    "P_trichocarpa": "Ptrichocarpa_533_v4.0.fa.fai",
    "S_bicolor": "Sbicolor_730_v5.0.fa.fai",
    "S_italica": "Sitalica_312_v2.fa.fai",
    "S_lycopersicum": "Slycopersicum_796_ITAG5.0.fa.fai",
    "V_vinifera": "Vvinifera_T2T_ref.fa.fai",
    "Z_mays": "Zmays_493_APGv4.fa.fai",
}

# Canonical species prefixes so annevo/helixer (and others) match
SPECIES_CANONICAL = {
    "Athaliana": "A_thaliana",
    "athaliana": "A_thaliana",
    "A_thaliana": "A_thaliana",
    "a_thaliana": "A_thaliana",
    "Bdistachyon": "B_distachyon",
    "bdistachyon": "B_distachyon",
    "B_distachyon": "B_distachyon",
    "b_distachyon": "B_distachyon",
    "BrapaO": "B_rapa",
    "brapao": "B_rapa",
    "B_rapa": "B_rapa",
    "b_rapa": "B_rapa",
    "Brapa": "B_rapa",
    "brapa": "B_rapa",
    "Gmax": "G_max",
    "gmax": "G_max",
    "G_max": "G_max",
    "g_max": "G_max",
    "Lsativa": "L_sativa",
    "lsativa": "L_sativa",
    "L_sativa": "L_sativa",
    "l_sativa": "L_sativa",
    "Osativa": "O_sativa",
    "osativa": "O_sativa",
    "O_sativa": "O_sativa",
    "o_sativa": "O_sativa",
    "Ppatens": "P_patens",
    "ppatens": "P_patens",
    "P_patens": "P_patens",
    "p_patens": "P_patens",
    "Ptrichocarpa": "P_trichocarpa",
    "ptrichocarpa": "P_trichocarpa",
    "P_trichocarpa": "P_trichocarpa",
    "p_trichocarpa": "P_trichocarpa",
    "Sbicolor": "S_bicolor",
    "sbicolor": "S_bicolor",
    "S_bicolor": "S_bicolor",
    "s_bicolor": "S_bicolor",
    "Sitalica": "S_italica",
    "sitalica": "S_italica",
    "S_italica": "S_italica",
    "s_italica": "S_italica",
    "Slycopersicum": "S_lycopersicum",
    "slycopersicum": "S_lycopersicum",
    "S_lycopersicum": "S_lycopersicum",
    "s_lycopersicum": "S_lycopersicum",
    "Vvinifera": "V_vinifera",
    "vvinifera": "V_vinifera",
    "V_vinifera": "V_vinifera",
    "v_vinifera": "V_vinifera",
    "Zmays": "Z_mays",
    "zmays": "Z_mays",
    "Z_mays": "Z_mays",
    "z_mays": "Z_mays",
}

# Pre-compiled regex patterns
RE_CLEAN_CHARS = re.compile(r"[^A-Za-z0-9_]")
RE_MULTI_UNDERSCORE = re.compile(r"_+")
RE_GFF_EXT = re.compile(r"\.gff3?$")
RE_FA_ANNOTATED = re.compile(r"\.fa_a+n+otated$")
RE_SPLIT_SPECIES = re.compile(r"[_.]")


def load_chrom_lengths(genome_dir, species_prefix):
    """Load chromosome lengths from a .fai file for the given species.
    Returns a dict {chrom_name: length} or empty dict if unavailable."""
    fai_file = SPECIES_TO_FAI.get(species_prefix)
    if not fai_file:
        return {}
    fai_path = os.path.join(genome_dir, fai_file)
    if not os.path.exists(fai_path):
        return {}
    chrom_lengths = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom_lengths[parts[0]] = int(parts[1])
    return chrom_lengths


def merge_intervals(intervals):
    """Merge overlapping or adjacent (touching) intervals.
    Returns list of [start, end] merged intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1] + 1:  # adjacent or overlapping
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def parse_attributes_gff3(attr_string):
    """Parse GFF3 style attributes (key=value;)"""
    attributes = {}
    for part in attr_string.split(';'):
        if '=' in part:
            idx = part.index('=')
            attributes[part[:idx]] = part[idx+1:]
    return attributes


def parse_attributes_gtf(attr_string):
    """Parse GTF style attributes (key "value";)"""
    attributes = {}
    for part in attr_string.split(';'):
        part = part.strip()
        if not part:
            continue
        space_idx = part.find(' ')
        if space_idx > 0:
            key = part[:space_idx]
            value = part[space_idx+1:].strip('"')
            attributes[key] = value
    return attributes


def parse_attributes(attr_string):
    """Fast attribute parser with format detection."""
    if not attr_string or attr_string == '.':
        return {}
    # Check format once and dispatch
    if ' "' in attr_string:
        return parse_attributes_gtf(attr_string)
    return parse_attributes_gff3(attr_string)


def format_attributes(attributes):
    """Format attributes dict to GFF3 string."""
    if not attributes:
        return "."

    parts = []
    # ID and Parent first
    if 'ID' in attributes:
        parts.append(f"ID={attributes['ID']}")
    if 'Parent' in attributes:
        parts.append(f"Parent={attributes['Parent']}")

    # Rest of attributes
    for k, v in attributes.items():
        if k != 'ID' and k != 'Parent':
            parts.append(f"{k}={v}")

    return ";".join(parts)


def infer_species(tool_name, filename):
    """Infer a clean species name from tool + filename."""
    base = os.path.basename(filename)

    if tool_name == "helixer":
        for suffix in ("_annotated_helixer.gff3", "_annotated_helixer.gff",
                       "_helixer.gff3", "_helixer.gff"):
            if base.endswith(suffix):
                species = base[:-len(suffix)]
                break
        else:
            species = os.path.splitext(base)[0]

    elif tool_name == "annevo":
        tmp = RE_GFF_EXT.sub("", base)
        tmp = RE_FA_ANNOTATED.sub("", tmp)
        species = tmp.split("_")[0]

    else:
        tmp = os.path.splitext(base)[0]
        species = RE_SPLIT_SPECIES.split(tmp)[0]

    # Clean and normalize
    species = RE_CLEAN_CHARS.sub("_", species)
    species = RE_MULTI_UNDERSCORE.sub("_", species).strip("_")

    return SPECIES_CANONICAL.get(species, species) or "species"


def _check_intron_lengths(exons_sorted):
    """Check that all introns (gaps between consecutive exons) are within
    MAX_INTRON_LENGTH.  Returns the length of the longest intron, or 0
    if there are fewer than 2 exons."""
    max_intron = 0
    for i in range(1, len(exons_sorted)):
        intron_len = exons_sorted[i]['start'] - exons_sorted[i - 1]['end'] - 1
        if intron_len > max_intron:
            max_intron = intron_len
    return max_intron


def _fix_inverted_cds(cds_features):
    """Detect and fix inverted/overlapping CDS features within a transcript.

    The model can hallucinate a CDS whose coordinates overlap with an adjacent
    CDS (an "inversion" — the exon boundaries are effectively reversed).
    Fix by trimming the overlapping CDS so it starts just after the previous
    one ends.  If trimming leaves a CDS shorter than 3 bp it is marked for
    removal.

    Operates in-place on the list.  Returns the number of features fixed."""
    if len(cds_features) < 2:
        return 0

    cds_features.sort(key=lambda c: c['start'])

    fixed = 0
    to_remove = []
    prev_idx = 0  # index of last surviving CDS
    for i in range(1, len(cds_features)):
        prev_end = cds_features[prev_idx]['end']
        cur = cds_features[i]
        if cur['start'] <= prev_end:
            # Trim current CDS to start right after previous surviving CDS
            cur['start'] = prev_end + 1
            fixed += 1
            if cur['start'] > cur['end'] or (cur['end'] - cur['start'] + 1) < 3:
                to_remove.append(cur)
                continue  # don't advance prev_idx
        prev_idx = i

    for cds in to_remove:
        cds_features.remove(cds)

    return fixed


def standardize_gff(input_file, output_file, species_prefix, tool_name, genome_dir=None):
    """Process and standardize a GFF/GTF file."""
    print(f"Processing {input_file}...")

    fixes = set()  # Use set from start for O(1) dedup

    # Load chromosome lengths for coordinate validation
    chrom_lengths = {}
    if genome_dir:
        chrom_lengths = load_chrom_lengths(genome_dir, species_prefix)
        if chrom_lengths:
            print(f"  Loaded {len(chrom_lengths)} chromosome lengths for {species_prefix}")
        else:
            print(f"  [WARN] No chromosome lengths found for {species_prefix}, using MAX_COORDINATE fallback")

    # Data structures for single-pass categorization
    genes = {}
    mrnas = {}
    other_features = []
    feature_map = {}

    # 1. Read and categorize all features in one pass
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line[0] == '#':
                continue

            parts = line.rstrip('\n\r').split('\t')
            if len(parts) != 9:
                continue

            seqid, source, type_, start, end, score, strand, phase, attributes_str = parts

            # Parse coordinates
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                continue

            if start > end:
                start, end = end, start
                fixes.add("Swapped coordinates for features")

            if start < 1:
                fixes.add("Skipped features with start coordinate < 1")
                continue

            # Validate against actual chromosome length if available
            if chrom_lengths and seqid in chrom_lengths:
                chrom_len = chrom_lengths[seqid]
                if start > chrom_len or end > chrom_len:
                    fixes.add("Skipped features exceeding chromosome length")
                    continue
            elif start > MAX_COORDINATE or end > MAX_COORDINATE:
                # Fallback: use hard max if no FAI data
                fixes.add(f"Skipped features with coordinates exceeding {MAX_COORDINATE:,}")
                continue

            # Check feature length is reasonable
            feature_length = end - start + 1
            if type_ == 'gene' and feature_length > MAX_GENE_LENGTH:
                fixes.add(f"Skipped genes longer than {MAX_GENE_LENGTH:,} bp")
                continue
            elif type_ in ('mRNA', 'transcript') and feature_length > MAX_MRNA_LENGTH:
                fixes.add(f"Skipped mRNAs/transcripts longer than {MAX_MRNA_LENGTH:,} bp")
                continue

            # Normalize type
            if type_ == 'transcript':
                type_ = 'mRNA'

            # Parse attributes
            attributes = parse_attributes(attributes_str)

            # GTF to GFF3 ID/Parent logic
            if 'ID' not in attributes:
                gene_id = attributes.get('gene_id')
                if gene_id:
                    if type_ == 'gene':
                        attributes['ID'] = gene_id
                    elif type_ == 'mRNA':
                        attributes['ID'] = attributes.get('transcript_id', f"{gene_id}.t1")
                        attributes['Parent'] = gene_id
                    else:
                        attributes['Parent'] = attributes.get('transcript_id', gene_id)
                        attributes['ID'] = f"temp_{line_num}"
                else:
                    attributes['ID'] = f"temp_{line_num}"

            # Normalize invalid strand values early; resolve later from parent/children.
            if strand not in ('+', '-'):
                strand = '.'

            # PASA expects CDS phase to be 0/1/2. Default invalid/missing phase to 0.
            if type_ == 'CDS' and phase not in ('0', '1', '2'):
                phase = '0'
                fixes.add("Normalized missing/invalid CDS phase to 0")

            feature = {
                'seqid': seqid,
                'source': source,
                'type': type_,
                'start': start,
                'end': end,
                'score': score,
                'strand': strand,
                'phase': phase,
                'attributes': attributes,
                'children': [],
            }

            fid = attributes['ID']
            feature_map[fid] = feature

            # Categorize immediately
            if type_ == 'gene':
                genes[fid] = feature
            elif type_ == 'mRNA':
                mrnas[fid] = feature
            else:
                other_features.append(feature)

    # 2. Build Hierarchy
    # Link mRNAs to genes
    for mrna_id, mrna in mrnas.items():
        parent_id = mrna['attributes'].get('Parent')
        if parent_id and parent_id in genes:
            genes[parent_id]['children'].append(mrna)
        else:
            # Create parent gene for orphan mRNA
            new_gene_id = f"gene_for_{mrna_id}"
            # Carry the child's provenance onto the synthetic parent. GM= records which
            # input locus a completion-mode prediction was prompted from, and this invented
            # gene row covers exactly the same locus as the mRNA beneath it. Dropping the
            # tag emits a gene that no downstream step can pair to an input, which inflates
            # gene counts and hides an equal amount of real loss from any check comparing
            # genes in against genes out — `33_run_polishing_benchmark.py`'s I-3 gate
            # caught this on the 2026-08-06 run.
            new_attributes = {'ID': new_gene_id}
            if 'GM' in mrna['attributes']:
                new_attributes['GM'] = mrna['attributes']['GM']
            new_gene = {
                'seqid': mrna['seqid'],
                'source': mrna['source'],
                'type': 'gene',
                'start': mrna['start'],
                'end': mrna['end'],
                'score': '.',
                'strand': mrna['strand'],
                'phase': '.',
                'attributes': new_attributes,
                'children': [mrna],
            }
            genes[new_gene_id] = new_gene
            feature_map[new_gene_id] = new_gene
            mrna['attributes']['Parent'] = new_gene_id
            fixes.add("Created parent genes for orphan mRNAs")

    # Link exons/CDS to mRNAs
    gene_direct_children = defaultdict(list)  # gene_id -> [features]
    for o in other_features:
        parent_id = o['attributes'].get('Parent')
        if not parent_id or parent_id not in feature_map:
            fixes.add("Dropped orphan features")
            continue

        parent = feature_map[parent_id]
        if parent['type'] == 'mRNA':
            parent['children'].append(o)
        elif parent['type'] == 'gene':
            gene_direct_children[parent_id].append(o)

    # Create one synthetic mRNA per gene for all its direct children
    for gene_id, direct_children in gene_direct_children.items():
        gene = feature_map[gene_id]
        new_mrna_id = f"mRNA_for_{gene_id}"
        new_mrna = {
            'seqid': gene['seqid'],
            'source': gene['source'],
            'type': 'mRNA',
            'start': min(c['start'] for c in direct_children),
            'end': max(c['end'] for c in direct_children),
            'score': '.',
            'strand': gene['strand'],
            'phase': '.',
            'attributes': {'ID': new_mrna_id, 'Parent': gene_id},
            'children': direct_children,
        }
        gene['children'].append(new_mrna)
        feature_map[new_mrna_id] = new_mrna
        for o in direct_children:
            o['attributes']['Parent'] = new_mrna_id
        fixes.add("Created intermediate mRNAs for features attached directly to genes")

    # 2.5. Create/fix exons, strand consistency, and translation viability checks
    # This fixes issues where transgenic model outputs overlapping CDS/UTR features
    filtered_genes = {}
    for gene_id, gene in genes.items():
        valid_mrnas = []
        for mrna in gene['children']:
            children = mrna['children']
            has_exon = any(c['type'] == 'exon' for c in children)
            has_cds = any(c['type'] == 'CDS' for c in children)
            utr_types = ('five_prime_UTR', 'three_prime_UTR')
            has_utr = any(c['type'] in utr_types for c in children)

            # Ensure mRNA strand is valid and propagated to children.
            # PASA fails hard if model orientation is undefined.
            strand_candidates = []
            if mrna['strand'] in ('+', '-'):
                strand_candidates.append(mrna['strand'])
            if gene['strand'] in ('+', '-'):
                strand_candidates.append(gene['strand'])
            strand_candidates.extend(
                c['strand'] for c in children if c['strand'] in ('+', '-')
            )
            mrna_strand = strand_candidates[0] if strand_candidates else None
            if mrna_strand is None:
                fixes.add("Removed mRNAs with invalid/undefined strand")
                continue
            mrna['strand'] = mrna_strand
            for child in children:
                if child['strand'] not in ('+', '-'):
                    child['strand'] = mrna_strand

            if not has_exon and has_cds:
                # No exons exist: create one exon per CDS, then extend with adjacent UTRs
                # This ensures exon count >= CDS count (required by PASA)
                cds_list = sorted(
                    [c for c in children if c['type'] == 'CDS'],
                    key=lambda x: x['start']
                )
                utr_list = [c for c in children if c['type'] in utr_types]

                # Use first CDS as template for seqid/source/strand
                template = cds_list[0]

                # Create one exon per CDS
                exon_intervals = [[c['start'], c['end']] for c in cds_list]

                # Extend exons to cover adjacent UTRs
                for utr in utr_list:
                    utr_start, utr_end = utr['start'], utr['end']
                    extended = False
                    for exon_iv in exon_intervals:
                        # UTR adjacent to or overlapping exon start
                        if utr_end + 1 >= exon_iv[0] and utr_start < exon_iv[0]:
                            exon_iv[0] = min(exon_iv[0], utr_start)
                            extended = True
                            break
                        # UTR adjacent to or overlapping exon end
                        if utr_start - 1 <= exon_iv[1] and utr_end > exon_iv[1]:
                            exon_iv[1] = max(exon_iv[1], utr_end)
                            extended = True
                            break
                    # If UTR not adjacent to any CDS exon, create a standalone exon for it
                    if not extended:
                        exon_intervals.append([utr_start, utr_end])

                # Sort and merge only truly overlapping exons (not just adjacent)
                exon_intervals.sort(key=lambda x: x[0])
                merged_exons = [exon_intervals[0]]
                for start, end in exon_intervals[1:]:
                    # Only merge if overlapping (not just adjacent)
                    if start <= merged_exons[-1][1]:
                        merged_exons[-1][1] = max(merged_exons[-1][1], end)
                    else:
                        merged_exons.append([start, end])

                # Verify we still have at least as many exons as CDS
                # If merging reduced count below CDS count, revert to per-CDS exons
                if len(merged_exons) < len(cds_list):
                    merged_exons = [[c['start'], c['end']] for c in cds_list]
                    fixes.add("Created per-CDS exons (overlapping features detected)")

                for i, (ms, me) in enumerate(merged_exons, 1):
                    exon = {
                        'seqid': template['seqid'],
                        'source': template['source'],
                        'type': 'exon',
                        'start': ms,
                        'end': me,
                        'score': '.',
                        'strand': template['strand'],
                        'phase': '.',
                        'attributes': {
                            'ID': f"merged_exon_{mrna['attributes']['ID']}.{i}",
                            'Parent': mrna['attributes']['ID']
                        },
                        'children': [],
                    }
                    mrna['children'].append(exon)
                if has_utr:
                    fixes.add("Created exons from CDS + UTR intervals")
                else:
                    fixes.add("Created exon features from CDS coordinates")
                has_exon = True

            elif has_exon and has_utr:
                # Exons exist but may not cover UTRs — extend first/last exon
                exons = [c for c in children if c['type'] == 'exon']
                utrs = [c for c in children if c['type'] in utr_types]

                exons_sorted = sorted(exons, key=lambda x: x['start'])

                for utr in utrs:
                    # Find the exon adjacent to this UTR and extend it
                    extended = False
                    for exon in exons_sorted:
                        # UTR immediately before exon (5'UTR on + strand, 3'UTR on - strand)
                        if utr['end'] + 1 == exon['start'] or (
                            utr['end'] >= exon['start'] and utr['start'] < exon['start']
                        ):
                            exon['start'] = min(exon['start'], utr['start'])
                            extended = True
                            break
                        # UTR immediately after exon (3'UTR on + strand, 5'UTR on - strand)
                        if exon['end'] + 1 == utr['start'] or (
                            utr['start'] <= exon['end'] and utr['end'] > exon['end']
                        ):
                            exon['end'] = max(exon['end'], utr['end'])
                            extended = True
                            break

                    if not extended:
                        # UTR not adjacent to any existing exon — create a new exon for it
                        new_exon = {
                            'seqid': utr['seqid'],
                            'source': utr['source'],
                            'type': 'exon',
                            'start': utr['start'],
                            'end': utr['end'],
                            'score': '.',
                            'strand': utr['strand'],
                            'phase': '.',
                            'attributes': {
                                'ID': f"exon_for_utr_{utr['attributes']['ID']}",
                                'Parent': mrna['attributes']['ID']
                            },
                            'children': [],
                        }
                        mrna['children'].append(new_exon)

                fixes.add("Extended exons to cover adjacent UTR regions")

            # Keep only transcript models that can be translated into at least one codon.
            # Drop entire mRNA (and all linked exon/CDS/UTR features) otherwise.
            cds_features = [c for c in mrna['children'] if c['type'] == 'CDS']
            if not cds_features:
                fixes.add("Removed mRNAs without CDS features")
                continue

            merged_cds = merge_intervals([(c['start'], c['end']) for c in cds_features])
            cds_len = sum((end - start + 1) for start, end in merged_cds)
            if cds_len < 3:
                fixes.add("Removed mRNAs with CDS length < 3 bp (non-translatable)")
                continue

            # Fix overlapping/inverted CDS by trimming overlaps and
            # removing CDS that become too short after trimming.
            n_fixed = _fix_inverted_cds(cds_features)
            if n_fixed:
                fixes.add("Fixed inverted/overlapping CDS feature(s) (trimmed coordinates)")
                # Remove CDS features that _fix_inverted_cds pruned from the list
                mrna['children'] = [
                    c for c in mrna['children']
                    if c['type'] != 'CDS' or c in cds_features
                ]
                # Re-check: if all CDS were removed the mRNA is non-translatable
                if not cds_features:
                    fixes.add("Removed mRNAs whose CDS collapsed after inversion fix")
                    continue

                # Rebuild exon features from the corrected CDS + remaining UTRs.
                # Drop old exons — they may correspond to the pre-fix CDS layout.
                mrna['children'] = [
                    c for c in mrna['children'] if c['type'] != 'exon'
                ]
                utr_types = ('five_prime_UTR', 'three_prime_UTR')
                utr_list = [c for c in mrna['children'] if c['type'] in utr_types]

                # One exon per CDS, extended to cover adjacent UTRs
                exon_intervals = [[c['start'], c['end']] for c in cds_features]
                for utr in utr_list:
                    extended = False
                    for exon_iv in exon_intervals:
                        if utr['end'] + 1 >= exon_iv[0] and utr['start'] < exon_iv[0]:
                            exon_iv[0] = min(exon_iv[0], utr['start'])
                            extended = True
                            break
                        if utr['start'] - 1 <= exon_iv[1] and utr['end'] > exon_iv[1]:
                            exon_iv[1] = max(exon_iv[1], utr['end'])
                            extended = True
                            break
                    if not extended:
                        exon_intervals.append([utr['start'], utr['end']])

                exon_intervals.sort(key=lambda x: x[0])
                merged_exons = [exon_intervals[0]]
                for s, e in exon_intervals[1:]:
                    if s <= merged_exons[-1][1]:
                        merged_exons[-1][1] = max(merged_exons[-1][1], e)
                    else:
                        merged_exons.append([s, e])

                # PASA requires exon count >= CDS count
                if len(merged_exons) < len(cds_features):
                    merged_exons = [[c['start'], c['end']] for c in cds_features]

                template = cds_features[0]
                mrna_id = mrna['attributes']['ID']
                for i, (es, ee) in enumerate(merged_exons, 1):
                    mrna['children'].append({
                        'seqid': template['seqid'],
                        'source': template['source'],
                        'type': 'exon',
                        'start': es,
                        'end': ee,
                        'score': '.',
                        'strand': template['strand'],
                        'phase': '.',
                        'attributes': {
                            'ID': f"fixed_exon_{mrna_id}.{i}",
                            'Parent': mrna_id,
                        },
                        'children': [],
                    })
                has_exon = True
                fixes.add("Rebuilt exon features after CDS inversion fix")

            # Reject mRNAs with biologically implausible intron lengths.
            # Prompt-mode hallucinations can produce megabase-spanning introns.
            exons_for_intron_check = sorted(
                [c for c in mrna['children'] if c['type'] == 'exon'],
                key=lambda x: x['start']
            )
            if len(exons_for_intron_check) >= 2:
                longest_intron = _check_intron_lengths(exons_for_intron_check)
                if longest_intron > MAX_INTRON_LENGTH:
                    fixes.add(f"Removed mRNAs with intron > {MAX_INTRON_LENGTH:,} bp")
                    continue

            if not has_exon:
                # CDS exist (checked above) but no exons — create one exon per CDS
                utr_types_final = ('five_prime_UTR', 'three_prime_UTR')
                cds_sorted = sorted(
                    [c for c in mrna['children'] if c['type'] == 'CDS'],
                    key=lambda c: c['start']
                )
                utr_remaining = [c for c in mrna['children'] if c['type'] in utr_types_final]
                template = cds_sorted[0]
                exon_ivs = [[c['start'], c['end']] for c in cds_sorted]
                for utr in utr_remaining:
                    attached = False
                    for iv in exon_ivs:
                        if utr['end'] + 1 >= iv[0] and utr['start'] < iv[0]:
                            iv[0] = min(iv[0], utr['start'])
                            attached = True
                            break
                        if utr['start'] - 1 <= iv[1] and utr['end'] > iv[1]:
                            iv[1] = max(iv[1], utr['end'])
                            attached = True
                            break
                    if not attached:
                        exon_ivs.append([utr['start'], utr['end']])
                exon_ivs.sort(key=lambda x: x[0])
                merged = [exon_ivs[0]]
                for s, e in exon_ivs[1:]:
                    if s <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([s, e])
                # PASA requires exon count >= CDS count
                if len(merged) < len(cds_sorted):
                    merged = [[c['start'], c['end']] for c in cds_sorted]
                mrna_id = mrna['attributes']['ID']
                for i, (es, ee) in enumerate(merged, 1):
                    mrna['children'].append({
                        'seqid': template['seqid'],
                        'source': template['source'],
                        'type': 'exon',
                        'start': es,
                        'end': ee,
                        'score': '.',
                        'strand': template['strand'],
                        'phase': '.',
                        'attributes': {
                            'ID': f"cds_exon_{mrna_id}.{i}",
                            'Parent': mrna_id,
                        },
                        'children': [],
                    })
                fixes.add("Created exon features for CDS-only mRNAs")

            valid_mrnas.append(mrna)

        if valid_mrnas:
            gene['children'] = valid_mrnas
            # Harmonize gene strand from valid child mRNAs.
            if gene['strand'] not in ('+', '-') and valid_mrnas:
                gene['strand'] = valid_mrnas[0]['strand']
                fixes.add("Set gene strand from child mRNA strand")
            filtered_genes[gene_id] = gene
        else:
            fixes.add("Removed genes with no valid mRNAs")

    genes = filtered_genes

    # 3. Sort genes and verify/fix coordinates
    sorted_genes = sorted(genes.values(), key=lambda x: (x['seqid'], x['start']))

    for gene in sorted_genes:
        min_start = gene['start']
        max_end = gene['end']

        for mrna in gene['children']:
            children = mrna['children']
            if children:
                # Calculate min/max in one pass
                m_min = min(c['start'] for c in children)
                m_max = max(c['end'] for c in children)

                if m_min < mrna['start'] or m_max > mrna['end']:
                    mrna['start'] = m_min
                    mrna['end'] = m_max
                    fixes.add("Expanded mRNA coordinates to cover children")
            else:
                m_min = mrna['start']
                m_max = mrna['end']

            min_start = min(min_start, m_min)
            max_end = max(max_end, m_max)

        if min_start < gene['start'] or max_end > gene['end']:
            gene['start'] = min_start
            gene['end'] = max_end
            fixes.add("Expanded gene coordinates to cover children")

    # 4. Uniform Renaming
    prefix = species_prefix

    for gene_counter, gene in enumerate(sorted_genes, 1):
        new_gene_id = f"{prefix}_g{gene_counter:06d}"
        gene['attributes']['ID'] = new_gene_id

        for mrna_counter, mrna in enumerate(gene['children'], 1):
            new_mrna_id = f"{new_gene_id}.t{mrna_counter}"
            mrna['attributes']['ID'] = new_mrna_id
            mrna['attributes']['Parent'] = new_gene_id

            # Sort children by start position
            mrna['children'].sort(key=lambda x: x['start'])

            # Counters for different types
            type_counters = defaultdict(int)

            for child in mrna['children']:
                ctype = child['type']
                type_counters[ctype] += 1
                child['attributes']['ID'] = f"{new_mrna_id}.{ctype}.{type_counters[ctype]}"
                child['attributes']['Parent'] = new_mrna_id

    fixes.add(f"Standardized all IDs to use species prefix '{prefix}'")
    fixes.add("Rebuilt Parent relationships")
    fixes.add("Sorted features by genomic coordinates")

    # 5. Write Output - use buffered writing
    lines = ["##gff-version 3\n", "# Fixes applied:\n"]

    sorted_fixes = sorted(fixes)
    for fix in sorted_fixes[:20]:
        lines.append(f"# - {fix}\n")
    if len(sorted_fixes) > 20:
        lines.append(f"# - ... and {len(sorted_fixes) - 20} more\n")

    # Build all output lines iteratively (avoid recursion)
    for gene in sorted_genes:
        lines.append(_feature_line(gene))
        for mrna in gene['children']:
            lines.append(_feature_line(mrna))
            for child in mrna['children']:
                lines.append(_feature_line(child))

    # Write all at once
    with open(output_file, 'w') as out:
        out.writelines(lines)


def _feature_line(feature):
    """Generate a GFF3 line for a feature."""
    attr_str = format_attributes(feature['attributes'])
    return (f"{feature['seqid']}\t{feature['source']}\t{feature['type']}\t"
            f"{feature['start']}\t{feature['end']}\t{feature['score']}\t"
            f"{feature['strand']}\t{feature['phase']}\t{attr_str}\n")


def main():
    # Parse command line arguments
    transgenic_only = "--transgenic-only" in sys.argv
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python standardize_gff.py [--transgenic-only]")
        print("  --transgenic-only  Only process transgenic result folders")
        sys.exit(0)

    root_dir = "/home/framazan/transgenic_comparison"
    results_root = os.path.join(root_dir, "results")
    genome_dir = os.path.join(root_dir, "genomes")
    output_dir = os.path.join(root_dir, "standardized_results")
    os.makedirs(output_dir, exist_ok=True)

    # Get all folders at once
    result_folders = [
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    ]

    for folder in result_folders:
        # Filter for transgenic-only if flag is set
        if transgenic_only and "transgenic" not in folder.lower():
            print(f"[SKIP] {folder} (not transgenic)")
            continue

        folder_path = os.path.join(results_root, folder)
        tool_name = folder.replace("_results", "")

        # Filter files
        files = [
            f for f in os.listdir(folder_path)
            if f.endswith((".gff", ".gff3", ".gtf"))
        ]

        for file_name in files:
            input_path = os.path.join(folder_path, file_name)
            species = infer_species(tool_name, file_name)

            clean_tool = RE_CLEAN_CHARS.sub("_", tool_name)
            clean_tool = RE_MULTI_UNDERSCORE.sub("_", clean_tool).strip("_")

            output_basename = f"{species}_{clean_tool}.gff3"
            output_path = os.path.join(output_dir, output_basename)

            standardize_gff(input_path, output_path, species, tool_name,
                            genome_dir=genome_dir)


if __name__ == "__main__":
    main()
