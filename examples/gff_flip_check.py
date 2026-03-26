import sys
from collections import defaultdict

def parse_gff_attributes(attr_str):
    attrs = {}
    for part in attr_str.strip().split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            attrs[k] = v
    return attrs

def main(gff_path):
    # Store parent features: {ID: (start, end, line, feature, line_num)}
    parents = {}
    # Store features with end < start
    invalid_features = []
    # Store all features for parent lookup
    features = defaultdict(list)
    # Store all lines for output
    all_lines = []

    with open(gff_path) as f:
        for i, line in enumerate(f, 1):
            all_lines.append((i, line.rstrip('\n')))
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            seqid, source, feature, start, end, score, strand, phase, attr = parts
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                continue
            attrs = parse_gff_attributes(attr)
            ID = attrs.get('ID')
            Parent = attrs.get('Parent')
            # Save parent features (gene, mRNA)
            if feature in ('gene', 'mRNA', 'transcript') and ID:
                parents[ID] = (start, end, line.strip(), feature, i)
            # Save all features for lookup
            if Parent:
                features[Parent].append((feature, start, end, line.strip(), i, ID))
            # Check for invalid coordinates
            if end < start:
                invalid_features.append({
                    'line_num': i,
                    'feature': feature,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'parent': Parent,
                    'ID': ID,
                    'line': line.strip(),
                })

    print(f"Found {len(invalid_features)} features with end < start.\n")
    flipped = 0
    outside_parent = 0
    no_parent = 0
    # Track parents to remove
    parents_to_remove = set()
    # Track features to remove
    features_to_remove = set()
    for feat in invalid_features:
        parent_id = feat['parent']
        swapped_start, swapped_end = feat['end'], feat['start']
        parent_ok = False
        if parent_id and parent_id in parents:
            p_start, p_end, _, _, _ = parents[parent_id]
            # Check if swapped interval fits within parent
            if p_start <= swapped_start <= swapped_end <= p_end:
                parent_ok = True
                flipped += 1
            else:
                outside_parent += 1
        else:
            no_parent += 1
        print(f"Line {feat['line_num']}: {feat['line']}")
        if parent_id and parent_id in parents:
            print(f"  Parent {parent_id}: {parents[parent_id][2]}")
            print(f"  Swapped interval: {swapped_start}-{swapped_end} (within parent: {parent_ok})")
        else:
            print(f"  No parent found for {parent_id}")
        print()
        # Mark parent and all its features for removal
        if parent_id:
            parents_to_remove.add(parent_id)
            # Remove grandparent if this is mRNA and parent is gene
            if parent_id in parents and parents[parent_id][3] == 'mRNA':
                # Find grandparent
                parent_attrs = parse_gff_attributes(parents[parent_id][2].split('\t')[-1])
                grandparent_id = parent_attrs.get('Parent')
                if grandparent_id:
                    parents_to_remove.add(grandparent_id)
        if feat['ID']:
            features_to_remove.add(feat['ID'])

    print(f"Summary:")
    print(f"  Flipped and fits parent: {flipped}")
    print(f"  Flipped but outside parent: {outside_parent}")
    print(f"  No parent found: {no_parent}")
    print(f"  Total invalid: {len(invalid_features)}")

    # Write cleaned file
    cleaned_path = gff_path.rsplit('.', 1)[0] + '.cleaned.gff'
    removed_lines = set()
    # Remove all lines for parents and their features
    for parent_id in parents_to_remove:
        # Remove parent line
        if parent_id in parents:
            removed_lines.add(parents[parent_id][4])
        # Remove all features with this parent
        for feat in features.get(parent_id, []):
            removed_lines.add(feat[4])
            # If feature is mRNA, remove its children too
            if feat[0] == 'mRNA' and feat[5]:
                for subfeat in features.get(feat[5], []):
                    removed_lines.add(subfeat[4])

    with open(cleaned_path, 'w') as out:
        for i, line in all_lines:
            if i not in removed_lines:
                out.write(line + '\n')
    print(f"\nCleaned file written to: {cleaned_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <gff_file>")
        sys.exit(1)
    main(sys.argv[1])
