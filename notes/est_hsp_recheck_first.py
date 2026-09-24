"""Read-only independent recheck of fetched job 6165460 (no engine execution)."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision/scripts/evidence'))
import est_hsp_expanded as diagnostic


def fasta(path):
    records = {}
    for block in path.read_text().split('>')[1:]:
        lines = block.splitlines()
        records[lines[0].split()[0]] = ''.join(lines[1:])
    return records


def main(root):
    root = Path(root)
    queries = fasta(root / 'queries.fa')
    provenance = json.loads((root / 'provenance.json').read_text())
    diagnostic.require(provenance['diamond_sha256'] == diagnostic.ENGINE_SHA256, 'recorded engine pin')
    results = []
    for source in ('orthodb', 'swissprot'):
        raw = root / source / (source + '.raw17.tsv')
        protein_path = root / source / (source + '.filtered.fa')
        proteins = fasta(protein_path)
        pins = provenance['sources'][source]
        for path, pin in [(raw, 'raw_sha256'), (root / 'queries.fa', 'query_sha256'),
                          (protein_path, 'filtered_fasta_sha256')]:
            diagnostic.require(diagnostic.digest(path) == pins[pin], str(path) + ' pin')
        for line in raw.read_text().splitlines():
            row = line.split('\t')
            result = diagnostic.interpret(row, queries[row[0]], proteins[row[2]])
            if result['counts']['F']:
                try:
                    diagnostic.interpret(row, queries[row[0]], proteins[row[2]], shift_sign=-1)
                except ValueError:
                    result['wrong_sign_rejected'] = True
                else:
                    raise ValueError('wrong-sign negative control survived')
            results.append(dict(source=source, query=row[0], counts=result['counts'],
                                metrics=result['metrics'], blocks=result['blocks'],
                                wrong_sign_rejected=result.get('wrong_sign_rejected')))
    print(json.dumps(dict(recorded_engine_sha256=diagnostic.ENGINE_SHA256,
                         live_binary_rehashed=False, certified=False, rows=results), indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
