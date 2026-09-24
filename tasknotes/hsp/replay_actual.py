"""Read-only preserved actual-row replay; run from worktree root with PYTHONPATH=."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from revision.scripts.evidence.est_hsp_contract import adapt_raw13, classify_rich17, SCOPE

ROOT = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/est_execution_20260916_1530/hsp_expanded_result')


def fasta(path):
    result = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]
            assert name and name not in result
            result[name] = ''
        else:
            assert name is not None
            result[name] += line
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = {name: sha(ROOT/name) for name in ('expanded.raw17.tsv', 'queries.fa', 'proteins.fa', 'diagnostic.json', 'manifest.json')}
queries, proteins = fasta(ROOT/'queries.fa'), fasta(ROOT/'proteins.fa')
counts, dispositions, output = Counter(), Counter(), []
for number, line in enumerate((ROOT/'expanded.raw17.tsv').read_text().splitlines(), 1):
    row = line.split('\t')
    dna, protein = queries[row[0]], proteins[row[2]]
    rich = adapt_raw13(row[:13], dna, protein)
    assert rich == row, (number, 'raw13 reconstruction differs')
    result = classify_rich17(row, dna, protein)
    counts.update(result['counts'])
    dispositions[result['diagnostic_disposition']] += 1
    output.append(dict(row=number, qseqid=row[0], sseqid=row[2],
                       identities=result['identities'],
                       raw13_sha256=hashlib.sha256('\t'.join(row[:13]).encode()).hexdigest(),
                       disposition=result['diagnostic_disposition']))
assert before == {name: sha(ROOT/name) for name in before}
print(json.dumps(dict(scope=SCOPE, fixture_root=str(ROOT), input_sha256=before,
                     rows=len(output), exact_raw13_to_rich17=len(output), counts=counts,
                     dispositions=dispositions, originals_unchanged=True,
                     row_results=output), indent=2, sort_keys=True))
