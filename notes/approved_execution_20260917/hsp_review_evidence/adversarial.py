"""Independent bounded checks. Run with worktree on PYTHONPATH; no writes."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from revision.scripts.evidence.est_hsp_contract import adapt_raw13, classify_rich17

checks = 0
def check(test):
    global checks
    assert test
    checks += 1

def rejects(row, dna, protein):
    global checks
    try:
        adapt_raw13(row, dna, protein)
    except ValueError:
        checks += 1
    else:
        raise AssertionError(('unexpected acceptance', row))

# Hand-specified mixed trace: 29M,1I,1D,2F, nonzero query and subject starts.
protein = 'RR' + 'K'*30 + 'R'
for reverse in (False, True):
    dna = ('T' if reverse else 'A')*95
    raw = ['query','95','subject','33', '93' if reverse else '3',
           '4' if reverse else '92','3','32','31','1e-5','100',
           '-3' if reverse else '3', '2-KK-\\-2/-25']
    rich = adapt_raw13(raw, dna, protein)
    result = classify_rich17(rich, dna, protein)
    check(rich[:13] == raw)
    check(result['counts'] == dict(M=29,I=1,D=1,F=2))
    check(result['accepted_by_hsp_rule'])
    positions = [2,5,8,12,15] + list(range(17,92,3))
    subjects = [2,3,None,5,6] + list(range(7,32))
    check([e['q0'] for e in result['codon_events']] == [94-p if reverse else p for p in positions])
    check([e['subject_aa_index'] for e in result['codon_events']] == subjects)
    check([e['q0'] for e in result['frameshifts']] == ([83,77] if reverse else [11,17]))
    check([e['effect'] for e in result['frameshifts']] == ['omitted','reused'])
    check(all(e['direction'] == (-1 if reverse else 1) for e in result['codon_events']))
    check(classify_rich17(rich,dna,protein,expected_identities=result['identities']) == result)
    for identity in ({}, {'dna_sha256':'x'}, {'unexpected':'x'}):
        try:
            classify_rich17(rich,dna,protein,expected_identities=identity)
        except ValueError:
            checks += 1
        else:
            raise AssertionError('bad identity accepted')
    for index, value in [(4,'96'),(5,'96'),(6,'-1'),(7,'34'),(7,'31'),
                         (8,'33'),(9,'-1'),(9,'NaN'),(9,'-inf'),(9,'1.0001e-5'),
                         (10,'nan'),(10,'-1'),(11,'2'),(12,'2-KK-\\K2/-25'),
                         (12,'2-KK-\\-2/-26'),(12,'2-KK-\\-2/-0'),
                         (12,'2-KK-\\-2/-025'),(12,'2-KK-\\-2/-25X')]:
        mutated = raw.copy()
        mutated[index] = value
        rejects(mutated,dna,protein)
    rejects(raw,dna,'RK'+protein[2:3]+'R'+protein[4:])

root = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/est_execution_20260916_1530/hsp_expanded_result')
report = json.loads((root/'diagnostic.json').read_text())
statuses = Counter(case['status'] for case in report['cases'])
check(statuses['NOT_ELICITED'] == 12)
print(json.dumps(dict(assertions_passed=checks, historical_case_statuses=statuses,
    historical_acceptance_policy=report['acceptance_policy']),indent=2))
