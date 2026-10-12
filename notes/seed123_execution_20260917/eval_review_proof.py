"""Bounded readonly review probes; only temporary fixture outputs are written."""
import copy
import json
import sys
import tempfile
from pathlib import Path
ROOT = Path('/tmp/mvp-seed123-integration-20260917')
sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'revision/scripts/mvp'), str(ROOT/'src')]
import test_mvp_seed123_eval as t
import mvp_score as s
import mvp_produce_counts as p

with tempfile.TemporaryDirectory(prefix='eval-independent-') as tmp:
    payload, panel = t.score_payload(Path(tmp))
    for where in ('top', 'metadata'):
        bad = copy.deepcopy(payload)
        target = bad if where == 'top' else bad.setdefault('metadata', {})
        target['run_id'] = 'b5_seed123'
        result = s.score(bad, panel)
        print('ACCEPTED_CONFLICT', where, 'input=b5_seed123', 'output='+result['run_id'])
        assert result['run_id'] == 'mvp_seed123'
    for profile in (True, 123, [], {}, None):
        try:
            p.evaluation_profile({'profile': profile})
        except ValueError:
            print('REJECT_PROFILE_TYPE', repr(profile))
        else:
            raise AssertionError(profile)
    path = Path(payload['pairs']['mvp_seed123'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(loss_off=10., ce_numerator_off=10., composite_loss_off=20.)
    rows[1].update(loss_off=2., ce_numerator_off=200., ce_numerator_on=100., ce_denominator_off=100., ce_denominator_on=100., composite_loss_off=4.)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = s.score(payload, panel)
    assert result['primary']['pooled_point'] == 5.
    assert result['composite_delta_diagnostic'] == 10.
    print('EQUAL_TILE_PRIMARY', result['primary']['pooled_point'], 'COMPOSITE', result['composite_delta_diagnostic'])
    counts = lambda tp, n: dict(f1_tp=tp, f1_predicted=n, f1_reference=n)
    unequal = [dict(seeds={'123': {'off':counts(0,1),'on':counts(1,1)}}),
               dict(seeds={'123': {'off':counts(0,99),'on':counts(0,99)}})]
    print('F1_COUNT_POOLED', s.metric(unequal,'on','F1',(123,)), 'PER_LOCUS_MEAN_WOULD_BE',50.)
    assert s.metric(unequal,'on','F1',(123,)) == 1.
