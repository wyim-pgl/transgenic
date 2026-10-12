import base64, hashlib, importlib.util, json, sys
from pathlib import Path
import pytest
p=Path(__file__).parents[1]/'est_sample_review_evidence/test_adversarial.py'
s=importlib.util.spec_from_file_location('original_review',p)
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

@pytest.mark.parametrize('field',['c1_provenance','c1_commands'])
def test_original_identity_bad_outcome_rejected(tmp_path,field):
    m=a.f.fixture(tmp_path);a.add_provenance(m,tmp_path,'engine-A')
    ar,rows=a.execute(m,tmp_path/'a')
    key=m['searches'][0][field]
    a.f.repin(m,key,json.dumps({'engine':'different','raw_sha256':m['files']['orthodb_raw']['sha256']}))
    br,other=a.execute(m,tmp_path/'b')
    assert ar['status']==br['status']=='COMPLETE'
    for kind in ('search','hsp'):
        x=[r['id'] for r in rows if r['type']==kind];y=[r['id'] for r in other if r['type']==kind]
        assert x[0]!=y[0] and x[-1]==y[-1]
    entry=next(r for r in other if r['type']=='search')
    assert entry['preimage'][field+'_sha256']==br['sources'][key]['sha256']
    assert br['sources'][key]['fresh_whole_file_verified'] and br['sources'][key]['expected_hash_validated']

def test_original_stderr_bad_outcome_rejected_after_decoding(tmp_path):
    m=a.f.fixture(tmp_path)
    r,_=a.execute(m,tmp_path/'out','import sys;sys.stderr.write("UNIQUE_BAM_" + "CORRUPTION_DIAGNOSTIC\\n");sys.exit(7)')
    payload=b'UNIQUE_BAM_CORRUPTION_DIAGNOSTIC\n';e=r['sam_stderr']
    assert r['status']=='FAILED' and r['sam_returncode']==7
    assert base64.b64decode(e['bytes_base64'])==payload
    assert e['total_bytes']==e['captured_bytes']==len(payload)
    assert e['sha256']==hashlib.sha256(payload).hexdigest() and e['eof'] and not e['truncated']

def test_stdout_eof_without_newline_and_empty_stderr(tmp_path):
    m=a.f.fixture(tmp_path)
    r,_=a.execute(m,tmp_path/'out','import sys;sys.stdout.buffer.write(open(sys.argv[1],"rb").read().rstrip(b"\\n"))')
    assert r['status']=='COMPLETE' and r['selected_bam_records_complete'] and r['sam_returncode']==0
    assert r['sam_stderr']['eof'] and r['sam_stderr']['total_bytes']==0

def test_closed_pipes_nonzero_exit_still_fails(tmp_path):
    m=a.f.fixture(tmp_path)
    r,_=a.execute(m,tmp_path/'out','import os,sys,time;sys.stdout.buffer.write(open(sys.argv[1],"rb").read());sys.stdout.flush();os.close(1);os.close(2);time.sleep(.05);sys.exit(9)')
    assert r['status']=='FAILED' and r['sam_returncode']==9
    assert not r['selected_bam_records_complete'] and r['sam_stderr']['eof']
