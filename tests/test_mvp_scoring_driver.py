import hashlib
import json
from pathlib import Path

import pytest

from test_mvp_scoring import script, ROOT


def setup_inputs(tmp_path):
    m = script('mvp_align_panel_longreads')
    root = tmp_path / 'root'
    (root / 'evidence').mkdir(parents=True)
    reads = root / 'actual.fa.gz'
    reads.write_bytes(b'actual read file bytes')
    cols = (ROOT / 'data/manifests/DATASET_ROLES.longread_v1.tsv').read_text().splitlines()[0]
    values = ['ont/Athaliana_FLIC_PRJNA1087576', 'SRR28341943', 'Athaliana', 'reference', 'PromethION', 'RNA-Seq', '1', '3', '', '', hashlib.md5(reads.read_bytes()).hexdigest(), 'b1_validation_only', 'fixture', 'fixture']
    roles = root / 'evidence/DATASET_ROLES.longread_v1.tsv'
    roles.write_text(cols + '\n' + '\t'.join(values) + '\n')
    return m, root, reads, roles


def test_actual_read_role_and_md5_gate(tmp_path):
    m, root, reads, roles = setup_inputs(tmp_path)
    row = m.validate_read(root, 'SRR28341943', reads)
    assert row['role'] == 'b1_validation_only'
    reads.write_bytes(b'changed')
    with pytest.raises(ValueError, match='md5'):
        m.validate_read(root, 'SRR28341943', reads)


@pytest.mark.parametrize('mutation', ['training', 'duplicate', 'header'])
def test_no_training_bypass_or_ambiguous_manifest(tmp_path, mutation):
    m, root, reads, roles = setup_inputs(tmp_path)
    text = roles.read_text()
    if mutation == 'training':
        text = text.replace('b1_validation_only', 'c2_training_eligible')
    elif mutation == 'duplicate':
        text += text.splitlines()[1] + '\n'
    else:
        text = text.replace('local_fa_md5', 'wrong')
    roles.write_text(text)
    with pytest.raises(ValueError):
        m.validate_read(root, 'SRR28341943', reads)


def test_output_namespace_cannot_redirect_to_legacy(tmp_path):
    m, root, reads, roles = setup_inputs(tmp_path)
    legacy = root / 'evidence/orientation_audit_v1'; legacy.mkdir()
    (root / 'evidence/mvp_scoring_20260916').symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match='namespace'):
        m.output_directory(root, 'SRR28341943')
    assert list(legacy.iterdir()) == []


def test_pass1_uses_mapq5_and_shared_window_not_gene(tmp_path):
    m = script('mvp_align_panel_longreads')
    from test_mvp_scoring import locus
    paf = ['keep\t100\t0\t100\t+\tChr1\t100000\t55000\t55100\t100\t100\t5\n',
           'drop\t100\t0\t100\t+\tChr1\t100000\t55000\t55100\t100\t100\t4\n',
           'edge\t100\t0\t100\t+\tChr1\t100000\t55860\t55960\t100\t100\t60\n']
    assert m.select_names(paf, [locus()]) == ['keep']
    assert m.pass2_command('ONT', False, 'ref', 'reads', 32)[1:] == ['-t', '32', '-ax', 'splice', '-k14', '-G', '200000', '--secondary=no', '--cs=long', 'ref', 'reads']
    assert '-uf' in m.pass2_command('ONT', True, 'ref', 'reads', 32)
    assert 'splice:hq' in m.pass2_command('HiFi', False, 'ref', 'reads', 32)


def test_real_orientation_scorer_nondefault_root_done_key_and_stale_rejection(tmp_path):
    m = script('mvp_align_panel_longreads')
    out = tmp_path / 'new_namespace'
    out.mkdir()
    sam = out / 'sample.sam'
    sam.write_text('@HD\tVN:1.6\n')
    gff = tmp_path / 'ref.gff'
    gff.write_text('##gff-version 3\n')
    reads = tmp_path / 'reads'; reads.write_text('reads')
    ref = tmp_path / 'ref'; ref.write_text('ref')
    legacy = tmp_path / 'orientation_audit_v1'; legacy.mkdir()
    marker = legacy / 'DONE'; marker.write_text('untouched')
    key = m.audit_key(reads, ref, gff)
    result = m.audit_from_sam(sam, gff, out, 'SRR28341943', 'dataset', key)
    assert result['status'] == 'UNRESOLVED'
    assert result['use_uf'] is False  # spec explicitly permits no -uf here
    assert m.read_audit(out, key, 'SRR28341943')['use_uf'] is False
    assert (out / 'DONE').read_text().splitlines()[0] == 'key=' + key
    assert marker.read_text() == 'untouched'
    with pytest.raises(ValueError):
        m.read_audit(out, 'stale', 'SRR28341943')
    with pytest.raises(ValueError):
        m.read_audit(out, key, 'wrongrun')
    result['use_uf'] = True
    (out / 'audit.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):
        m.read_audit(out, key, 'SRR28341943')
