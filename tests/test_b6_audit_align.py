"""A46 rule 4(b) audit alignment driver: dry-run command derivation and fail-closed gates."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from revision.scripts.evidence import b6_audit_runs as runs

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'revision/scripts/evidence'
SBATCH = EVIDENCE / 'b6_audit_align.sbatch'
# Verbatim CMD line of longread_align_pacbio.sbatch (not yet tracked in this branch).
PACBIO_CMD = 'CMD="minimap2 -t $THREADS -ax splice:hq -G 200000 --secondary=no --cs=long $REF $FA"'


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def driver_cmd_line(name, fallback=None):
    path = EVIDENCE / name
    if not path.exists():
        return fallback
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip().startswith('CMD="minimap2')]
    assert len(lines) == 1, name
    return lines[0]


def primary_command(line, **env):
    """Expand the primary driver's own CMD assignment with bash."""
    script = ''.join(f'{k}={v!r}; ' for k, v in env.items()) + line + '; printf %s "$CMD"'
    return subprocess.run(['bash', '-c', script], check=True, capture_output=True, text=True).stdout


def write(path, text, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with gzip.open(path, 'wt') as f:
            f.write(text)
    else:
        path.write_text(text)
    return path


def roles_row(first, second):
    return '\t'.join([first, second] + ['x'] * 9 + ['c2_training_eligible', 'product']) + '\n'


@pytest.fixture
def root(tmp_path):
    r = tmp_path / 'T'
    manifest = ['species_id\tspecies\tv\tfasta\tfasta_md5\tgff\tgff_md5\tnote\n']
    est_roles, lr_roles = [], []
    for sp in runs.SPECIES:
        ref = write(r / f'genomes/{sp}.fa', f'>Chr1\n{"ACGT" * 30}\n')
        gff = write(r / f'genomes/{sp}.gff3', '##gff-version 3\n')
        manifest.append(f'{sp}\t{sp}\tv1\t{ref}\t{md5(ref)}\t{gff}\t{md5(gff)}\t.\n')
        est = write(r / f'evidence/est/{sp}/univec/est.univec.fa.gz', f'>e1\n{"A" * 120}\n', gz=True)
        write(r / f'evidence/est/{sp}/univec/DONE', 'ok\n')
        est_roles.append(roles_row(f'est/{sp}', 'x'))
        write(r / f'evidence/est_align/{sp}/DONE', f'est_md5={md5(est)}\narm=primary\nmin_len=100\n')
        cmd = primary_command(driver_cmd_line('est_align.sbatch'), REF=str(ref), EST=str(est))
        write(r / f'evidence/est_align/{sp}/PROVENANCE.txt', f'species        {sp}\ncommand        {cmd}\n')
    write(r / 'transgenic/data/manifests/b5_species_v1.tsv', ''.join(manifest))
    write(r / 'evidence/DATASET_ROLES.est_v1.tsv', ''.join(est_roles))
    ont_list, pb_list = [], []
    for run_id, status in (('SRR1', 'PASS'), ('SRR2', 'FAIL')):
        sp, ds = 'Athaliana', 'dsA'
        reads = write(r / f'evidence/training/ont/{sp}/{ds}/{run_id}.fa.gz', '>r\nACGT\n', gz=True)
        ont_list.append(f'{sp}\ttraining/ont/{sp}/{ds}\t{run_id}\tevidence/training/ont/{sp}/{ds}/{run_id}.fa.gz\n')
        lr_roles.append(roles_row(f'training/ont/{sp}/{ds}', run_id))
        ref, gff = r / f'genomes/{sp}.fa', r / f'genomes/{sp}.gff3'
        audit = r / f'evidence/orientation_audit_v1/{sp}/{ds}/{run_id}'
        body = json.dumps(dict(status=status, use_uf=status == 'PASS', reads_md5=md5(reads),
                               reference_md5=md5(ref), gff_md5=md5(gff), protocol='B1 A39'))
        write(audit / 'audit.json', body)
        write(audit / 'DONE', f'key={md5(reads)}:{md5(ref)}:{md5(gff)}:scorer\nrun={run_id}\n'
                              f'status={status}\naudit_md5={md5(audit / "audit.json")}\n')
        uf = '-uf' if status == 'PASS' else ''
        primary = r / f'evidence/longread_align_v1/{sp}/{ds}/{run_id}'
        write(primary / 'DONE', f'key={md5(reads)}:{md5(ref)}:{md5(audit / "audit.json")}\n'
                                f'run={run_id}\nuf={uf or "none"}\n')
        cmd = primary_command(driver_cmd_line('longread_align.sbatch'), THREADS='32', UF=uf,
                              REF=str(ref), FA=str(reads))
        write(primary / 'PROVENANCE.txt', f'run            {run_id}\ncommand        {cmd}\n')
    sp, ds, run_id = 'Ptrichocarpa', 'dsP', 'SRR3'
    reads = write(r / f'evidence/training/pacbio/{sp}/{ds}/{run_id}.fa.gz', '>r\nACGT\n', gz=True)
    pb_list.append(f'{sp}\ttraining/pacbio/{sp}/{ds}\t{run_id}\tevidence/training/pacbio/{sp}/{ds}/{run_id}.fa.gz\n')
    lr_roles.append(roles_row(f'training/pacbio/{sp}/{ds}', run_id))
    ref = r / f'genomes/{sp}.fa'
    primary = r / f'evidence/longread_align_v1/{sp}/{ds}/{run_id}'
    write(primary / 'DONE', f'key={md5(reads)}:{md5(ref)}:splice_hq\nrun={run_id}\npreset=splice:hq\n')
    cmd = primary_command(driver_cmd_line('longread_align_pacbio.sbatch', PACBIO_CMD), THREADS='32',
                          REF=str(ref), FA=str(reads))
    write(primary / 'PROVENANCE.txt', f'command        {cmd}\n')
    write(r / 'evidence/longread_runs_v1.tsv', ''.join(ont_list))
    write(r / 'evidence/longread_runs_pacbio_v1.tsv', ''.join(pb_list))
    write(r / 'evidence/DATASET_ROLES.longread_v1.tsv', ''.join(lr_roles))
    return r


def build_runlist(root):
    out = root / 'evidence/audit_align_v1/runs.tsv'
    out.parent.mkdir(parents=True)
    assert runs.main(['--root', str(root), '--out', str(out), '--expect', '9,2,1']) == 0
    return out


def dry(root, runlist, task):
    env = dict(B6_ROOT=str(root), B6_RUNLIST=str(runlist), B6_DRY_RUN='1', B6_PY=sys.executable,
               SLURM_ARRAY_TASK_ID=str(task), PATH='/usr/bin:/bin')
    p = subprocess.run(['bash', str(SBATCH)], env=env, capture_output=True, text=True)
    out = dict(line.split('\t', 1) for line in p.stdout.splitlines() if '\t' in line)
    return p.returncode, out, p.stderr


def test_run_list_covers_sources_in_order_with_verdicts(root):
    rows = [l.split('\t') for l in build_runlist(root).read_text().splitlines()]
    assert [r[1] for r in rows] == ['est'] * 9 + ['ont', 'ont', 'pacbio']
    assert [r[0] for r in rows] == [str(i) for i in range(12)]
    assert [r[6] for r in rows[9:]] == ['-uf', 'none', 'none']
    assert rows[0][2:5] == ['Athaliana', 'est', 'est_primary']


def test_audit_command_differs_from_primary_only_in_secondary_flags(root):
    runlist = build_runlist(root)
    seen = []
    for task in range(12):
        code, out, err = dry(root, runlist, task)
        assert code == 0, err
        primary, audit = out['PRIMARY'].split(), out['AUDIT'].split()
        i = primary.index('--secondary=no')
        assert audit == primary[:i] + ['--secondary=yes', '-N', '50', '-p', '0.5'] + primary[i + 1:]
        assert '--secondary=no' not in audit
        key = out['KEY'].split(':')
        assert key[2] == hashlib.md5(out['AUDIT'].encode()).hexdigest()
        seen.append((out['TASK'].split('\t')[1], audit))
    assert all('splice:hq' in a and '-uf' not in a for s, a in seen if s in ('est', 'pacbio'))
    assert ['-uf' in a for s, a in seen if s == 'ont'] == [True, False]
    assert not (root / 'evidence/audit_align_v1/Athaliana').exists()   # dry run writes nothing


def test_frozen_est_command_is_the_protocol_literal(root):
    code, out, _ = dry(root, build_runlist(root), 0)
    assert code == 0
    assert re.fullmatch(r'minimap2 -t 32 -ax splice:hq -G 200000 --secondary=yes -N 50 -p 0\.5 '
                        r'--cs=long \S+\.fa \S+/est\.univec\.fa\.gz', out['AUDIT'])


def test_refuses_run_list_verdict_that_disagrees_with_the_audit(root):
    runlist = build_runlist(root)
    rows = runlist.read_text().splitlines()
    rows[9] = rows[9].replace('\t-uf\t', '\tnone\t')
    runlist.write_text('\n'.join(rows) + '\n')
    code, _, err = dry(root, runlist, 9)
    assert code != 0 and 'orientation verdict' in err


def test_refuses_when_recorded_primary_command_drifted(root):
    runlist = build_runlist(root)
    prov = root / 'evidence/longread_align_v1/Athaliana/dsA/SRR2/PROVENANCE.txt'
    prov.write_text(prov.read_text().replace('-k14', '-k15'))
    code, _, err = dry(root, runlist, 10)
    assert code != 0 and 'differs from the recorded' in err


def test_refuses_changed_reads_and_wrong_role(root):
    runlist = build_runlist(root)
    write(root / 'evidence/est/Gmax/univec/est.univec.fa.gz', f'>e1\n{"C" * 120}\n', gz=True)
    code, _, err = dry(root, runlist, runs.SPECIES.index('Gmax'))
    assert code != 0 and 'different screened FASTA' in err
    roles = root / 'evidence/DATASET_ROLES.longread_v1.tsv'
    roles.write_text(roles.read_text().replace('SRR3\t' + 'x\t' * 9 + 'c2_training_eligible',
                                               'SRR3\t' + 'x\t' * 9 + 'excluded'))
    code, _, err = dry(root, runlist, 11)
    assert code != 0 and 'c2_training_eligible rows' in err


def test_run_list_builder_refuses_primary_uf_mismatch(root, tmp_path):
    done = root / 'evidence/longread_align_v1/Athaliana/dsA/SRR1/DONE'
    done.write_text(done.read_text().replace('uf=-uf', 'uf=none'))
    with pytest.raises(ValueError, match='primary alignment used uf'):
        runs.build(root, dict(est=9, ont=2, pacbio=1))


def test_list_prints_every_task_output_path_and_matches_the_driver(root, capsys):
    assert runs.main(['--root', str(root), '--list', '--expect', '9,2,1']) == 0
    lines = [l.split('\t') for l in capsys.readouterr().out.splitlines()]
    assert [l[0] for l in lines] == [str(i) for i in range(12)]
    assert lines[0][3] == 'evidence/audit_align_v1/Athaliana/est/est_primary/est_primary.audit.bam'
    assert all(l[3] == f'evidence/audit_align_v1/{sp}/est/est_primary/est_primary.audit.bam'
               for l, sp in zip(lines[:9], runs.SPECIES))
    assert lines[9][3] == 'evidence/audit_align_v1/Athaliana/dsA/SRR1/SRR1.audit.bam'
    assert not (root / 'evidence/audit_align_v1').exists()          # --list writes nothing
    runlist = build_runlist(root)
    for task in (0, 9, 11):
        code, out, err = dry(root, runlist, task)
        assert code == 0, err
        assert out['BAM'] == str(root / lines[task][3])
