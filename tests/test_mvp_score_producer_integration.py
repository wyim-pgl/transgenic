"""Full350 transport/receipt integration, synthetic authority/candidate/tool fixtures."""
import json
from pathlib import Path

import pytest
from test_mvp_score_producer import p, w, ROOT, dump, finalized_fixture, fake_gffcompare, prediction_rows


def test_full_producer_actual_finalizer_frozen_panel_synthetic_biology(tmp_path, monkeypatch):
    from transgenic.training import mvp_cli, mvp_cli_runtime
    panel_path = ROOT/'revision/protocols/b6_panels_v1.tsv'
    panel = w.load_panel(panel_path)
    # No biological reference authority or candidate acceptance is asserted here.
    candidate = {}
    for name in ('acceptance','inventory','receipts','corpus','evidence'):
        path = tmp_path/(name+'.json'); path.write_text('{}')
        candidate[name] = str(path)
    candidate['acceptance_sha256'] = w.sha256_file(candidate['acceptance'])
    hashes = {k:w.sha256_file(candidate[k]) for k in ('corpus','evidence')}
    hashes['panel'] = w.sha256_file(panel_path)
    monkeypatch.setattr(mvp_cli,'validate_candidate_records',lambda *a: object())
    monkeypatch.setattr(mvp_cli_runtime,'verify_candidate_artifacts',lambda *a: {'valid_rns':[1],'sha256':hashes})
    reference = tmp_path/'reference.gff'
    lines = ['##gff-version 3']
    for l in panel:
        for kind, attrs, start, end in [('gene',f'ID={l.gene_id}',l.start1,l.fin1),
                                      ('mRNA',f'ID=t{l.gene_id};Parent={l.gene_id}',l.start1,l.fin1),
                                      ('exon',f'Parent=t{l.gene_id}',l.start1,l.start1+9),
                                      ('exon',f'Parent=t{l.gene_id}',l.fin1-9,l.fin1)]:
            lines.append(f'{l.chromosome}\ttest\t{kind}\t{start}\t{end}\t.\t{l.strand}\t.\t{attrs}')
    reference.write_text('\n'.join(lines)+'\n')
    fasta = tmp_path/'ref.fa'; fasta.write_text('>synthetic\nA\n')
    monkeypatch.setattr(p,'frozen_reference',lambda a,b: {'test_fixture_only':True,'files':{'fasta':p.file_identity(a),'gff':p.file_identity(b)}})
    monkeypatch.setattr(p,'fasta_lengths',lambda _: {l.chromosome:100000000 for l in panel})
    rows, _ = prediction_rows(tmp_path,panel)
    runs, checkpoints = {}, {}
    for seed in (1,2,3):
        run = tmp_path/f'seed{seed}'; run.mkdir()
        # Three runs consume ONE serialized baseline. Re-saving equal tensors
        # can reorder safetensors alias metadata and change its byte identity.
        shared_source = tmp_path/'seed1/source/model.safetensors' if seed > 1 else None
        final, _, code = finalized_fixture(run,seed,hashes,source_weights=shared_source)
        runs[str(seed)] = {'final':str(final),'code_manifest':str(code)}
        checkpoints[seed] = w.sha256_file(final/'model.safetensors')
    checkpoints[123] = w.sha256_file(tmp_path/'seed1/source/model.safetensors')
    for row in rows: row['checkpoint_sha256'] = checkpoints[row['seed']]
    executable = fake_gffcompare(tmp_path)
    monkeypatch.setattr(p.shutil,'which',lambda _: executable)
    payload = {'panels':str(panel_path),'fasta':str(fasta),'reference':str(reference),
               'candidate':candidate,'runs':runs,'predictions':rows,
               'decoder':{'files':{str(Path(__file__)):w.sha256_file(__file__)},'commands':[['synthetic-test-only']]}}
    input_path = dump(tmp_path/'evaluation.json',payload)
    out = tmp_path/'result'
    result = p.produce(input_path,out)
    assert result['three_seed_identity_verified'] is True
    assert result['count_provenance_status'] == 'DERIVED_FROM_RAW_GFF_IN_THIS_INVOCATION'
    assert result['structural_available'] is False and result['decision'] is None
    assert len(result['baseline_reference_counts']) == 50
    receipt = json.loads((out/'receipt.json').read_text())
    assert len(receipt['comparisons']) == 350
    assert receipt['independent_read_splice_status'] == 'ABSENT_NOT_NEGATIVE'
    assert receipt['coverage_policy'] == 'PENDING_AUTHOR'
    assert receipt['withholding_status'] == 'UNASSESSABLE'
    assert receipt['runs']['2']['seed'] == 2
    assert len(json.loads((out/'counts.json').read_text())['loci']) == 50
    with pytest.raises(FileExistsError): p.produce(input_path,out)
    # Declared GFF bytes cannot be modified after decoding.
    Path(rows[0]['path']).write_text('mutated')
    with pytest.raises(ValueError,match='hash mismatch'): p.produce(input_path,tmp_path/'other')
