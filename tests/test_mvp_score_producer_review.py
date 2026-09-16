"""Round-1 review regressions: contradictory self-rehashed artifacts must fail."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from test_mvp_score_producer import p, w, ROOT, dump, finalized_fixture


def reseal(final, metadata):
    from transgenic.training.mvp_finalize import _canonical_json
    metadata['initialization_audit']['audit_sha256'] = hashlib.sha256(
        _canonical_json(metadata['initialization_audit']['audit'])).hexdigest()
    dump(final/'metadata.json',metadata)
    manifest=p.read_json(final/'manifest.json')
    for name in manifest['files']:
        identity=p.file_identity(final/name)
        manifest['files'][name]={k:identity[k] for k in ('sha256','size')}
    dump(final/'manifest.json',manifest)


@pytest.mark.parametrize('mutation', ['one_file','omit_loader','packages_empty','packages_missing','packages_extra','package_blank'])
def test_r1_rehashed_incomplete_code_manifest_rejected(tmp_path, mutation):
    final, hashes, code=finalized_fixture(tmp_path)
    value=p.read_json(code)
    if mutation == 'one_file': value['files']={str(Path(__file__)):w.sha256_file(__file__)}
    elif mutation == 'omit_loader': value['files'].pop(str(ROOT/'src/transgenic/training/weights_init.py'))
    elif mutation == 'packages_empty': value['external_packages']={}
    elif mutation == 'packages_missing': value['external_packages'].pop('duckdb')
    elif mutation == 'packages_extra': value['external_packages']['extra']='1'
    else: value['external_packages']['duckdb']=' '
    dump(code,value)
    metadata=p.read_json(final/'metadata.json')
    metadata['sif_code_identity']['code']['sha256']=w.sha256_file(code)
    reseal(final,metadata)
    with pytest.raises(ValueError): p.verify_final(final,1,[1],hashes,code)


@pytest.mark.parametrize('mutation', ['weights','recipe_missing','recipe_digest','implementation_path','implementation_hash','source_config','semantics','safetensors_metadata'])
def test_r2_rehashed_source_contradictions_rejected(tmp_path, mutation):
    from transgenic.training import weights_init as wi
    final, hashes, code=finalized_fixture(tmp_path)
    metadata=p.read_json(final/'metadata.json')
    audit=metadata['initialization_audit']['audit']
    changed=None
    if mutation == 'weights':
        changed='model.safetensors'; Path(audit['files'][changed]['path']).write_bytes(b'not safetensors')
    elif mutation in ('recipe_missing','recipe_digest'):
        changed='run_config.json'; path=Path(audit['files'][changed]['path']); value=p.read_json(path)
        if mutation == 'recipe_missing': value.pop('recipe')
        else: value['recipe']['name']='changed recipe, same architecture'
        dump(path,value)
    elif mutation == 'implementation_path': audit['implementation']['b5_runtime.py']['path']=str(tmp_path/'nonexistent/b5_runtime.py')
    elif mutation == 'implementation_hash': audit['implementation']['b5_runtime.py']['sha256']='0'*64
    elif mutation == 'source_config':
        changed='config.json'; path=Path(audit['files'][changed]['path']); value=p.read_json(path)
        value['decoder_d_model']+=1; dump(path,value)
    elif mutation == 'semantics':
        for side in ('source_semantics','target_semantics'):
            fields=audit[side]['fields']; fields['vocab_size']['value']+=1
            audit[side]=wi._manifest_record(fields)
    else: audit['load']['safetensors_metadata']={'invented':'metadata'}
    if changed:
        entry=audit['files'][changed]
        entry['before']=wi._snapshot(Path(entry['path'])); entry['after']=dict(entry['before'])
    reseal(final,metadata)
    with pytest.raises((ValueError,FileNotFoundError)):
        p.verify_final(final,1,[1],hashes,code)


@pytest.mark.parametrize('text', ['not JSON','[]','{"evidence_channels":8,"evidence_channels":8}','inconsistent'])
def test_r5_rehashed_final_config_rejected(tmp_path,text):
    final, hashes, code=finalized_fixture(tmp_path)
    if text == 'inconsistent':
        value=p.read_json(final/'config.json'); value['decoder_d_model']+=1; text=json.dumps(value)
    (final/'config.json').write_text(text)
    reseal(final,p.read_json(final/'metadata.json'))
    with pytest.raises(ValueError): p.verify_final(final,1,[1],hashes,code)


def test_real_loader_positive_and_bound_checkout_not_current_closure(tmp_path):
    final, hashes, code=finalized_fixture(tmp_path)
    assert p.verify_final(final,1,[1],hashes,code)['seed'] == 1
    # Independent training checkout, including a package module not in scorer's
    # checkout. Schema closure is computed there; no historical package version
    # is compared with currently installed scoring packages.
    checkout=tmp_path/'training-checkout'
    shutil.copytree(ROOT/'src/transgenic',checkout/'src/transgenic',ignore=shutil.ignore_patterns('__pycache__'))
    (checkout/'train').mkdir()
    shutil.copyfile(ROOT/'train/train_HyenaTransgenic_MVP.py',checkout/'train/train_HyenaTransgenic_MVP.py')
    extra=checkout/'src/transgenic/training/extra_training_source.py'; extra.write_text('# training checkout fixture\n')
    value=p.read_json(code)
    value['files']={str(checkout/Path(path).relative_to(ROOT)):digest for path,digest in value['files'].items()}
    value['files'][str(extra)]=w.sha256_file(extra)
    value['external_packages']['duckdb']='historical-duckdb-not-in-finalizer'
    dump(code,value)
    metadata=p.read_json(final/'metadata.json')
    for identity in metadata['initialization_audit']['audit']['implementation'].values():
        identity['path']=str(checkout/Path(identity['path']).relative_to(ROOT))
    metadata['sif_code_identity']['code']['sha256']=w.sha256_file(code)
    reseal(final,metadata)
    assert p.verify_final(final,1,[1],hashes,code)['seed'] == 1
    # Even a newly added unlisted module invalidates the exact training closure.
    unlisted=checkout/'src/transgenic/training/unlisted.py'; unlisted.write_text('# not in manifest\n')
    with pytest.raises(ValueError, match='exact bound training source closure'):
        p.verify_final(final,1,[1],hashes,code)
    unlisted.unlink()
    extra.unlink()
    with pytest.raises(ValueError, match='exact bound training source closure'):
        p.verify_final(final,1,[1],hashes,code)


@pytest.mark.parametrize('ambiguous',['t ', ' t', 't\u00a0'])
def test_r3_whitespace_transcript_collision_rejected(tmp_path, ambiguous):
    path=tmp_path/'p.gff3'
    path.write_text('##gff-version 3\nChr1\tx\tgene\t1\t100\t.\t+\t.\tID=g\n'
                    'Chr1\tx\tmRNA\t10\t20\t.\t+\t.\tID=t;Parent=g\n'
                    'Chr1\tx\texon\t10\t20\t.\t+\t.\tParent=t\n'
                    f'Chr1\tx\tmRNA\t30\t40\t.\t+\t.\tParent=g;ID={ambiguous}\n'
                    f'Chr1\tx\texon\t30\t40\t.\t+\t.\tParent={ambiguous}\n')
    with pytest.raises(ValueError): p.read_gff(path)


@pytest.mark.parametrize('loss',['gene','transcript'])
def test_r3_parser_inventory_conservation(tmp_path, monkeypatch, loss):
    from types import SimpleNamespace
    from test_mvp_score_producer import gff
    path=gff(tmp_path/'p.gff', [('t','+',[(30001,30100)],[(30001,30100)])])
    parsed=p.read_gff(path)
    if loss == 'gene': parsed.clear()
    else: parsed['g']['transcripts'].clear()
    monkeypatch.setattr(p,'legacy_module',lambda _: SimpleNamespace(load_gff=lambda _: parsed))
    with pytest.raises(ValueError,match='inventory changed'): p.read_gff(path)


@pytest.mark.parametrize('text',['',' \n\t\n','# unfinished\n'])
def test_r4_uncompleted_empty_gff_rejected(tmp_path,text):
    path=tmp_path/'p.gff3'; path.write_text(text)
    with pytest.raises(ValueError): p.read_gff(path)
    path.write_text('##gff-version 3\n')
    assert p.read_gff(path) == {}
