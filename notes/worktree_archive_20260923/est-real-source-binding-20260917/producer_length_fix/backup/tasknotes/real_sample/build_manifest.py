"""Build actual pins from controller-captured metadata, without reading remote data.
Usage: python build_manifest.py CAPTURE_PATH OUTPUT_JSON (exclusive output).
"""
import hashlib
import json
from pathlib import Path
import re
import sys


def build(capture):
    data = json.loads(Path(capture).read_text())
    root = '/data/gpfs/assoc/pgl/data/Transgenic/evidence'
    base = root + '/est_blastx/sitalica_pilot_20260914'
    files, searches = {}, []
    for source in ('orthodb', 'swissprot'):
        path = base + '/' + source
        prov_path = path + '/provenance.json'
        text = data[prov_path]
        prov = json.loads(text)
        key = source + '_provenance'
        files[key] = dict(path=prov_path, sha256=hashlib.sha256(text.encode()).hexdigest())
        commands_path = path + '/commands.json'
        files[source + '_commands'] = dict(path=commands_path, sha256=hashlib.sha256(data[commands_path].encode()).hexdigest())
        for role, suffix, field in [('raw', '.raw13.tsv', 'raw_sha256'), ('protein', '.filtered.fa', 'filtered_fasta_sha256')]:
            files[source + '_' + role] = dict(path=path + '/' + source + suffix, sha256=prov[field],
                                             receipt=dict(file=key, field=[field]))
        searches.append(dict(source=source, raw=source + '_raw', protein=source + '_protein',
                             c1_provenance=key, c1_commands=source + '_commands'))
        query = dict(path=prov['inputs']['query'], sha256=prov['inputs']['query_sha256'],
                     receipt=dict(file=key, field=['inputs', 'query_sha256']))
        if 'screened' in files:
            assert files['screened']['sha256'] == query['sha256']
        else:
            files['screened'] = query
    align_path = root + '/est_align/Sitalica/PROVENANCE.txt'
    align_text = data[align_path]
    align = {line.split()[0]: line.split()[1] for line in align_text.splitlines() if len(line.split()) >= 2}
    files['align_provenance'] = dict(path=align_path, sha256=hashlib.sha256(align_text.encode()).hexdigest())
    for key, field in [('original', 'est_raw'), ('bam', 'bam'), ('reference', 'reference')]:
        files[key] = dict(path=align[field], md5=align[field + '_md5'],
                          receipt=dict(file='align_provenance', text_key=field + '_md5', algorithm='md5'))
    files['screened']['md5'] = align['est_md5']
    files['report'] = dict(path=root + '/est/Sitalica/univec/report.tsv')
    files['samtools'] = dict(path='/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/samtools',
                            sha256='a579aed31a13841cc0950a487b59729355d0b9fbcc39293e0b90b960133217bc')
    return dict(species='Sitalica', arm='primary', files=files, searches=searches,
                selection=dict(source='orthodb', count=16, prefix_lines=10000),
                metadata_capture_sha256=hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
                runtime=dict(python='/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11',
                             python_version='3.11.14', reader_samtools='1.21', reader_htslib='1.23',
                             original_producer_samtools='1.3.1'),
                report_pin_state='no historical report hash captured; fresh whole-file SHA256 and before/after stat at compute; report relations cross-checked against pinned FASTAs',
                a11_status='unassessable', training_eligible=False, full_c0=False)


if __name__ == '__main__':
    manifest = build(sys.argv[1])
    with open(sys.argv[2], 'x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
