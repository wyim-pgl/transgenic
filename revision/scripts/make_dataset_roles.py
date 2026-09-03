#!/usr/bin/env python3
"""Build evidence/DATASET_ROLES.tsv — the fail-closed role manifest of protocol A18.3 (issue #5).

Every dataset AND every run gets exactly one role: b1_validation_only | c2_training_eligible | excluded.
Builders (C0 ingestion, C2 label generation, B1 scorer) refuse any run that is not listed here, so this
file has to be complete before the first alignment (protocol line 213: the matrix and this manifest are
checksummed into §1 before the first alignment).

Roles are not inferred from the directory a file happens to sit in. Two of them are deliberate exceptions
that a directory-based rule would get wrong:
  - ont/Athaliana_cui2020_PRJNA594286 was fetched by the *validation* driver but is c2_training_eligible
    (A18.3), and is excluded from the A. thaliana B1 replication.
  - the PacBio tree under RETIRED_DO_NOT_USE/ is excluded by author decision of 2026-09-03 (issue #60):
    all 51 runs are _subreads, and protocol §3/v1.3/v1.5 admits PacBio only at CCS/FLNC level.

Checksums, author decision of 2026-09-03. `source_checksum` is the **ENA-published fastq_md5**, which
identifies the object at the source and is stable across re-downloads. The md5 our fetcher writes is a
different thing — it hashes the FASTA we produced after `zcat | seqkit fq2fa | gzip`, so it changes
between two downloads of identical data (gzip stores an mtime in its header; two fetches of Cui 2020
differ in file md5 and agree exactly once decompressed) and cannot be compared to anything the source
publishes. That value is kept as `local_fa_md5`, which is what identifies the artifact we actually hold.

Known limit, recorded rather than papered over: the fetcher streams the fastq and never stores it, so
the ENA md5 documents what was fetched but cannot be re-verified against what we keep without
downloading again. Verifying the stream at fetch time (tee to md5sum before zcat) is the fix and is not
implemented. GenBank efetch publishes no per-batch checksum, so the EST rows carry authority `none`.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

ROOT_DEFAULT = "/data/gpfs/assoc/pgl/data/Transgenic/evidence"

TRAINING_SPECIES = ("Athaliana", "Bdistachyon", "Gmax", "Osativa", "Ppatens",
                    "Ptrichocarpa", "Sbicolor", "Sitalica", "Vvinifera")
TEST_SPECIES = ("Zmays", "Slycopersicum")

ROLES = ("b1_validation_only", "c2_training_eligible", "excluded")

# Genotype stratum per dataset. Not derivable from any field, so it is stated once, here, with the
# reason. A18.4 gives it a weight: reference 1.0, non_reference / hybrid_pooled / unknown 0.5.
GENOTYPE = {
    "est/Athaliana": ("reference", "GenBank gbdiv_est, predominantly Col-0; not resolvable per accession"),
    "ont/Athaliana_FLIC_PRJNA1087576": ("reference", "Col-0"),
    "ont/Athaliana_cui2020_PRJNA594286": ("reference", "Col-0 rosette"),
    "ont/Zmays_roottip_PRJNA822071": ("reference", "B73 root tip"),
    "ont/Slycopersicum_heinz_PRJEB37834": ("reference", "Heinz"),
    "pacbio/Athaliana_zhang2023_PRJNA911826": ("reference", "Col-0 WT runs only, mutants excluded at fetch"),
    "pacbio/Zmays_B73_ccs_PRJNA1470126": ("reference", "B73"),
    "pacbio/Zmays_kinnex_hybrid_PRJNA1290227": ("hybrid_pooled", "B73 x Mo17 hybrid; A7-P pooled stratum"),
    "pacbio/Zmays_wang2018_PRJEB22122": ("reference", "B73"),
    "pacbio/Zmays_wang2020_zenodo2611319": ("hybrid_pooled", "all-genotype FLNC; B73-only selection required by §3.1 before scoring"),
    "training/ont/Athaliana/cui2020_PRJNA594286": ("reference", "Col-0 rosette"),
    "training/ont/Athaliana/col0_DRP009401": ("reference", "Col-0 subset of PRJDB14952; C24 and F1 runs deliberately not taken"),
    "training/ont/Gmax/wm82_graft_PRJNA648759": ("reference", "Williams 82"),
    "training/ont/Gmax/wm82_seed_PRJNA416810": ("reference", "Williams 82"),
    "training/ont/Gmax/scn_roots_PRJNA803218": ("non_reference", "genotype 09-138"),
    "training/ont/Osativa/nip_dRNA_6tissue_PRJNA752930": ("reference", "Nipponbare"),
    "training/ont/Osativa/nip_pool_PRJNA953663": ("reference", "Nipponbare; mixed BioProject, WGS run rejected by the A18.3 strategy filter (#63)"),
    "training/ont/Osativa/nip_sheath_PRJNA1044249": ("reference", "Nipponbare sheath"),
    "training/ont/Osativa/indica_flagleaf_PRJNA1291274": ("non_reference", "indica SY63/MH63/ZS97"),
    "training/ont/Ppatens/dRNA_gametophore_PRJNA681088": ("reference", "Gransden gametophore"),
    "training/ont/Ptrichocarpa/sdx_dRNA_PRJNA517295": ("non_reference", "SDX genotype"),
    "training/ont/Ptrichocarpa/sdx_drought_PRJNA672182": ("non_reference", "SDX genotype"),
    "training/ont/Sitalica/ci846_salt_PRJNA1097621": ("non_reference", "Ci846"),
    "training/ont/Vvinifera/pinotnoir_berry_PRJNA776245": ("reference", "Pinot noir, the PN40024 background"),
    "training/ont/Vvinifera/callus_PRJNA732451": ("unknown", "callus line not stated"),
}

# A run that exists in two dataset paths must be declared here, naming which copy is canonical, or the
# manifest refuses to build. A18.3's acceptance criterion is that every run occurs exactly once; a
# duplicate that nobody noticed is the same evidence counted twice. Key: run accession -> canonical dataset.
# 저자 결정 2026-09-03: 중복된 사본은 쓰지 않는다. 정본은 프로토콜 A18.3이 **이름으로 지목한** 경로
# ("ont/Athaliana_cui2020_PRJNA594286 … has role c2_training_eligible")이므로 그쪽을 남기고, 학습
# 드라이버가 따로 받은 사본은 role=excluded로 내린다. 두 사본은 압축을 풀면 완전히 동일하다(실측);
# 파일 md5가 달랐던 것은 gzip 헤더의 타임스탬프 때문이다.
DUPLICATE_RESOLUTION: dict = {
    "SRR10611193": "ont/Athaliana_cui2020_PRJNA594286",
    "SRR10611194": "ont/Athaliana_cui2020_PRJNA594286",
    "SRR10611195": "ont/Athaliana_cui2020_PRJNA594286",
}
DUPLICATE_BASIS = ("저자 결정 2026-09-03: 같은 런의 중복 사본은 쓰지 않는다. 정본은 프로토콜 A18.3이 "
                   "이름으로 지목한 경로이며 내용은 두 사본이 동일하다")


# Datasets whose role is not the default for their location, with the clause that sets it.
ROLE_OVERRIDE = {
    "ont/Athaliana_cui2020_PRJNA594286": ("c2_training_eligible",
                                          "A18.3: fetched by the validation driver but training-eligible; "
                                          "excluded from the A. thaliana B1 replication"),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_md5_file(path):
    """<run>.md5 as written by longread_fetch.sh / est_fetch.sh: 'md5  filename'."""
    try:
        with open(path) as fh:
            return fh.read().split()[0]
    except Exception:
        return ""


def filereport(dataset_dir):
    """run_accession -> (instrument_model, library_strategy, read_count, fastq_md5) from the ENA filereport."""
    out = {}
    p = os.path.join(dataset_dir, "filereport.tsv")
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r.get("run_accession", "")] = (r.get("instrument_model", ""), r.get("library_strategy", ""),
                                               r.get("read_count", ""), r.get("fastq_md5", ""))
    return out


def species_of(dataset_key, fallback=""):
    for sp in TRAINING_SPECIES + TEST_SPECIES:
        if re.search(rf"\b{sp}\b", dataset_key) or f"/{sp}/" in dataset_key or dataset_key.split("/")[-1].startswith(sp):
            return sp
    return fallback


def rows_for_est(root):
    for sp in sorted(os.listdir(os.path.join(root, "est"))):
        d = os.path.join(root, "est", sp)
        fa = os.path.join(d, "est.fa.gz")
        if not os.path.isdir(d) or not os.path.exists(fa):
            continue
        key = f"est/{sp}"
        total = ""
        for name in ("COUNT", "TOTAL"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                total = open(p).read().strip()
                break
        role = "b1_validation_only" if sp in TEST_SPECIES else "c2_training_eligible"
        gt, gt_note = GENOTYPE.get(key, ("unknown", "not stated"))
        yield dict(dataset=key, run="est.fa.gz", species=sp, genotype_stratum=gt, instrument="GenBank_EST",
                   data_type="EST", expected_files="1", expected_reads=total,
                   source_checksum="", source_checksum_authority="none",
                   local_fa_md5=read_md5_file(fa + ".md5"), role=role,
                   basis="A14: EST of the training species is training evidence; test-species EST is validation only",
                   note=gt_note + "; GenBank efetch publishes no per-batch checksum")


# Datasets that are not one-file-per-run. Zenodo 2611319 ships a single FLNC FASTA plus the paper's GFF,
# marked flnc.DONE / gff.DONE, so the per-run walker cannot find a <run>.md5 for them.
NON_RUN_DATASETS = {
    "pacbio/Zmays_wang2020_zenodo2611319": [
        ("F1maize.flnc.fa.gz", "F1maize.flnc.md5", "FLNC"),
        ("F1maize.FINAL.gff", None, "reference_gff"),
        ("F1maize.demux_FL_count.txt", None, "demultiplex_counts"),
    ],
}


def rows_for_runs(root, rel, default_role, data_type, basis):
    base = os.path.join(root, rel)
    if not os.path.isdir(base):
        return
    for dirpath, _dirnames, filenames in os.walk(base):
        key = os.path.relpath(dirpath, root)
        if key in NON_RUN_DATASETS:
            gt, gt_note = GENOTYPE.get(key, ("unknown", "not stated"))
            role, note_extra = ROLE_OVERRIDE.get(key, (default_role, ""))
            for fname, md5file, kind in NON_RUN_DATASETS[key]:
                fp = os.path.join(dirpath, fname)
                if not os.path.exists(fp):
                    continue
                ck = read_md5_file(os.path.join(dirpath, md5file)) if md5file else md5(fp)
                yield dict(dataset=key, run=fname, species=species_of(key), genotype_stratum=gt,
                           instrument="Sequel", data_type=kind, expected_files="1", expected_reads="",
                           source_checksum=ck, source_checksum_authority="Zenodo_published" if md5file else "local_only",
                           local_fa_md5=ck, role=role, basis=note_extra or basis, note=gt_note)
            continue
        dones = sorted(f[:-5] for f in filenames if f.endswith(".DONE"))
        if not dones:
            continue
        meta = filereport(dirpath)
        role, note_extra = default_role, ""
        if key in ROLE_OVERRIDE:
            role, note_extra = ROLE_OVERRIDE[key]
        gt, gt_note = GENOTYPE.get(key, ("unknown", "not stated"))
        for run in dones:
            inst, strat, reads, ena_md5 = meta.get(run, ("", "", "", ""))
            yield dict(dataset=key, run=run, species=species_of(key), genotype_stratum=gt,
                       instrument=inst or "unknown", data_type=strat or data_type,
                       expected_files="1", expected_reads=reads,
                       source_checksum=ena_md5, source_checksum_authority="ENA_fastq_md5" if ena_md5 else "none",
                       local_fa_md5=read_md5_file(os.path.join(dirpath, run + ".md5")),
                       role=role, basis=note_extra or basis, note=gt_note)


def demote_duplicates(rows):
    """중복 사본을 excluded로 내린다 — 매니페스트가 같은 증거를 두 번 세지 않게."""
    n = 0
    for r in rows:
        canonical = DUPLICATE_RESOLUTION.get(r["run"])
        if canonical and r["dataset"] != canonical:
            r["role"] = "excluded"
            r["basis"] = DUPLICATE_BASIS + f" ({canonical})"
            n += 1
    return n


def build(root):
    rows = []
    rows += list(rows_for_est(root))
    rows += list(rows_for_runs(root, "training/ont", "c2_training_eligible", "ONT",
                               "A14: training-species ONT is training evidence"))
    rows += list(rows_for_runs(root, "ont", "b1_validation_only", "ONT",
                               "A14: validation-only long reads"))
    rows += list(rows_for_runs(root, "pacbio", "b1_validation_only", "PacBio",
                               "A14: test-species validation accepts any PacBio generation"))
    rows += list(rows_for_runs(root, "RETIRED_DO_NOT_USE", "excluded", "PacBio",
                               "issue #60, author decision 2026-09-03: subreads-only, "
                               "protocol §3/v1.3/v1.5 admits CCS/FLNC level only"))
    demote_duplicates(rows)
    return rows


FIELDS = ["dataset", "run", "species", "genotype_stratum", "instrument", "data_type",
          "expected_files", "expected_reads", "source_checksum", "source_checksum_authority",
          "local_fa_md5", "role", "basis", "note"]


def validate(rows):
    """The manifest is only worth having if it is complete and unambiguous. Refuse to write otherwise."""
    v = []
    seen = {}
    for r in rows:
        k = (r["dataset"], r["run"])
        if k in seen:
            v.append(f"duplicate entry {k}")
        seen[k] = r
        if r["role"] not in ROLES:
            v.append(f"{k}: invalid role {r['role']!r}")
        if not r["species"]:
            v.append(f"{k}: no species")
        if r["source_checksum_authority"] == "none" and not r["local_fa_md5"]:
            v.append(f"{k}: no checksum of any kind")
        if r["source_checksum_authority"] == "ENA_fastq_md5" and not r["source_checksum"]:
            v.append(f"{k}: claims an ENA checksum but carries none")
    # A18.3: every run occurs exactly once. The (dataset, run) check above cannot see a run that was
    # downloaded twice into two different dataset paths, which is what happened to Cui 2020.
    by_run: dict = {}
    for r in rows:
        if r["data_type"] in ("reference_gff", "demultiplex_counts") or r["run"] == "est.fa.gz":
            continue
        by_run.setdefault(r["run"], []).append(r["dataset"])
    for run, datasets in sorted(by_run.items()):
        if len(datasets) < 2:
            continue
        canonical = DUPLICATE_RESOLUTION.get(run)
        if canonical is None:
            v.append(f"run {run} appears in {len(datasets)} datasets {sorted(datasets)} and no canonical "
                     f"copy is declared in DUPLICATE_RESOLUTION")
        elif canonical not in datasets:
            v.append(f"run {run}: declared canonical {canonical!r} is not one of {sorted(datasets)}")
        else:
            for r in rows:
                if r["run"] == run and r["dataset"] != canonical and r["role"] != "excluded":
                    v.append(f"run {run}: non-canonical copy in {r['dataset']} is {r['role']}, must be excluded")

    for sp in TEST_SPECIES:
        bad = [r for r in rows if r["species"] == sp and r["role"] == "c2_training_eligible"]
        if bad:
            v.append(f"test species {sp} marked training-eligible: {[r['run'] for r in bad]}")
    if not any(r["dataset"] == "ont/Athaliana_cui2020_PRJNA594286" and r["role"] == "c2_training_eligible"
               for r in rows):
        v.append("A18.3: ont/Athaliana_cui2020_PRJNA594286 must be c2_training_eligible")
    for ds in ("ont/Athaliana_FLIC_PRJNA1087576", "pacbio/Athaliana_zhang2023_PRJNA911826"):
        bad = [r for r in rows if r["dataset"] == ds and r["role"] != "b1_validation_only"]
        if bad:
            v.append(f"A14: {ds} must be b1_validation_only")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--out", default=None, help="default: <root>/DATASET_ROLES.tsv")
    ap.add_argument("--summary", default=None, help="write a JSON summary next to the manifest")
    ap.add_argument("--scope", choices=["all", "est"], default="all",
                    help="est = only the EST rows. Author decision 2026-09-03: EST is complete and cannot "
                         "change, so its roles are frozen first and the long-read scope gets its own v2 "
                         "freeze once collection finishes (protocol line 213 asks the manifest to be "
                         "checksummed into §1 before the first alignment)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "DATASET_ROLES.tsv")
    rows = build(a.root)
    violations = validate(rows)                      # 검증은 항상 전체에 대해 한다
    if a.scope == "est":
        rows = [r for r in rows if r["run"] == "est.fa.gz"]
    if violations:
        print("manifest refused:", file=sys.stderr)
        for x in violations:
            print("  " + x, file=sys.stderr)
        sys.exit(1)
    rows.sort(key=lambda r: (r["role"], r["species"], r["dataset"], r["run"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"rows": len(rows), "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
               "by_role": {}, "by_role_species": {}}
    for r in rows:
        summary["by_role"][r["role"]] = summary["by_role"].get(r["role"], 0) + 1
        summary["by_role_species"].setdefault(r["role"], {})
        summary["by_role_species"][r["role"]][r["species"]] = \
            summary["by_role_species"][r["role"]].get(r["species"], 0) + 1
    if a.summary:
        with open(a.summary, "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
