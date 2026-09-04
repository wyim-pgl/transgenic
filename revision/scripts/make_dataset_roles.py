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
import shlex
import subprocess
import hashlib
import json
import os
import re
import sys

ROOT_DEFAULT = "/data/gpfs/assoc/pgl/data/Transgenic/evidence"

TRAINING_SPECIES = ("Athaliana", "Bdistachyon", "Gmax", "Osativa", "Ppatens",
                    "Ptrichocarpa", "Sbicolor", "Sitalica", "Vvinifera")
TEST_SPECIES = ("Zmays", "Slycopersicum")

# A30 (frozen text): the Swiss-Prot resource is recorded "with the role sensitivity_set (a fourth
# role value; builders still fail closed on unknown roles)". The tuple held three values, so the row
# A30.5 step 1 tells this script to write was rejected by validate(). Unknown roles still fail.
ROLES = ("b1_validation_only", "c2_training_eligible", "excluded", "sensitivity_set")

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


# ----------------------------------------------------------------------------------------------
# Non-run resources (issue #66, author decision 2026-09-03)
# ----------------------------------------------------------------------------------------------
# Everything above is *scanned*: the builder walks the evidence tree and emits one row per
# sequencing run. Two objects the protocol depends on are not runs and are not in that tree --
# the cross-species protein resource (A19/#44) and the Swiss-Prot sensitivity set (A30) -- so
# they had no way into the manifest at all, and hand-editing the TSV does not survive the next
# regeneration. They are *declared* here instead.
#
# Declared, but checkable: both files live on pgl-gpu, outside the tree this builder walks, so
# they cannot be scanned into the manifest -- but that host is reachable over ssh, so the
# declaration is verified rather than trusted. `--verify-resources` recomputes each md5 from the
# file and refuses to write on a mismatch *or* on a file it could not read; without the flag the
# run says so on stderr. A resource whose file changes without this table changing is exactly the
# drift the manifest exists to catch, and now the check can see it.
RESOURCE_DATA_TYPES = ("protein_resource", "sensitivity_resource")

RESOURCES = (
    dict(dataset="protein/orthodb_v12_viridiplantae_stage2",
         run="odb12_Viridiplantae.filtered.fa.gz",
         species="cross_species", genotype_stratum="n/a", instrument="n/a",
         data_type="protein_resource", expected_files="1", expected_reads="12115085",
         source_checksum="", source_checksum_authority="none",
         local_fa_md5="453cb32b02e0799950d7d5f4de5f62ac",
         role="c2_training_eligible",
         basis="A19: cross-species protein resource for C2 CDS-family labels; leakage filter A19.1",
         note="pgl-gpu /home/pgl/scratch1/wyim/transgenic_data/orthodb_filtered_stage2/; "
              "12,115,085 of 12,204,762 sequences over 408 taxa; zero sequences from the evaluated "
              "species (taxids 3702/4577/4081), removed in stage 1; 3,178,195,033 B; "
              "md5 recomputed from the file 2026-09-03 and re-verified live over ssh, matches "
              "filter_summary.json; "
              "frozen into protocol section 1 (issue #66)"),
    dict(dataset="protein/swissprot_plants_2026_02",
         run="uniprot_sprot_plants.dat.gz",
         species="cross_species", genotype_stratum="n/a", instrument="n/a",
         data_type="sensitivity_resource", expected_files="1", expected_reads="42096",
         source_checksum="", source_checksum_authority="none",
         local_fa_md5="552089d0642b6c17f3486140a73a0163",
         role="sensitivity_set",
         basis="A30: separately labelled sensitivity/audit set for start codon, stop codon and "
               "phase; never a label resource (A19 unchanged)",
         note="pgl-gpu /home/pgl/scratch1/wyim/transgenic_data/protein/; UniProtKB/Swiss-Prot "
              "Release 2026_02 of 10-Jun-2026; 44,909 entries, 42,096 Viridiplantae; 58,889,942 B; "
              "sha256 09116f0b9db67ecc47bd4393ca6a417f311c0c6499b4614a9f581c0f5ead092e; "
              "md5 recomputed from the file 2026-09-03, re-verified live over ssh"),
)


# Where each declared resource actually lives, so the declaration can be checked rather than
# trusted. The builder runs on pronghorn and the files sit on pgl-gpu, but that host is reachable
# over ssh, so "the builder cannot see them" was a reason to write the check, not to skip it.
RESOURCE_LOCATIONS = {
    "odb12_Viridiplantae.filtered.fa.gz":
        ("gpu", "/home/pgl/scratch1/wyim/transgenic_data/orthodb_filtered_stage2/"
                "odb12_Viridiplantae.filtered.fa.gz"),
    "uniprot_sprot_plants.dat.gz":
        ("gpu", "/home/pgl/scratch1/wyim/transgenic_data/protein/uniprot_sprot_plants.dat.gz"),
}


def _ssh_md5(host, path):
    """md5 of a remote file. Raises if the host or the file cannot be reached."""
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"md5sum -- {shlex.quote(path)}"],
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0 or not out.stdout.strip():
        raise OSError((out.stderr or out.stdout).strip() or f"md5sum failed on {host}:{path}")
    return out.stdout.split()[0]


def verify_resources(runner=_ssh_md5):
    """Recompute every declared resource's md5 from the file and compare with the declaration.

    Fails closed in both directions that matter: a checksum that no longer matches, and a file we
    could not read at all. The second case is the subtle one -- a check that quietly passes when it
    could not look is worse than no check, because it converts 'unknown' into 'verified'.
    """
    v = []
    for r in RESOURCES:
        host, path = RESOURCE_LOCATIONS[r["run"]]
        try:
            got = runner(host, path)
        except Exception as e:                      # noqa: BLE001 - any failure to look is a failure
            v.append(f"{r['dataset']}: {host}:{path} could not be read ({e})")
            continue
        if got != r["local_fa_md5"]:
            v.append(f"{r['dataset']}: md5 mismatch at {host}:{path} "
                     f"(declared {r['local_fa_md5']}, file has {got})")
    return v


def rows_for_resources():
    """Declared non-run resources. Copies, so a caller cannot mutate the declaration."""
    for r in RESOURCES:
        yield dict(r)


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
    rows += list(rows_for_resources())
    demote_duplicates(rows)
    return rows


FIELDS = ["dataset", "run", "species", "genotype_stratum", "instrument", "data_type",
          "expected_files", "expected_reads", "source_checksum", "source_checksum_authority",
          "local_fa_md5", "role", "basis", "note"]


def scope_filter(rows, scope):
    """Select the rows of one freeze scope.

    Protocol section 1 freezes the manifest in scopes rather than all at once, because the two
    bodies of evidence complete at different times: EST finished downloading on 2026-09-01 and was
    frozen as DATASET_ROLES.est_v1.tsv, while long-read collection was still running and a single
    freeze would have gone stale within the hour.

    The long-read scope is everything that is neither EST nor a declared non-run resource. It keeps
    a dataset's auxiliary files (the Wang 2020 reference GFF and demultiplex counts) because they
    belong to that dataset, and it keeps `excluded` rows on purpose: a freeze that records only what
    was kept cannot be audited against what was rejected.
    """
    if scope == "est":
        return [r for r in rows if r["run"] == "est.fa.gz"]
    if scope == "longread":
        return [r for r in rows
                if r["run"] != "est.fa.gz" and r["data_type"] not in RESOURCE_DATA_TYPES]
    return list(rows)


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
        if r["data_type"] in ("reference_gff", "demultiplex_counts") + RESOURCE_DATA_TYPES or r["run"] == "est.fa.gz":
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
    ap.add_argument("--scope", choices=["all", "est", "longread"], default="all",
                    help="est = only the EST rows. Author decision 2026-09-03: EST is complete and cannot "
                         "change, so its roles are frozen first and the long-read scope gets its own v2 "
                         "freeze once collection finishes (protocol line 213 asks the manifest to be "
                         "checksummed into §1 before the first alignment)")
    ap.add_argument("--verify-resources", action="store_true",
                    help="recompute the declared resources' md5 from the files themselves (over ssh) "
                         "and refuse to write on a mismatch or on a file that cannot be read")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "DATASET_ROLES.tsv")
    rows = build(a.root)
    violations = validate(rows)                      # 검증은 항상 전체에 대해 한다
    if a.verify_resources:
        violations += verify_resources()
    else:
        print(f"note: {len(RESOURCES)} declared resource(s) were not checked against their files "
              f"this run; pass --verify-resources to recompute their md5", file=sys.stderr)
    rows = scope_filter(rows, a.scope)
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
