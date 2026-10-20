"""Strict small B6 proof predicates shared by allocated and Python3.6 host tools.

No files, environment, tensors or external packages are read here. Callers must
independently bind expected source/job and actual comparison/candidate bytes.
Allocated publication additionally recomputes all tensors/traces/history.
"""
import hashlib
import json
import re

PROFILE = "b6_fresh400"
STATE_KEYS = {"optimizer.bin", "scheduler.bin"} | {
    "random_states_%d.pkl" % rank for rank in range(4)
}
TRACE_KEYS = {
    "inputs",
    "shapes",
    "rng_before",
    "rng_after",
    "loss",
    "context",
    "observed",
    "gradients",
    "boundary_rng",
}
RANKS = {str(rank) for rank in range(4)}
PHASE_RANKS = {phase + "/rank%d" % rank for phase in ("continuous", "resumed") for rank in range(4)}
TIERS = {30720, 61440, 129024}
POSITION = dict(epoch=0, step=9840, global_step=410, pending_micro=0)
SNAPSHOT = dict(epoch=0, step=9600, global_step=400, pending_micro=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value):
    return hashlib.sha256(encoded(value).encode("utf8")).hexdigest()


def sha(value, label):
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "invalid " + label + " SHA256",
    )


def mapping(value, label, keys=None):
    require(type(value) is dict, "missing " + label)
    if keys is not None:
        require(set(value) == set(keys), "incomplete " + label)
    return value


def identity(value):
    mapping(value, "B6 identity")
    expected = dict(
        arm="B6",
        seed=123,
        aug_seed=20261037,
        world_size=4,
        n_pilot=4099,
        stop_after_updates=4099,
        mixed_precision="bf16",
        checkpointing="nonreentrant",
        deterministic=True,
    )
    require(
        all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()),
        "wrong B6 identity",
    )
    for key in ("recipe_sha256", "init_weights_sha256"):
        sha(value.get(key), key)


def validate_report(report, expected_source, expected_job):
    """Reject contradictory/partial small reports, regardless of declared hashes.

    expected_source is the independently pinned40-digit source SHA; expected_job
    is the original validation allocation's decimal string (not the current job).
    Returns report on success and raises ValueError otherwise.
    """
    require(
        type(expected_source) is str and re.fullmatch("[0-9a-f]{40}", expected_source),
        "invalid expected source",
    )
    require(
        type(expected_job) is str and re.fullmatch("[0-9]+", expected_job),
        "invalid expected validation job",
    )
    mapping(report, "B6 comparison")
    envelope = dict(
        version=1,
        kind="b6_history_comparison",
        profile=PROFILE,
        arm="B6",
        source_commit=expected_source,
        validation_job_id=expected_job,
    )
    require(
        all(type(report.get(k)) is type(v) and report[k] == v for k, v in envelope.items()),
        "wrong B6 comparison envelope",
    )
    require(
        all(
            report.get(key) is True
            for key in ("passed", "model_equal", "history_equal", "finite_losses")
        ),
        "failed B6 comparison",
    )
    require(report.get("different_model_keys") == [], "model mismatch list is not empty")
    states = mapping(report.get("states_equal"), "state equality", STATE_KEYS)
    require(
        all(value is True for value in states.values()), "optimizer/scheduler/rank RNG mismatch"
    )
    traces = mapping(report.get("traces"), "rank trace mismatch lists", RANKS)
    for row in traces.values():
        mapping(row, "trace mismatch fields", TRACE_KEYS)
        require(
            all(type(value) is list and value == [] for value in row.values()), "trace mismatch"
        )
    for key in ("checkpoint_digests", "checkpoint_history_sha256"):
        hashes = mapping(report.get(key), key, {"continuous", "resumed"})
        for value in hashes.values():
            sha(value, key)
        require(hashes["continuous"] == hashes["resumed"], key + " differ")
    for key in ("verified_record_sha256", "fresh_start_sha256", "runtime_fingerprint_sha256"):
        sha(report.get(key), key)
    trace_hashes = mapping(report.get("trace_sha256"), "trace digests", PHASE_RANKS)
    for value in trace_hashes.values():
        sha(value, "trace")
    row_counts = mapping(report.get("trace_row_counts"), "trace row counts", PHASE_RANKS)
    for name, row in row_counts.items():
        expected = (
            dict(forward=9840, accumulation_boundary=410)
            if name.startswith("continuous/")
            else dict(forward=240, accumulation_boundary=10)
        )
        require(encoded(row) == encoded(expected), "incomplete trace rows")
    bindings = mapping(
        report.get("checkpoint_bindings"), "checkpoint bindings", {"continuous", "at400", "resumed"}
    )
    identities, fingerprints = [], []
    for name, row in bindings.items():
        mapping(row, "checkpoint binding")
        require(
            encoded(row.get("position")) == encoded(SNAPSHOT if name == "at400" else POSITION),
            "wrong diagnostic checkpoint position",
        )
        for key in ("meta_sha256", "identity_sha256", "history_sha256"):
            sha(row.get(key), key)
        identity(row.get("identity"))
        identities.append(encoded(row["identity"]))
        fingerprint = mapping(row.get("fingerprint"), "runtime fingerprint")
        expected = dict(
            world_size=4,
            backend="inductor",
            mixed_precision="bf16",
            checkpointing="nonreentrant",
            deterministic=True,
        )
        require(
            all(
                type(fingerprint.get(k)) is type(v) and fingerprint[k] == v
                for k, v in expected.items()
            ),
            "wrong runtime fingerprint",
        )
        require(
            fingerprint.get("effective_caches", {}).get("fx_local") is True
            and fingerprint.get("effective_caches", {}).get("aot_local") is True,
            "FX/AOT cache is not enabled",
        )
        require(
            fingerprint.get("inductor", {}).get("force_disable_caches") is False,
            "compiler cache force-disabled",
        )
        env = fingerprint.get("cache_env", {})
        require(
            env.get("TORCHINDUCTOR_CACHE_DIR") == "/work/compiler_cache/b6_history_v1/inductor"
            and env.get("TRITON_CACHE_DIR") == "/work/compiler_cache/b6_history_v1/triton",
            "wrong B6 cache namespace",
        )
        require(
            fingerprint.get("recipe", {}).get("_sha256") == row["identity"]["recipe_sha256"],
            "runtime recipe mismatch",
        )
        fingerprints.append(encoded(fingerprint))
        for key in ("compiler_events", "history_entries"):
            require(type(row.get(key)) is int and row[key] > 0, "missing observed compiler events")
        files = mapping(
            row.get("state_file_sha256"),
            "complete checkpoint file digests",
            STATE_KEYS | {"model.safetensors"},
        )
        for value in files.values():
            sha(value, "state file")
        if name != "at400":
            require(
                row["history_sha256"] == report["checkpoint_history_sha256"][name],
                "history binding mismatch",
            )
    require(
        len(set(identities)) == len(set(fingerprints)) == 1,
        "cross-run identity/fingerprint mismatch",
    )
    require(
        canonical_sha(bindings["resumed"]["fingerprint"]) == report["runtime_fingerprint_sha256"],
        "fingerprint digest mismatch",
    )
    observed = mapping(report.get("observed_evidence"), "observed evidence", PHASE_RANKS)
    tiers, tracks, chains = set(), 0, 0
    for row in observed.values():
        mapping(row, "observed evidence summary")
        require(
            type(row.get("dna_tiers")) is list and set(row["dna_tiers"]) <= TIERS,
            "invalid observed tiers",
        )
        for key in ("tracks_nonzero", "valid_chain_elements"):
            require(type(row.get(key)) is int and row[key] >= 0, "invalid observed evidence counts")
        tiers.update(row["dna_tiers"])
        tracks += row["tracks_nonzero"]
        chains += row["valid_chain_elements"]
    require(
        tiers == TIERS and tracks > 0 and chains > 0,
        "missing global B6 evidence/chain/tier observations",
    )
    return report


def validate_ready(receipt, report, expected):
    """Validate ready fields against report AND independent actual-file bindings.

    expected must contain exactly the full ready envelope: version, kind,
    source_commit, validation_job_id, comparison_sha256, verified_record_sha256,
    provenance, runtime_fingerprint_sha256, candidate. provenance contains image,
    source-manifest and helper-manifest SHA256s. candidate contains relative_run,
    relative_checkpoint, epoch, step, global_step, pending_micro and raw meta,
    identity, history plus canonical full-checkpoint SHA256s. The caller obtains
    these values independently; copying an untrusted receipt into expected does
    not verify files. This function still rejects a rehashed false comparison.
    """
    keys = {
        "version",
        "kind",
        "source_commit",
        "validation_job_id",
        "comparison_sha256",
        "verified_record_sha256",
        "provenance",
        "runtime_fingerprint_sha256",
        "candidate",
    }
    mapping(expected, "independent expected ready fields", keys)
    mapping(receipt, "ready envelope", keys)
    validate_report(report, expected["source_commit"], expected["validation_job_id"])
    require(encoded(receipt) == encoded(expected), "ready differs from independently bound files")
    require(
        type(receipt["version"]) is int
        and receipt["version"] == 1
        and receipt["kind"] == "b6_history_validation_ready",
        "wrong B6 ready kind/version",
    )
    for key in ("comparison_sha256", "verified_record_sha256", "runtime_fingerprint_sha256"):
        sha(receipt[key], key)
    require(
        receipt["verified_record_sha256"] == report["verified_record_sha256"]
        and receipt["runtime_fingerprint_sha256"] == report["runtime_fingerprint_sha256"],
        "ready verification/runtime binding mismatch",
    )
    provenance = mapping(
        receipt["provenance"],
        "ready provenance",
        {"image_sha256", "source_manifest_sha256", "helper_manifest_sha256"},
    )
    for value in provenance.values():
        sha(value, "provenance")
    candidate = mapping(receipt["candidate"], "candidate")
    binding = report["checkpoint_bindings"]["resumed"]
    expected_candidate = dict(
        relative_run="resumed", relative_checkpoint="latest_state", **POSITION
    )
    expected_candidate.update(
        {key: binding[key] for key in ("meta_sha256", "identity_sha256", "history_sha256")}
    )
    expected_candidate["canonical_checkpoint_sha256"] = report["checkpoint_digests"]["resumed"]
    require(
        encoded(candidate) == encoded(expected_candidate), "candidate contradicts B6 comparison"
    )
    return receipt
