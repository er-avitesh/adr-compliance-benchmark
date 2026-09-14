#!/usr/bin/env python3
"""Read-only verification of the ADR benchmark replication package."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "Fully_Compliant",
    "Mostly_Compliant",
    "Partially_Compliant",
    "Not_Compliant",
)
SC_WEIGHTS = (10, 15, 15, 15, 20, 15, 10)
DQ_WEIGHTS = (10, 15, 15, 10, 25, 15, 10)
EXPECTED_CLASS_COUNTS = {
    "Fully_Compliant": 63,
    "Mostly_Compliant": 53,
    "Partially_Compliant": 33,
    "Not_Compliant": 13,
}
EXPECTED_PROMPT_EXEMPLARS = {
    "alphagov_govuk-aws_0012-security-groups-in-terraform",
    "adr_madr_0001-use-CC0-or-MIT-as-license",
    "argoproj_argo-cd_deep-links",
    "synthetic_not_compliant_v1",
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recorded_path(root: Path, value: str) -> Path:
    return root.joinpath(*value.replace("\\", "/").split("/"))


def input_exceptions(root: Path) -> dict[str, dict]:
    path = root / "release" / "input_text_exceptions.json"
    if not path.is_file():
        return {}
    return {item["adr_id"]: item for item in read_json(path).get("exceptions", [])}


def public_data_transforms(root: Path) -> dict[str, dict]:
    path = root / "release" / "public_data_transforms.json"
    if not path.is_file():
        return {}
    return {
        item["path"].replace("\\", "/"): item
        for item in read_json(path).get("transforms", [])
    }


def model_input_text(adr_id: str, stored_text: str, exceptions: dict[str, dict]) -> str:
    if adr_id not in exceptions:
        return stored_text
    return stored_text.encode("utf-8").decode("windows-1252")


def exact_scores(record: dict) -> tuple[Fraction, Fraction, Fraction, str, str, str]:
    sc = Fraction(sum(weight * int(record[f"C{i}"]) for i, weight in enumerate(SC_WEIGHTS, 1)))
    dq_numerator = sum(weight * int(record[f"Q{i}"]) for i, weight in enumerate(DQ_WEIGHTS, 1))
    dq = Fraction(dq_numerator, 3)
    composite = Fraction(2 * int(sc) + dq_numerator, 5)

    def dimension(value: Fraction) -> str:
        if value >= 85:
            return "Fully_Compliant"
        if value >= 70:
            return "Mostly_Compliant"
        if value >= 45:
            return "Partially_Compliant"
        return "Not_Compliant"

    if composite < 45 or sc < 40:
        overall = "Not_Compliant"
    elif composite >= 85 and sc >= 80 and dq >= 80:
        overall = "Fully_Compliant"
    elif composite >= 70 and sc >= 60 and dq >= 60:
        overall = "Mostly_Compliant"
    else:
        overall = "Partially_Compliant"
    return sc, dq, composite, dimension(sc), dimension(dq), overall


class Audit:
    def __init__(self):
        self.passes: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passes if condition else self.failures).append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def verify_release_manifest(root: Path, spec: dict, audit: Audit, allow_dirty_manifest: bool = False) -> None:
    manifest_path = root / "RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        audit.warnings.append("No RELEASE_MANIFEST.json found; auditing the source tree rather than a built release.")
        return

    manifest = read_json(manifest_path)
    for key in (
        "canonical_run_id",
        "source_run_id",
        "protocol_version",
        "scoring_version",
    ):
        audit.check(manifest.get(key) == spec[key], f"Release manifest {key} matches the release specification")
    if allow_dirty_manifest and manifest.get("working_tree_clean") is not True:
        audit.warnings.append("Built package is marked as a dirty local test build.")
    else:
        audit.check(manifest.get("working_tree_clean") is True, "Publication release was built from a clean worktree")

    listed = set()
    for item in manifest.get("files", []):
        relative = item.get("path", "")
        listed.add(relative)
        target = recorded_path(root, relative)
        audit.check(target.is_file(), f"Manifest file exists: {relative}")
        if target.is_file():
            audit.check(target.stat().st_size == item.get("bytes"), f"Manifest size matches: {relative}")
            audit.check(digest(target) == item.get("sha256"), f"Manifest hash matches: {relative}")

    sums_path = root / "SHA256SUMS.txt"
    audit.check(sums_path.is_file(), "SHA256SUMS.txt is present")
    if sums_path.is_file():
        sums = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value, relative = line.split("  ", 1)
            sums[relative] = value
        expected = listed | {"RELEASE_MANIFEST.json"}
        audit.check(set(sums) == expected, "SHA256SUMS.txt covers every payload file and the release manifest")
        for relative, value in sums.items():
            target = recorded_path(root, relative)
            audit.check(target.is_file() and digest(target) == value, f"SHA256SUMS entry matches: {relative}")

    forbidden = set(spec["forbidden_release_path_parts"])
    payload_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected_files = listed | {"RELEASE_MANIFEST.json", "SHA256SUMS.txt"}
    audit.check(payload_files == expected_files, "Built release contains no unlisted files")
    for relative in payload_files:
        parts = set(Path(relative).parts)
        audit.check(not (parts & forbidden), f"Release path is allowed: {relative}")
        audit.check(not relative.lower().endswith((".pyc", ".pyo")), f"Release excludes bytecode: {relative}")


def verify_inputs(root: Path, spec: dict, audit: Audit) -> tuple[dict, dict, dict, list[dict]]:
    eval_path = root / "results" / "eval_set.json"
    primary_path = root / "results" / "human_ground_truth.json"
    second_path = root / "results" / "human_ground_truth_interrator.json"
    for path in (eval_path, primary_path, second_path):
        audit.check(path.is_file(), f"Required input exists: {path.relative_to(root)}")
    if not all(path.is_file() for path in (eval_path, primary_path, second_path)):
        return {}, {}, {}, []

    evaluation = read_json(eval_path)
    primary = read_json(primary_path)
    second = read_json(second_path)
    adrs = evaluation.get("adrs", [])
    ids = [item.get("id") for item in adrs]
    unique_ids = set(ids)

    audit.check(evaluation.get("n") == spec["evaluation_set_size"], "Evaluation manifest declares 162 ADRs")
    audit.check(len(adrs) == spec["evaluation_set_size"], "Evaluation manifest contains 162 ADR records")
    audit.check(len(unique_ids) == len(ids), "Evaluation ADR identifiers are unique")
    audit.check(unique_ids <= set(primary), "Every evaluation ADR has a primary human assessment")
    audit.check(len({item.get("source_repo") for item in adrs}) == 14, "Evaluation set spans 14 source repositories")

    counts = Counter(primary[item]["overall"] for item in ids if item in primary)
    audit.check(dict(counts) == EXPECTED_CLASS_COUNTS, "Recorded primary class distribution is 63/53/33/13")
    audit.check(evaluation.get("class_distribution") == EXPECTED_CLASS_COUNTS, "Manifest class distribution matches primary labels")

    held_out = set(evaluation.get("heldout_few_shot_exemplar_ids", []))
    audit.check(len(held_out) == 4, "Four source-manifest holdouts are declared")
    audit.check(not (held_out & unique_ids), "Source-manifest holdouts are excluded from evaluation")

    texts = {}
    for adr_id in unique_ids | (held_out - {"synthetic_not_compliant_v1"}):
        path = root / "results" / "adrs" / f"{adr_id}.json"
        audit.check(path.is_file(), f"Frozen ADR exists: {adr_id}")
        if path.is_file():
            item = read_json(path)
            audit.check(item.get("id") == adr_id, f"Frozen ADR identifier matches filename: {adr_id}")
            texts[adr_id] = item.get("text", "")

    exceptions = input_exceptions(root)
    audit.check(len(exceptions) == 2, "Two historical text-decoding exceptions are documented")
    for adr_id, item in exceptions.items():
        stored = texts.get(adr_id, "")
        transformed = model_input_text(adr_id, stored, exceptions)
        audit.check(len(stored) == item.get("stored_text_chars"), f"Stored text length matches exception record: {adr_id}")
        audit.check(hashlib.sha256(stored.encode("utf-8")).hexdigest() == item.get("stored_text_sha256"), f"Stored text hash matches exception record: {adr_id}")
        audit.check(len(transformed) == item.get("model_input_chars"), f"Model input length matches exception record: {adr_id}")
        audit.check(hashlib.sha256(transformed.encode("utf-8")).hexdigest() == item.get("model_input_sha256"), f"Model input hash matches exception record: {adr_id}")

    audit.check(len(second) == 41, "Second human assessment contains 41 ADRs")
    audit.check(set(second) <= unique_ids, "Second assessments are limited to evaluation ADRs")
    recorded_matches = 0
    derived_matches = 0
    second_internal_mismatches = 0
    primary_internal_mismatches = 0
    for adr_id, second_record in second.items():
        if adr_id not in primary:
            continue
        primary_record = primary[adr_id]
        recorded_matches += primary_record.get("overall") == second_record.get("overall")
        primary_derived = exact_scores(primary_record)[-1]
        second_derived = exact_scores(second_record)[-1]
        derived_matches += primary_derived == second_derived
        primary_internal_mismatches += primary_record.get("overall") != primary_derived
        second_internal_mismatches += second_record.get("overall") != second_derived
    audit.check(recorded_matches == 37, "Recorded primary and second overall labels match for 37 of 41 ADRs")
    audit.check(derived_matches == 27, "Criterion-derived primary and second classes match for 27 of 41 ADRs")
    audit.check(second_internal_mismatches == 11, "Second file contains 11 recorded-versus-derived label differences")
    audit.check(primary_internal_mismatches == 0, "Primary labels are formula-consistent for the 41 second-review ADRs")

    transforms = public_data_transforms(root)
    if transforms:
        primary_transform = transforms.get("results/human_ground_truth.json", {})
        second_transform = transforms.get("results/human_ground_truth_interrator.json", {})
        audit.check(len(primary) == 166, "Public primary file is limited to evaluation ADRs and declared holdouts")
        audit.check(
            set(primary) == unique_ids | (held_out - {"synthetic_not_compliant_v1"}),
            "Public primary file contains only evaluation ADRs and declared holdouts",
        )
        audit.check(
            set(record.get("reviewer_id") for record in primary.values()) == {f"P{index:02d}" for index in range(1, 8)},
            "Primary reviewer identities use seven anonymous public codes",
        )
        audit.check(
            set(record.get("reviewer_id") for record in second.values()) == {f"S{index:02d}" for index in range(1, 4)},
            "Second reviewer identities use three anonymous public codes",
        )
        audit.check(
            primary_transform.get("public_sha256") == digest(primary_path),
            "Primary privacy-transform public hash matches",
        )
        audit.check(
            second_transform.get("public_sha256") == digest(second_path),
            "Second privacy-transform public hash matches",
        )
        evidence_relative = (
            "results/runs/v31_fulltext_gpt_medium_exact_scoring/analysis/"
            "manuscript_evidence/evidence.json"
        )
        evidence_path = recorded_path(root, evidence_relative)
        evidence_transform = transforms.get(evidence_relative, {})
        evidence = read_json(evidence_path) if evidence_path.is_file() else {}
        reviewer_counts = evidence.get("reviewer_counts", {})
        audit.check(
            set(reviewer_counts.get("first", {})) == {f"P{index:02d}" for index in range(1, 8)},
            "Manuscript evidence uses anonymous primary reviewer codes",
        )
        audit.check(
            set(reviewer_counts.get("second", {})) == {f"S{index:02d}" for index in range(1, 4)},
            "Manuscript evidence uses anonymous second reviewer codes",
        )
        audit.check(
            evidence_transform.get("public_sha256") == digest(evidence_path),
            "Manuscript-evidence privacy-transform public hash matches",
        )
    return evaluation, primary, second, adrs


def verify_source_licenses(root: Path, evaluation: dict, audit: Audit) -> None:
    """Verify that every distributed ADR maps to documented source terms."""
    inventory_path = root / "release" / "source_licenses.json"
    notices_path = root / "THIRD_PARTY_NOTICES.md"
    audit.check(inventory_path.is_file(), "Machine-readable source-license inventory exists")
    audit.check(notices_path.is_file(), "Third-party attribution notice exists")
    if not inventory_path.is_file() or not evaluation:
        return

    inventory = read_json(inventory_path)
    mappings = {
        item.get("repository"): item
        for item in inventory.get("repositories", [])
    }
    held_out = set(evaluation.get("heldout_few_shot_exemplar_ids", []))
    ids = {item["id"] for item in evaluation.get("adrs", [])} | held_out
    repo_counts = Counter()
    missing = []
    for adr_id in sorted(ids):
        path = root / "results" / "adrs" / f"{adr_id}.json"
        if not path.is_file():
            missing.append(f"missing distributed ADR {adr_id}")
            continue
        record = read_json(path)
        repository = record.get("source_repo")
        repo_counts[repository] += 1
        if repository not in mappings:
            missing.append(f"no license mapping for {repository}")
        if not record.get("url"):
            missing.append(f"no source URL for {adr_id}")

    audit.check(not missing, "Every distributed ADR has source and license metadata")
    audit.failures.extend(missing)
    audit.check(len(ids) == inventory.get("distributed_adr_records"), "License inventory covers 166 distributed ADR records")
    audit.check(len(evaluation.get("adrs", [])) == inventory.get("evaluated_adr_records"), "License inventory covers 162 evaluated ADR records")
    recorded_counts = {
        item["repository"]: item.get("distributed_adr_count")
        for item in inventory.get("repositories", [])
    }
    audit.check(dict(repo_counts) == recorded_counts, "Per-repository license counts match distributed ADR files")
    audit.check(
        all(item.get("license_expression") and item.get("license_url") for item in mappings.values()),
        "Every source repository has a license expression and evidence URL",
    )


def verify_run_config(root: Path, spec: dict, audit: Audit) -> tuple[Path, dict]:
    run = root / "results" / "runs" / spec["canonical_run_id"]
    config_path = run / "run_config.json"
    audit.check(config_path.is_file(), "Canonical run configuration exists")
    if not config_path.is_file():
        return run, {}
    config = read_json(config_path)
    expected = {
        "run_id": spec["canonical_run_id"],
        "source_run_id": spec["source_run_id"],
        "protocol_version": spec["protocol_version"],
        "scoring_version": spec["scoring_version"],
        "input_policy": spec["input_policy"],
        "evaluation_set_size": spec["evaluation_set_size"],
        "repetitions": spec["repetitions"],
        "active_mistral_variant": "ministral-3-8b-api",
        "analysis_only": True,
        "rescore_complete": True,
        "model_api_calls": 0,
    }
    for key, value in expected.items():
        audit.check(config.get(key) == value, f"Canonical run config {key} is {value!r}")
    audit.check(len(config.get("protocol_fingerprints", {})) == 12, "Canonical run records 12 protocol fingerprints")
    return run, config


def verify_raw_results(
    root: Path,
    run: Path,
    config: dict,
    spec: dict,
    evaluation: dict,
    primary: dict,
    adrs: list[dict],
    audit: Audit,
) -> None:
    if not config or not evaluation or not primary:
        audit.failures.append("Raw-result verification skipped because required inputs are missing")
        return

    eval_ids = [item["id"] for item in adrs]
    eval_id_set = set(eval_ids)
    text_by_id = {}
    exceptions = input_exceptions(root)
    for adr_id in eval_ids:
        path = root / "results" / "adrs" / f"{adr_id}.json"
        if path.is_file():
            stored_text = read_json(path).get("text", "")
            text_by_id[adr_id] = model_input_text(adr_id, stored_text, exceptions)

    total = 0
    truncated = 0
    invalid = []
    fingerprints = config["protocol_fingerprints"]
    for model in spec["models"]:
        for strategy in spec["strategies"]:
            key = f"{model}/{strategy}"
            path = run / "raw_results" / f"{model}_{strategy}.json"
            if not path.is_file():
                invalid.append(f"missing file {path.name}")
                continue
            repetitions = read_json(path)
            if len(repetitions) != spec["repetitions"]:
                invalid.append(f"{key}: expected 3 repetitions, found {len(repetitions)}")
                continue
            for rep_index, rows in enumerate(repetitions, 1):
                if len(rows) != spec["evaluation_set_size"]:
                    invalid.append(f"{key} repetition {rep_index}: expected 162 rows, found {len(rows)}")
                    continue
                row_ids = [row.get("adr_id") for row in rows]
                if len(set(row_ids)) != len(row_ids) or set(row_ids) != eval_id_set:
                    invalid.append(f"{key} repetition {rep_index}: ADR identifiers differ from eval_set.json")
                for row in rows:
                    total += 1
                    adr_id = row.get("adr_id")
                    try:
                        sc, dq, composite, sc_class, dq_class, overall = exact_scores(
                            {**row["predicted_sc_checks"], **row["predicted_dq_scores"]}
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        invalid.append(f"{key}/{adr_id}: invalid criterion vector ({error})")
                        continue
                    expected_hash = hashlib.sha256(text_by_id.get(adr_id, "").encode("utf-8")).hexdigest()
                    checks = (
                        row.get("result_schema_version") == 4,
                        row.get("protocol_version") == spec["protocol_version"],
                        row.get("protocol_fingerprint") == fingerprints.get(key),
                        row.get("run_id") == spec["source_run_id"],
                        row.get("analysis_run_id") == spec["canonical_run_id"],
                        row.get("scoring_version") == spec["scoring_version"],
                        row.get("model_key") == model,
                        row.get("strategy") == strategy,
                        row.get("actual") == primary.get(adr_id, {}).get("overall"),
                        row.get("predicted") == overall,
                        row.get("predicted_overall_from_criteria") == overall,
                        row.get("predicted_sc_class") == sc_class,
                        row.get("predicted_dq_class") == dq_class,
                        abs(float(row.get("predicted_sc_score")) - float(sc)) < 1e-12,
                        abs(float(row.get("predicted_dq_score")) - float(dq)) < 1e-12,
                        abs(float(row.get("predicted_composite_score")) - float(composite)) < 1e-12,
                        row.get("correct") is (row.get("actual") == overall),
                        row.get("parse_success") is True,
                        row.get("criterion_parse_success") is True,
                        row.get("error") in (None, ""),
                        row.get("input_was_truncated") is False,
                        row.get("adr_input_chars") == row.get("adr_source_chars") == len(text_by_id.get(adr_id, "")),
                        row.get("adr_input_sha256") == expected_hash,
                    )
                    if not all(checks):
                        invalid.append(f"{key}/{adr_id}: row invariant failed")
                    truncated += row.get("input_was_truncated") is not False

    audit.check(not invalid, "All raw rows satisfy protocol, input, parse, and exact-scoring invariants")
    if invalid:
        audit.failures.extend(invalid[:25])
        if len(invalid) > 25:
            audit.failures.append(f"{len(invalid) - 25} additional raw-row failures omitted")
    audit.check(total == spec["expected_evaluations"], "Canonical run contains 5,832 evaluations")
    audit.check(truncated == 0, "Canonical run contains no truncated ADR inputs")


def verify_analysis(root: Path, run: Path, config: dict, spec: dict, audit: Audit) -> None:
    required = (
        "cost_analysis.json",
        "criterion_level_performance.json",
        "error_analysis.json",
        "exact_rescoring_report.json",
        "exact_scoring_verification.json",
        "manuscript_results_summary.json",
        "metrics_summary.json",
        "model_reproducibility.json",
        "pairwise_statistical_tests.json",
        "performance_details.json",
        "rubric_manifest.json",
        "rubric_sensitivity_performance.json",
        "rule_based_baseline.json",
        "rule_hybrid_comparison.json",
        "threshold_sensitivity.json",
        "manuscript_evidence/evidence.json",
        "manuscript_assets/asset_provenance.json",
        "manuscript_assets/figure_1_distribution.png",
        "manuscript_assets/figure_2_performance.png",
        "manuscript_assets/figure_3_confusion.png",
        "manuscript_assets/figure_4_criteria.png",
        "manuscript_assets/overall_performance.csv",
        "manuscript_assets/sensitivity.csv",
    )
    for relative in required:
        audit.check((run / "analysis" / relative).is_file(), f"Canonical analysis exists: {relative}")

    report_path = run / "analysis" / "exact_rescoring_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        audit.check(report.get("api_calls") == 0, "Exact rescoring made zero API calls")
        audit.check(report.get("source_files_unchanged") is True, "Exact rescoring reports unchanged source files")
        audit.check(report.get("counts", {}).get("evaluations") == 5832, "Exact rescoring covers 5,832 evaluations")
        audit.check(report.get("counts", {}).get("overall_changes") == 40, "Exact rescoring records 40 overall changes")
        transforms = public_data_transforms(root)
        for relative, value in report.get("source_sha256", {}).items():
            path = recorded_path(root, relative)
            normalized = relative.replace("\\", "/")
            transform = transforms.get(normalized)
            matches_source = path.is_file() and digest(path) == value
            matches_public_transform = (
                path.is_file()
                and transform is not None
                and transform.get("source_sha256") == value
                and transform.get("public_sha256") == digest(path)
            )
            audit.check(
                matches_source or matches_public_transform,
                f"Rescoring source or documented public transform matches: {relative}",
            )
        for name, value in report.get("corrected_raw_sha256", {}).items():
            path = run / "raw_results" / name
            audit.check(path.is_file() and digest(path) == value, f"Corrected raw hash matches: {name}")

    verification_path = run / "analysis" / "exact_scoring_verification.json"
    if verification_path.is_file():
        verification = read_json(verification_path)
        audit.check(verification.get("evaluations") == 5832, "Independent verification covers 5,832 evaluations")
        audit.check(verification.get("overall_changes") == 40, "Independent verification reproduces 40 changes")
        audit.check(verification.get("failures") == [], "Saved independent exact-scoring verification has no failures")

    model_path = run / "analysis" / "model_reproducibility.json"
    if model_path.is_file():
        model_report = read_json(model_path)
        models = {item["model_key"]: item for item in model_report.get("models", [])}
        expected_ids = {
            "gpt-5.5": "gpt-5.5-2026-04-23",
            "claude-sonnet-4-6": "claude-sonnet-4-6",
            "ministral-3-8b": "ministral-8b-2512",
            "gemini-2.5-pro": "gemini-2.5-pro",
        }
        audit.check(set(models) == set(expected_ids), "Reproducibility manifest contains the four publication models")
        for model, model_id in expected_ids.items():
            audit.check(models.get(model, {}).get("exact_model_id") == model_id, f"Exact model ID matches for {model}")
            audit.check(models.get(model, {}).get("model_access_date") == spec["model_access_date"], f"Model access date matches for {model}")
        audit.check(models.get("gpt-5.5", {}).get("reasoning_setting") == "reasoning_effort=medium", "GPT reasoning effort is medium")
        audit.check(model_report.get("input_policy") == "full_text", "Model manifest records full-text input")
        audit.check(model_report.get("pricing_verified_date") == spec["pricing_verified_date"], "Model pricing date is recorded")

    cost_path = run / "analysis" / "cost_analysis.json"
    if cost_path.is_file():
        cost = read_json(cost_path)
        audit.check(cost.get("pricing_verified_date") == spec["pricing_verified_date"], "Cost analysis uses dated pricing inputs")
        access_dates = {
            item.get("model_access_date")
            for item in cost.get("model_access_summary", {}).values()
        }
        audit.check(access_dates == {spec["model_access_date"]}, "Cost analysis reports one matching model access date")

    source_run = root / "results" / "runs" / spec["source_run_id"]
    source_required = (
        "run_config.json",
        "analysis/preflight_report.json",
        "analysis/prompt_manifest.json",
        "analysis/prompt_leakage_audit.json",
        "analysis/manuscript_dataset_summary.json",
        "batch_jobs/batch_manifest.json",
    )
    for relative in source_required:
        audit.check((source_run / relative).is_file(), f"Original provider-run evidence exists: {relative}")

    prompt_path = source_run / "analysis" / "prompt_manifest.json"
    if prompt_path.is_file():
        prompt = read_json(prompt_path)
        actual = set(prompt.get("few_shot", {}).get("exemplar_ids", []))
        audit.check(actual == EXPECTED_PROMPT_EXEMPLARS, "Retained prompt manifest identifies the four actual exemplars")
        source_declared = set(
            read_json(root / "results" / "eval_set.json").get(
                "heldout_few_shot_exemplar_ids", []
            )
        )
        audit.check(
            source_declared - actual == {
                "alphagov_govuk-aws_0001-record-architecture-decisions"
            },
            "The one unused source-manifest holdout is explicitly identifiable",
        )

    if (root / "RELEASE_MANIFEST.json").is_file():
        stale = (
            "adr_dataset.json",
            "dataset_report.json",
            "find_adr_paths.py",
            "results/adr_manifest.json",
        )
        for relative in stale:
            audit.check(
                not recorded_path(root, relative).exists(),
                f"Reviewer release excludes historical acquisition artifact: {relative}",
            )


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_entrypoints(root: Path, audit: Audit) -> None:
    """Confirm that public CLIs start without performing experiment work."""
    commands = (
        ("adr_benchmark.py", "--help"),
        ("scripts/verify_validation_run.py", "--help"),
        ("scripts/build_release.py", "--help"),
    )
    for relative, argument in commands:
        path = recorded_path(root, relative)
        if not path.is_file():
            audit.failures.append(f"Public CLI is missing: {relative}")
            continue
        result = subprocess.run(
            [sys.executable, "-B", str(path), argument],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        audit.check(
            result.returncode == 0,
            f"Public CLI starts successfully: {relative} {argument}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT, help="Source tree or built release root")
    parser.add_argument("--allow-dirty-manifest", action="store_true", help="Permit a manifest marked as a local dirty test build")
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit()

    spec_path = root / "release" / "release_spec.json"
    audit.check(spec_path.is_file(), "Release specification exists")
    if not spec_path.is_file():
        spec = {}
    else:
        spec = read_json(spec_path)

    if spec:
        verify_release_manifest(root, spec, audit, args.allow_dirty_manifest)
        evaluation, primary, _second, adrs = verify_inputs(root, spec, audit)
        verify_source_licenses(root, evaluation, audit)
        run, config = verify_run_config(root, spec, audit)
        verify_raw_results(root, run, config, spec, evaluation, primary, adrs, audit)
        verify_analysis(root, run, config, spec, audit)
        verify_entrypoints(root, audit)

    revision = git_revision(root)
    if revision:
        print(f"Source commit: {revision}")
    print(f"Checks passed: {len(audit.passes)}")
    if audit.warnings:
        print(f"Warnings: {len(audit.warnings)}")
        for item in audit.warnings:
            print(f"WARN: {item}")
    if audit.failures:
        print(f"Failures: {len(audit.failures)}")
        for item in audit.failures:
            print(f"FAIL: {item}")
        return 1
    print("RELEASE VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
