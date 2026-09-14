"""Create an analysis-only exact-scoring revision without any model requests."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import adr_scoring as rubric


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)


def run_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid run ID")
    return ROOT / "results" / "runs" / run_id


def rescore_row(row, analysis_run_id):
    """Copy derived fields only; never edit criteria, labels or request evidence."""
    updated = copy.deepcopy(row)
    fields = rubric.prediction_fields(row["predicted_sc_checks"], row["predicted_dq_scores"])
    updated.update(fields)
    updated["correct"] = updated["predicted"] == updated["actual"]
    updated["analysis_run_id"] = analysis_run_id
    return updated


def rescore(source_run_id, run_id, reasoning_effort="medium"):
    import adr_benchmark as benchmark

    source, destination = run_path(source_run_id), run_path(run_id)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Choose a new, nonexistent destination run ID. Nothing was overwritten.")
    original_config = read(source / "run_config.json")
    if original_config.get("analysis_only"):
        raise ValueError("Use the original model run as the source, not an analysis revision.")
    benchmark.configure_run_paths(run_id)
    benchmark.configure_mistral_variant(original_config["active_mistral_variant"])
    benchmark.configure_gpt_reasoning_effort(reasoning_effort)
    fingerprints = {f"{model}/{strategy}": benchmark.protocol_fingerprint(model, strategy)
                    for model in benchmark.MODELS for strategy in benchmark.STRATEGIES}
    if fingerprints != original_config["protocol_fingerprints"]:
        raise ValueError("Source request protocol differs from the current configuration. Do not resubmit requests.")
    refs = [ROOT / "results" / name for name in
            ("eval_set.json", "human_ground_truth.json", "human_ground_truth_interrator.json")]
    if digest(refs[0]) != original_config["evaluation_set_sha256"]:
        raise ValueError("Evaluation manifest changed since the original run.")
    if digest(refs[1]) != original_config["ground_truth_sha256"]:
        raise ValueError("Primary human submissions changed since the original run.")
    evaluation = read(refs[0])["adrs"]
    ids = [row["id"] for row in evaluation]
    if len(ids) != benchmark.N_EVAL or len(set(ids)) != len(ids):
        raise ValueError("Evaluation set is not the complete frozen set.")
    truth = read(refs[1])
    inputs = sorted([p for p in source.rglob("*") if p.is_file()] + refs +
                    [ROOT / "results/adrs" / f"{adr_id}.json" for adr_id in ids])
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    expected_files = {f"{model}_{strategy}.json" for model in benchmark.MODELS
                      for strategy in benchmark.STRATEGIES}
    actual_files = {path.name for path in (source / "raw_results").glob("*.json")
                    if path.name != "all_results.json"}
    if actual_files != expected_files:
        raise ValueError("Source must contain exactly the expected model/strategy result files.")
    combined, original_combined, changes, counts = {}, {}, [], Counter()
    for model in benchmark.MODELS:
        combined[model], original_combined[model] = {}, {}
        for strategy in benchmark.STRATEGIES:
            key = f"{model}/{strategy}"
            repetitions = read(source / "raw_results" / f"{model}_{strategy}.json")
            valid, reason = benchmark._validate_result_shape(repetitions, ids, benchmark.N_REPS, True)
            if not valid:
                raise ValueError(f"{key}: {reason}")
            original_combined[model][strategy] = repetitions
            updated_repetitions = []
            for rep_index, rep in enumerate(repetitions, 1):
                updated_rep = []
                for row_index, row in enumerate(rep, 1):
                    if row.get("model_key") != model or row.get("strategy") != strategy:
                        raise ValueError(f"Configuration mismatch: {key}, rep {rep_index}, row {row_index}")
                    if row.get("actual") != truth[row["adr_id"]]["overall"]:
                        raise ValueError(f"Recorded reference differs from primary submission: {row['adr_id']}")
                    updated = rescore_row(row, run_id)
                    reason = benchmark._result_row_validation_reason(updated, rep_index, row_index)
                    if reason:
                        raise ValueError(f"{key}: {reason}")
                    changed = [field for field in ("predicted", "predicted_sc_class", "predicted_dq_class")
                               if row.get(field) != updated[field]]
                    if changed:
                        derived_keys = rubric.prediction_fields(row["predicted_sc_checks"], row["predicted_dq_scores"])
                        changes.append({"configuration": key, "repetition": rep_index,
                                        "row": row_index, "adr_id": row["adr_id"],
                                        "changed_classes": changed,
                                        "before": {field: row.get(field) for field in derived_keys},
                                        "after": {field: updated[field] for field in derived_keys}})
                    counts["evaluations"] += 1
                    counts["overall_changes"] += row.get("predicted") != updated["predicted"]
                    updated_rep.append(updated)
                updated_repetitions.append(updated_rep)
            combined[model][strategy] = updated_repetitions
    merged_path = source / "raw_results/all_results.json"
    if merged_path.exists() and read(merged_path) != original_combined:
        raise ValueError("Original merged results disagree with individual configuration files.")
    human_audits = {}
    for path in refs[1:]:
        labels = read(path)
        human_audits[path.name] = {
            "submissions_unchanged": True,
            "recorded_vs_exact_overall_discrepancies": [adr_id for adr_id, row in labels.items()
                if rubric.evaluate_criteria(row)["overall"] != row["overall"]],
            "exact_class_distribution": dict(Counter(rubric.evaluate_criteria(row)["overall"]
                                                      for row in labels.values())),
        }
    if any(digest(ROOT / relative) != sha for relative, sha in hashes.items()):
        raise ValueError("A source changed during validation; no revision was written.")

    # Validate everything before creating a destination. An incomplete manifest blocks analysis.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "raw_results").mkdir()
    (destination / "analysis").mkdir()
    config = {**original_config, "run_id": run_id, "source_run_id": source_run_id,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "source_created_at": original_config["created_at"],
              "scoring_version": rubric.SCORING_VERSION, "analysis_only": True,
              "rescore_complete": False, "model_api_calls": 0}
    write(destination / "run_config.json", config)
    for model, strategies in combined.items():
        for strategy, repetitions in strategies.items():
            write(destination / "raw_results" / f"{model}_{strategy}.json", repetitions)
    write(destination / "raw_results/all_results.json", combined)
    unchanged = all(digest(ROOT / relative) == sha for relative, sha in hashes.items())
    report = {"source_run_id": source_run_id, "analysis_run_id": run_id,
              "scoring_version": rubric.SCORING_VERSION, "api_calls": 0,
              "source_files_unchanged": unchanged, "source_sha256": hashes,
              "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                              (ROOT / "adr_scoring.py", ROOT / "adr_benchmark.py", Path(__file__))},
              "counts": dict(counts), "human_submissions": human_audits,
              "class_changes": changes,
              "corrected_raw_sha256": {p.name: digest(p) for p in
                                       (destination / "raw_results").glob("*.json")},
              "reference_policy": "Recorded primary overall labels retained; dimension labels derived exactly.",
              "provenance_policy": "Row run_id, criteria, request fingerprints, human labels and API metadata retained."}
    write(destination / "analysis/exact_rescoring_report.json", report)
    if not unchanged:
        raise ValueError("Source changed during output creation. Revision is incomplete and blocked.")
    config["rescore_complete"] = True
    temp_config = destination / "run_config.complete.json"
    write(temp_config, config)
    temp_config.replace(destination / "run_config.json")
    print(json.dumps({"destination": str(destination), **dict(counts),
                      "source_files_unchanged": unchanged, "api_calls": 0}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True, help="New, nonexistent analysis run ID")
    parser.add_argument("--gpt-reasoning-effort", default="medium")
    args = parser.parse_args()
    os.chdir(ROOT)
    rescore(args.source_run_id, args.run_id, args.gpt_reasoning_effort)
