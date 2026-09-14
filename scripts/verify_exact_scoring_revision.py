"""Independently verify an exact-scoring revision and its source preservation."""
import argparse
import json
import statistics
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import adr_scoring as rubric
from scripts.rescore_saved_results import digest, read, run_path

CLASSES = ["Fully_Compliant", "Mostly_Compliant", "Partially_Compliant", "Not_Compliant"]


def integer_oracle(checks, scores):
    sc = sum(w * checks[f"C{i}"] for i, w in enumerate((10, 15, 15, 15, 20, 15, 10), 1))
    dq_numerator = sum(w * scores[f"Q{i}"] for i, w in enumerate((10, 15, 15, 10, 25, 15, 10), 1))
    composite_numerator = 2 * sc + dq_numerator
    if composite_numerator < 225 or sc < 40:
        return "Not_Compliant"
    if composite_numerator >= 425 and sc >= 80 and dq_numerator >= 240:
        return "Fully_Compliant"
    if composite_numerator >= 350 and sc >= 60 and dq_numerator >= 180:
        return "Mostly_Compliant"
    return "Partially_Compliant"


def verify(run_id):
    run = run_path(run_id)
    config = rubric.require_exact_run(run)
    audit = read(run / "analysis/exact_rescoring_report.json")
    failures = []
    for relative, sha in audit["source_sha256"].items():
        if digest(ROOT / relative) != sha:
            failures.append(f"Source changed: {relative}")
    for name, sha in audit["corrected_raw_sha256"].items():
        if digest(run / "raw_results" / name) != sha:
            failures.append(f"Corrected raw file changed: {name}")
    source = run_path(config["source_run_id"])
    performance = read(run / "analysis/performance_details.json")["configurations"]
    combined = read(run / "raw_results/all_results.json")
    total, changes = 0, 0
    comparisons = []
    for model, strategies in combined.items():
        for strategy, repetitions in strategies.items():
            key = f"{model}/{strategy}"
            original = read(source / "raw_results" / f"{model}_{strategy}.json")
            if repetitions != read(run / "raw_results" / f"{model}_{strategy}.json"):
                failures.append(f"Combined file mismatch: {key}")
            old_f1, new_f1, new_kappa, config_changes = [], [], [], 0
            for old_rep, new_rep in zip(original, repetitions, strict=True):
                for old, new in zip(old_rep, new_rep, strict=True):
                    total += 1
                    config_changes += old["predicted"] != new["predicted"]
                    fields = rubric.prediction_fields(old["predicted_sc_checks"], old["predicted_dq_scores"])
                    allowed = set(fields) | {"correct", "analysis_run_id"}
                    if {k: v for k, v in old.items() if k not in allowed} != {k: v for k, v in new.items() if k not in allowed}:
                        failures.append(f"Nonderived data changed: {key}/{new['adr_id']}")
                    if any(new.get(k) != v for k, v in fields.items()):
                        failures.append(f"Derived field mismatch: {key}/{new['adr_id']}")
                    if new["predicted"] != integer_oracle(new["predicted_sc_checks"], new["predicted_dq_scores"]):
                        failures.append(f"Integer oracle mismatch: {key}/{new['adr_id']}")
                truth = [row["actual"] for row in new_rep]
                predicted = [row["predicted"] for row in new_rep]
                old_f1.append(float(f1_score(truth, [row["predicted"] for row in old_rep], labels=CLASSES, average="macro", zero_division=0)))
                new_f1.append(float(f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)))
                new_kappa.append(float(cohen_kappa_score(truth, predicted, labels=CLASSES)))
            for metric, values in (("macro_f1", new_f1), ("cohen_kappa", new_kappa)):
                stored = performance[key]["overall_summary"][metric]
                for field, expected in (("mean", statistics.mean(values)), ("std", statistics.stdev(values))):
                    if abs(stored[field] - expected) > 1e-12:
                        failures.append(f"Metric mismatch: {key}/{metric}/{field}")
            changes += config_changes
            comparisons.append({"configuration": key, "changed_predictions": config_changes,
                                "old_macro_f1": statistics.mean(old_f1),
                                "new_macro_f1": statistics.mean(new_f1),
                                "new_kappa": statistics.mean(new_kappa)})
    if total != audit["counts"]["evaluations"] or changes != audit["counts"]["overall_changes"]:
        failures.append("Evaluation/change counts differ from rescoring report")
    sensitivity = read(run / "analysis/rubric_sensitivity_performance.json")
    if sensitivity["validation"]["baseline_model_predictions_different_from_saved"]:
        failures.append("Sensitivity baseline differs from corrected predictions")
    report = {"run_id": run_id, "evaluations": total, "overall_changes": changes,
              "source_files_checked": len(audit["source_sha256"]), "failures": failures,
              "independent_check": "Integer-numerator classifier and independent per-repetition metric calculation",
              "configurations": comparisons,
              "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                              [ROOT / "adr_scoring.py", ROOT / "adr_benchmark.py"] +
                              sorted((ROOT / "scripts").glob("*scor*.py")) +
                              [ROOT / "scripts/analyze_rubric_sensitivity.py",
                               ROOT / "scripts/analyze_rule_hybrid_comparison.py",
                               ROOT / "scripts/prepare_manuscript_v31_evidence.py",
                               ROOT / "scripts/build_manuscript_v32.py"]}}
    (run / "analysis/exact_scoring_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("code_sha256", "configurations")}, indent=2))
    if failures:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    verify(parser.parse_args().run_id)
