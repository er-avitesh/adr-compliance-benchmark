"""Compare deterministic SC rules, LLM-only predictions, and a hybrid system.

The hybrid uses rule-derived SC scores and LLM-derived DQ scores with the same
40/60 composite formula and structural floors as the benchmark classifier.
No model API calls are made.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import adr_scoring as rubric
RESULTS = ROOT / "results"
DEFAULT_RUN_ID = "v31_fulltext_gpt_medium"
CLASSES = [
    "Fully_Compliant",
    "Mostly_Compliant",
    "Partially_Compliant",
    "Not_Compliant",
]
MODEL_LABELS = {
    "gpt-5.5": "GPT-5.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "ministral-3-8b": "Ministral 3 8B",
}
STRATEGY_LABELS = {
    "zero_shot": "ZS",
    "few_shot": "FS",
    "chain_of_thought": "CoT",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def classify(sc_score: float, dq_score: float) -> str:
    return rubric.overall_class(sc_score, dq_score)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    values = []
    for label in CLASSES:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return statistics.mean(values)


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    n = len(y_true)
    observed = sum(a == b for a, b in zip(y_true, y_pred)) / n
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = sum(true_counts[label] * pred_counts[label] for label in CLASSES) / (n * n)
    return float("nan") if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def metric_summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def configuration_label(configuration_key: str) -> str:
    model, strategy = configuration_key.split("/", 1)
    return f"{MODEL_LABELS.get(model, model)} {STRATEGY_LABELS.get(strategy, strategy)}"


def main(run_id: str) -> int:
    run_dir = RESULTS / "runs" / run_id
    rubric.require_exact_run(run_dir)
    analysis = run_dir / "analysis"
    baseline = load_json(analysis / "rule_based_baseline.json")
    raw_results = load_json(run_dir / "raw_results" / "all_results.json")
    performance = load_json(analysis / "performance_details.json")
    criterion = load_json(analysis / "criterion_level_performance.json")
    rule_rows = {row["adr_id"]: row for row in baseline["rows"]}
    if len(rule_rows) != 162:
        raise ValueError(f"Expected 162 rule rows, found {len(rule_rows)}")

    hybrid_results = []
    for model, strategies in raw_results.items():
        for strategy, repetitions in strategies.items():
            repetition_rows = []
            for repetition_index, repetition in enumerate(repetitions, start=1):
                y_true = []
                y_pred = []
                for row in repetition:
                    rule_row = rule_rows[row["adr_id"]]
                    dq_score = rubric.quality_score(row["predicted_dq_scores"])
                    sc_score = rubric.structural_score(rule_row["rule_checks"])
                    y_true.append(rule_row["human_overall"])
                    y_pred.append(classify(sc_score, dq_score))
                repetition_rows.append({
                    "repetition": repetition_index,
                    "n": len(y_true),
                    "macro_f1": macro_f1(y_true, y_pred),
                    "cohen_kappa": cohen_kappa(y_true, y_pred),
                })
            hybrid_results.append({
                "configuration_key": f"{model}/{strategy}",
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "macro_f1": metric_summary([row["macro_f1"] for row in repetition_rows]),
                "cohen_kappa": metric_summary([row["cohen_kappa"] for row in repetition_rows]),
                "repetitions": repetition_rows,
            })

    hybrid_results.sort(key=lambda row: (-row["macro_f1"]["mean"], row["configuration_key"]))
    best_hybrid_f1 = hybrid_results[0]
    best_hybrid_kappa = max(
        hybrid_results,
        key=lambda row: (row["cohen_kappa"]["mean"], row["configuration_key"]),
    )

    llm_overall = []
    for key, config in performance["configurations"].items():
        summary = config["overall_summary"]
        llm_overall.append({
            "configuration_key": key,
            "macro_f1": summary["macro_f1"]["mean"],
            "cohen_kappa": summary["cohen_kappa"]["mean"],
        })
    best_llm_f1 = max(llm_overall, key=lambda row: row["macro_f1"])
    best_llm_kappa = max(llm_overall, key=lambda row: row["cohen_kappa"])

    llm_sc = []
    for key, config in criterion["configurations"].items():
        summary = config["dimension_confusion_matrices"]["sc_class"]["summary"]
        llm_sc.append({
            "configuration_key": key,
            "macro_f1": summary["macro_f1"]["mean"],
            "cohen_kappa": summary["cohen_kappa"]["mean"],
        })
    best_llm_sc_f1 = max(llm_sc, key=lambda row: row["macro_f1"])
    best_llm_sc_kappa = max(llm_sc, key=lambda row: row["cohen_kappa"])

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Comparable rule-only, LLM-only, and hybrid baseline analysis.",
        "method": {
            "scoring_version": rubric.SCORING_VERSION,
            "api_calls": 0,
            "n_adrs": 162,
            "rule_only_target": "Structural Completeness class",
            "overall_target": "Four-class overall compliance",
            "hybrid_definition": (
                "Rule-derived SC score plus LLM criterion-derived DQ score, combined using "
                "CS = 0.40 * SC + 0.60 * DQ and the benchmark structural floors."
            ),
            "standard_deviation": "sample SD across three repetitions",
            "important_comparability_note": (
                "The rule-only baseline does not produce a Decision Quality score and is therefore "
                "not reported as an overall-compliance classifier."
            ),
        },
        "rule_only_structural_completeness": baseline["rule_based_structural_completeness"],
        "best_llm_structural_macro_f1": best_llm_sc_f1,
        "best_llm_structural_kappa": best_llm_sc_kappa,
        "best_llm_overall_macro_f1": best_llm_f1,
        "best_llm_overall_kappa": best_llm_kappa,
        "best_hybrid_overall_macro_f1": best_hybrid_f1,
        "best_hybrid_overall_kappa": best_hybrid_kappa,
        "all_hybrid_configurations": hybrid_results,
        "legacy_hybrid_note": (
            "The older hybrid_rule_sc_plus_llm_dq section in rule_based_baseline.json combines "
            "dimension classes by selecting the lower class. It is retained for audit history but "
            "is not used here because it does not apply the benchmark composite formula."
        ),
    }

    json_path = analysis / "rule_hybrid_comparison.json"
    csv_path = analysis / "rule_hybrid_manuscript_table.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    table_rows = [
        [
            "Rule-only",
            "Structural Completeness",
            "Deterministic C1-C7 rules",
            baseline["rule_based_structural_completeness"]["macro_f1"],
            baseline["rule_based_structural_completeness"]["cohen_kappa"],
        ],
        [
            "LLM-only",
            "Structural Completeness",
            configuration_label(best_llm_sc_f1["configuration_key"]),
            best_llm_sc_f1["macro_f1"],
            next(
                row["cohen_kappa"]
                for row in llm_sc
                if row["configuration_key"] == best_llm_sc_f1["configuration_key"]
            ),
        ],
        [
            "LLM-only",
            "Overall compliance",
            configuration_label(best_llm_f1["configuration_key"]),
            best_llm_f1["macro_f1"],
            next(
                row["cohen_kappa"]
                for row in llm_overall
                if row["configuration_key"] == best_llm_f1["configuration_key"]
            ),
        ],
        [
            "LLM-only",
            "Overall compliance",
            configuration_label(best_llm_kappa["configuration_key"]),
            next(
                row["macro_f1"]
                for row in llm_overall
                if row["configuration_key"] == best_llm_kappa["configuration_key"]
            ),
            best_llm_kappa["cohen_kappa"],
        ],
        [
            "Hybrid",
            "Overall compliance",
            f"Rule SC + {best_hybrid_f1['model_label']} {best_hybrid_f1['strategy_label']} DQ",
            best_hybrid_f1["macro_f1"]["mean"],
            best_hybrid_f1["cohen_kappa"]["mean"],
        ],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Approach", "Evaluation target", "Configuration", "Macro-F1", "Kappa"])
        writer.writerows(table_rows)

    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare rule-only, LLM-only, and hybrid performance for a run."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    raise SystemExit(main(args.run_id))
