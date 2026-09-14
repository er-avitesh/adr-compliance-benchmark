"""Recompute benchmark performance under alternative ADR rubric settings.

This analysis uses saved C1-C7 and Q1-Q7 outputs. It does not call a model API.
Both human and model labels are re-derived under each scenario so the compared
labels always use the same rubric.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import statistics
import sys
from fractions import Fraction
from collections import Counter, defaultdict
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
SC_WEIGHTS = rubric.SC_WEIGHTS
DQ_WEIGHTS = rubric.DQ_WEIGHTS
EQUAL_SC_WEIGHTS = {criterion: Fraction(1, 7) for criterion in SC_WEIGHTS}
EQUAL_DQ_WEIGHTS = {criterion: Fraction(1, 7) for criterion in DQ_WEIGHTS}
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
SCENARIOS = [
    {
        "name": "Baseline 40/60",
        "key": "baseline_40_60_floor_40",
        "sc_weights": SC_WEIGHTS,
        "dq_weights": DQ_WEIGHTS,
        "sc_composite_weight": 0.40,
        "structural_floor": 40.0,
    },
    {
        "name": "Equal SC criteria",
        "key": "equal_sc_criteria",
        "sc_weights": EQUAL_SC_WEIGHTS,
        "dq_weights": DQ_WEIGHTS,
        "sc_composite_weight": 0.40,
        "structural_floor": 40.0,
    },
    {
        "name": "Equal DQ criteria",
        "key": "equal_dq_criteria",
        "sc_weights": SC_WEIGHTS,
        "dq_weights": EQUAL_DQ_WEIGHTS,
        "sc_composite_weight": 0.40,
        "structural_floor": 40.0,
    },
    {
        "name": "Balanced 50/50",
        "key": "balanced_50_50",
        "sc_weights": SC_WEIGHTS,
        "dq_weights": DQ_WEIGHTS,
        "sc_composite_weight": 0.50,
        "structural_floor": 40.0,
    },
    {
        "name": "No SC floor",
        "key": "no_structural_floor",
        "sc_weights": SC_WEIGHTS,
        "dq_weights": DQ_WEIGHTS,
        "sc_composite_weight": 0.40,
        "structural_floor": 0.0,
    },
    {
        "name": "SC floor 50",
        "key": "structural_floor_50",
        "sc_weights": SC_WEIGHTS,
        "dq_weights": DQ_WEIGHTS,
        "sc_composite_weight": 0.40,
        "structural_floor": 50.0,
    },
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sc_score(checks: dict, weights: dict) -> Fraction:
    return rubric.structural_score(checks, weights)


def dq_score(scores: dict, weights: dict) -> Fraction:
    return rubric.quality_score(scores, weights)


def classify(sc: float, dq: float, sc_weight: float, structural_floor: float) -> str:
    return rubric.overall_class(sc, dq, sc_weight, structural_floor)


def scenario_label(checks: dict, scores: dict, scenario: dict) -> str:
    sc = sc_score(checks, scenario["sc_weights"])
    dq = dq_score(scores, scenario["dq_weights"])
    return classify(sc, dq, scenario["sc_composite_weight"], scenario["structural_floor"])


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


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def rank_map(rows: list[dict], metric: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-row[metric], row["configuration_key"]))
    return {row["configuration_key"]: index + 1 for index, row in enumerate(ordered)}


def spearman_from_ranks(left: dict[str, int], right: dict[str, int]) -> float:
    keys = sorted(set(left) & set(right))
    n = len(keys)
    if n < 2:
        return float("nan")
    squared = sum((left[key] - right[key]) ** 2 for key in keys)
    return 1.0 - (6.0 * squared) / (n * (n * n - 1))


def rounded(value: float, digits=None):
    return None if not math.isfinite(value) else float(value) if digits is None else round(value, digits)


def main(run_id: str) -> int:
    run_dir = RESULTS / "runs" / run_id
    rubric.require_exact_run(run_dir)
    analysis = run_dir / "analysis"
    ground_truth = load_json(RESULTS / "human_ground_truth.json")
    eval_set = load_json(RESULTS / "eval_set.json")
    all_results = load_json(run_dir / "raw_results" / "all_results.json")
    eval_ids = [item["id"] for item in eval_set["adrs"]]
    if len(eval_ids) != 162:
        raise ValueError(f"Expected 162 evaluation ADRs, found {len(eval_ids)}")

    human_criteria = {}
    for adr_id in eval_ids:
        label = ground_truth[adr_id]
        human_criteria[adr_id] = (
            {key: label[key] for key in SC_WEIGHTS},
            {key: label[key] for key in DQ_WEIGHTS},
        )

    scenario_reports = []
    baseline_human_labels = None
    baseline_prediction_mismatches = 0
    baseline_macro_ranks = None
    baseline_kappa_ranks = None

    for scenario in SCENARIOS:
        human_labels = {
            adr_id: scenario_label(*human_criteria[adr_id], scenario)
            for adr_id in eval_ids
        }
        if baseline_human_labels is None:
            baseline_human_labels = human_labels
        human_changes = sum(
            human_labels[adr_id] != baseline_human_labels[adr_id]
            for adr_id in eval_ids
        )

        configuration_rows = []
        family_metrics = defaultdict(lambda: {"macro_f1": [], "kappa": []})
        for model, strategies in all_results.items():
            for strategy, repetitions in strategies.items():
                repetition_metrics = []
                for repetition_index, repetition in enumerate(repetitions, start=1):
                    rows_by_id = {row["adr_id"]: row for row in repetition}
                    y_true = []
                    y_pred = []
                    for adr_id in eval_ids:
                        row = rows_by_id.get(adr_id)
                        if row is None:
                            raise ValueError(f"Missing {adr_id} in {model}/{strategy}/rep{repetition_index}")
                        checks = row.get("predicted_sc_checks")
                        scores = row.get("predicted_dq_scores")
                        if not isinstance(checks, dict) or not isinstance(scores, dict):
                            raise ValueError(
                                f"Missing criteria for {adr_id} in {model}/{strategy}/rep{repetition_index}"
                            )
                        predicted = scenario_label(checks, scores, scenario)
                        y_true.append(human_labels[adr_id])
                        y_pred.append(predicted)
                        if scenario["key"] == "baseline_40_60_floor_40":
                            baseline_prediction_mismatches += predicted != row.get("predicted")
                    repetition_metrics.append({
                        "repetition": repetition_index,
                        "n": len(y_true),
                        "macro_f1": macro_f1(y_true, y_pred),
                        "cohen_kappa": cohen_kappa(y_true, y_pred),
                    })

                f1_mean, f1_sd = mean_sd([row["macro_f1"] for row in repetition_metrics])
                kappa_mean, kappa_sd = mean_sd([row["cohen_kappa"] for row in repetition_metrics])
                config_key = f"{model}/{strategy}"
                config_row = {
                    "configuration_key": config_key,
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                    "macro_f1_mean": rounded(f1_mean),
                    "macro_f1_sample_sd": rounded(f1_sd),
                    "cohen_kappa_mean": rounded(kappa_mean),
                    "cohen_kappa_sample_sd": rounded(kappa_sd),
                    "repetitions": [
                        {
                            **row,
                            "macro_f1": rounded(row["macro_f1"]),
                            "cohen_kappa": rounded(row["cohen_kappa"]),
                        }
                        for row in repetition_metrics
                    ],
                }
                configuration_rows.append(config_row)
                family_metrics[model]["macro_f1"].append(f1_mean)
                family_metrics[model]["kappa"].append(kappa_mean)

        macro_ranks = rank_map(configuration_rows, "macro_f1_mean")
        kappa_ranks = rank_map(configuration_rows, "cohen_kappa_mean")
        for row in configuration_rows:
            row["macro_f1_rank"] = macro_ranks[row["configuration_key"]]
            row["cohen_kappa_rank"] = kappa_ranks[row["configuration_key"]]
        configuration_rows.sort(key=lambda row: row["macro_f1_rank"])

        if baseline_macro_ranks is None:
            baseline_macro_ranks = macro_ranks
            baseline_kappa_ranks = kappa_ranks

        family_rows = []
        for model, metrics in family_metrics.items():
            family_rows.append({
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "macro_f1_mean_across_strategies": rounded(statistics.mean(metrics["macro_f1"])),
                "cohen_kappa_mean_across_strategies": rounded(statistics.mean(metrics["kappa"])),
            })
        family_rows.sort(key=lambda row: (-row["macro_f1_mean_across_strategies"], row["model"]))
        for index, row in enumerate(family_rows, start=1):
            row["macro_f1_family_rank"] = index

        top_f1 = min(configuration_rows, key=lambda row: row["macro_f1_rank"])
        top_kappa = min(configuration_rows, key=lambda row: row["cohen_kappa_rank"])
        scenario_reports.append({
            "scenario": scenario["name"],
            "scenario_key": scenario["key"],
            "settings": {
                "sc_weights": {key: float(value) for key, value in scenario["sc_weights"].items()},
                "dq_weights": {key: float(value) for key, value in scenario["dq_weights"].items()},
                "exact_sc_weights": {key: str(rubric.exact(value)) for key, value in scenario["sc_weights"].items()},
                "exact_dq_weights": {key: str(rubric.exact(value)) for key, value in scenario["dq_weights"].items()},
                "sc_composite_weight": scenario["sc_composite_weight"],
                "dq_composite_weight": 1.0 - scenario["sc_composite_weight"],
                "structural_floor": scenario["structural_floor"],
            },
            "human_class_distribution": {label: Counter(human_labels.values())[label] for label in CLASSES},
            "human_label_changes_vs_baseline": human_changes,
            "top_macro_f1_configuration": top_f1["configuration_key"],
            "top_macro_f1": top_f1["macro_f1_mean"],
            "top_cohen_kappa_configuration": top_kappa["configuration_key"],
            "top_cohen_kappa": top_kappa["cohen_kappa_mean"],
            "macro_f1_rank_correlation_vs_baseline": rounded(
                spearman_from_ranks(baseline_macro_ranks, macro_ranks)
            ),
            "cohen_kappa_rank_correlation_vs_baseline": rounded(
                spearman_from_ranks(baseline_kappa_ranks, kappa_ranks)
            ),
            "configuration_results": configuration_rows,
            "model_family_results": family_rows,
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Performance sensitivity to alternative criterion weights, composite weights, "
            "and structural floors using saved criterion-level human and model outputs."
        ),
        "method": {
            "scoring_version": rubric.SCORING_VERSION,
            "api_calls": 0,
            "n_adrs": len(eval_ids),
            "n_configurations": 12,
            "repetitions_per_configuration": 3,
            "labels_recomputed_for_both_sides": True,
            "macro_f1_classes": CLASSES,
            "cohen_kappa_weighting": "unweighted",
            "standard_deviation": "sample SD across three repetitions",
        },
        "validation": {
            "baseline_human_labels_matching_recorded": sum(
                baseline_human_labels[adr_id] == ground_truth[adr_id]["overall"]
                for adr_id in eval_ids
            ),
            "baseline_model_predictions_different_from_saved": baseline_prediction_mismatches,
        },
        "scenarios": scenario_reports,
    }

    analysis.mkdir(parents=True, exist_ok=True)
    json_path = analysis / "rubric_sensitivity_performance.json"
    csv_path = analysis / "rubric_sensitivity_manuscript_table.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Scenario",
            "Human distribution F/M/P/N",
            "Changed human labels",
            "Best macro-F1 configuration",
            "Macro-F1",
            "Best kappa configuration",
            "Kappa",
            "Macro-F1 rank rho",
            "Kappa rank rho",
        ])
        for report in scenario_reports:
            distribution = report["human_class_distribution"]
            writer.writerow([
                report["scenario"],
                "/".join(str(distribution[label]) for label in CLASSES),
                report["human_label_changes_vs_baseline"],
                report["top_macro_f1_configuration"],
                f"{report['top_macro_f1']:.3f}",
                report["top_cohen_kappa_configuration"],
                f"{report['top_cohen_kappa']:.3f}",
                f"{report['macro_f1_rank_correlation_vs_baseline']:.3f}",
                f"{report['cohen_kappa_rank_correlation_vs_baseline']:.3f}",
            ])

    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute rubric sensitivity from a completed experiment run."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    raise SystemExit(main(args.run_id))
