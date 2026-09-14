#!/usr/bin/env python3
"""Regenerate publication tables and figures from a saved exact-scoring run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MPL_CACHE = Path(tempfile.gettempdir()) / "adr_benchmark_matplotlib_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adr_scoring as rubric
from scripts.rescore_saved_results import run_path


MODELS = ("gpt-5.5", "claude-sonnet-4-6", "gemini-2.5-pro", "ministral-3-8b")
MODEL_NAMES = {
    "gpt-5.5": "GPT-5.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "ministral-3-8b": "Ministral 3 8B",
}
STRATEGIES = ("zero_shot", "few_shot", "chain_of_thought")
STRATEGY_NAMES = {"zero_shot": "ZS", "few_shot": "FS", "chain_of_thought": "CoT"}
LABELS = ("Fully_Compliant", "Mostly_Compliant", "Partially_Compliant", "Not_Compliant")
COLORS = ("#24557A", "#2F7D6E", "#B07A20", "#A64B4B")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def configuration_name(key: str) -> str:
    model, strategy = key.split("/")
    return f"{MODEL_NAMES[model]} {STRATEGY_NAMES[strategy]}"


def performance_rows(performance: dict) -> list[list[str]]:
    rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            summary = performance[f"{model}/{strategy}"]["overall_summary"]

            def cell(name: str) -> str:
                return f"{summary[name]['mean']:.3f} ({summary[name]['std']:.3f})"

            rows.append(
                [
                    MODEL_NAMES[model],
                    STRATEGY_NAMES[strategy],
                    cell("macro_f1"),
                    cell("macro_precision"),
                    cell("macro_recall"),
                    cell("cohen_kappa"),
                ]
            )
    rows.append(["Majority baseline", "Always Fully", "0.140", "0.097", "0.250", "0.000"])
    return rows


def sensitivity_rows(scenarios: list[dict]) -> list[list[str]]:
    rows = []
    for item in scenarios:
        rows.append(
            [
                item["scenario"],
                str(item["human_label_changes_vs_baseline"]),
                f"{configuration_name(item['top_macro_f1_configuration'])}, {item['top_macro_f1']:.3f}",
                f"{configuration_name(item['top_cohen_kappa_configuration'])}, {item['top_cohen_kappa']:.3f}",
                f"{item['macro_f1_rank_correlation_vs_baseline']:.3f} / "
                f"{item['cohen_kappa_rank_correlation_vs_baseline']:.3f}",
            ]
        )
    return rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate(run: Path, output: Path) -> None:
    rubric.require_exact_run(run)
    analysis = run / "analysis"
    evidence_path = analysis / "manuscript_evidence" / "evidence.json"
    input_paths = {
        "performance": analysis / "performance_details.json",
        "criteria": analysis / "criterion_level_performance.json",
        "sensitivity": analysis / "rubric_sensitivity_performance.json",
        "pairwise": analysis / "pairwise_statistical_tests.json",
        "evidence": evidence_path,
    }
    missing = [path for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing analysis inputs: {missing}")

    performance = read_json(input_paths["performance"])["configurations"]
    criteria = read_json(input_paths["criteria"])["configurations"]
    scenarios = read_json(input_paths["sensitivity"])["scenarios"]
    evidence = read_json(evidence_path)
    if evidence.get("scoring_version") != rubric.SCORING_VERSION:
        raise ValueError("Evidence was not generated with exact rational scoring")

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(3.55, 2.2), layout="constrained")
    for row, key in enumerate(("overall", "sc_class", "dq_class")):
        left = 0
        for column, label in enumerate(LABELS):
            count = evidence["human_distributions"][key].get(label, 0)
            ax.barh(row, count, left=left, color=COLORS[column], height=0.6, edgecolor="white")
            ax.text(left + count / 2, row, str(count), ha="center", va="center", color="white", fontsize=7, weight="bold")
            left += count
    ax.set(yticks=range(3), yticklabels=("Overall", "SC", "DQ"), xlim=(0, 162), xlabel="ADRs")
    ax.invert_yaxis()
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS],
        ("Fully", "Mostly", "Partially", "Not"),
        ncol=4,
        frameon=False,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
    )
    fig.savefig(output / "figure_1_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.55, 2.55), layout="constrained")
    x = np.arange(len(MODELS))
    width = 0.23
    prompt_colors = ("#24557A", "#2F7D6E", "#B07A20")
    for index, strategy in enumerate(STRATEGIES):
        values = [performance[f"{model}/{strategy}"]["overall_summary"]["macro_f1"]["mean"] for model in MODELS]
        bars = ax.bar(x + (index - 1) * width, values, width, label=STRATEGY_NAMES[strategy], color=prompt_colors[index])
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.2f}", ha="center", va="bottom", fontsize=5.7)
    ax.axhline(0.14, color="#555555", linewidth=0.9, linestyle="--", label="Majority")
    ax.set_xticks(x, ("GPT-5.5", "Claude", "Gemini", "Ministral"), fontsize=6.8)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 0.64)
    ax.legend(ncol=4, frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.savefig(output / "figure_2_performance.png", dpi=300)
    plt.close(fig)

    dimensions = (("overall", "Overall"), ("sc_class", "SC"), ("dq_class", "DQ"))
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.75), layout="constrained")
    for ax, (key, label) in zip(axes.ravel(), dimensions):
        counts = np.asarray(
            criteria["gpt-5.5/few_shot"]["dimension_confusion_matrices"][key]["summed_confusion_matrix"]["matrix"]
        )
        row_normalized = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(row_normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (row, column), value in np.ndenumerate(counts):
            ax.text(
                column,
                row,
                f"{value}\n{row_normalized[row, column] * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=4.8,
                color="white" if row_normalized[row, column] > 0.53 else "black",
            )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=("F", "M", "P", "N"),
            yticklabels=("F", "M", "P", "N"),
            xlabel="Pred.",
            ylabel="Human",
            title=label,
        )
        ax.tick_params(length=0)
    axes.ravel()[3].axis("off")
    fig.savefig(output / "figure_3_confusion.png", dpi=300)
    plt.close(fig)

    configurations = [f"{model}/{strategy}" for model in MODELS for strategy in STRATEGIES]
    matrix = []
    for key in configurations:
        item = criteria[key]
        matrix.append(
            [item["structural_criteria"][f"C{i}"]["summary"]["accuracy"]["mean"] for i in range(1, 8)]
            + [item["decision_quality_criteria"][f"Q{i}"]["summary"]["exact_accuracy"]["mean"] for i in range(1, 8)]
        )
    fig, ax = plt.subplots(figsize=(3.55, 3.2), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis", aspect="auto")
    for (row, column), value in np.ndenumerate(np.asarray(matrix)):
        ax.text(column, row, f"{value * 100:.0f}", ha="center", va="center", fontsize=4.2, color="white" if value < 0.48 else "black")
    ax.axvline(6.5, color="white", linewidth=1.7)
    ax.set_xticks(range(14), [f"C{i}" for i in range(1, 8)] + [f"Q{i}" for i in range(1, 8)], fontsize=5.3)
    labels = [
        configuration_name(key)
        .replace("Claude Sonnet 4.6", "Claude")
        .replace("Gemini 2.5 Pro", "Gemini")
        .replace("Ministral 3 8B", "Ministral")
        for key in configurations
    ]
    ax.set_yticks(range(12), labels, fontsize=4.8)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Exact agreement", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5)
    fig.savefig(output / "figure_4_criteria.png", dpi=300)
    plt.close(fig)

    write_csv(
        output / "overall_performance.csv",
        ["Model", "Prompt", "Macro-F1 (sample SD)", "Precision (sample SD)", "Recall (sample SD)", "Kappa (sample SD)"],
        performance_rows(performance),
    )
    write_csv(
        output / "sensitivity.csv",
        ["Scenario", "Changed human labels", "Best macro-F1", "Best kappa", "Rank correlation F1 / kappa"],
        sensitivity_rows(scenarios),
    )

    provenance = {
        "run_id": run.name,
        "scoring_version": rubric.SCORING_VERSION,
        "generator": "scripts/generate_publication_assets.py",
        "generator_sha256": digest(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "input_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in input_paths.values()},
    }
    (output / "asset_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v31_fulltext_gpt_medium_exact_scoring")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate destination; canonical assets are never overwritten")
    parser.add_argument("--replace-canonical", action="store_true", help="Maintainer-only: explicitly refresh the canonical asset directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    canonical = run_path(args.run_id) / "analysis" / "manuscript_assets"
    if output == canonical.resolve() and not args.replace_canonical:
        parser.error("Choose a separate output directory; canonical assets are read-only evidence")
    generate(run_path(args.run_id), output)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
