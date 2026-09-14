#!/usr/bin/env python3
"""Verify a nonpublication validation run without modifying its artifacts."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adr_benchmark as benchmark  # noqa: E402


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify(run_id: str, expected_adrs: int, expected_reps: int) -> list[str]:
    """Return validation failures; an empty list means the run is complete."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        return ["invalid run ID"]

    run_dir = ROOT / "results" / "runs" / run_id
    config_path = run_dir / "run_config.json"
    eval_path = run_dir / "eval_set.json"
    failures = []

    if not config_path.is_file():
        failures.append(f"missing {config_path.relative_to(ROOT)}")
    if not eval_path.is_file():
        failures.append(f"missing {eval_path.relative_to(ROOT)}")
    if failures:
        return failures

    config = read_json(config_path)
    evaluation = read_json(eval_path)
    expected_ids = [row.get("id") for row in evaluation.get("adrs", [])]

    try:
        benchmark.configure_run_paths(run_id)
        benchmark.configure_mistral_variant(
            config.get("active_mistral_variant", "ministral-3-8b-api")
        )
        benchmark.configure_gpt_reasoning_effort(
            config.get("gpt_reasoning_effort", "medium")
        )
        benchmark.configure_validation_mode(
            expected_adrs, int(evaluation.get("seed"))
        )
    except (OSError, TypeError, ValueError) as exc:
        failures.append(f"cannot activate validation protocol: {exc}")
        return failures

    checks = {
        "validation_mode is true": config.get("validation_mode") is True,
        "publication_compatible is false": config.get("publication_compatible") is False,
        "evaluation size matches": config.get("evaluation_set_size") == expected_adrs,
        "repetition count matches": config.get("repetitions") == expected_reps,
        "run-local manifest size matches": len(expected_ids) == expected_adrs,
        "run-local ADR IDs are unique": len(set(expected_ids)) == len(expected_ids),
        "all compliance classes are represented": (
            set(evaluation.get("class_distribution", {})) == set(benchmark.CLASSES)
        ),
        "validation protocol is identified": "-validation-" in str(
            config.get("protocol_version", "")
        ),
    }
    failures.extend(message for message, passed in checks.items() if not passed)

    raw_dir = run_dir / "raw_results"
    complete_files = 0
    total_rows = 0
    for model_name in benchmark.MODELS:
        for strategy in benchmark.STRATEGIES:
            relative = Path("raw_results") / f"{model_name}_{strategy}.json"
            path = run_dir / relative
            label = f"{model_name}/{strategy}"
            if not path.is_file():
                failures.append(f"{label}: missing {relative.as_posix()}")
                continue
            data = read_json(path)
            valid, reason = benchmark._validate_result_reps(
                data, expected_ids, expected_reps, require_complete=True
            )
            if not valid:
                failures.append(f"{label}: {reason}")
                continue
            complete_files += 1
            total_rows += sum(len(rep) for rep in data)

    expected_files = len(benchmark.MODELS) * len(benchmark.STRATEGIES)
    expected_rows = expected_files * expected_reps * expected_adrs
    if complete_files != expected_files:
        failures.append(
            f"complete configurations: {complete_files}/{expected_files}"
        )
    if total_rows != expected_rows:
        failures.append(f"validated rows: {total_rows}/{expected_rows}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-adrs", type=int, default=5)
    parser.add_argument("--expected-reps", type=int, default=3)
    args = parser.parse_args()

    failures = verify(args.run_id, args.expected_adrs, args.expected_reps)
    if failures:
        print("VALIDATION RUN INCOMPLETE")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    configurations = len(benchmark.MODELS) * len(benchmark.STRATEGIES)
    rows = configurations * args.expected_reps * args.expected_adrs
    print("VALIDATION RUN COMPLETE")
    print(f"  Run ID: {args.run_id}")
    print(f"  Configurations: {configurations}")
    print(f"  Repetitions: {args.expected_reps}")
    print(f"  ADRs per repetition: {args.expected_adrs}")
    print(f"  Validated rows: {rows}")
    print("  Publication compatible: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
