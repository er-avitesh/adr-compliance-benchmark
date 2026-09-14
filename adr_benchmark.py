#!/usr/bin/env python3
"""
ADR COMPLIANCE BENCHMARK — Experiment Runner
=============================================
4 models × 3 strategies × 3 reps × 162 ADRs = 5,832 calls.
Eval set is drawn via stratified sampling (compliance class x template
variant x domain) — see select_eval_adrs() — rather than ranked by the
GPT-4o preliminary prelabel score, so the sample isn't skewed toward
already-compliant ADRs.

Each model/strategy pair saves its own result file independently so runs can be
split across sessions. Use --phase merge to combine and analyze when ready.

Usage:
  export OPENAI_API_KEY="sk-..."   ANTHROPIC_API_KEY="sk-ant-..."
  export MISTRAL_API_KEY="..."     GEMINI_API_KEY="..."

  # Step by step
  python adr_benchmark.py --phase fetch    --n-eval 162
  python adr_benchmark.py --phase annotate
  python adr_benchmark.py --phase freeze   --run-id v31_fulltext_gpt_medium
  python adr_benchmark.py --phase preflight --run-id v31_fulltext_gpt_medium
  python adr_benchmark.py --phase repro    # model/API reproducibility metadata
  python adr_benchmark.py --phase smoke    # one pre-flight API call per model/strategy
  python adr_benchmark.py --phase run      --n-eval 162   # all models × strategies
  python adr_benchmark.py --phase merge                   # combine files + analyze

  # Targeted run — one or several model/strategy pairs
  python adr_benchmark.py --run gemini-2.5-pro/zero_shot --n-eval 162
  python adr_benchmark.py --run gemini-2.5-pro/zero_shot,gpt-5.5/chain_of_thought
  python adr_benchmark.py --phase smoke
"""

import os
import sys
import json
import hashlib
import time
import re
import random
import argparse
import shutil
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
from fractions import Fraction
import adr_scoring as rubric


def _parse_local_env_line(line: str):
    """Parse standard or PowerShell-style dotenv syntax without exposing values."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if key.lower().startswith("$env:"):
        key = key[5:].strip()
    value = value.strip().strip('"').strip("'")
    return (key, value) if key else None


def load_local_env(env_path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries from a local .env file if present."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_local_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_DIR = Path("results")
RUNS_DIR = EXPERIMENT_DIR / "runs"
ADRS_DIR = EXPERIMENT_DIR / "adrs"
PUBLICATION_EVAL_SET_PATH = EXPERIMENT_DIR / "eval_set.json"
EVAL_SET_PATH = PUBLICATION_EVAL_SET_PATH
GROUND_TRUTH_PATH = EXPERIMENT_DIR / "human_ground_truth.json"
DEFAULT_RUN_ID = "v31_fulltext_gpt_medium"
ACTIVE_RUN_ID = os.environ.get("ADR_RUN_ID", DEFAULT_RUN_ID)
RUN_DIR = RUNS_DIR / ACTIVE_RUN_ID
RUN_CONFIG_PATH = RUN_DIR / "run_config.json"
RESULTS_DIR = RUN_DIR / "raw_results"
ANALYSIS_DIR = RUN_DIR / "analysis"
SMOKE_TESTS_DIR = RUN_DIR / "smoke_tests"
BATCH_DIR = RUN_DIR / "batch_jobs"
BATCH_REQUESTS_DIR = BATCH_DIR / "requests"
BATCH_OUTPUTS_DIR = BATCH_DIR / "outputs"
BATCH_MANIFEST_PATH = BATCH_DIR / "batch_manifest.json"
PROMPT_MANIFEST_PATH = ANALYSIS_DIR / "prompt_manifest.json"
MODEL_REPRODUCIBILITY_PATH = ANALYSIS_DIR / "model_reproducibility.json"
PROMPT_LEAKAGE_AUDIT_PATH = ANALYSIS_DIR / "prompt_leakage_audit.json"
RULE_BASELINE_PATH = ANALYSIS_DIR / "rule_based_baseline.json"
PAIRWISE_STATS_PATH = ANALYSIS_DIR / "pairwise_statistical_tests.json"
PERFORMANCE_DETAILS_PATH = ANALYSIS_DIR / "performance_details.json"
CRITERION_PERFORMANCE_PATH = ANALYSIS_DIR / "criterion_level_performance.json"
ERROR_ANALYSIS_PATH = ANALYSIS_DIR / "error_analysis.json"
COST_ANALYSIS_PATH = ANALYSIS_DIR / "cost_analysis.json"
RUBRIC_MANIFEST_PATH = ANALYSIS_DIR / "rubric_manifest.json"
THRESHOLD_SENSITIVITY_PATH = ANALYSIS_DIR / "threshold_sensitivity.json"
MANUSCRIPT_DATASET_SUMMARY_PATH = ANALYSIS_DIR / "manuscript_dataset_summary.json"
MANUSCRIPT_RESULTS_SUMMARY_PATH = ANALYSIS_DIR / "manuscript_results_summary.json"
PREFLIGHT_REPORT_PATH = ANALYSIS_DIR / "preflight_report.json"

N_EVAL = 162         # Frozen evaluation-set size used by the reported benchmark.
N_REPS = 3           # Frozen repetition count used by the manuscript.
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
ADR_INPUT_POLICY = "full_text"
RESULT_SCHEMA_VERSION = 4
PROTOCOL_VERSION = "v31-fulltext-criterion-derived-1"
ACTIVE_PROTOCOL_VERSION = PROTOCOL_VERSION
ACTIVE_N_EVAL = N_EVAL
ACTIVE_N_REPS = N_REPS
VALIDATION_MODE = False
PRICING_VERIFIED_DATE = "2026-09-10"
FROZEN_INPUT_TEXT_TRANSFORMS = {
    "loopdive_js2_0001-hybrid-compilation-strategy": "utf8-bytes-decoded-as-windows-1252",
    "loopdive_js2_0004-aot": "utf8-bytes-decoded-as-windows-1252",
}
CACHED_TOKEN_TREATMENT = (
    "No prompt cache was explicitly configured. Retained OpenAI response metadata "
    "records automatic cached input tokens. The estimate applies the full applicable "
    "input rate to all input tokens and does not apply a cached-input discount."
)
UNRETAINED_USAGE_TREATMENT = (
    "Unsuccessful requests without retained provider usage metadata are excluded."
)

# Fixed few-shot exemplars are held out from results/eval_set.json. They were
# selected to cover all four final compliance classes and are documented in
# The run-specific prompt manifest records these for reproducibility.
FEW_SHOT_EXEMPLAR_IDS = [
    "alphagov_govuk-aws_0012-security-groups-in-terraform",  # Fully_Compliant
    "adr_madr_0001-use-CC0-or-MIT-as-license",               # Mostly_Compliant
    "argoproj_argo-cd_deep-links",                           # Partially_Compliant
    "synthetic_not_compliant_v1",                            # Not_Compliant
]

SYNTHETIC_FEW_SHOT_EXEMPLARS = {
    "synthetic_not_compliant_v1": {
        "id": "synthetic_not_compliant_v1",
        "text": "# ADR-001\n\nWe made a decision.\n",
        "label": "Not_Compliant",
        "label_payload": {
            "C1": False, "C2": False, "C3": False, "C4": False,
            "C5": False, "C6": False, "C7": False,
            "Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0,
            "Q5": 0, "Q6": 0, "Q7": 0,
        },
        "source_repo": "synthetic-control",
        "variant": "SYNTHETIC",
        "domain": "Controlled negative exemplar",
    },
}


def configure_run_paths(run_id: str) -> str:
    """Route mutable outputs to an isolated, validated run directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id or ""):
        raise ValueError(
            "run_id must contain only letters, numbers, dot, underscore, or dash"
        )

    global ACTIVE_RUN_ID, RUN_DIR, RUN_CONFIG_PATH, RESULTS_DIR, ANALYSIS_DIR, SMOKE_TESTS_DIR
    global BATCH_DIR, BATCH_REQUESTS_DIR, BATCH_OUTPUTS_DIR, BATCH_MANIFEST_PATH
    global PROMPT_MANIFEST_PATH, MODEL_REPRODUCIBILITY_PATH
    global PROMPT_LEAKAGE_AUDIT_PATH, RULE_BASELINE_PATH, PAIRWISE_STATS_PATH
    global PERFORMANCE_DETAILS_PATH, CRITERION_PERFORMANCE_PATH
    global ERROR_ANALYSIS_PATH, COST_ANALYSIS_PATH, RUBRIC_MANIFEST_PATH
    global THRESHOLD_SENSITIVITY_PATH, MANUSCRIPT_DATASET_SUMMARY_PATH
    global MANUSCRIPT_RESULTS_SUMMARY_PATH, PREFLIGHT_REPORT_PATH
    global EVAL_SET_PATH, ACTIVE_PROTOCOL_VERSION, ACTIVE_N_EVAL, ACTIVE_N_REPS
    global VALIDATION_MODE

    ACTIVE_RUN_ID = run_id
    RUN_DIR = RUNS_DIR / run_id
    RUN_CONFIG_PATH = RUN_DIR / "run_config.json"
    RESULTS_DIR = RUN_DIR / "raw_results"
    ANALYSIS_DIR = RUN_DIR / "analysis"
    SMOKE_TESTS_DIR = RUN_DIR / "smoke_tests"
    BATCH_DIR = RUN_DIR / "batch_jobs"
    BATCH_REQUESTS_DIR = BATCH_DIR / "requests"
    BATCH_OUTPUTS_DIR = BATCH_DIR / "outputs"
    BATCH_MANIFEST_PATH = BATCH_DIR / "batch_manifest.json"
    PROMPT_MANIFEST_PATH = ANALYSIS_DIR / "prompt_manifest.json"
    MODEL_REPRODUCIBILITY_PATH = ANALYSIS_DIR / "model_reproducibility.json"
    PROMPT_LEAKAGE_AUDIT_PATH = ANALYSIS_DIR / "prompt_leakage_audit.json"
    RULE_BASELINE_PATH = ANALYSIS_DIR / "rule_based_baseline.json"
    PAIRWISE_STATS_PATH = ANALYSIS_DIR / "pairwise_statistical_tests.json"
    PERFORMANCE_DETAILS_PATH = ANALYSIS_DIR / "performance_details.json"
    CRITERION_PERFORMANCE_PATH = ANALYSIS_DIR / "criterion_level_performance.json"
    ERROR_ANALYSIS_PATH = ANALYSIS_DIR / "error_analysis.json"
    COST_ANALYSIS_PATH = ANALYSIS_DIR / "cost_analysis.json"
    RUBRIC_MANIFEST_PATH = ANALYSIS_DIR / "rubric_manifest.json"
    THRESHOLD_SENSITIVITY_PATH = ANALYSIS_DIR / "threshold_sensitivity.json"
    MANUSCRIPT_DATASET_SUMMARY_PATH = ANALYSIS_DIR / "manuscript_dataset_summary.json"
    MANUSCRIPT_RESULTS_SUMMARY_PATH = ANALYSIS_DIR / "manuscript_results_summary.json"
    PREFLIGHT_REPORT_PATH = ANALYSIS_DIR / "preflight_report.json"
    EVAL_SET_PATH = PUBLICATION_EVAL_SET_PATH
    ACTIVE_PROTOCOL_VERSION = PROTOCOL_VERSION
    ACTIVE_N_EVAL = N_EVAL
    ACTIVE_N_REPS = N_REPS
    VALIDATION_MODE = False
    return ACTIVE_RUN_ID


def configure_validation_mode(size: int, seed: int) -> Dict:
    """Create or verify a deterministic run-local subset for functional testing."""
    if ACTIVE_RUN_ID == DEFAULT_RUN_ID:
        raise ValueError("Validation mode requires a new, noncanonical --run-id")
    if size < len(CLASSES):
        raise ValueError(
            f"validation size must be at least {len(CLASSES)} to cover every class"
        )

    source = json.loads(PUBLICATION_EVAL_SET_PATH.read_text(encoding="utf-8"))
    source_rows = source.get("adrs", [])
    if size > len(source_rows):
        raise ValueError(
            f"validation size {size} exceeds the frozen evaluation set ({len(source_rows)})"
        )

    rng = random.Random(seed)
    rows_by_class = {
        class_name: [
            row for row in source_rows if row.get("ground_truth") == class_name
        ]
        for class_name in CLASSES
    }
    missing_classes = [name for name, rows in rows_by_class.items() if not rows]
    if missing_classes:
        raise ValueError(
            "Frozen evaluation set cannot support class-covered validation: "
            + ", ".join(missing_classes)
        )

    selected_ids = {
        rng.choice(rows_by_class[class_name])["id"] for class_name in CLASSES
    }
    remaining = [row for row in source_rows if row["id"] not in selected_ids]
    rng.shuffle(remaining)
    selected_ids.update(row["id"] for row in remaining[: size - len(selected_ids)])
    selected = [row for row in source_rows if row["id"] in selected_ids]

    source_hash = _sha256_file(PUBLICATION_EVAL_SET_PATH)
    payload = {
        "selected_at": None,
        "n": len(selected),
        "seed": seed,
        "method": "deterministic class-covered subset of frozen publication evaluation set",
        "validation_mode": True,
        "publication_compatible": False,
        "source_evaluation_manifest": str(PUBLICATION_EVAL_SET_PATH),
        "source_evaluation_set_sha256": source_hash,
        "class_distribution": dict(Counter(row["ground_truth"] for row in selected)),
        "heldout_few_shot_exemplar_ids": FEW_SHOT_EXEMPLAR_IDS,
        "source_declared_heldout_ids": source.get(
            "heldout_few_shot_exemplar_ids", []
        ),
        "adrs": selected,
    }

    validation_path = RUN_DIR / "eval_set.json"
    if validation_path.exists():
        saved = json.loads(validation_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "n", "seed", "method", "validation_mode", "publication_compatible",
            "source_evaluation_set_sha256", "class_distribution", "adrs",
        )
        if any(saved.get(key) != payload.get(key) for key in comparable_keys):
            raise ValueError(
                f"Run ID '{ACTIVE_RUN_ID}' already has a different validation subset"
            )
        payload = saved
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        payload["selected_at"] = datetime.now().isoformat()
        validation_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    global EVAL_SET_PATH, ACTIVE_PROTOCOL_VERSION, ACTIVE_N_EVAL, ACTIVE_N_REPS
    global VALIDATION_MODE
    EVAL_SET_PATH = validation_path
    ACTIVE_PROTOCOL_VERSION = f"{PROTOCOL_VERSION}-validation-n{size}-r{N_REPS}"
    ACTIVE_N_EVAL = size
    ACTIVE_N_REPS = N_REPS
    VALIDATION_MODE = True
    return payload

MODELS = {
    "gpt-5.5": {
        "provider": "openai",
        "model": "gpt-5.5-2026-04-23",
        "use_max_completion_tokens": True,
        "max_completion_tokens": 8192,
        "reasoning_effort": "medium",
        "context_window_tokens": 1050000,
        "no_temperature": True,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "context_window_tokens": 1000000,
        "no_temperature": True,
    },
    "mistral-7b": {
        "provider": "openai",  # OpenAI-compatible local/vLLM endpoint
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_url": os.environ.get("MISTRAL_BASE_URL", "http://localhost:8000/v1"),
        "api_key_env": "MISTRAL_API_KEY",
        "api_key_default": "EMPTY",
        "context_window_tokens": 32768,
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "model": "gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
        "max_output_tokens": 8192,  # thinking model needs headroom for reasoning tokens
        "context_window_tokens": 1048576,
    },
}

STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]
CLASSES = ["Fully_Compliant", "Mostly_Compliant", "Partially_Compliant", "Not_Compliant"]
SC_WEIGHTS = {
    "C1": 0.10,
    "C2": 0.15,
    "C3": 0.15,
    "C4": 0.15,
    "C5": 0.20,
    "C6": 0.15,
    "C7": 0.10,
}
DQ_WEIGHTS = {
    "Q1": 0.10,
    "Q2": 0.15,
    "Q3": 0.15,
    "Q4": 0.10,
    "Q5": 0.25,
    "Q6": 0.15,
    "Q7": 0.10,
}

STRUCTURAL_CRITERIA = {
    "C1": "Title present and descriptive",
    "C2": "Context or problem statement articulated",
    "C3": "Decision drivers listed",
    "C4": "At least two considered options",
    "C5": "Decision outcome with justification",
    "C6": "Consequences documented",
    "C7": "Status field present and valid",
}

DECISION_QUALITY_CRITERIA = {
    "Q1": "Problem relevance",
    "Q2": "Option viability, not strawmen",
    "Q3": "Criteria completeness",
    "Q4": "Criteria prioritization",
    "Q5": "Rationale soundness",
    "Q6": "Consequence objectivity",
    "Q7": "Actionability",
}

DQ_SCORE_ANCHORS = {
    0: {
        "label": "Absent",
        "meaning": "No document evidence supports the criterion.",
    },
    1: {
        "label": "Weak",
        "meaning": "The criterion is mentioned but is generic, superficial, or unsupported.",
    },
    2: {
        "label": "Adequate",
        "meaning": "The criterion is addressed with project-specific reasoning, with some omissions or limited depth.",
    },
    3: {
        "label": "Strong",
        "meaning": "The criterion is addressed clearly, specifically, and with defensible architectural reasoning.",
    },
}

COST_ASSUMPTIONS_USD_PER_MILLION_TOKENS = {
    "gpt-5.5": {
        "input": 5.00,
        "output": 30.00,
        "basis": "OpenAI API standard token pricing",
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.5",
        "pricing_date": PRICING_VERIFIED_DATE,
        "pricing_verified_date": PRICING_VERIFIED_DATE,
        "batch_multiplier": 0.5,
        "batch_input": 2.50,
        "batch_output": 15.00,
        "batch_pricing_source": "https://platform.openai.com/docs/api-reference/batch/object",
        "subscription_or_tier": "API pay-as-you-go or configured project tier",
        "cached_token_treatment": CACHED_TOKEN_TREATMENT,
        "local_compute_assumption": "Not applicable",
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "basis": "Anthropic API standard token pricing",
        "pricing_source": "https://www.anthropic.com/claude/sonnet",
        "pricing_date": PRICING_VERIFIED_DATE,
        "pricing_verified_date": PRICING_VERIFIED_DATE,
        "batch_multiplier": 0.5,
        "batch_input": 1.50,
        "batch_output": 7.50,
        "batch_pricing_source": "https://www.anthropic.com/claude/sonnet",
        "subscription_or_tier": "API pay-as-you-go or configured project tier",
        "cached_token_treatment": CACHED_TOKEN_TREATMENT,
        "local_compute_assumption": "Not applicable",
    },
    "mistral-7b": {
        "input": 0.00,
        "output": 0.00,
        "basis": "Local/vLLM deployment; API cost treated as zero",
        "pricing_source": "Local deployment assumption, not managed API pricing",
        "pricing_date": "Not applicable",
        "pricing_verified_date": "Not applicable",
        "batch_multiplier": None,
        "batch_input": None,
        "batch_output": None,
        "batch_pricing_source": None,
        "subscription_or_tier": "Local inference",
        "cached_token_treatment": "Not applicable",
        "local_compute_assumption": "API-token cost is zero; hardware and electricity costs must be reported separately if included",
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "basis": (
            "Google Gemini Developer API paid-tier pricing for prompts below "
            "200,000 tokens"
        ),
        "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "pricing_date": PRICING_VERIFIED_DATE,
        "pricing_verified_date": PRICING_VERIFIED_DATE,
        "batch_multiplier": 0.5,
        "batch_input": 0.625,
        "batch_output": 5.00,
        "batch_pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "subscription_or_tier": "API pay-as-you-go or configured project tier",
        "cached_token_treatment": CACHED_TOKEN_TREATMENT,
        "local_compute_assumption": "Not applicable",
    },
}

MODEL_REPRODUCIBILITY_NOTES = {
    "gpt-5.5": {
        "paper_label": "GPT-5.5",
        "provider": "OpenAI",
        "access_mode": "OpenAI Chat Completions API",
        "api_version": "SDK default; no explicit API version parameter in request",
        "endpoint_or_deployment": "OpenAI hosted endpoint",
        "temperature": "provider default; temperature parameter not sent",
        "top_p": "provider default; top_p parameter not sent",
        "max_output": "max_completion_tokens=8192",
        "structured_output_mode": "Prompt-enforced JSON only; no provider-native structured-output mode",
        "reasoning_setting": "reasoning_effort=medium",
    },
    "claude-sonnet-4-6": {
        "paper_label": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "access_mode": "Anthropic Messages API",
        "api_version": "SDK default; no explicit API version parameter in request",
        "endpoint_or_deployment": "Anthropic hosted endpoint",
        "temperature": "provider default; temperature parameter not sent",
        "top_p": "provider default; top_p parameter not sent",
        "max_output": "max_tokens=4096",
        "structured_output_mode": "Prompt-enforced JSON only; no provider-native structured-output mode",
        "reasoning_setting": "Provider default; no explicit thinking/reasoning parameter sent",
    },
    "mistral-7b": {
        "paper_label": "Mistral 7B",
        "provider": "OpenAI-compatible local/vLLM endpoint",
        "access_mode": "OpenAI-compatible Chat Completions API",
        "api_version": "Local server API compatibility mode",
        "endpoint_or_deployment": "MISTRAL_BASE_URL or http://localhost:8000/v1",
        "temperature": "0",
        "top_p": "provider default; top_p parameter not sent",
        "max_output": "max_tokens=1024",
        "structured_output_mode": "Prompt-enforced JSON only; no provider-native structured-output mode",
        "reasoning_setting": "None",
    },
    "gemini-2.5-pro": {
        "paper_label": "Gemini 2.5 Pro",
        "provider": "Google",
        "access_mode": "Google GenAI generate_content API",
        "api_version": "SDK default; no explicit API version parameter in request",
        "endpoint_or_deployment": "Google hosted endpoint",
        "temperature": "provider default; temperature parameter not sent",
        "top_p": "provider default; top_p parameter not sent",
        "max_output": "max_output_tokens=8192",
        "structured_output_mode": "Prompt-enforced JSON only; no provider-native structured-output mode",
        "reasoning_setting": "Provider default; no explicit thinking/reasoning parameter sent",
    },
}

MISTRAL_MODEL_VARIANTS = {
    "mistral-7b-local": {
        "model_key": "mistral-7b",
        "model": {
            "provider": "openai",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "base_url": os.environ.get("MISTRAL_BASE_URL", "http://localhost:8000/v1"),
            "api_key_env": "MISTRAL_API_KEY",
            "api_key_default": "EMPTY",
            "context_window_tokens": 32768,
        },
        "cost": {
            "input": 0.00,
            "output": 0.00,
            "basis": "Local/vLLM deployment; API cost treated as zero",
            "pricing_source": "Local deployment assumption, not managed API pricing",
            "pricing_date": "Not applicable",
            "pricing_verified_date": "Not applicable",
            "batch_multiplier": None,
            "batch_input": None,
            "batch_output": None,
            "batch_pricing_source": None,
            "subscription_or_tier": "Local inference",
            "cached_token_treatment": "Not applicable",
            "local_compute_assumption": "API-token cost is zero; hardware and electricity costs must be reported separately if included",
        },
        "notes": {
            "paper_label": "Mistral 7B",
            "provider": "OpenAI-compatible local/vLLM endpoint",
            "access_mode": "OpenAI-compatible Chat Completions API",
            "api_version": "Local server API compatibility mode",
            "endpoint_or_deployment": "MISTRAL_BASE_URL or http://localhost:8000/v1",
            "temperature": "0",
            "top_p": "provider default; top_p parameter not sent",
            "max_output": "max_tokens=1024",
            "structured_output_mode": "Prompt-enforced JSON only; no provider-native structured-output mode",
            "reasoning_setting": "None",
        },
    },
    "ministral-3-8b-api": {
        "model_key": "ministral-3-8b",
        "model": {
            "provider": "openai",
            "model": "ministral-8b-2512",
            "base_url": os.environ.get("MINISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            "api_key_env": "MISTRAL_API_KEY",
            "max_tokens": 2048,
            "context_window_tokens": 256000,
        },
        "cost": {
            "input": 0.15,
            "output": 0.15,
            "basis": "Hosted Mistral API standard token pricing for Ministral 3 8B",
            "pricing_source": "https://docs.mistral.ai/models/ministral-3-8b-25-12",
            "pricing_date": PRICING_VERIFIED_DATE,
            "pricing_verified_date": PRICING_VERIFIED_DATE,
            "batch_multiplier": None,
            "batch_input": None,
            "batch_output": None,
            "batch_pricing_source": None,
            "subscription_or_tier": "Mistral API pay-as-you-go or configured workspace tier",
            "cached_token_treatment": CACHED_TOKEN_TREATMENT,
            "local_compute_assumption": "Not applicable",
        },
        "notes": {
            "paper_label": "Ministral 3 8B",
            "provider": "Mistral AI",
            "access_mode": "Hosted Mistral Chat Completions API",
            "api_version": "Mistral v1 chat completions API through OpenAI-compatible client",
            "endpoint_or_deployment": "https://api.mistral.ai/v1/chat/completions, or MINISTRAL_BASE_URL override",
            "temperature": "0",
            "top_p": "provider default; top_p parameter not sent",
            "max_output": "max_tokens=2048",
            "structured_output_mode": "Prompt-enforced JSON only; provider-native structured-output mode not enabled",
            "reasoning_setting": "None",
        },
    },
}
MISTRAL_VARIANT_KEYS = {
    variant["model_key"] for variant in MISTRAL_MODEL_VARIANTS.values()
}
ACTIVE_MISTRAL_VARIANT = "ministral-3-8b-api"


def _insert_model_before_gemini(model_key: str, model_cfg: Dict) -> None:
    """Keep model ordering stable when switching Mistral variants."""
    existing = {
        key: value
        for key, value in MODELS.items()
        if key not in MISTRAL_VARIANT_KEYS
    }
    ordered = {}
    inserted = False
    for key, value in existing.items():
        if key == "gemini-2.5-pro":
            ordered[model_key] = model_cfg
            inserted = True
        ordered[key] = value
    if not inserted:
        ordered[model_key] = model_cfg
    MODELS.clear()
    MODELS.update(ordered)


def configure_mistral_variant(variant_name: str) -> str:
    """Select local Mistral 7B or hosted Ministral 3 8B for this process."""
    global ACTIVE_MISTRAL_VARIANT
    if variant_name not in MISTRAL_MODEL_VARIANTS:
        choices = ", ".join(MISTRAL_MODEL_VARIANTS)
        raise ValueError(f"Unknown Mistral variant '{variant_name}'. Choose one of: {choices}")

    variant = MISTRAL_MODEL_VARIANTS[variant_name]
    for key in MISTRAL_VARIANT_KEYS:
        MODELS.pop(key, None)
        COST_ASSUMPTIONS_USD_PER_MILLION_TOKENS.pop(key, None)
        MODEL_REPRODUCIBILITY_NOTES.pop(key, None)

    model_key = variant["model_key"]
    _insert_model_before_gemini(model_key, dict(variant["model"]))
    COST_ASSUMPTIONS_USD_PER_MILLION_TOKENS[model_key] = dict(variant["cost"])
    MODEL_REPRODUCIBILITY_NOTES[model_key] = dict(variant["notes"])
    ACTIVE_MISTRAL_VARIANT = variant_name
    return model_key


def configure_gpt_reasoning_effort(effort: str) -> None:
    """Pin GPT reasoning effort and keep reproducibility metadata synchronized."""
    allowed = {"none", "low", "medium", "high", "xhigh"}
    if effort not in allowed:
        raise ValueError(f"Unsupported GPT reasoning effort: {effort}")
    MODELS["gpt-5.5"]["reasoning_effort"] = effort
    MODEL_REPRODUCIBILITY_NOTES["gpt-5.5"][
        "reasoning_setting"
    ] = f"reasoning_effort={effort}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_descriptor(model_name: str, strategy: str) -> Dict:
    """Return the canonical configuration that defines one experiment pair."""
    cfg = MODELS[model_name]
    request_cfg = {
        key: value
        for key, value in cfg.items()
        if key not in {"api_key_env", "api_key_default"}
    }
    return {
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "input_policy": ADR_INPUT_POLICY,
        "model_key": model_name,
        "strategy": strategy,
        "request_configuration": request_cfg,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_template": _prompt_template(strategy),
        "json_output_schema": JSON_FORMAT,
        "few_shot_exemplar_ids": FEW_SHOT_EXEMPLAR_IDS,
        "synthetic_few_shot_exemplars": SYNTHETIC_FEW_SHOT_EXEMPLARS,
        "sc_weights": SC_WEIGHTS,
        "dq_weights": DQ_WEIGHTS,
        "dimension_thresholds": [45, 70, 85],
        "overall_rule": {
            "not_compliant": "CS < 45 or SC < 40",
            "fully_compliant": "CS >= 85 and SC >= 80 and DQ >= 80",
            "mostly_compliant": "CS >= 70 and SC >= 60 and DQ >= 60",
            "partially_compliant": "otherwise",
        },
        "eval_set_sha256": _sha256_file(EVAL_SET_PATH),
        "ground_truth_sha256": _sha256_file(GROUND_TRUTH_PATH),
    }


def protocol_fingerprint(model_name: str, strategy: str) -> str:
    payload = json.dumps(
        protocol_descriptor(model_name, strategy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _sha256_text(payload)


def ensure_run_configuration() -> Dict:
    """Create an immutable run manifest or reject a conflicting reuse."""
    payload = {
        "run_id": ACTIVE_RUN_ID,
        "scoring_version": rubric.SCORING_VERSION,
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "input_policy": ADR_INPUT_POLICY,
        "evaluation_set_size": ACTIVE_N_EVAL,
        "repetitions": ACTIVE_N_REPS,
        "evaluation_set_sha256": _sha256_file(EVAL_SET_PATH),
        "ground_truth_sha256": _sha256_file(GROUND_TRUTH_PATH),
        "active_mistral_variant": ACTIVE_MISTRAL_VARIANT,
        "protocol_fingerprints": {
            f"{model_name}/{strategy}": protocol_fingerprint(model_name, strategy)
            for model_name in MODELS
            for strategy in STRATEGIES
        },
    }
    if VALIDATION_MODE:
        payload.update({
            "validation_mode": True,
            "publication_compatible": False,
            "gpt_reasoning_effort": MODELS["gpt-5.5"]["reasoning_effort"],
            "source_evaluation_set_sha256": _sha256_file(
                PUBLICATION_EVAL_SET_PATH
            ),
        })

    if RUN_CONFIG_PATH.exists():
        saved = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
        comparable = {key: saved.get(key) for key in payload}
        if comparable != payload:
            raise ValueError(
                f"Run ID '{ACTIVE_RUN_ID}' already belongs to a different "
                "protocol or scoring version. Rescore saved criteria into a new --run-id."
            )
        if saved.get("analysis_only") and not saved.get("rescore_complete"):
            raise ValueError("Offline rescoring is incomplete; analysis is blocked.")
        return saved

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    saved = {"created_at": datetime.now().isoformat(), **payload}
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(saved, fp, indent=2)
    return saved


def build_input_audit(adr_text: str, prompt: str) -> Dict:
    """Record enough input evidence to prove that the complete ADR was sent."""
    request_text = SYSTEM_PROMPT + "\n" + prompt
    return {
        "adr_source_chars": len(adr_text),
        "adr_input_chars": len(adr_text),
        "adr_input_utf8_bytes": len(adr_text.encode("utf-8")),
        "adr_input_sha256": _sha256_text(adr_text),
        "prompt_chars": len(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "prompt_sha256": _sha256_text(prompt),
        "request_text_sha256": _sha256_text(request_text),
        "input_was_truncated": False,
    }


def attach_run_metadata(result: Dict, model_name: str, strategy: str,
                        adr_text: str, prompt: str) -> Dict:
    """Attach protocol and full-input evidence before a result row is built."""
    result["strategy"] = strategy
    result["protocol_fingerprint"] = protocol_fingerprint(model_name, strategy)
    result["input_audit"] = build_input_audit(adr_text, prompt)
    return result


def _max_output_tokens(model_name: str) -> int:
    cfg = MODELS[model_name]
    return int(
        cfg.get("max_completion_tokens")
        or cfg.get("max_output_tokens")
        or cfg.get("max_tokens")
        or 1024
    )


def validate_prompt_budget(model_name: str, prompt: str) -> Dict:
    """Use UTF-8 bytes as a conservative upper bound for input tokens."""
    cfg = MODELS[model_name]
    context_tokens = int(cfg["context_window_tokens"])
    input_upper_bound = len((SYSTEM_PROMPT + "\n" + prompt).encode("utf-8"))
    output_reserve = _max_output_tokens(model_name)
    fits = input_upper_bound + output_reserve <= context_tokens
    return {
        "context_window_tokens": context_tokens,
        "conservative_input_token_upper_bound": input_upper_bound,
        "reserved_output_tokens": output_reserve,
        "fits": fits,
    }


ACQUISITION_MANIFEST_PATH = Path("adr_dataset.json")


def _load_acquisition_manifest() -> List[Dict]:
    """Load the historical fetch manifest only when acquisition is requested."""
    if not ACQUISITION_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Historical corpus acquisition is not included in the reviewer "
            "package. Use the frozen results/adrs inputs for replication."
        )
    with ACQUISITION_MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _archive_stale_result(result_file: Path, reason: str) -> None:
    """Move stale/incompatible raw results out of the active results directory."""
    archive_dir = RUN_DIR / "_archive" / f"stale_raw_results_{datetime.now():%Y%m%d_%H%M%S}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / result_file.name
    shutil.move(str(result_file), str(target))
    print(f"  STALE {result_file.name} — {reason}. Archived to {target}")


def _validate_result_reps(rep_results_list, expected_ids: List[str], n_reps: int,
                          require_complete: bool = False):
    """
    Validate resumable raw result files against the active eval_set.

    A valid partial file may have fewer than n_reps repetitions, but every
    stored repetition must contain exactly the active eval_set ADR IDs in order.
    This prevents old 3 x 100 files from being treated as complete for the
    current 162-ADR evaluation.
    """
    valid, reason = _validate_result_shape(rep_results_list, expected_ids, n_reps,
                                           require_complete=require_complete)
    if not valid:
        return valid, reason

    for rep_idx, rep in enumerate(rep_results_list, 1):
        for row_idx, row in enumerate(rep, 1):
            reason = _result_row_validation_reason(row, rep_idx, row_idx)
            if reason:
                return False, reason

    return True, "ok"


def _validate_result_shape(rep_results_list, expected_ids: List[str], n_reps: int,
                           require_complete: bool = False):
    """Validate result file shape and ADR identity without checking row quality."""
    if not isinstance(rep_results_list, list):
        return False, "top-level JSON is not a list"
    if require_complete and len(rep_results_list) != n_reps:
        return False, f"has {len(rep_results_list)} reps but expected {n_reps}"
    if len(rep_results_list) > n_reps:
        return False, f"has {len(rep_results_list)} reps but expected at most {n_reps}"

    for rep_idx, rep in enumerate(rep_results_list, 1):
        if not isinstance(rep, list):
            return False, f"rep {rep_idx} is not a list"
        if len(rep) != len(expected_ids):
            return False, f"rep {rep_idx} has {len(rep)} ADRs but expected {len(expected_ids)}"
        rep_ids = [row.get("adr_id") if isinstance(row, dict) else None for row in rep]
        if rep_ids != expected_ids:
            return False, f"rep {rep_idx} ADR IDs do not match active eval_set.json"
        for row_idx, row in enumerate(rep, 1):
            if not isinstance(row, dict):
                return False, f"rep {rep_idx} row {row_idx} is not an object"

    return True, "ok"


def _result_row_validation_reason(row: Dict, rep_idx: int, row_idx: int):
    """Return a merge-blocking reason for an incomplete saved result row."""
    if row.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        return (
            f"rep {rep_idx} row {row_idx} uses old result schema; "
            "criterion-level C1-C7/Q1-Q7 fields are required"
        )
    model_name = row.get("model_key") or (row.get("model_metadata") or {}).get("model_key")
    strategy = row.get("strategy")
    if model_name not in MODELS or strategy not in STRATEGIES:
        return f"rep {rep_idx} row {row_idx} lacks valid model/strategy metadata"
    expected_protocol = protocol_fingerprint(model_name, strategy)
    if row.get("protocol_fingerprint") != expected_protocol:
        return f"rep {rep_idx} row {row_idx} belongs to a different protocol"
    if row.get("input_was_truncated") is not False:
        return f"rep {rep_idx} row {row_idx} does not prove full-text input"
    if row.get("adr_input_chars") != row.get("adr_source_chars"):
        return f"rep {rep_idx} row {row_idx} has unequal source and input lengths"
    if not row.get("adr_input_sha256") or not row.get("prompt_sha256"):
        return f"rep {rep_idx} row {row_idx} lacks input hashes"
    if "predicted_sc_checks" not in row or "predicted_dq_scores" not in row:
        return (
            f"rep {rep_idx} row {row_idx} lacks criterion-level "
            "prediction fields"
        )
    if row.get("error"):
        return f"rep {rep_idx} row {row_idx} has API error"
    if not row.get("parse_success"):
        return f"rep {rep_idx} row {row_idx} lacks an overall prediction"
    if not row.get("criterion_parse_success"):
        return f"rep {rep_idx} row {row_idx} lacks complete C1-C7/Q1-Q7 predictions"
    if _is_truncated_finish_reason(row.get("finish_reason")):
        return f"rep {rep_idx} row {row_idx} ended with truncated finish_reason"
    if row.get("scoring_version") != rubric.SCORING_VERSION:
        return f"rep {rep_idx} row {row_idx} requires offline exact rescoring"
    try:
        expected = rubric.prediction_fields(row["predicted_sc_checks"], row["predicted_dq_scores"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return f"rep {rep_idx} row {row_idx} has invalid criteria: {error}"
    for key, value in expected.items():
        if row.get(key) != value:
            return f"rep {rep_idx} row {row_idx} has inconsistent derived field {key}"
    if row.get("correct") != (row.get("predicted") == row.get("actual")):
        return f"rep {rep_idx} row {row_idx} has inconsistent correctness"
    return None


# ============================================================
# PHASE 1: FETCH REAL ADRS FROM GITHUB
# ============================================================

def is_high_quality(text: str) -> bool:
    t = text.lower()

    score = 0

    # Context / Problem
    if any(k in t for k in ["context", "problem", "background"]):
        score += 1

    # Decision
    if "decision" in t:
        score += 1

    # Consequences / Impact
    if any(k in t for k in ["consequence", "impact", "trade-off", "tradeoff"]):
        score += 1

    # Alternatives
    if any(k in t for k in ["alternative", "option", "considered"]):
        score += 1

    # Length bonus (important)
    if len(t.split()) > 150:
        score += 1

    return score >= 2   # 🔥 relaxed from 3 → 2


def classify_variant(text: str) -> str:
    t = text.lower()

    if "decision drivers" in t and "considered options" in t:
        return "MADR"
    elif "context" in t and "consequences" in t:
        return "NYGARD"
    elif "decision" in t:
        return "LIGHTWEIGHT"
    return "OTHER"
def fetch_adrs_from_github(target_count=None):
    import urllib.request
    import time

    adr_dataset = _load_acquisition_manifest()

    if target_count is None:
        target_count = len(adr_dataset)

    ADRS_DIR.mkdir(parents=True, exist_ok=True)

    fetched = []
    repo_counts = defaultdict(int)

    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    print("\n================ FETCH ADRs ================")
    print(f"  Source: adr_dataset.json ({len(adr_dataset)} entries)\n")

    for entry in adr_dataset:
        if len(fetched) >= target_count:
            break

        owner = entry["owner"]
        repo  = entry["repo"]
        path  = entry["path"]
        filename = path.split("/")[-1]

        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"

        try:
            req = urllib.request.Request(raw_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            word_count = len(content.split())
            if word_count < 80:
                continue

            adr_id = f"{owner}_{repo}_{filename.replace('.md', '')}"
            adr_id = re.sub(r'[^a-zA-Z0-9_-]', '_', adr_id)

            adr_data = {
                "id": adr_id,
                "source_repo": f"{owner}/{repo}",
                "filename": filename,
                "url": f"https://github.com/{owner}/{repo}/blob/HEAD/{path}",
                "word_count": word_count,
                "variant": classify_variant(content),
                "text": content,
                "fetched_at": datetime.now().isoformat(),
            }

            with open(ADRS_DIR / f"{adr_id}.json", "w") as fp:
                json.dump(adr_data, fp, indent=2)

            fetched.append(adr_data)
            repo_counts[f"{owner}/{repo}"] += 1

            print(f"  OK  {owner}/{repo} | {filename} ({word_count} words)")
            time.sleep(0.4)

        except Exception as e:
            print(f"  SKIP  {owner}/{repo}/{path} -- {e}")
            continue

    print(f"\nTOTAL ADRs fetched: {len(fetched)}")

    # ==============================
    # DATASET ANALYSIS
    # ==============================
    from collections import Counter

    variant_counts = Counter()
    lengths = []

    for adr in fetched:
        variant_counts[adr["variant"]] += 1
        lengths.append(adr["word_count"])

    print("\nRepo distribution:")
    for k, v in repo_counts.items():
        print(f"  {k}: {v}")

    print("\nVariant distribution:")
    for k, v in variant_counts.items():
        print(f"  {k}: {v}")

    print(f"\nAvg length: {sum(lengths)//len(lengths) if lengths else 0}")

    # ==============================
    # SAVE REPORT
    # ==============================
    report = {
        "total_adrs": len(fetched),
        "repo_distribution": dict(repo_counts),
        "variant_distribution": dict(variant_counts),
        "avg_length": int(sum(lengths)/len(lengths)) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }

    with open("dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved dataset_report.json")

    return fetched


def load_adrs():
    """Load ADRs with the publication protocol's frozen input decoding."""
    adrs = []
    if not ADRS_DIR.exists():
        return adrs
    for f in sorted(ADRS_DIR.glob("*.json")):
        with f.open(encoding="utf-8") as fp:
            adr = json.load(fp)
        transform = FROZEN_INPUT_TEXT_TRANSFORMS.get(adr.get("id"))
        if transform == "utf8-bytes-decoded-as-windows-1252":
            adr["text"] = adr["text"].encode("utf-8").decode("windows-1252")
        adrs.append(adr)
    return adrs


# ============================================================
# PHASE 2: GPT-4o PRE-ANNOTATION
# ============================================================

ANNOTATION_SYSTEM_PROMPT = """You are an expert software architect performing preliminary ADR pre-annotation
for a research workflow. These labels are not final benchmark ground truth.
You must be extremely precise and consistent.

You are evaluating Architecture Decision Records (ADRs) against two dimensions:

STRUCTURAL COMPLETENESS (7 binary checks):
C1: Title present and descriptive (not generic like "ADR-001")
C2: Context/Problem Statement clearly articulated
C3: Decision Drivers explicitly listed
C4: At least two genuinely different Considered Options presented
C5: Decision Outcome stated with explicit justification
C6: Consequences (both positive AND negative) documented
C7: Status field present with valid value (proposed/accepted/deprecated/superseded)

DECISION QUALITY (7 criteria):
Q1: Problem is relevant and significant enough for an ADR
Q2: Alternatives are genuine (not strawmen or obviously inferior)
Q3: Decision criteria are complete and well-defined
Q4: When criteria conflict, they are prioritized
Q5: Rationale is sound and convincing
Q6: Consequences are reported objectively (both positive and negative)
Q7: Solution is described in an actionable way

DQ scores use a 0-3 anchor scale:
0 Absent: no document evidence supports the criterion.
1 Weak: mentioned but generic, superficial, or unsupported.
2 Adequate: project-specific reasoning is present, with some omissions or limited depth.
3 Strong: clear, specific, and defensible architectural reasoning is present.

SC = (0.10*C1 + 0.15*C2 + 0.15*C3 + 0.15*C4 + 0.20*C5 + 0.15*C6 + 0.10*C7) * 100.
DQ = ((0.10*Q1 + 0.15*Q2 + 0.15*Q3 + 0.10*Q4 + 0.25*Q5 + 0.15*Q6 + 0.10*Q7) / 3) * 100.
CS = 0.40*SC + 0.60*DQ.

OVERALL prelabel tiers:
Fully_Compliant: CS >= 85 with SC >= 80 and DQ >= 80
Mostly_Compliant: 70 <= CS < 85 with SC >= 60 and DQ >= 60
Partially_Compliant: 45 <= CS < 70, or SC < 60 while DQ >= 60
Not_Compliant: CS < 45, or SC < 40

Be strict, but derive the final overall class from the criterion scores and formulas above.
Respond ONLY with valid JSON, no other text."""

ANNOTATION_PROMPT_TEMPLATE = """Evaluate this ADR:

---
{adr_text}
---

Respond with ONLY this JSON (no markdown, no explanation):
{{"C1":true|false,"C2":true|false,"C3":true|false,"C4":true|false,"C5":true|false,"C6":true|false,"C7":true|false,"Q1":0|1|2|3,"Q2":0|1|2|3,"Q3":0|1|2|3,"Q4":0|1|2|3,"Q5":0|1|2|3,"Q6":0|1|2|3,"Q7":0|1|2|3,"sc_score":0-100,"dq_score":0-100,"composite_score":0-100,"sc_class":"Fully_Compliant|Mostly_Compliant|Partially_Compliant|Not_Compliant","dq_class":"Fully_Compliant|Mostly_Compliant|Partially_Compliant|Not_Compliant","overall":"Fully_Compliant|Mostly_Compliant|Partially_Compliant|Not_Compliant"}}"""

GPT4O_PRELABEL_TEMPERATURES = (0.3, 0.5)


def generate_gpt4o_prelabels(adrs: List[Dict], prelabel_model="gpt-4o-2024-08-06"):
    """Use GPT-4o for preliminary pre-labeling and self-consistency checks.

    The two runs intentionally use mild temperature variation to flag unstable
    prelabels for human review. They are not independent annotators, and their
    outputs are never used as final benchmark ground truth.
    """
    from openai import OpenAI
    client = OpenAI()

    print(f"\n{'='*60}")
    print(f"PHASE 2: Generating GPT-4o prelabels for {len(adrs)} ADRs ({prelabel_model})")
    print(f"{'='*60}")

    prelabel_path = EXPERIMENT_DIR / "prelabels_gpt4o.json"
    prelabels = {}

    # Load existing if resuming
    if prelabel_path.exists():
        with open(prelabel_path) as fp:
            prelabels = json.load(fp)
        print(f"  Loaded {len(prelabels)} existing GPT-4o prelabels")

    for i, adr in enumerate(adrs):
        if adr["id"] in prelabels:
            continue

        print(f"  [{i+1}/{len(adrs)}] Pre-labeling {adr['id']}...")

        # Truncate very long ADRs to avoid token limits
        adr_text = adr["text"][:4000]

        # Run GPT-4o twice to estimate same-model prelabel stability.
        # This is workflow support only, not inter-rater reliability.
        annotations = []
        for run, temperature in enumerate(GPT4O_PRELABEL_TEMPERATURES):
            try:
                response = client.chat.completions.create(
                    model=prelabel_model,
                    messages=[
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {"role": "user", "content": ANNOTATION_PROMPT_TEMPLATE.format(adr_text=adr_text)},
                    ],
                    temperature=temperature,
                    max_tokens=500,
                )
                raw = response.choices[0].message.content.strip()

                # Clean and parse
                raw = re.sub(r'^```json\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
                parsed = json.loads(raw)
                annotations.append(parsed)

            except Exception as e:
                print(f"    ERROR run {run+1} failed: {e}")
                annotations.append(None)

            time.sleep(RATE_LIMIT_DELAY)

        # Store first prelabel and record same-model self-consistency.
        if annotations[0]:
            gt = annotations[0]
            consistency = "agree"
            if annotations[1] and annotations[0].get("overall") != annotations[1].get("overall"):
                consistency = "disagree"

            prelabels[adr["id"]] = {
                "overall": gt.get("overall", "Partially_Compliant"),
                "sc_class": gt.get("sc_class", "Partially_Compliant"),
                "dq_class": gt.get("dq_class", "Partially_Compliant"),
                "sc_score": gt.get("sc_score"),
                "dq_score": gt.get("dq_score"),
                "composite_score": gt.get("composite_score"),
                "details": gt,
                "prelabel_model": prelabel_model,
                "prelabel_temperatures": list(GPT4O_PRELABEL_TEMPERATURES),
                "gpt4o_self_consistency": consistency,
                "prelabeled_at": datetime.now().isoformat(),
            }

        # Save incrementally
        with open(prelabel_path, "w") as fp:
            json.dump(prelabels, fp, indent=2)

    # Report distribution
    classes = [v["overall"] for v in prelabels.values()]
    print(f"\n  GPT-4o Prelabel Distribution:")
    for cls in CLASSES:
        n = classes.count(cls)
        print(f"    {cls}: {n} ({n/len(classes)*100:.0f}%)")

    agreements = [v["gpt4o_self_consistency"] for v in prelabels.values()]
    agree_rate = agreements.count("agree") / len(agreements)
    print(f"  GPT-4o self-consistency: {agree_rate:.1%}")

    return prelabels


# ============================================================
# PHASE 3: RUN ACTUAL LLM EXPERIMENTS
# ============================================================

SYSTEM_PROMPT = """You are a senior software architect with 15+ years of experience reviewing 
Architecture Decision Records (ADRs). Be precise and evidence-based."""

# Final evaluation uses the four compliance tiers reported in the manuscript.
# GPT-4o prelabels above remain preliminary and are not used for final benchmark
# scoring.
RUBRIC = """COMPLIANCE RUBRIC:
STRUCTURAL COMPLETENESS (7 checks):
C1: Title present and descriptive
C2: Context/Problem Statement articulated
C3: Decision Drivers listed
C4: At least two Considered Options
C5: Decision Outcome with justification
C6: Consequences (positive and negative) documented
C7: Status field present and valid

DECISION QUALITY (7 criteria):
Q1: Problem relevance  Q2: Option viability (not strawmen)
Q3: Criteria completeness  Q4: Criteria prioritization
Q5: Rationale soundness  Q6: Consequence objectivity  Q7: Actionability

DQ SCORING ANCHORS (0-3 for each Q criterion):
0 Absent: no document evidence supports the criterion.
1 Weak: mentioned but generic, superficial, or unsupported.
2 Adequate: project-specific reasoning is present, with some omissions or limited depth.
3 Strong: clear, specific, and defensible architectural reasoning is present.

Structural checks are binary but not equally weighted. Core architectural
knowledge fields, such as context, options, decision justification, and
consequences, carry higher weight than title/status metadata.

Judge only the evidence for C1-C7 and Q1-Q7. The benchmark pipeline applies
the frozen weights and ordered classification rule after parsing these
criterion judgments. Do not calculate or return SC, DQ, CS, dimension classes,
or an overall compliance class."""

JSON_FORMAT = """Respond ONLY with valid JSON.
DO NOT use markdown or code blocks.
{"C1":true|false,"C2":true|false,"C3":true|false,"C4":true|false,"C5":true|false,"C6":true|false,"C7":true|false,"Q1":0|1|2|3,"Q2":0|1|2|3,"Q3":0|1|2|3,"Q4":0|1|2|3,"Q5":0|1|2|3,"Q6":0|1|2|3,"Q7":0|1|2|3,"confidence":0.0-1.0}"""


def _prompt_template(strategy: str) -> str:
    """Return the prompt template with placeholders for reproducibility docs."""
    if strategy == "zero_shot":
        return f"""{RUBRIC}

Evaluate this ADR:
---
{{FULL_ADR_TEXT}}
---
{JSON_FORMAT}"""

    if strategy == "few_shot":
        return f"""{RUBRIC}

Here are annotated examples:
{{FEW_SHOT_EXAMPLES}}

Now evaluate this ADR:
---
{{FULL_ADR_TEXT}}
---
{JSON_FORMAT}"""

    if strategy == "chain_of_thought":
        return f"""{RUBRIC}

Evaluate this ADR step by step, but keep the reasoning concise and output only
the final JSON object. Use the steps internally to assign C1-C7 and Q1-Q7,
identify ambiguities, and assign the criterion judgments.

ADR:
---
{{FULL_ADR_TEXT}}
---

Return EXACTLY this JSON object and no additional prose:
{JSON_FORMAT}"""

    raise ValueError(f"Unknown strategy: {strategy}")


def _criterion_label_payload(label: Dict) -> Dict:
    """Return the criterion-level label payload used in few-shot examples."""
    payload = {}
    for criterion in SC_WEIGHTS:
        payload[criterion] = bool(label.get(criterion, False))
    for criterion in DQ_WEIGHTS:
        value = label.get(criterion, 0)
        try:
            payload[criterion] = int(value)
        except (TypeError, ValueError):
            payload[criterion] = 0

    return payload


def build_few_shot_examples(adrs: List[Dict], ground_truth: Dict,
                            eval_adrs: List[Dict] = None) -> List[Dict]:
    """Load and validate fixed few-shot exemplars."""
    adrs_by_id = {a["id"]: a for a in adrs}
    eval_ids = {a["id"] for a in eval_adrs or []}
    overlap = sorted(set(FEW_SHOT_EXEMPLAR_IDS) & eval_ids)
    if overlap:
        raise ValueError(
            "Few-shot exemplars must be excluded from results/eval_set.json; "
            f"overlap found: {overlap}"
        )

    examples = []
    missing = []
    missing_labels = []
    for adr_id in FEW_SHOT_EXEMPLAR_IDS:
        if adr_id in SYNTHETIC_FEW_SHOT_EXEMPLARS:
            examples.append(dict(SYNTHETIC_FEW_SHOT_EXEMPLARS[adr_id]))
            continue
        adr = adrs_by_id.get(adr_id)
        if adr is None:
            missing.append(adr_id)
            continue
        label = ground_truth.get(adr_id, {}).get("overall")
        if not label:
            missing_labels.append(adr_id)
            continue
        label_payload = _criterion_label_payload(ground_truth[adr_id])
        examples.append({
            "id": adr_id,
            "text": adr["text"],
            "label": label,
            "label_payload": label_payload,
            "source_repo": adr.get("source_repo"),
            "variant": adr.get("variant", "OTHER"),
            "domain": get_domain(adr.get("source_repo", "")),
        })

    if missing:
        raise ValueError(f"Few-shot exemplar ADR files missing: {missing}")
    if missing_labels:
        raise ValueError(f"Few-shot exemplar labels missing: {missing_labels}")

    represented = {ex["label"] for ex in examples}
    expected = set(CLASSES)
    if represented != expected:
        raise ValueError(
            "Few-shot exemplars must cover all compliance classes; "
            f"represented={sorted(represented)}, expected={sorted(expected)}"
        )

    return examples


def write_prompt_manifest(adrs: List[Dict], ground_truth: Dict,
                          eval_adrs: List[Dict]) -> None:
    """Write prompt templates, schema, parser, and exemplar metadata."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    examples = build_few_shot_examples(adrs, ground_truth, eval_adrs)
    eval_ids = {a["id"] for a in eval_adrs}
    overlap = sorted({ex["id"] for ex in examples} & eval_ids)

    manifest = {
        "written_at": datetime.now().isoformat(),
        "purpose": "Prompt transparency manifest for benchmark reproducibility.",
        "run_id": ACTIVE_RUN_ID,
        "run_directory": str(RUN_DIR),
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "strategies": STRATEGIES,
        "system_prompt": SYSTEM_PROMPT,
        "rubric": RUBRIC,
        "rubric_manifest_path": str(RUBRIC_MANIFEST_PATH),
        "structural_completeness_weights": SC_WEIGHTS,
        "decision_quality_weights": DQ_WEIGHTS,
        "decision_quality_score_anchors": DQ_SCORE_ANCHORS,
        "json_output_schema": JSON_FORMAT,
        "required_output_fields": {
            "structural_checks": list(SC_WEIGHTS.keys()),
            "decision_quality_scores": list(DQ_WEIGHTS.keys()),
            "optional_fields": ["confidence"],
        },
        "adr_input_policy": ADR_INPUT_POLICY,
        "adr_text_truncation_chars": None,
        "prompt_templates": {
            strategy: _prompt_template(strategy)
            for strategy in STRATEGIES
        },
        "protocol_fingerprints": {
            f"{model_name}/{strategy}": protocol_fingerprint(model_name, strategy)
            for model_name in MODELS
            for strategy in STRATEGIES
        },
        "chain_of_thought_operationalization": {
            "definition": (
                "Chain-of-thought is operationalized as a reproducible prompt "
                "condition, not as access to equivalent internal reasoning "
                "processes across providers."
            ),
            "supplied_instruction": (
                "The prompt instructs the model to assess structural checks "
                "C1-C7, decision-quality criteria Q1-Q7, and ambiguities before "
                "ending with the required JSON object."
            ),
            "provider_reasoning_modes": (
                "Provider-native reasoning controls are not treated as equivalent "
                "across vendors. GPT reasoning effort is explicitly pinned for each "
                "run; other provider settings are documented in the run-specific "
                "model reproducibility manifest."
            ),
            "scoring_rule": (
                "The model supplies C1-C7 and Q1-Q7 judgments. The runner computes "
                "SC, DQ, CS, dimension classes, and the overall label using the "
                "frozen deterministic scoring functions."
            ),
        },
        "few_shot": {
            "selection_rule": (
                "Three fixed human-labeled held-out ADRs plus one controlled "
                "synthetic negative exemplar, one per compliance class. None is "
                "part of results/eval_set.json."
            ),
            "exemplar_ids": [ex["id"] for ex in examples],
            "exemplar_count": len(examples),
            "covers_all_compliance_classes": True,
            "excluded_from_eval_set": not overlap,
            "eval_set_overlap": overlap,
            "examples": [
                {
                    "id": ex["id"],
                    "label": ex["label"],
                    "label_payload": ex["label_payload"],
                    "source_repo": ex["source_repo"],
                    "variant": ex["variant"],
                    "domain": ex["domain"],
                    "input_chars": len(ex["text"]),
                    "input_sha256": hashlib.sha256(
                        ex["text"].encode("utf-8")
                    ).hexdigest(),
                    "full_text_supplied": True,
                }
                for ex in examples
            ],
        },
        "parser": {
            "function": "extract_prediction(raw)",
            "procedure": [
                "Strip leading and trailing whitespace.",
                "Remove optional Markdown code fences.",
                "Extract the first JSON object from the response.",
                "Read C1-C7 structural checks and normalize them to true/false values.",
                "Read Q1-Q7 decision-quality scores and normalize them to integers on the 0-3 anchor scale.",
                "Reject incomplete criterion vectors and retry the model request.",
                "Compute SC, DQ, CS, dimension classes, and overall class from the complete criterion vector.",
                "Save C1-C7, Q1-Q7, derived scores and classes, input audit fields, and parse flags in each raw result row.",
                "Return an incomplete prediction and mark parse_success=false when valid criterion JSON cannot be extracted.",
            ],
            "allowed_labels": CLASSES,
            "decision_quality_score_labels": {
                "0": "Absent",
                "1": "Weak",
                "2": "Adequate",
                "3": "Strong",
            },
        },
    }

    if overlap:
        raise ValueError(f"Few-shot exemplars overlap eval_set.json: {overlap}")

    with open(PROMPT_MANIFEST_PATH, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"  Prompt manifest written to {PROMPT_MANIFEST_PATH}")


def _normalize_for_similarity(text: str) -> str:
    """Normalize ADR text for deterministic near-duplicate checks."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def _token_jaccard_similarity(a: str, b: str) -> float:
    """Return token Jaccard similarity for two ADR texts."""
    a_tokens = set(_normalize_for_similarity(a).split())
    b_tokens = set(_normalize_for_similarity(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _adr_source_key(adr: Dict) -> str:
    """Return the strongest available source identifier for an ADR."""
    return adr.get("url") or adr.get("path") or adr.get("source_path") or adr.get("filename")


def write_prompt_leakage_audit(adrs: List[Dict], ground_truth: Dict,
                               eval_adrs: List[Dict],
                               similarity_threshold: float = 0.80) -> Dict:
    """Audit few-shot exemplars for leakage into the evaluation set."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    examples = build_few_shot_examples(adrs, ground_truth, eval_adrs)
    adrs_by_id = {a["id"]: a for a in adrs}
    eval_ids = {a["id"] for a in eval_adrs}
    eval_by_id = {a["id"]: a for a in eval_adrs}

    id_overlap = sorted({ex["id"] for ex in examples} & eval_ids)
    source_path_overlap = []
    same_repo_overlap = []
    near_duplicate_overlap = []
    closest_pairs = []

    for ex in examples:
        ex_adr = adrs_by_id.get(ex["id"], ex)
        ex_source = _adr_source_key(ex_adr)
        ex_repo = ex_adr.get("source_repo")

        for eval_id, eval_adr in eval_by_id.items():
            eval_source = _adr_source_key(eval_adr)
            eval_repo = eval_adr.get("source_repo")

            if ex_source and eval_source and ex_source == eval_source:
                source_path_overlap.append({
                    "few_shot_id": ex["id"],
                    "eval_id": eval_id,
                    "source": ex_source,
                })

            if ex_repo and eval_repo and ex_repo == eval_repo:
                same_repo_overlap.append({
                    "few_shot_id": ex["id"],
                    "eval_id": eval_id,
                    "source_repo": ex_repo,
                })

            similarity = _token_jaccard_similarity(
                ex_adr.get("text", ""),
                eval_adr.get("text", ""),
            )
            closest_pairs.append({
                "few_shot_id": ex["id"],
                "eval_id": eval_id,
                "similarity": round(similarity, 3),
            })
            if similarity >= similarity_threshold:
                near_duplicate_overlap.append({
                    "few_shot_id": ex["id"],
                    "eval_id": eval_id,
                    "similarity": round(similarity, 3),
                })

    audit = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Prompt leakage audit for few-shot exemplars versus the active "
            "evaluation set."
        ),
        "few_shot_exemplar_ids": [ex["id"] for ex in examples],
        "evaluation_set_size": len(eval_adrs),
        "checks": {
            "exact_adr_id_overlap_count": len(id_overlap),
            "exact_adr_id_overlap": id_overlap,
            "source_path_overlap_count": len(source_path_overlap),
            "source_path_overlap": source_path_overlap,
            "near_duplicate_text_overlap_count": len(near_duplicate_overlap),
            "near_duplicate_similarity_threshold": similarity_threshold,
            "near_duplicate_text_overlap": near_duplicate_overlap,
            "closest_text_pairs": sorted(
                closest_pairs,
                key=lambda item: item["similarity"],
                reverse=True,
            )[:20],
            "same_repo_overlap_count": len(same_repo_overlap),
            "same_repo_overlap_allowed": True,
            "same_repo_overlap_note": (
                "Repository-level overlap is allowed because the benchmark intentionally "
                "contains multiple ADRs from the same real-world projects. Leakage control "
                "is applied at exact ADR ID, source path, and near-duplicate text levels."
            ),
        },
        "leakage_passed": (
            not id_overlap
            and not source_path_overlap
            and not near_duplicate_overlap
        ),
    }

    with open(PROMPT_LEAKAGE_AUDIT_PATH, "w", encoding="utf-8") as fp:
        json.dump(audit, fp, indent=2)
    print(f"  Prompt leakage audit written to {PROMPT_LEAKAGE_AUDIT_PATH}")
    return audit


def run_preflight(adrs: List[Dict], ground_truth: Dict,
                  eval_adrs: List[Dict]) -> Dict:
    """Validate the frozen full-text protocol before any paid API request."""
    ensure_run_configuration()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    eval_ids = [adr["id"] for adr in eval_adrs]
    if len(eval_ids) != ACTIVE_N_EVAL or len(set(eval_ids)) != len(eval_ids):
        failures.append(
            f"evaluation set must contain {ACTIVE_N_EVAL} unique ADRs; found {len(eval_ids)}"
        )

    label_mismatches = []
    for adr in eval_adrs:
        label = ground_truth.get(adr["id"])
        if not label:
            label_mismatches.append({"adr_id": adr["id"], "reason": "missing label"})
            continue
        derived = classify_overall_from_scores(
            compute_sc_score_from_label(label),
            compute_dq_score_from_label(label),
        )
        if derived != label.get("overall"):
            label_mismatches.append({
                "adr_id": adr["id"],
                "stored": label.get("overall"),
                "derived": derived,
            })
    if label_mismatches:
        failures.append(
            f"{len(label_mismatches)} human labels disagree with the frozen scoring rule"
        )

    examples = build_few_shot_examples(adrs, ground_truth, eval_adrs)
    leakage = write_prompt_leakage_audit(adrs, ground_truth, eval_adrs)
    if not leakage["leakage_passed"]:
        failures.append("few-shot leakage audit failed")

    prompt_summaries = {}
    input_failures = []
    for model_name in MODELS:
        for strategy in STRATEGIES:
            chars = []
            byte_upper_bounds = []
            for adr in eval_adrs:
                prompt = make_prompt(adr["text"], strategy, examples)
                if adr["text"] not in prompt:
                    input_failures.append({
                        "model": model_name,
                        "strategy": strategy,
                        "adr_id": adr["id"],
                        "reason": "complete ADR text not found in prompt",
                    })
                budget = validate_prompt_budget(model_name, prompt)
                if not budget["fits"]:
                    input_failures.append({
                        "model": model_name,
                        "strategy": strategy,
                        "adr_id": adr["id"],
                        "reason": "conservative prompt budget exceeds context window",
                        **budget,
                    })
                chars.append(len(prompt))
                byte_upper_bounds.append(
                    budget["conservative_input_token_upper_bound"]
                )
            prompt_summaries[f"{model_name}/{strategy}"] = {
                "protocol_fingerprint": protocol_fingerprint(model_name, strategy),
                "minimum_prompt_chars": min(chars),
                "maximum_prompt_chars": max(chars),
                "maximum_conservative_input_token_upper_bound": max(byte_upper_bounds),
                "context_window_tokens": MODELS[model_name]["context_window_tokens"],
                "reserved_output_tokens": _max_output_tokens(model_name),
            }
    if input_failures:
        failures.append(f"{len(input_failures)} prompt input or budget checks failed")

    report = {
        "written_at": datetime.now().isoformat(),
        "run_id": ACTIVE_RUN_ID,
        "run_directory": str(RUN_DIR),
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "input_policy": ADR_INPUT_POLICY,
        "evaluation_set_size": len(eval_adrs),
        "evaluation_set_sha256": _sha256_file(EVAL_SET_PATH),
        "ground_truth_sha256": _sha256_file(GROUND_TRUTH_PATH),
        "maximum_adr_chars": max(len(adr["text"]) for adr in eval_adrs),
        "all_adrs_supplied_in_full": not input_failures,
        "human_label_mismatches": label_mismatches,
        "few_shot_exemplars": [ex["id"] for ex in examples],
        "few_shot_leakage_passed": leakage["leakage_passed"],
        "prompt_summaries": prompt_summaries,
        "input_failures": input_failures,
        "failures": failures,
        "passed": not failures,
    }
    with open(PREFLIGHT_REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Preflight report written to {PREFLIGHT_REPORT_PATH}")
    if failures:
        raise ValueError("Preflight failed: " + "; ".join(failures))
    return report


def _iter_result_rows(value):
    """Yield saved evaluation rows from merged or per-configuration JSON."""
    if isinstance(value, dict):
        if "adr_id" in value:
            yield value
            return
        for nested in value.values():
            yield from _iter_result_rows(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_result_rows(nested)


def _summarize_model_access(value) -> Dict:
    """Summarize observed access timestamps without changing saved rows."""
    by_model = defaultdict(list)
    for row in _iter_result_rows(value):
        model_name = row.get("model_key") or row.get("model")
        accessed_at = row.get("accessed_at")
        if model_name and isinstance(accessed_at, str) and accessed_at.strip():
            by_model[model_name].append(accessed_at.strip())

    summary = {}
    for model_name, timestamps in sorted(by_model.items()):
        ordered = sorted(timestamps)
        dates = sorted({timestamp[:10] for timestamp in ordered})
        summary[model_name] = {
            "model_access_date": dates[0] if len(dates) == 1 else dates,
            "first_observed_accessed_at": ordered[0],
            "last_observed_accessed_at": ordered[-1],
            "saved_evaluation_rows": len(ordered),
            "timestamp_semantics": (
                "Saved response access or collection timestamp. For batch rows, "
                "provider execution may have occurred before collection."
            ),
        }
    return summary


def _load_saved_access_summary() -> Dict:
    """Load observed access dates when a completed merged result is available."""
    path = RESULTS_DIR / "all_results.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fp:
            return _summarize_model_access(json.load(fp))
    except (OSError, ValueError, TypeError):
        return {}


def model_reproducibility_row(model_name: str) -> Dict:
    """Return the configured reproducibility metadata for a model."""
    cfg = MODELS[model_name]
    notes = MODEL_REPRODUCIBILITY_NOTES[model_name]
    cost = COST_ASSUMPTIONS_USD_PER_MILLION_TOKENS[model_name]

    return {
        "model_key": model_name,
        "paper_label": notes["paper_label"],
        "provider": notes["provider"],
        "exact_model_id": cfg["model"],
        "access_mode": notes["access_mode"],
        "api_version": notes["api_version"],
        "endpoint_or_deployment": notes["endpoint_or_deployment"],
        "configured_base_url": cfg.get("base_url"),
        "context_window_tokens": cfg.get("context_window_tokens"),
        "access_date": "Recorded per final API run in raw result metadata; manifest generated before rerun.",
        "temperature": notes["temperature"],
        "top_p": notes["top_p"],
        "max_output": notes["max_output"],
        "structured_output_mode": notes["structured_output_mode"],
        "reasoning_setting": notes["reasoning_setting"],
        "system_prompt": "Shared SYSTEM_PROMPT in prompt manifest",
        "prompt_strategies": STRATEGIES,
        "retry_policy": (
            "Up to 3 total API attempts for transient provider/network errors, "
            "including 429/rate-limit, upstream reset, timeout, and overflow errors; "
            "waits 10 seconds, then 30 seconds on final API retry. The runner also "
            "retries incomplete model outputs that omit required C1-C7/Q1-Q7 fields "
            "or end because of output truncation."
        ),
        "rate_limit_delay_seconds": RATE_LIMIT_DELAY,
        "cost_assumption_usd_per_million_input_tokens": cost["input"],
        "cost_assumption_usd_per_million_output_tokens": cost["output"],
        "cost_assumption_basis": cost["basis"],
        "pricing_source": cost.get("pricing_source"),
        "pricing_date": cost.get("pricing_date"),
        "pricing_verified_date": cost.get("pricing_verified_date"),
        "batch_cost_multiplier": cost.get("batch_multiplier"),
        "batch_input_usd_per_million_tokens": cost.get("batch_input"),
        "batch_output_usd_per_million_tokens": cost.get("batch_output"),
        "batch_pricing_source": cost.get("batch_pricing_source"),
        "subscription_or_tier": cost.get("subscription_or_tier"),
        "cached_token_treatment": cost.get("cached_token_treatment"),
        "local_compute_assumption": cost.get("local_compute_assumption"),
    }


def write_model_reproducibility_manifest() -> None:
    """Write model/API configuration metadata before the final rerun."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    observed_access = _load_saved_access_summary()
    model_rows = []
    for model_name in MODELS:
        row = model_reproducibility_row(model_name)
        row.update(observed_access.get(model_name, {}))
        model_rows.append(row)
    manifest = {
        "written_at": datetime.now().isoformat(),
        "purpose": "Model and API reproducibility metadata for the ADR benchmark.",
        "run_id": ACTIVE_RUN_ID,
        "run_directory": str(RUN_DIR),
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "input_policy": ADR_INPUT_POLICY,
        "active_mistral_variant": ACTIVE_MISTRAL_VARIANT,
        "mistral_variant_options": {
            name: {
                "model_key": variant["model_key"],
                "exact_model_id": variant["model"]["model"],
                "provider": variant["notes"]["provider"],
                "endpoint_or_deployment": variant["notes"]["endpoint_or_deployment"],
            }
            for name, variant in MISTRAL_MODEL_VARIANTS.items()
        },
        "pricing_verified_date": PRICING_VERIFIED_DATE,
        "cached_token_treatment": CACHED_TOKEN_TREATMENT,
        "unsuccessful_call_usage_treatment": UNRETAINED_USAGE_TREATMENT,
        "models": model_rows,
        "protocol_fingerprints": {
            f"{model_name}/{strategy}": protocol_fingerprint(model_name, strategy)
            for model_name in MODELS
            for strategy in STRATEGIES
        },
        "shared_settings": {
            "n_repetitions": ACTIVE_N_REPS,
            "n_eval_default": ACTIVE_N_EVAL,
            "validation_mode": VALIDATION_MODE,
            "publication_compatible": not VALIDATION_MODE,
            "strategies": STRATEGIES,
            "structured_output_note": (
                "All models use prompt-enforced JSON and the same parser. No provider-native "
                "structured-output mode is configured in this runner."
            ),
            "access_date_note": (
                "Observed dates are derived from saved accessed_at fields when merged "
                "results are available. Batch timestamps may represent collection rather "
                "than the provider's exact execution time."
            ),
        },
    }
    with open(MODEL_REPRODUCIBILITY_PATH, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"  Model reproducibility manifest written to {MODEL_REPRODUCIBILITY_PATH}")


def make_prompt(adr_text: str, strategy: str, few_shot_examples: List[Dict] = None):
    """Build the prompt for a given strategy."""
    if not isinstance(adr_text, str) or not adr_text.strip():
        raise ValueError("ADR text must be a non-empty string")

    if strategy == "zero_shot":
        return f"""{RUBRIC}

Evaluate this ADR:
---
{adr_text}
---
{JSON_FORMAT}"""

    elif strategy == "few_shot":
        examples_text = ""
        if few_shot_examples:
            for i, ex in enumerate(few_shot_examples, 1):
                label_json = json.dumps(ex["label_payload"], separators=(",", ":"))
                examples_text += f"""
Example {i}:
ADR:
---
{ex['text']}
---
Criterion judgments: {label_json}
"""
        return f"""{RUBRIC}

Here are annotated examples:
{examples_text}

Now evaluate this ADR:
---
{adr_text}
---
{JSON_FORMAT}"""

    elif strategy == "chain_of_thought":
        return f"""{RUBRIC}

Evaluate this ADR step by step, but keep the reasoning concise and output only
the final JSON object. Use the steps internally to assign C1-C7 and Q1-Q7,
identify ambiguities, and verify that every judgment is supported by the ADR.

ADR:
---
{adr_text}
---

Return EXACTLY this JSON object and no additional prose:
{JSON_FORMAT}"""


def call_llm(model_name: str, prompt: str, _retries: int = 2, _attempt: int = 1):
    """Call an LLM API and return result with timing."""
    cfg = MODELS[model_name]
    start = time.time()
    accessed_at = datetime.now().isoformat()
    request_metadata = model_reproducibility_row(model_name)

    try:
        if cfg["provider"] == "openai":
            from openai import OpenAI
            kwargs = {}
            if "base_url" in cfg:
                kwargs["base_url"] = cfg["base_url"]
            if "api_key_env" in cfg:
                kwargs["api_key"] = os.environ.get(
                    cfg["api_key_env"],
                    cfg.get("api_key_default", "")
                )
            client = OpenAI(**kwargs)

            create_kwargs = {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            if not cfg.get("no_temperature"):
                create_kwargs["temperature"] = 0
            if cfg.get("use_max_completion_tokens"):
                create_kwargs["max_completion_tokens"] = cfg.get("max_completion_tokens", 1024)
            else:
                create_kwargs["max_tokens"] = cfg.get("max_tokens", 1024)
            if cfg.get("reasoning_effort"):
                create_kwargs["reasoning_effort"] = cfg["reasoning_effort"]
            request_metadata["request_parameters"] = {
                k: v for k, v in create_kwargs.items()
                if k not in {"messages"}
            }
            response = client.chat.completions.create(**create_kwargs)
            raw = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            completion_details = getattr(response.usage, "completion_tokens_details", None)
            prompt_details = getattr(response.usage, "prompt_tokens_details", None)
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "completion_tokens_details": (
                    completion_details.model_dump()
                    if hasattr(completion_details, "model_dump")
                    else completion_details
                ),
                "prompt_tokens_details": (
                    prompt_details.model_dump()
                    if hasattr(prompt_details, "model_dump")
                    else prompt_details
                ),
            }

        elif cfg["provider"] == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            create_kwargs = {
                "model": cfg["model"],
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cfg.get("max_tokens", 1024),
            }
            if not cfg.get("no_temperature"):
                create_kwargs["temperature"] = 0
            request_metadata["request_parameters"] = {
                k: v for k, v in create_kwargs.items()
                if k not in {"messages", "system"}
            }
            response = client.messages.create(**create_kwargs)
            raw = response.content[0].text
            finish_reason = response.stop_reason
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        elif cfg["provider"] == "gemini":
            from google import genai as google_genai
            from google.genai import types as genai_types
            client = google_genai.Client(api_key=os.environ.get(cfg["api_key_env"], ""))
            response = client.models.generate_content(
                model=cfg["model"],
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=cfg.get("max_output_tokens", 1024),
                ),
            )
            request_metadata["request_parameters"] = {
                "model": cfg["model"],
                "max_output_tokens": cfg.get("max_output_tokens", 1024),
                "system_instruction": "SYSTEM_PROMPT",
            }
            raw = response.text
            if raw is None:
                finish_reason = "unknown"
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)
                raise ValueError(f"empty response from Gemini (finish_reason={finish_reason})")
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates else None
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        else:
            return {"error": f"Unknown provider: {cfg['provider']}"}

        elapsed = time.time() - start

        return {
            "raw": raw,
            "usage": usage,
            "latency": round(elapsed, 2),
            "error": None,
            "accessed_at": accessed_at,
            "model_metadata": request_metadata,
            "finish_reason": finish_reason,
            "attempts": _attempt,
            "retry_count": _attempt - 1,
        }

    except Exception as e:
        err = str(e)
        print(f"\nAPI ERROR [{model_name}]:", err)
        quota_exhausted = _is_quota_or_billing_exhausted(err)

        is_transient = _is_transient_api_error(err) and not quota_exhausted
        if is_transient and _retries > 0:
            wait = 30 if _retries == 1 else 10
            print(f"  Transient API error — waiting {wait}s then retrying ({_retries} left)...")
            time.sleep(wait)
            return call_llm(model_name, prompt, _retries=_retries - 1, _attempt=_attempt + 1)

        return {
            "raw": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency": round(time.time() - start, 2),
            "error": err,
            "quota_exhausted": quota_exhausted,
            "accessed_at": accessed_at,
            "model_metadata": request_metadata,
            "attempts": _attempt,
            "retry_count": _attempt - 1,
        }


def normalize_compliance_class(value):
    """Map model or human label text to one of the four benchmark classes."""
    if value is None:
        return None
    val = str(value).lower()
    if "fully" in val:
        return "Fully_Compliant"
    if "mostly" in val:
        return "Mostly_Compliant"
    if "partially" in val:
        return "Partially_Compliant"
    if "not" in val or "non" in val:
        return "Not_Compliant"
    return None


def score_to_compliance_class(score: float) -> str:
    """Classify a dimension score using the benchmark's four-tier thresholds."""
    return rubric.dimension_class(score)


def classify_overall_from_scores(sc_score: float, dq_score: float,
                                 sc_weight: float = 0.40,
                                 structural_floor: float = 40.0) -> str:
    """Apply the benchmark composite rule to SC and DQ scores."""
    return rubric.overall_class(sc_score, dq_score, sc_weight, structural_floor)


def compute_composite_score(sc_score: float, dq_score: float,
                            sc_weight: float = 0.40) -> Fraction:
    """Compute the benchmark composite score from SC and DQ."""
    return rubric.composite_score(sc_score, dq_score, sc_weight)


def compute_sc_score_from_label(label: Dict, weights: Dict[str, float] = None) -> Fraction:
    """Compute SC score from C1-C7 binary labels."""
    return rubric.structural_score(label, weights)


def compute_dq_score_from_label(label: Dict, weights: Dict[str, float] = None) -> Fraction:
    """Compute DQ score from Q1-Q7 integer labels on the 0-3 anchor scale."""
    return rubric.quality_score(label, weights)


def write_rubric_manifest() -> Dict:
    """Write the frozen rubric definitions used by prompts and analysis."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "written_at": datetime.now().isoformat(),
        "purpose": "Frozen ADR compliance rubric for manuscript and replication-package alignment.",
        "scoring_version": rubric.SCORING_VERSION,
        "arithmetic": "Exact rational arithmetic for classification; rounding only for display.",
        "dimension_thresholds": {"Fully_Compliant": ">= 85", "Mostly_Compliant": ">= 70 and < 85", "Partially_Compliant": ">= 45 and < 70", "Not_Compliant": "< 45"},
        "structural_completeness": {
            "description": "Binary C1-C7 checks for the presence of required ADR evidence.",
            "criteria": STRUCTURAL_CRITERIA,
            "weights": SC_WEIGHTS,
            "score_formula": "SC = sum(W_i * C_i) * 100, where C_i is 1 when present and 0 otherwise.",
            "weighting_rationale": (
                "SC checks are binary but not equally weighted. Context/problem, options, "
                "decision justification, and consequences receive higher weight because they "
                "carry more architectural knowledge than title/status metadata."
            ),
        },
        "decision_quality": {
            "description": "Q1-Q7 semantic-quality criteria scored on a 0-3 evidence scale.",
            "criteria": DECISION_QUALITY_CRITERIA,
            "weights": DQ_WEIGHTS,
            "anchors": DQ_SCORE_ANCHORS,
            "score_formula": "DQ = (sum(V_j * Q_j) / 3) * 100, where Q_j is scored 0-3.",
            "anchor_note": (
                "The same 0-3 scale is applied to every DQ criterion, with criterion-specific "
                "interpretation. For example, strong Q5 requires a rationale that connects the "
                "selected option to constraints and trade-offs, while weak Q5 merely asserts a "
                "preference without defensible reasoning."
            ),
        },
        "composite_classification": {
            "formula": "CS = 0.40 * SC + 0.60 * DQ",
            "rules": {
                "Not_Compliant": "CS < 45, or SC < 40 (evaluated first)",
                "Fully_Compliant": "CS >= 85 with SC >= 80 and DQ >= 80",
                "Mostly_Compliant": "CS >= 70 with SC >= 60 and DQ >= 60, if no earlier rule matches",
                "Partially_Compliant": "All remaining cases",
            },
            "structural_floor_rationale": (
                "The SC floor prevents ADRs with missing core structural evidence from being "
                "classified as fully compliant solely because the available rationale text is strong."
            ),
        },
        "construct_validity_note": (
            "The rubric operationalizes MADR and Zimmermann-derived criteria for this benchmark. "
            "Alternative weights or thresholds may change class distributions, so threshold "
            "sensitivity is reported separately."
        ),
    }
    with open(RUBRIC_MANIFEST_PATH, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"  Rubric manifest written to {RUBRIC_MANIFEST_PATH}")
    return manifest


def write_threshold_sensitivity(ground_truth: Dict, eval_adrs: List[Dict] = None) -> Dict:
    """Write class-distribution sensitivity under alternative rubric settings."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if eval_adrs is not None:
        labels = {
            adr["id"]: ground_truth[adr["id"]]
            for adr in eval_adrs
            if adr["id"] in ground_truth
        }
    else:
        labels = ground_truth

    equal_sc = {c: Fraction(1, 7) for c in SC_WEIGHTS}
    equal_dq = {q: Fraction(1, 7) for q in DQ_WEIGHTS}
    scenarios = [
        {
            "name": "current_weighted_40_60_structural_floor_40",
            "sc_weights": SC_WEIGHTS,
            "dq_weights": DQ_WEIGHTS,
            "sc_composite_weight": 0.40,
            "structural_floor": 40.0,
        },
        {
            "name": "equal_sc_weights_current_dq_weights",
            "sc_weights": equal_sc,
            "dq_weights": DQ_WEIGHTS,
            "sc_composite_weight": 0.40,
            "structural_floor": 40.0,
        },
        {
            "name": "current_sc_weights_equal_dq_weights",
            "sc_weights": SC_WEIGHTS,
            "dq_weights": equal_dq,
            "sc_composite_weight": 0.40,
            "structural_floor": 40.0,
        },
        {
            "name": "balanced_sc_dq_50_50",
            "sc_weights": SC_WEIGHTS,
            "dq_weights": DQ_WEIGHTS,
            "sc_composite_weight": 0.50,
            "structural_floor": 40.0,
        },
        {
            "name": "no_structural_floor",
            "sc_weights": SC_WEIGHTS,
            "dq_weights": DQ_WEIGHTS,
            "sc_composite_weight": 0.40,
            "structural_floor": 0.0,
        },
        {
            "name": "stricter_structural_floor_50",
            "sc_weights": SC_WEIGHTS,
            "dq_weights": DQ_WEIGHTS,
            "sc_composite_weight": 0.40,
            "structural_floor": 50.0,
        },
    ]

    scenario_outputs = []
    baseline_predictions = None
    for scenario in scenarios:
        predictions = {}
        score_rows = []
        for adr_id, label in labels.items():
            sc_score = compute_sc_score_from_label(label, scenario["sc_weights"])
            dq_score = compute_dq_score_from_label(label, scenario["dq_weights"])
            overall = classify_overall_from_scores(
                sc_score,
                dq_score,
                sc_weight=scenario["sc_composite_weight"],
                structural_floor=scenario["structural_floor"],
            )
            predictions[adr_id] = overall
            score_rows.append({
                "adr_id": adr_id,
                "sc_score": float(sc_score),
                "dq_score": float(dq_score),
                "overall": overall,
            })

        if baseline_predictions is None:
            baseline_predictions = predictions
            changes_vs_current = 0
        else:
            changes_vs_current = sum(
                1 for adr_id, pred in predictions.items()
                if baseline_predictions.get(adr_id) != pred
            )

        scenario_outputs.append({
            "name": scenario["name"],
            "n": len(predictions),
            "class_distribution": dict(Counter(predictions.values())),
            "changes_vs_current_weighted_rule": changes_vs_current,
            "sc_composite_weight": scenario["sc_composite_weight"],
            "dq_composite_weight": round(1.0 - scenario["sc_composite_weight"], 2),
            "structural_floor": scenario["structural_floor"],
            "score_rows": score_rows,
        })

    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Sensitivity of human-label class distributions to alternative SC/DQ "
            "weights and structural-floor settings. This does not call any model API."
        ),
        "n_labels": len(labels),
        "baseline_recorded_human_distribution": dict(
            Counter(label.get("overall", "Unknown") for label in labels.values())
        ),
        "scenarios": scenario_outputs,
    }
    with open(THRESHOLD_SENSITIVITY_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Threshold sensitivity written to {THRESHOLD_SENSITIVITY_PATH}")
    return report


def _count_and_percent(items: List[str]) -> Dict:
    total = len(items)
    counts = Counter(items)
    return {
        key: {
            "count": count,
            "percent": round(float(count / total * 100), 1) if total else 0.0,
        }
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    }


def _majority_class_macro_f1(labels: List[str]) -> Dict:
    """Compute the always-majority baseline for manuscript consistency checks."""
    from sklearn.metrics import precision_recall_fscore_support

    if not labels:
        return {
            "majority_class": None,
            "macro_f1": None,
            "weighted_f1": None,
        }
    majority_class = Counter(labels).most_common(1)[0][0]
    predictions = [majority_class for _ in labels]
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=CLASSES, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=CLASSES, average="weighted", zero_division=0
    )
    return {
        "majority_class": majority_class,
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }


def _format_class_distribution_sentence(labels: List[str]) -> str:
    """Build a copy-ready class-distribution sentence from the frozen labels."""
    if not labels:
        return "The frozen evaluation set contains 0 ADRs."
    distribution = _count_and_percent(labels)
    parts = []
    for cls in CLASSES:
        if cls in distribution:
            clean = cls.replace("_", " ")
            parts.append(
                f"{distribution[cls]['count']} {clean} "
                f"({distribution[cls]['percent']}%)"
            )
    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The frozen {len(labels)}-ADR evaluation set contains {joined} ADRs."


def write_manuscript_dataset_summary(adrs: List[Dict], ground_truth: Dict,
                                     eval_adrs: List[Dict]) -> Dict:
    """Write dataset numbers intended to be copied into the manuscript."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [ground_truth[adr["id"]]["overall"] for adr in eval_adrs]
    repo_counts = Counter(adr["source_repo"] for adr in eval_adrs)
    variant_counts = Counter(adr.get("variant", "OTHER") for adr in eval_adrs)
    domain_counts = Counter(get_domain(adr["source_repo"]) for adr in eval_adrs)
    lengths = [adr.get("word_count", len(adr.get("text", "").split())) for adr in eval_adrs]

    summary = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Single-source dataset statistics for manuscript text, tables, and "
            "figure captions. Use this file to avoid stale 100-ADR, 176-ADR, "
            "or old class-distribution claims."
        ),
        "source_files": {
            "evaluation_manifest": str(EVAL_SET_PATH),
            "sampling_audit": str(ANALYSIS_DIR / "sampling_audit.json"),
            "human_labels": str(EXPERIMENT_DIR / "human_ground_truth.json"),
        },
        "evaluation_set": {
            "n": len(eval_adrs),
            "class_distribution": _count_and_percent(labels),
            "majority_class_baseline": _majority_class_macro_f1(labels),
            "repository_distribution": _count_and_percent(
                [adr["source_repo"] for adr in eval_adrs]
            ),
            "template_variant_distribution": _count_and_percent(
                [adr.get("variant", "OTHER") for adr in eval_adrs]
            ),
            "domain_distribution": _count_and_percent(
                [get_domain(adr["source_repo"]) for adr in eval_adrs]
            ),
            "length_words": {
                "mean": round(float(np.mean(lengths)), 1) if lengths else 0.0,
                "min": int(min(lengths)) if lengths else 0,
                "max": int(max(lengths)) if lengths else 0,
            },
        },
        "plain_language_values_for_manuscript": {
            "class_distribution_sentence": _format_class_distribution_sentence(labels),
            "repository_count_sentence": (
                f"The evaluation set spans {len(repo_counts)} licensed source repositories, "
                f"{len(variant_counts)} ADR format categories, and {len(domain_counts)} application domains."
            ),
            "majority_baseline_sentence": (
                "For this four-class evaluation set, the majority-class baseline always "
                f"predicts {Counter(labels).most_common(1)[0][0].replace('_', ' ')} "
                f"and has macro-F1 = {_majority_class_macro_f1(labels)['macro_f1']}."
            ) if labels else "",
        },
        "stale_terms_to_audit": [
            "100 ADRs",
            "100-ADR",
            "176 ADRs",
            "194 ADRs",
            "12% Compliant",
            "86% Partially Compliant",
            "2% Non-Compliant",
            "60% Partially Compliant",
            "7/60/33",
            "Partially Compliant class dominates",
            "majority-class baseline always predicting Partially Compliant",
            "Non-Compliant class contains only two",
        ],
        "notes": [
            "Dataset distributions do not require a model/API run once eval_set.json and human_ground_truth.json are frozen.",
            "Performance values, best-model claims, and confusion matrices still require the final complete rerun and merge.",
        ],
    }

    with open(MANUSCRIPT_DATASET_SUMMARY_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  Manuscript dataset summary written to {MANUSCRIPT_DATASET_SUMMARY_PATH}")
    return summary


def combine_dimension_classes(sc_class: str, dq_class: str) -> str:
    """Combine SC and DQ classes conservatively for the hybrid baseline."""
    ranks = {
        "Not_Compliant": 0,
        "Partially_Compliant": 1,
        "Mostly_Compliant": 2,
        "Fully_Compliant": 3,
    }
    inverse = {v: k for k, v in ranks.items()}
    sc_rank = ranks.get(sc_class)
    dq_rank = ranks.get(dq_class)
    if sc_rank is None or dq_rank is None:
        return None
    return inverse[min(sc_rank, dq_rank)]


def _empty_prediction() -> Dict:
    return {
        "overall": None,
        "sc_class": None,
        "dq_class": None,
        "confidence": None,
        "sc_checks": {criterion: None for criterion in SC_WEIGHTS},
        "dq_scores": {criterion: None for criterion in DQ_WEIGHTS},
        "sc_score": None,
        "dq_score": None,
        "composite_score": None,
        "sc_score_from_criteria": None,
        "dq_score_from_criteria": None,
        "composite_score_from_criteria": None,
        "sc_class_from_criteria": None,
        "dq_class_from_criteria": None,
        "overall_from_criteria": None,
        "overall_raw": None,
        "parsed_json": False,
        "criterion_parse_success": False,
    }


def _nested_value(parsed: Dict, key: str, containers: List[str]):
    if key in parsed:
        return parsed.get(key)
    for container in containers:
        value = parsed.get(container)
        if isinstance(value, dict) and key in value:
            return value.get(key)
    return None


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if value is None:
        return None
    val = str(value).strip().lower()
    true_values = {"true", "t", "yes", "y", "pass", "passed", "present", "met", "1"}
    false_values = {
        "false", "f", "no", "n", "fail", "failed", "absent", "missing",
        "not met", "not_met", "0",
    }
    if val in true_values:
        return True
    if val in false_values:
        return False
    return None


def _normalize_dq_score(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = int(round(float(value)))
        return score if 0 <= score <= 3 else None
    val = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if re.fullmatch(r"[0-3](?:\.0+)?", val):
        return int(float(val))
    if val in {"absent", "not_met", "missing", "none", "no_evidence"}:
        return 0
    if val in {"weak", "partially_met", "partial", "limited", "superficial"}:
        return 1
    if val in {"adequate", "mostly_met", "mostly", "acceptable"}:
        return 2
    if val in {"strong", "met", "fully_met", "complete"}:
        return 3
    return None


def _normalize_score(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        score = float(match.group(0))
    if 0 <= score <= 100:
        return round(score, 1)
    return None


def _normalize_confidence(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= confidence <= 1:
        return round(confidence, 4)
    if 1 < confidence <= 100:
        return round(confidence / 100, 4)
    return None


def _computed_sc_score_from_checks(checks: Dict):
    if any(checks.get(criterion) is None for criterion in SC_WEIGHTS):
        return None
    return rubric.structural_score(checks)


def _computed_dq_score_from_scores(scores: Dict):
    if any(scores.get(criterion) is None for criterion in DQ_WEIGHTS):
        return None
    return rubric.quality_score(scores)


def _criteria_available(prediction: Dict) -> bool:
    return (
        all(prediction["sc_checks"].get(criterion) is not None for criterion in SC_WEIGHTS)
        and all(prediction["dq_scores"].get(criterion) is not None for criterion in DQ_WEIGHTS)
    )


def extract_prediction(raw):
    """Extract overall, dimension, and criterion-level predictions."""
    empty = _empty_prediction()
    if not raw:
        return empty

    try:
        cleaned = raw.strip()

        # Remove markdown fences like ```json ... ```
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
            cleaned = re.sub(r"```$", "", cleaned)
            cleaned = cleaned.strip()

        # Extract JSON (greedy match)
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                sc_checks = {
                    criterion: _normalize_bool(
                        _nested_value(
                            parsed,
                            criterion,
                            ["structural_checks", "sc_checks", "criteria"],
                        )
                    )
                    for criterion in SC_WEIGHTS
                }
                dq_scores = {
                    criterion: _normalize_dq_score(
                        _nested_value(
                            parsed,
                            criterion,
                            ["decision_quality_scores", "dq_scores", "quality_scores", "criteria"],
                        )
                    )
                    for criterion in DQ_WEIGHTS
                }
                sc_score_from_criteria = _computed_sc_score_from_checks(sc_checks)
                dq_score_from_criteria = _computed_dq_score_from_scores(dq_scores)
                sc_score = _normalize_score(parsed.get("sc_score"))
                dq_score = _normalize_score(parsed.get("dq_score"))
                if sc_score is None:
                    sc_score = sc_score_from_criteria
                if dq_score is None:
                    dq_score = dq_score_from_criteria
                composite_score_from_criteria = (
                    compute_composite_score(sc_score_from_criteria, dq_score_from_criteria)
                    if sc_score_from_criteria is not None and dq_score_from_criteria is not None
                    else None
                )
                composite_score = _normalize_score(parsed.get("composite_score"))
                if composite_score is None:
                    composite_score = composite_score_from_criteria
                sc_class_from_criteria = (
                    score_to_compliance_class(sc_score_from_criteria)
                    if sc_score_from_criteria is not None else None
                )
                dq_class_from_criteria = (
                    score_to_compliance_class(dq_score_from_criteria)
                    if dq_score_from_criteria is not None else None
                )
                overall_from_criteria = (
                    classify_overall_from_scores(sc_score_from_criteria, dq_score_from_criteria)
                    if sc_score_from_criteria is not None and dq_score_from_criteria is not None
                    else None
                )
                overall_raw = normalize_compliance_class(parsed.get("overall"))
                criteria_complete = sc_score_from_criteria is not None and dq_score_from_criteria is not None

                prediction = {
                    "overall": overall_from_criteria if criteria_complete else overall_raw,
                    "overall_raw": overall_raw,
                    "sc_class": sc_class_from_criteria or normalize_compliance_class(parsed.get("sc_class")),
                    "dq_class": dq_class_from_criteria or normalize_compliance_class(parsed.get("dq_class")),
                    "confidence": _normalize_confidence(parsed.get("confidence")),
                    "sc_checks": sc_checks,
                    "dq_scores": dq_scores,
                    "sc_score": sc_score,
                    "dq_score": dq_score,
                    "composite_score": composite_score,
                    "sc_score_from_criteria": sc_score_from_criteria,
                    "dq_score_from_criteria": dq_score_from_criteria,
                    "composite_score_from_criteria": composite_score_from_criteria,
                    "sc_class_from_criteria": sc_class_from_criteria,
                    "dq_class_from_criteria": dq_class_from_criteria,
                    "overall_from_criteria": overall_from_criteria,
                    "parsed_json": True,
                    "criterion_parse_success": False,
                }
                prediction["criterion_parse_success"] = _criteria_available(prediction)
                if criteria_complete:
                    evaluated = rubric.evaluate_criteria(sc_checks, dq_scores)
                    prediction["scoring_version"] = rubric.SCORING_VERSION
                    prediction["exact_scores"] = evaluated["exact_scores"]
                    for key in ("sc_score", "dq_score", "composite_score"):
                        prediction[key] = evaluated[key]
                # Fractions remain internal; exact values are also recorded as strings.
                for key, value in prediction.items():
                    if isinstance(value, Fraction):
                        prediction[key] = float(value)
                return prediction

    except Exception:
        pass

    raw_lower = raw.lower()
    fallback = empty.copy()

    if "fully_compliant" in raw_lower or "fully compliant" in raw_lower:
        fallback["overall"] = "Fully_Compliant"
    elif "mostly_compliant" in raw_lower or "mostly compliant" in raw_lower:
        fallback["overall"] = "Mostly_Compliant"
    elif "partially_compliant" in raw_lower or "partially compliant" in raw_lower:
        fallback["overall"] = "Partially_Compliant"
    elif (
        "not_compliant" in raw_lower or "not compliant" in raw_lower
        or "non_compliant" in raw_lower or "non-compliant" in raw_lower
    ):
        fallback["overall"] = "Not_Compliant"

    fallback["overall_raw"] = fallback["overall"]
    return fallback


def extract_classification(raw):
    return extract_prediction(raw).get("overall")


def build_result_row(adr: Dict, actual: str, prediction: Dict, result: Dict) -> Dict:
    """Build the schema-versioned row saved by final runs and smoke tests."""
    predicted = prediction.get("overall")
    usage = result.get("usage", {"input_tokens": 0, "output_tokens": 0})
    input_audit = result.get("input_audit") or {}
    model_metadata = result.get("model_metadata") or {}
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "scoring_version": prediction.get("scoring_version"),
        "predicted_exact_scores": prediction.get("exact_scores"),
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "protocol_fingerprint": result.get("protocol_fingerprint"),
        "run_id": ACTIVE_RUN_ID,
        "model_key": model_metadata.get("model_key"),
        "strategy": result.get("strategy"),
        "adr_id": adr["id"],
        "actual": actual,
        "predicted": predicted,
        "predicted_sc_class": prediction.get("sc_class"),
        "predicted_dq_class": prediction.get("dq_class"),
        "predicted_sc_checks": prediction.get("sc_checks"),
        "predicted_dq_scores": prediction.get("dq_scores"),
        "predicted_sc_score": prediction.get("sc_score"),
        "predicted_dq_score": prediction.get("dq_score"),
        "predicted_composite_score": prediction.get("composite_score"),
        "predicted_sc_score_from_criteria": prediction.get("sc_score_from_criteria"),
        "predicted_dq_score_from_criteria": prediction.get("dq_score_from_criteria"),
        "predicted_composite_score_from_criteria": prediction.get("composite_score_from_criteria"),
        "predicted_sc_class_from_criteria": prediction.get("sc_class_from_criteria"),
        "predicted_dq_class_from_criteria": prediction.get("dq_class_from_criteria"),
        "predicted_overall_from_criteria": prediction.get("overall_from_criteria"),
        "predicted_overall_raw": prediction.get("overall_raw"),
        "predicted_confidence": prediction.get("confidence"),
        "correct": predicted == actual if predicted else False,
        "accessed_at": result.get("accessed_at"),
        "model_metadata": model_metadata,
        "finish_reason": result.get("finish_reason"),
        "usage_details": {
            key: value
            for key, value in usage.items()
            if key not in {"input_tokens", "output_tokens"}
        },
        "latency": result.get("latency"),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "adr_source_chars": input_audit.get("adr_source_chars"),
        "adr_input_chars": input_audit.get("adr_input_chars"),
        "adr_input_utf8_bytes": input_audit.get("adr_input_utf8_bytes"),
        "adr_input_sha256": input_audit.get("adr_input_sha256"),
        "prompt_chars": input_audit.get("prompt_chars"),
        "prompt_utf8_bytes": input_audit.get("prompt_utf8_bytes"),
        "prompt_sha256": input_audit.get("prompt_sha256"),
        "request_text_sha256": input_audit.get("request_text_sha256"),
        "input_was_truncated": input_audit.get("input_was_truncated"),
        "attempts": result.get("attempts", 1),
        "retry_count": result.get("retry_count", 0),
        "output_retry_count": result.get("output_retry_count", 0),
        "attempt_history": result.get("attempt_history"),
        "billing_mode": result.get("billing_mode", "sync"),
        "cost_multiplier": result.get("cost_multiplier", 1.0),
        "batch_metadata": result.get("batch_metadata"),
        "error": result.get("error"),
        "parse_success": predicted is not None,
        "parsed_json": prediction.get("parsed_json", False),
        "criterion_parse_success": prediction.get("criterion_parse_success", False),
    }


def _pending_repair_row(adr: Dict, actual: str, model_name: str,
                        strategy: str, rep_idx: int, reason: str) -> Dict:
    """Create a schema-valid placeholder row that the normal repair path reruns."""
    return build_result_row(
        adr,
        actual,
        {},
        {
            "accessed_at": datetime.now().isoformat(),
            "model_metadata": model_reproducibility_row(model_name),
            "strategy": strategy,
            "protocol_fingerprint": protocol_fingerprint(model_name, strategy),
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "attempts": 0,
            "retry_count": 0,
            "output_retry_count": 0,
            "billing_mode": "repair_placeholder",
            "cost_multiplier": 0.0,
            "error": f"pending repair: {reason}; rep={rep_idx}",
        },
    )


def prepare_raw_results_for_manifest_repair(
    adrs: List[Dict],
    ground_truth: Dict,
    targets: List[tuple] = None,
    n_reps: int = N_REPS,
) -> Dict:
    """
    Rewrite raw result files so they match the active eval_set shape.

    This is used after a small manifest correction. It preserves reusable rows
    for unchanged ADR IDs, drops rows for ADRs no longer in the manifest, and
    inserts repair placeholders for newly added or missing ADRs. The normal
    targeted runner then repairs only placeholder, API-error, or incomplete rows.
    """
    eval_adrs = _load_eval_adrs_from_manifest(adrs)
    expected_ids = [adr["id"] for adr in eval_adrs]
    expected_set = set(expected_ids)
    active_targets = targets if targets is not None else [
        (model_name, strategy)
        for model_name in MODELS
        for strategy in STRATEGIES
    ]

    archive_dir = (
        RUN_DIR
        / "_archive"
        / f"raw_results_before_manifest_repair_{datetime.now():%Y%m%d_%H%M%S}"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "written_at": datetime.now().isoformat(),
        "eval_set_n": len(expected_ids),
        "n_reps": n_reps,
        "archive_dir": str(archive_dir),
        "targets": {},
    }

    for model_name, strategy in active_targets:
        result_file = RESULTS_DIR / f"{model_name}_{strategy}.json"
        old_reps = []
        file_status = "missing"
        if result_file.exists():
            shutil.copy2(result_file, archive_dir / result_file.name)
            try:
                with open(result_file, encoding="utf-8") as fp:
                    old_reps = json.load(fp)
                file_status = "loaded"
            except Exception as exc:
                file_status = f"json_error: {exc}"
                old_reps = []

        new_reps = []
        target_report = {
            "file_status": file_status,
            "raw_result_file": str(result_file),
            "old_rep_count": len(old_reps) if isinstance(old_reps, list) else 0,
            "reused_rows": 0,
            "placeholder_rows": 0,
            "stale_extra_rows_dropped": 0,
            "incomplete_rows_to_repair": 0,
            "per_rep": [],
        }

        if not isinstance(old_reps, list):
            old_reps = []

        for rep_idx in range(1, n_reps + 1):
            old_rep = old_reps[rep_idx - 1] if rep_idx <= len(old_reps) else []
            if not isinstance(old_rep, list):
                old_rep = []

            old_by_id = {
                row.get("adr_id"): row
                for row in old_rep
                if isinstance(row, dict) and row.get("adr_id")
            }
            stale_extras = [
                row.get("adr_id")
                for row in old_rep
                if isinstance(row, dict) and row.get("adr_id") not in expected_set
            ]

            new_rep = []
            rep_reused = 0
            rep_placeholders = 0
            for adr in eval_adrs:
                actual = ground_truth[adr["id"]]["overall"]
                existing = old_by_id.get(adr["id"])
                if existing is None:
                    row = _pending_repair_row(
                        adr, actual, model_name, strategy, rep_idx,
                        "ADR not present in previous raw result"
                    )
                    rep_placeholders += 1
                else:
                    row = dict(existing)
                    row["actual"] = actual
                    row["correct"] = (
                        row.get("predicted") == actual
                        if row.get("predicted")
                        else False
                    )
                    rep_reused += 1
                new_rep.append(row)

            rep_incomplete = sum(
                1
                for row_idx, row in enumerate(new_rep, 1)
                if _result_row_validation_reason(row, rep_idx, row_idx)
            )
            new_reps.append(new_rep)

            target_report["reused_rows"] += rep_reused
            target_report["placeholder_rows"] += rep_placeholders
            target_report["stale_extra_rows_dropped"] += len(stale_extras)
            target_report["incomplete_rows_to_repair"] += rep_incomplete
            target_report["per_rep"].append({
                "rep": rep_idx,
                "reused_rows": rep_reused,
                "placeholder_rows": rep_placeholders,
                "stale_extra_rows_dropped": len(stale_extras),
                "incomplete_rows_to_repair": rep_incomplete,
                "stale_extra_preview": stale_extras[:10],
            })

        valid_shape, shape_reason = _validate_result_shape(
            new_reps,
            expected_ids,
            n_reps,
            require_complete=True,
        )
        target_report["shape_valid"] = valid_shape
        target_report["shape_validation_reason"] = shape_reason
        if not valid_shape:
            raise RuntimeError(
                f"Repair preparation produced invalid shape for "
                f"{model_name}/{strategy}: {shape_reason}"
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _save_rep_results(result_file, new_reps)
        report["targets"][f"{model_name}_{strategy}"] = target_report
        print(
            f"  PREP {model_name}/{strategy}: reused "
            f"{target_report['reused_rows']} rows, placeholders "
            f"{target_report['placeholder_rows']}, incomplete to repair "
            f"{target_report['incomplete_rows_to_repair']}"
        )

    report_path = ANALYSIS_DIR / "raw_result_manifest_repair_plan.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Raw-result repair plan written to {report_path}")
    return report


def _first_heading_or_line(text: str) -> str:
    """Return the most plausible ADR title from Markdown text."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if not stripped.startswith(("-", "*", "|", "`")):
            return stripped
    return ""


def _has_heading_or_keyword(text_lower: str, terms: List[str]) -> bool:
    for term in terms:
        heading_pattern = rf"(?m)^\s*#+\s*{re.escape(term)}\b"
        if re.search(heading_pattern, text_lower):
            return True
        if term in text_lower:
            return True
    return False


def _count_option_markers(text_lower: str) -> int:
    option_markers = re.findall(
        r"(?m)^\s*(?:#+\s*)?(?:option|alternative)\s+[0-9a-zivx]+[\).:\- ]",
        text_lower,
    )
    bullet_markers = re.findall(
        r"(?m)^\s*[-*]\s+(?:option|alternative)\b",
        text_lower,
    )
    numbered_markers = re.findall(
        r"(?m)^\s*\d+[\).]\s+",
        text_lower,
    )
    return len(option_markers) + len(bullet_markers) + len(numbered_markers)


def rule_based_structural_checks(text: str) -> Dict[str, bool]:
    """
    Deterministic structural baseline for C1-C7.

    These checks intentionally use transparent section and keyword patterns.
    They are a baseline for structural completeness only, not a semantic
    substitute for human decision-quality review.
    """
    text_lower = text.lower()
    title = _first_heading_or_line(text)
    title_words = re.findall(r"[a-zA-Z0-9]+", title)
    generic_title = bool(re.fullmatch(r"(adr|decision|record|template|index)[\s\-_#0-9]*", title.lower()))

    status_values = [
        "accepted", "proposed", "draft", "deprecated", "superseded",
        "rejected", "amended", "decided",
    ]
    status_present = bool(
        re.search(r"(?m)^\s*(?:#+\s*)?status\s*[:\-]?\s*(accepted|proposed|draft|deprecated|superseded|rejected|amended|decided)\b", text_lower)
        or re.search(r"\bstatus\s*[:\-]\s*(accepted|proposed|draft|deprecated|superseded|rejected|amended|decided)\b", text_lower)
        or any(re.search(rf"(?m)^\s*#+\s*{value}\b", text_lower) for value in status_values)
    )

    decision_present = _has_heading_or_keyword(
        text_lower,
        ["decision", "decision outcome", "outcome", "chosen option", "we decided", "we will"],
    )
    rationale_present = any(
        term in text_lower
        for term in ["because", "therefore", "so that", "rationale", "reason", "justification"]
    )
    options_present = (
        _count_option_markers(text_lower) >= 2
        or _has_heading_or_keyword(text_lower, ["considered options", "options", "alternatives"])
        and len(re.findall(r"\b(option|alternative)\b", text_lower)) >= 2
    )

    return {
        "C1": len(title_words) >= 3 and not generic_title,
        "C2": _has_heading_or_keyword(
            text_lower,
            ["context", "problem", "background", "motivation", "issue", "need"],
        ),
        "C3": _has_heading_or_keyword(
            text_lower,
            ["decision drivers", "drivers", "forces", "constraints", "criteria", "requirements", "considerations"],
        ),
        "C4": options_present,
        "C5": decision_present and rationale_present,
        "C6": _has_heading_or_keyword(
            text_lower,
            ["consequences", "positive consequences", "negative consequences", "pros", "cons", "trade-offs", "tradeoffs", "implications"],
        ),
        "C7": status_present,
    }


def score_structural_checks(checks: Dict[str, bool]) -> float:
    return float(rubric.structural_score(checks))


def _load_eval_adrs_from_manifest(adrs: List[Dict] = None) -> List[Dict]:
    """Load ADR records in the exact order recorded by results/eval_set.json."""
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"{EVAL_SET_PATH} not found")
    if adrs is None:
        adrs = load_adrs()
    with open(EVAL_SET_PATH, encoding="utf-8") as fp:
        manifest = json.load(fp)
    adrs_by_id = {a["id"]: a for a in adrs}
    missing = [row["id"] for row in manifest.get("adrs", []) if row["id"] not in adrs_by_id]
    if missing:
        raise ValueError(f"{len(missing)} eval_set ADR IDs are missing from results/adrs")
    return [adrs_by_id[row["id"]] for row in manifest.get("adrs", [])]


def write_rule_based_baseline(eval_adrs: List[Dict], ground_truth: Dict,
                              all_results: Dict = None) -> Dict:
    """Write rule-only SC and optional hybrid rule-SC plus LLM-DQ metrics."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        precision_recall_fscore_support,
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for adr in eval_adrs:
        label = ground_truth[adr["id"]]
        checks = rule_based_structural_checks(adr.get("text", ""))
        sc_score = score_structural_checks(checks)
        human_sc_score = compute_sc_score_from_label(label)
        rows.append({
            "adr_id": adr["id"],
            "source_repo": adr.get("source_repo"),
            "variant": adr.get("variant", "OTHER"),
            "domain": get_domain(adr.get("source_repo", "")),
            "rule_checks": checks,
            "rule_sc_score": sc_score,
            "rule_sc_class": score_to_compliance_class(sc_score),
            "human_checks": {c: bool(label.get(c)) for c in SC_WEIGHTS},
            "human_sc_score": float(human_sc_score),
            "human_sc_class": score_to_compliance_class(human_sc_score),
            "human_dq_class": score_to_compliance_class(compute_dq_score_from_label(label)),
            "human_overall": label.get("overall"),
        })

    y_true_sc = [r["human_sc_class"] for r in rows]
    y_pred_sc = [r["rule_sc_class"] for r in rows]
    p_sc, r_sc, f1_sc, _ = precision_recall_fscore_support(
        y_true_sc, y_pred_sc, labels=CLASSES, average="macro", zero_division=0
    )

    criterion_agreement = {}
    for criterion in SC_WEIGHTS:
        matches = [
            r["rule_checks"][criterion] == r["human_checks"][criterion]
            for r in rows
        ]
        criterion_agreement[criterion] = {
            "accuracy": float(sum(matches) / len(matches)),
            "rule_pass_rate": float(sum(r["rule_checks"][criterion] for r in rows) / len(rows)),
            "human_pass_rate": float(sum(r["human_checks"][criterion] for r in rows) / len(rows)),
        }

    baseline = {
        "written_at": datetime.now().isoformat(),
        "n_eval": len(rows),
        "purpose": (
            "Deterministic baseline for Structural Completeness only. Decision "
            "Quality remains evaluated by human labels and LLM outputs because it "
            "requires semantic judgement."
        ),
        "rule_based_structural_completeness": {
            "accuracy": float(accuracy_score(y_true_sc, y_pred_sc)),
            "macro_precision": float(p_sc),
            "macro_recall": float(r_sc),
            "macro_f1": float(f1_sc),
            "cohen_kappa": float(cohen_kappa_score(y_true_sc, y_pred_sc, labels=CLASSES)),
            "class_distribution_predicted": dict(Counter(y_pred_sc)),
            "class_distribution_human_sc": dict(Counter(y_true_sc)),
        },
        "criterion_agreement": criterion_agreement,
        "rule_definitions": {
            "C1": "first heading or first non-empty line has at least three alphanumeric tokens and is not a generic template/index title",
            "C2": "context/problem/background/motivation/issue/need heading or keyword present",
            "C3": "decision-driver, force, constraint, criterion, requirement, or consideration heading/keyword present",
            "C4": "at least two option/alternative markers, or an options/alternatives section with repeated option terminology",
            "C5": "decision/outcome wording plus because/therefore/so that/rationale/reason/justification wording",
            "C6": "consequence, positive/negative consequence, pro/con, trade-off, tradeoff, or implication wording present",
            "C7": "status field/heading with accepted/proposed/draft/deprecated/superseded/rejected/amended/decided",
        },
        "rows": rows,
    }

    if all_results:
        baseline["hybrid_rule_sc_plus_llm_dq"] = _compute_hybrid_baseline(rows, all_results)
    else:
        baseline["hybrid_rule_sc_plus_llm_dq"] = {
            "status": "not_computed_until_model_results_are_available",
            "note": "Run --phase merge after final model evaluation to add hybrid rule-SC plus LLM-DQ metrics.",
        }

    with open(RULE_BASELINE_PATH, "w", encoding="utf-8") as fp:
        json.dump(baseline, fp, indent=2)
    print(f"  Rule-based structural baseline written to {RULE_BASELINE_PATH}")
    return baseline


def _compute_hybrid_baseline(rule_rows: List[Dict], all_results: Dict) -> Dict:
    """Combine deterministic rule SC classes with parsed LLM DQ classes."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        precision_recall_fscore_support,
    )

    rule_by_id = {r["adr_id"]: r for r in rule_rows}
    hybrid = {}
    missing_dq_predictions = 0

    for model_name, strategies in all_results.items():
        hybrid[model_name] = {}
        for strategy, reps in strategies.items():
            rep_metrics = []
            for rep_data in reps:
                y_true, y_pred = [], []
                for row in rep_data:
                    rule_row = rule_by_id.get(row.get("adr_id"))
                    dq_class = normalize_compliance_class(row.get("predicted_dq_class"))
                    if not rule_row or not dq_class:
                        missing_dq_predictions += 1
                        continue
                    hybrid_pred = combine_dimension_classes(rule_row["rule_sc_class"], dq_class)
                    if not hybrid_pred:
                        continue
                    y_true.append(rule_row["human_overall"])
                    y_pred.append(hybrid_pred)

                if len(y_true) < 10 and not (VALIDATION_MODE and y_true):
                    continue

                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
                )
                rep_metrics.append({
                    "n": len(y_true),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_precision": float(p),
                    "macro_recall": float(r),
                    "macro_f1": float(f1),
                    "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=CLASSES)),
                })

            if rep_metrics:
                hybrid[model_name][strategy] = {
                    "reps": len(rep_metrics),
                    "n_per_rep": [m["n"] for m in rep_metrics],
                    "accuracy_mean": round(float(np.mean([m["accuracy"] for m in rep_metrics])), 3),
                    "macro_f1_mean": round(float(np.mean([m["macro_f1"] for m in rep_metrics])), 3),
                    "macro_f1_std": round(float(np.std(
                        [m["macro_f1"] for m in rep_metrics], ddof=1
                    )), 3) if len(rep_metrics) > 1 else 0.0,
                    "cohen_kappa_mean": round(float(np.mean([m["cohen_kappa"] for m in rep_metrics])), 3),
                    "cohen_kappa_std": round(float(np.std(
                        [m["cohen_kappa"] for m in rep_metrics], ddof=1
                    )), 3) if len(rep_metrics) > 1 else 0.0,
                }
            else:
                hybrid[model_name][strategy] = {
                    "status": "not_computed",
                    "reason": "No parsed predicted_dq_class values available for this run.",
                }

    return {
        "status": "computed" if any(
            isinstance(v, dict) and "macro_f1_mean" in v
            for strategies in hybrid.values()
            for v in strategies.values()
        ) else "not_computed",
        "note": (
            "Hybrid predictions use deterministic rule_sc_class and each LLM row's "
            "predicted_dq_class. Older raw results must be rerun if they do not "
            "contain predicted_dq_class."
        ),
        "missing_dq_predictions": missing_dq_predictions,
        "by_model_strategy": hybrid,
    }


def _exact_mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant correct/incorrect pairs."""
    from scipy.stats import binomtest

    discordant = b + c
    if discordant == 0:
        return 1.0
    return float(binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)


def _configuration_rows(all_results: Dict) -> Dict:
    """Flatten result rows by model, strategy, and repetition."""
    configs = {}
    for model_name, strategies in all_results.items():
        for strategy, reps in strategies.items():
            key = f"{model_name}/{strategy}"
            configs[key] = {
                "model": model_name,
                "strategy": strategy,
                "reps": reps,
            }
    return configs


def _metrics_from_confusion_batches(confusions: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorized macro-F1 and Cohen's kappa for B x class x class matrices."""
    matrices = np.asarray(confusions, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis, :, :]

    true_support = matrices.sum(axis=2)
    predicted_support = matrices.sum(axis=1)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive),
        where=true_support != 0,
    )
    per_class_f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    macro_f1 = per_class_f1.mean(axis=1)

    totals = matrices.sum(axis=(1, 2))
    observed = np.divide(
        true_positive.sum(axis=1),
        totals,
        out=np.zeros_like(totals),
        where=totals != 0,
    )
    expected = np.divide(
        (true_support * predicted_support).sum(axis=1),
        totals ** 2,
        out=np.zeros_like(totals),
        where=totals != 0,
    )
    kappa = np.divide(
        observed - expected,
        1 - expected,
        out=np.full_like(observed, np.nan),
        where=(1 - expected) != 0,
    )
    return {"macro_f1": macro_f1, "cohen_kappa": kappa}


def _paired_adr_metric_inference(cfg_a: Dict, cfg_b: Dict,
                                 n_resamples: int = 10000,
                                 seed: int = 20260908) -> Dict:
    """Compare configurations while treating ADR, not repeated output, as the unit."""
    reps_to_compare = min(len(cfg_a["reps"]), len(cfg_b["reps"]))
    if not reps_to_compare:
        return {"status": "not_computed", "reason": "No repetitions to compare."}

    rep_maps_a = [
        {row["adr_id"]: row for row in rep if row.get("predicted") and row.get("actual")}
        for rep in cfg_a["reps"][:reps_to_compare]
    ]
    rep_maps_b = [
        {row["adr_id"]: row for row in rep if row.get("predicted") and row.get("actual")}
        for rep in cfg_b["reps"][:reps_to_compare]
    ]
    id_sets = [set(rows) for rows in rep_maps_a + rep_maps_b]
    common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
    if len(common_ids) < 2:
        return {
            "status": "not_computed",
            "reason": "Fewer than two ADRs have predictions in every compared repetition.",
        }

    class_to_index = {label: idx for idx, label in enumerate(CLASSES)}
    reference_actual = []
    for adr_id in common_ids:
        labels = {
            rows[adr_id]["actual"]
            for rows in rep_maps_a + rep_maps_b
        }
        if len(labels) != 1 or next(iter(labels)) not in class_to_index:
            raise ValueError(f"Inconsistent or invalid actual label for {adr_id}: {labels}")
        reference_actual.append(class_to_index[next(iter(labels))])

    y_true = np.asarray(reference_actual, dtype=np.int16)
    predictions_a = [
        np.asarray([class_to_index[rows[adr_id]["predicted"]] for adr_id in common_ids], dtype=np.int16)
        for rows in rep_maps_a
    ]
    predictions_b = [
        np.asarray([class_to_index[rows[adr_id]["predicted"]] for adr_id in common_ids], dtype=np.int16)
        for rows in rep_maps_b
    ]
    n_adrs = len(common_ids)
    eye = np.eye(len(CLASSES) ** 2, dtype=np.int16)

    rng = np.random.default_rng(seed)
    bootstrap_counts = rng.multinomial(
        n_adrs,
        np.full(n_adrs, 1.0 / n_adrs),
        size=n_resamples,
    )
    adr_swaps = rng.integers(0, 2, size=(n_resamples, n_adrs), dtype=np.int16)

    observed_by_config = {"a": defaultdict(list), "b": defaultdict(list)}
    bootstrap_by_config = {"a": defaultdict(list), "b": defaultdict(list)}
    randomized_by_config = {"a": defaultdict(list), "b": defaultdict(list)}

    for pred_a, pred_b in zip(predictions_a, predictions_b):
        encoded_a = y_true * len(CLASSES) + pred_a
        encoded_b = y_true * len(CLASSES) + pred_b
        indicators_a = eye[encoded_a]
        indicators_b = eye[encoded_b]
        observed_a = indicators_a.sum(axis=0).reshape(len(CLASSES), len(CLASSES))
        observed_b = indicators_b.sum(axis=0).reshape(len(CLASSES), len(CLASSES))
        bootstrap_a = (bootstrap_counts @ indicators_a).reshape(
            n_resamples, len(CLASSES), len(CLASSES)
        )
        bootstrap_b = (bootstrap_counts @ indicators_b).reshape(
            n_resamples, len(CLASSES), len(CLASSES)
        )

        delta = indicators_b.astype(np.int32) - indicators_a.astype(np.int32)
        randomized_a = (
            observed_a.reshape(1, -1) + adr_swaps.astype(np.int32) @ delta
        ).reshape(n_resamples, len(CLASSES), len(CLASSES))
        randomized_b = (
            observed_b.reshape(1, -1) - adr_swaps.astype(np.int32) @ delta
        ).reshape(n_resamples, len(CLASSES), len(CLASSES))

        for metric, values in _metrics_from_confusion_batches(observed_a).items():
            observed_by_config["a"][metric].append(values[0])
        for metric, values in _metrics_from_confusion_batches(observed_b).items():
            observed_by_config["b"][metric].append(values[0])
        for metric, values in _metrics_from_confusion_batches(bootstrap_a).items():
            bootstrap_by_config["a"][metric].append(values)
        for metric, values in _metrics_from_confusion_batches(bootstrap_b).items():
            bootstrap_by_config["b"][metric].append(values)
        for metric, values in _metrics_from_confusion_batches(randomized_a).items():
            randomized_by_config["a"][metric].append(values)
        for metric, values in _metrics_from_confusion_batches(randomized_b).items():
            randomized_by_config["b"][metric].append(values)

    metrics = {}

    def mean_across_repetitions(values: List[np.ndarray]) -> np.ndarray:
        """Average finite repetition values without warning on empty columns."""
        stacked = np.vstack(values).astype(float)
        counts = np.sum(np.isfinite(stacked), axis=0)
        totals = np.nansum(stacked, axis=0)
        return np.divide(
            totals,
            counts,
            out=np.full(totals.shape, np.nan, dtype=float),
            where=counts > 0,
        )

    for metric in ("macro_f1", "cohen_kappa"):
        observed_a = float(np.nanmean(observed_by_config["a"][metric]))
        observed_b = float(np.nanmean(observed_by_config["b"][metric]))
        observed_difference = observed_a - observed_b
        bootstrap_difference = (
            mean_across_repetitions(bootstrap_by_config["a"][metric])
            - mean_across_repetitions(bootstrap_by_config["b"][metric])
        )
        randomized_difference = (
            mean_across_repetitions(randomized_by_config["a"][metric])
            - mean_across_repetitions(randomized_by_config["b"][metric])
        )
        finite_bootstrap = bootstrap_difference[np.isfinite(bootstrap_difference)]
        finite_randomized = randomized_difference[np.isfinite(randomized_difference)]
        p_value = (
            (1 + int(np.sum(np.abs(finite_randomized) >= abs(observed_difference))))
            / (len(finite_randomized) + 1)
            if len(finite_randomized)
            else None
        )
        metrics[metric] = {
            "config_a_mean_across_repetitions": round(observed_a, 6),
            "config_b_mean_across_repetitions": round(observed_b, 6),
            "difference_a_minus_b": round(observed_difference, 6),
            "adr_cluster_bootstrap_ci95": {
                "low": round(float(np.percentile(finite_bootstrap, 2.5)), 6),
                "high": round(float(np.percentile(finite_bootstrap, 97.5)), 6),
            } if len(finite_bootstrap) else {"low": None, "high": None},
            "adr_level_paired_randomization_p_unadjusted": (
                round(float(p_value), 6) if p_value is not None else None
            ),
        }

    return {
        "status": "computed",
        "unit_of_resampling": "ADR",
        "n_adr_clusters": n_adrs,
        "repetitions_compared": reps_to_compare,
        "bootstrap_resamples": n_resamples,
        "randomization_resamples": n_resamples,
        "randomization_scheme": (
            "For each resample, configuration assignments are swapped as one block "
            "for each ADR across all compared repetitions."
        ),
        "metrics": metrics,
    }


def _summary_stats(values: List[float]) -> Dict:
    """Mean, sample standard deviation, and 95% t interval for repetition values."""
    from scipy.stats import t

    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        half_width = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        ci_low = mean - half_width
        ci_high = mean + half_width
    else:
        ci_low = None
        ci_high = None

    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def write_detailed_performance_report(all_results: Dict) -> Dict:
    """Write per-class metrics, confidence intervals, and confusion matrices."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    configs = _configuration_rows(all_results)
    report_configs = {}

    for config_name, config in sorted(configs.items()):
        rep_summaries = []
        overall_values = defaultdict(list)
        class_values = {
            cls: defaultdict(list)
            for cls in CLASSES
        }
        summed_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

        for rep_idx, rep_data in enumerate(config["reps"], 1):
            rows = [row for row in rep_data if row.get("predicted")]
            y_true = [row["actual"] for row in rows]
            y_pred = [row["predicted"] for row in rows]
            if len(y_true) < 10 and not (VALIDATION_MODE and y_true):
                rep_summaries.append({
                    "rep": rep_idx,
                    "status": "skipped_too_few_parsed_predictions",
                    "n_parsed": len(y_true),
                })
                continue

            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
            )
            _, _, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0
            )
            p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
                y_true, y_pred, labels=CLASSES, average=None, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
            summed_cm += cm
            accuracy = accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred, labels=CLASSES)

            overall_values["accuracy"].append(accuracy)
            overall_values["macro_precision"].append(p_macro)
            overall_values["macro_recall"].append(r_macro)
            overall_values["macro_f1"].append(f1_macro)
            overall_values["weighted_f1"].append(f1_weighted)
            overall_values["cohen_kappa"].append(kappa)

            per_class = {}
            for idx, cls in enumerate(CLASSES):
                per_class[cls] = {
                    "precision": float(p_cls[idx]),
                    "recall": float(r_cls[idx]),
                    "f1": float(f1_cls[idx]),
                    "support": int(support_cls[idx]),
                }
                class_values[cls]["precision"].append(p_cls[idx])
                class_values[cls]["recall"].append(r_cls[idx])
                class_values[cls]["f1"].append(f1_cls[idx])
                class_values[cls]["support"].append(float(support_cls[idx]))

            rep_summaries.append({
                "rep": rep_idx,
                "n_parsed": len(y_true),
                "parse_failures": len(rep_data) - len(rows),
                "accuracy": float(accuracy),
                "macro_precision": float(p_macro),
                "macro_recall": float(r_macro),
                "macro_f1": float(f1_macro),
                "weighted_f1": float(f1_weighted),
                "cohen_kappa": float(kappa),
                "per_class": per_class,
                "confusion_matrix": {
                    "labels": CLASSES,
                    "matrix": cm.tolist(),
                },
            })

        report_configs[config_name] = {
            "model": config["model"],
            "strategy": config["strategy"],
            "repetitions": len(config["reps"]),
            "overall_summary": {
                metric: _summary_stats(values)
                for metric, values in overall_values.items()
            },
            "per_class_summary": {
                cls: {
                    metric: _summary_stats(values)
                    for metric, values in metrics.items()
                }
                for cls, metrics in class_values.items()
            },
            "summed_confusion_matrix": {
                "labels": CLASSES,
                "matrix": summed_cm.tolist(),
            },
            "per_repetition": rep_summaries,
        }

    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Detailed model-performance report for manuscript tables: per-class "
            "precision, recall, F1, repetition summaries, 95% confidence intervals, "
            "and confusion matrices."
        ),
        "confidence_interval_method": (
            "95% t interval across completed repetitions for metrics with more "
            "than one repetition; null when only one repetition is available."
        ),
        "labels": CLASSES,
        "configurations": report_configs,
    }

    with open(PERFORMANCE_DETAILS_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Detailed performance report written to {PERFORMANCE_DETAILS_PATH}")
    return report


def _safe_metric(value, digits=None):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value) if digits is None else round(float(value), digits)


def _safe_cohen_kappa(y_true: List, y_pred: List, labels: List, weights: str = None):
    from sklearn.metrics import cohen_kappa_score

    if not y_true or (len(set(y_true)) == 1 and len(set(y_pred)) == 1):
        return None
    try:
        return _safe_metric(cohen_kappa_score(
            y_true, y_pred, labels=labels, weights=weights
        ))
    except Exception:
        return None


def _binary_criterion_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    if not y_true:
        return {
            "status": "not_computed",
            "n": 0,
            "reason": "No parsed model values for this criterion.",
            "confusion_matrix": {"labels": ["fail", "pass"], "matrix": [[0, 0], [0, 0]]},
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[False, True], average="binary",
        pos_label=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    return {
        "status": "computed",
        "n": len(y_true),
        "accuracy": _safe_metric(accuracy_score(y_true, y_pred)),
        "precision_pass": _safe_metric(precision),
        "recall_pass": _safe_metric(recall),
        "f1_pass": _safe_metric(f1),
        "cohen_kappa": _safe_cohen_kappa(y_true, y_pred, labels=[False, True]),
        "human_pass_rate": _safe_metric(sum(y_true) / len(y_true)),
        "model_pass_rate": _safe_metric(sum(y_pred) / len(y_pred)),
        "confusion_matrix": {
            "labels": ["fail", "pass"],
            "matrix": cm.tolist(),
        },
    }


def _ordinal_criterion_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    labels = [0, 1, 2, 3]
    if not y_true:
        return {
            "status": "not_computed",
            "n": 0,
            "reason": "No parsed model values for this criterion.",
            "confusion_matrix": {"labels": labels, "matrix": np.zeros((4, 4), dtype=int).tolist()},
        }

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    abs_errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
    per_score = {
        str(label): {
            "precision": _safe_metric(p_cls[idx]),
            "recall": _safe_metric(r_cls[idx]),
            "f1": _safe_metric(f1_cls[idx]),
            "support": int(support_cls[idx]),
        }
        for idx, label in enumerate(labels)
    }
    return {
        "status": "computed",
        "n": len(y_true),
        "exact_accuracy": _safe_metric(accuracy_score(y_true, y_pred)),
        "macro_precision": _safe_metric(p_macro),
        "macro_recall": _safe_metric(r_macro),
        "macro_f1": _safe_metric(f1_macro),
        "cohen_kappa": _safe_cohen_kappa(y_true, y_pred, labels=labels),
        "quadratic_weighted_kappa": _safe_cohen_kappa(
            y_true, y_pred, labels=labels, weights="quadratic"
        ),
        "mean_absolute_error": _safe_metric(float(np.mean(abs_errors))),
        "human_mean_score": _safe_metric(float(np.mean(y_true))),
        "model_mean_score": _safe_metric(float(np.mean(y_pred))),
        "per_score": per_score,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


def _class_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    if not y_true:
        return {
            "status": "not_computed",
            "n": 0,
            "reason": "No parsed model labels for this class dimension.",
            "confusion_matrix": {
                "labels": labels,
                "matrix": np.zeros((len(labels), len(labels)), dtype=int).tolist(),
            },
        }

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        label: {
            "precision": _safe_metric(p_cls[idx]),
            "recall": _safe_metric(r_cls[idx]),
            "f1": _safe_metric(f1_cls[idx]),
            "support": int(support_cls[idx]),
        }
        for idx, label in enumerate(labels)
    }
    return {
        "status": "computed",
        "n": len(y_true),
        "accuracy": _safe_metric(accuracy_score(y_true, y_pred)),
        "macro_precision": _safe_metric(p_macro),
        "macro_recall": _safe_metric(r_macro),
        "macro_f1": _safe_metric(f1_macro),
        "cohen_kappa": _safe_cohen_kappa(y_true, y_pred, labels=labels),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


def _human_dimension_class(label: Dict, dimension: str):
    if dimension == "overall":
        return normalize_compliance_class(label.get("overall") or label.get("classification"))
    if dimension == "sc_class":
        score = compute_sc_score_from_label(label)
        return score_to_compliance_class(score)
    if dimension == "dq_class":
        score = compute_dq_score_from_label(label)
        return score_to_compliance_class(score)
    return None


def _predicted_dimension_class(row: Dict, dimension: str):
    if dimension == "overall":
        return normalize_compliance_class(row.get("predicted"))
    if dimension == "sc_class":
        return (
            normalize_compliance_class(row.get("predicted_sc_class"))
            or normalize_compliance_class(row.get("predicted_sc_class_from_criteria"))
        )
    if dimension == "dq_class":
        return (
            normalize_compliance_class(row.get("predicted_dq_class"))
            or normalize_compliance_class(row.get("predicted_dq_class_from_criteria"))
        )
    return None


def _summed_matrix(per_repetition: List[Dict], labels: List) -> List[List[int]]:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row in per_repetition:
        if row.get("status") != "computed":
            continue
        cm = row.get("confusion_matrix", {}).get("matrix")
        if cm:
            matrix += np.array(cm, dtype=int)
    return matrix.tolist()


def _summarize_repetition_metrics(per_repetition: List[Dict],
                                  metric_names: List[str]) -> Dict:
    return {
        metric: _summary_stats([
            row.get(metric)
            for row in per_repetition
            if row.get("status") == "computed" and row.get(metric) is not None
        ])
        for metric in metric_names
    }


def write_criterion_level_performance_report(all_results: Dict,
                                             ground_truth: Dict) -> Dict:
    """Write C1-C7, Q1-Q7, and SC/DQ/overall confusion matrix metrics."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configs = _configuration_rows(all_results)
    report_configs = {}

    for config_name, config in sorted(configs.items()):
        structural_criteria = {}
        decision_quality_criteria = {}
        dimension_confusions = {}
        parse_repetitions = []

        for criterion in SC_WEIGHTS:
            per_repetition = []
            for rep_idx, rep_data in enumerate(config["reps"], 1):
                y_true, y_pred = [], []
                for row in rep_data:
                    label = ground_truth.get(row.get("adr_id"))
                    pred_checks = row.get("predicted_sc_checks")
                    if not label or not isinstance(pred_checks, dict):
                        continue
                    actual = _normalize_bool(label.get(criterion))
                    predicted = _normalize_bool(pred_checks.get(criterion))
                    if actual is None or predicted is None:
                        continue
                    y_true.append(actual)
                    y_pred.append(predicted)
                metrics = _binary_criterion_metrics(y_true, y_pred)
                metrics["rep"] = rep_idx
                per_repetition.append(metrics)

            structural_criteria[criterion] = {
                "criterion": STRUCTURAL_CRITERIA.get(criterion, criterion),
                "summary": _summarize_repetition_metrics(
                    per_repetition,
                    [
                        "accuracy", "precision_pass", "recall_pass",
                        "f1_pass", "cohen_kappa", "human_pass_rate",
                        "model_pass_rate",
                    ],
                ),
                "summed_confusion_matrix": {
                    "labels": ["fail", "pass"],
                    "matrix": _summed_matrix(per_repetition, ["fail", "pass"]),
                },
                "per_repetition": per_repetition,
            }

        for criterion in DQ_WEIGHTS:
            per_repetition = []
            for rep_idx, rep_data in enumerate(config["reps"], 1):
                y_true, y_pred = [], []
                for row in rep_data:
                    label = ground_truth.get(row.get("adr_id"))
                    pred_scores = row.get("predicted_dq_scores")
                    if not label or not isinstance(pred_scores, dict):
                        continue
                    actual = _normalize_dq_score(label.get(criterion))
                    predicted = _normalize_dq_score(pred_scores.get(criterion))
                    if actual is None or predicted is None:
                        continue
                    y_true.append(actual)
                    y_pred.append(predicted)
                metrics = _ordinal_criterion_metrics(y_true, y_pred)
                metrics["rep"] = rep_idx
                per_repetition.append(metrics)

            decision_quality_criteria[criterion] = {
                "criterion": DECISION_QUALITY_CRITERIA.get(criterion, criterion),
                "anchors": DQ_SCORE_ANCHORS,
                "summary": _summarize_repetition_metrics(
                    per_repetition,
                    [
                        "exact_accuracy", "macro_precision", "macro_recall",
                        "macro_f1", "cohen_kappa",
                        "quadratic_weighted_kappa", "mean_absolute_error",
                        "human_mean_score", "model_mean_score",
                    ],
                ),
                "summed_confusion_matrix": {
                    "labels": [0, 1, 2, 3],
                    "matrix": _summed_matrix(per_repetition, [0, 1, 2, 3]),
                },
                "per_repetition": per_repetition,
            }

        for dimension in ["sc_class", "dq_class", "overall"]:
            per_repetition = []
            for rep_idx, rep_data in enumerate(config["reps"], 1):
                y_true, y_pred = [], []
                for row in rep_data:
                    label = ground_truth.get(row.get("adr_id"))
                    if not label:
                        continue
                    actual = _human_dimension_class(label, dimension)
                    predicted = _predicted_dimension_class(row, dimension)
                    if not actual or not predicted:
                        continue
                    y_true.append(actual)
                    y_pred.append(predicted)
                metrics = _class_metrics(y_true, y_pred, CLASSES)
                metrics["rep"] = rep_idx
                per_repetition.append(metrics)

            dimension_confusions[dimension] = {
                "summary": _summarize_repetition_metrics(
                    per_repetition,
                    [
                        "accuracy", "macro_precision", "macro_recall",
                        "macro_f1", "cohen_kappa",
                    ],
                ),
                "summed_confusion_matrix": {
                    "labels": CLASSES,
                    "matrix": _summed_matrix(per_repetition, CLASSES),
                },
                "per_repetition": per_repetition,
            }

        for rep_idx, rep_data in enumerate(config["reps"], 1):
            total = len(rep_data)
            parsed_json = sum(1 for row in rep_data if row.get("parsed_json"))
            parsed_overall = sum(1 for row in rep_data if row.get("parse_success"))
            parsed_criteria = sum(1 for row in rep_data if row.get("criterion_parse_success"))
            missing_sc = {
                criterion: sum(
                    1 for row in rep_data
                    if not isinstance(row.get("predicted_sc_checks"), dict)
                    or row["predicted_sc_checks"].get(criterion) is None
                )
                for criterion in SC_WEIGHTS
            }
            missing_dq = {
                criterion: sum(
                    1 for row in rep_data
                    if not isinstance(row.get("predicted_dq_scores"), dict)
                    or row["predicted_dq_scores"].get(criterion) is None
                )
                for criterion in DQ_WEIGHTS
            }
            parse_repetitions.append({
                "rep": rep_idx,
                "n_rows": total,
                "parsed_json": parsed_json,
                "parsed_overall": parsed_overall,
                "parsed_criteria": parsed_criteria,
                "json_parse_failure_rate": _safe_metric(1 - (parsed_json / total)) if total else None,
                "overall_parse_failure_rate": _safe_metric(1 - (parsed_overall / total)) if total else None,
                "criterion_parse_failure_rate": _safe_metric(1 - (parsed_criteria / total)) if total else None,
                "missing_structural_fields": missing_sc,
                "missing_decision_quality_fields": missing_dq,
            })

        total_rows = sum(row["n_rows"] for row in parse_repetitions)
        report_configs[config_name] = {
            "model": config["model"],
            "strategy": config["strategy"],
            "repetitions": len(config["reps"]),
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "parse_coverage": {
                "per_repetition": parse_repetitions,
                "total_rows": total_rows,
                "parsed_json": sum(row["parsed_json"] for row in parse_repetitions),
                "parsed_overall": sum(row["parsed_overall"] for row in parse_repetitions),
                "parsed_criteria": sum(row["parsed_criteria"] for row in parse_repetitions),
                "criterion_parse_success_rate": _safe_metric(
                    sum(row["parsed_criteria"] for row in parse_repetitions) / total_rows
                ) if total_rows else None,
            },
            "structural_criteria": structural_criteria,
            "decision_quality_criteria": decision_quality_criteria,
            "dimension_confusion_matrices": dimension_confusions,
        }

    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Criterion-level model performance for C1-C7 and Q1-Q7, plus "
            "SC, DQ, and overall confusion matrices against human reference labels."
        ),
        "generated_after_final_run": True,
        "result_schema_version_required": RESULT_SCHEMA_VERSION,
        "source_files": {
            "all_results": str(RESULTS_DIR / "all_results.json"),
            "human_ground_truth": str(EXPERIMENT_DIR / "human_ground_truth.json"),
            "evaluation_manifest": str(EVAL_SET_PATH),
        },
        "metric_notes": {
            "structural_criteria": (
                "C1-C7 are binary pass/fail metrics. Precision, recall, and F1 "
                "treat pass as the positive class."
            ),
            "decision_quality_criteria": (
                "Q1-Q7 use the ordinal 0-3 anchor scale. Exact accuracy, macro-F1, "
                "Cohen's kappa, quadratic weighted kappa, and mean absolute error "
                "are reported."
            ),
            "dimension_confusion_matrices": (
                "SC and DQ class matrices compare predicted dimension classes with "
                "human dimension classes derived from submitted SC and DQ criteria. "
                "The overall matrix compares final predicted labels with human "
                "overall labels."
            ),
        },
        "configurations": report_configs,
    }

    with open(CRITERION_PERFORMANCE_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Criterion-level performance report written to {CRITERION_PERFORMANCE_PATH}")
    return report


def _adr_snippet(text: str, limit: int = 700) -> str:
    """Compact ADR text for representative error tables."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _error_category(row: Dict, label: Dict) -> str:
    actual = row.get("actual")
    predicted = row.get("predicted")
    q5 = label.get("Q5")
    q2 = label.get("Q2")
    q4 = label.get("Q4")
    notes = str(label.get("notes", "")).lower()

    compliant_like = {"Fully_Compliant", "Mostly_Compliant"}
    if actual == "Not_Compliant" and predicted != "Not_Compliant":
        return "non_compliant_miss"
    if (
        (actual in compliant_like and predicted == "Partially_Compliant")
        or (actual == "Partially_Compliant" and predicted in compliant_like)
    ):
        return "compliant_partially_confusion"
    if (
        actual in {"Partially_Compliant", "Not_Compliant"}
        and predicted in compliant_like
        and (
            q5 in (0, 1) or q2 in (0, 1) or q4 in (0, 1)
            or "weak" in notes or "rationale" in notes or "superficial" in notes
        )
    ):
        return "weak_rationale_miss"
    if actual in compliant_like and predicted in {"Partially_Compliant", "Not_Compliant"}:
        return "overly_strict_classification"
    return "other_misclassification"


def write_error_analysis_report(all_results: Dict, ground_truth: Dict,
                                top_k_configs: int = 3,
                                examples_per_category: int = 6) -> Dict:
    """Write error-visualization data for manuscript figures and examples."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    eval_adrs = _load_eval_adrs_from_manifest()
    adrs_by_id = {adr["id"]: adr for adr in eval_adrs}
    configs = _configuration_rows(all_results)
    config_scores = []

    for config_name, config in sorted(configs.items()):
        rep_f1s = []
        for rep_data in config["reps"]:
            rows = [row for row in rep_data if row.get("predicted")]
            y_true = [row["actual"] for row in rows]
            y_pred = [row["predicted"] for row in rows]
            if len(y_true) < 10 and not (VALIDATION_MODE and y_true):
                continue
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
            )
            rep_f1s.append(f1)
        if rep_f1s:
            config_scores.append({
                "config": config_name,
                "model": config["model"],
                "strategy": config["strategy"],
                "macro_f1_mean": round(float(np.mean(rep_f1s)), 4),
                "macro_f1_std": round(float(np.std(rep_f1s, ddof=1)), 4) if len(rep_f1s) > 1 else 0.0,
            })

    config_scores.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    best_configs = config_scores[:top_k_configs]
    best_names = {row["config"] for row in best_configs}

    by_config = {}
    representative_pool = []
    for config_name, config in sorted(configs.items()):
        if config_name not in best_names:
            continue

        summed_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        actual_counts = Counter()
        error_counts_by_actual = Counter()
        error_counts_by_prediction = Counter()
        confusion_pairs = Counter()

        for rep_idx, rep_data in enumerate(config["reps"], 1):
            rows = [row for row in rep_data if row.get("predicted")]
            y_true = [row["actual"] for row in rows]
            y_pred = [row["predicted"] for row in rows]
            if y_true:
                summed_cm += confusion_matrix(y_true, y_pred, labels=CLASSES)

            for row in rows:
                actual = row["actual"]
                predicted = row["predicted"]
                actual_counts[actual] += 1
                if actual == predicted:
                    continue
                error_counts_by_actual[actual] += 1
                error_counts_by_prediction[predicted] += 1
                confusion_pairs[f"{actual} -> {predicted}"] += 1

                adr = adrs_by_id.get(row["adr_id"], {})
                label = ground_truth.get(row["adr_id"], {})
                category = _error_category(row, label)
                representative_pool.append({
                    "category": category,
                    "config": config_name,
                    "model": config["model"],
                    "strategy": config["strategy"],
                    "rep": rep_idx,
                    "adr_id": row["adr_id"],
                    "source_repo": adr.get("source_repo"),
                    "variant": adr.get("variant", "OTHER"),
                    "domain": get_domain(adr.get("source_repo", "")),
                    "actual": actual,
                    "predicted": predicted,
                    "predicted_sc_class": row.get("predicted_sc_class"),
                    "predicted_dq_class": row.get("predicted_dq_class"),
                    "predicted_sc_checks": row.get("predicted_sc_checks"),
                    "predicted_dq_scores": row.get("predicted_dq_scores"),
                    "human_sc_score": label.get("sc_score"),
                    "human_dq_score": label.get("dq_score"),
                    "human_q5_rationale_soundness": label.get("Q5"),
                    "human_q6_consequence_objectivity": label.get("Q6"),
                    "human_notes": label.get("notes"),
                    "adr_snippet": _adr_snippet(adr.get("text", "")),
                })

        per_class = {}
        for cls in CLASSES:
            support = actual_counts[cls]
            errors = error_counts_by_actual[cls]
            per_class[cls] = {
                "support_across_repetitions": support,
                "misclassified": errors,
                "error_rate": round(float(errors / support), 4) if support else 0.0,
            }

        by_config[config_name] = {
            "model": config["model"],
            "strategy": config["strategy"],
            "summed_confusion_matrix": {
                "labels": CLASSES,
                "matrix": summed_cm.tolist(),
            },
            "per_class_error_analysis": per_class,
            "most_common_confusions": [
                {"confusion": pair, "count": count}
                for pair, count in confusion_pairs.most_common(12)
            ],
            "false_negative_actual_classes": dict(error_counts_by_actual),
            "false_positive_predicted_classes": dict(error_counts_by_prediction),
        }

    representative_examples = defaultdict(list)
    for item in sorted(representative_pool, key=lambda row: (row["config"], row["rep"], row["adr_id"])):
        bucket = representative_examples[item["category"]]
        if len(bucket) < examples_per_category:
            bucket.append(item)

    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Error-visualization data for reviewer-requested confusion matrices, "
            "per-class error analysis, and representative misclassified ADR examples."
        ),
        "generated_after_final_run": True,
        "selection_rule": (
            f"Top {top_k_configs} model-strategy configurations by mean macro-F1 "
            "across completed repetitions."
        ),
        "best_configurations": best_configs,
        "visualizations_to_prepare_for_manuscript": [
            "confusion-matrix heatmaps for the best configurations",
            "per-class error-rate bar chart",
            "representative misclassification table",
            "focused examples for Compliant/Partially Compliant confusion",
            "focused examples where weak or superficial rationale was missed",
        ],
        "by_best_configuration": by_config,
        "representative_misclassifications": dict(representative_examples),
    }

    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Error analysis report written to {ERROR_ANALYSIS_PATH}")
    return report


def _display_model_name(model_name: str) -> str:
    notes = MODEL_REPRODUCIBILITY_NOTES.get(model_name, {})
    return notes.get("paper_label", model_name)


def _display_strategy_name(strategy: str) -> str:
    return {
        "zero_shot": "ZS",
        "few_shot": "FS",
        "chain_of_thought": "CoT",
    }.get(strategy, strategy)


def _format_metric(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _format_metric_pm(mean, std, digits: int = 3) -> str:
    if mean is None:
        return "NA"
    return f"{float(mean):.{digits}f} +/- {float(std or 0):.{digits}f}"


def _sorted_metric_rows(metrics_summary: Dict) -> List[Dict]:
    rows = []
    for model_name, strategies in sorted(metrics_summary.items()):
        for strategy, metrics in sorted(strategies.items()):
            rows.append({
                "config": f"{model_name}/{strategy}",
                "model": model_name,
                "model_label": _display_model_name(model_name),
                "strategy": strategy,
                "strategy_label": _display_strategy_name(strategy),
                "macro_f1_mean": metrics.get("f1_mean"),
                "macro_f1_std": metrics.get("f1_std"),
                "weighted_f1_mean": metrics.get("weighted_f1_mean"),
                "weighted_f1_std": metrics.get("weighted_f1_std"),
                "macro_precision_mean": metrics.get("prec_mean"),
                "macro_precision_std": metrics.get("prec_std"),
                "macro_recall_mean": metrics.get("rec_mean"),
                "macro_recall_std": metrics.get("rec_std"),
                "cohen_kappa_mean": metrics.get("kappa_mean"),
                "cohen_kappa_std": metrics.get("kappa_std"),
                "parse_failure_rate": metrics.get("parse_failure_rate"),
                "total_responses": metrics.get("total_responses"),
                "parse_failures": metrics.get("parse_failures"),
            })
    return rows


def _best_metric_rows(rows: List[Dict], metric: str) -> List[Dict]:
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return []
    best = max(float(row[metric]) for row in valid)
    return [row for row in valid if abs(float(row[metric]) - best) < 1e-12]


def _format_config_list(rows: List[Dict], metric: str, digits: int = 3) -> str:
    if not rows:
        return "NA"
    names = [
        f"{row['model_label']} {row['strategy_label']}"
        for row in rows
    ]
    value = _format_metric(rows[0].get(metric), digits)
    if len(names) == 1:
        return f"{names[0]} ({value})"
    return f"{', '.join(names[:-1])}, and {names[-1]} ({value})"


def _main_error_pattern(error_report: Dict, best_f1_rows: List[Dict]) -> str:
    if not error_report or not best_f1_rows:
        return "the main residual error pattern reported in error_analysis.json"
    best_config = best_f1_rows[0]["config"]
    by_config = error_report.get("by_best_configuration", {})
    confusions = by_config.get(best_config, {}).get("most_common_confusions", [])
    if confusions:
        top = confusions[0]
        return f"{top['confusion']} errors"
    examples = error_report.get("representative_misclassifications", {})
    if examples:
        return next(iter(examples.keys())).replace("_", " ")
    return "the main residual error pattern reported in error_analysis.json"


def write_manuscript_results_summary(metrics_summary: Dict, performance_report: Dict,
                                     error_report: Dict, ground_truth: Dict) -> Dict:
    """Write one source of truth for central manuscript performance claims."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _sorted_metric_rows(metrics_summary)
    best_f1 = _best_metric_rows(rows, "macro_f1_mean")
    best_kappa = _best_metric_rows(rows, "cohen_kappa_mean")

    try:
        eval_adrs = _load_eval_adrs_from_manifest()
        labels = [ground_truth[adr["id"]]["overall"] for adr in eval_adrs]
        majority_baseline = _majority_class_macro_f1(labels)
    except Exception:
        majority_baseline = {
            "majority_class": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    f1_config_text = _format_config_list(best_f1, "macro_f1_mean")
    kappa_config_text = _format_config_list(best_kappa, "cohen_kappa_mean")
    best_f1_value = best_f1[0]["macro_f1_mean"] if best_f1 else None
    best_kappa_value = best_kappa[0]["cohen_kappa_mean"] if best_kappa else None
    error_pattern = _main_error_pattern(error_report, best_f1)

    table_rows = []
    for row in rows:
        table_rows.append({
            "model": row["model_label"],
            "strategy": row["strategy_label"],
            "macro_f1": _format_metric_pm(row["macro_f1_mean"], row["macro_f1_std"]),
            "weighted_f1": _format_metric_pm(
                row["weighted_f1_mean"], row["weighted_f1_std"]
            ),
            "macro_precision": _format_metric_pm(
                row["macro_precision_mean"], row["macro_precision_std"]
            ),
            "macro_recall": _format_metric_pm(
                row["macro_recall_mean"], row["macro_recall_std"]
            ),
            "cohen_kappa": _format_metric_pm(
                row["cohen_kappa_mean"], row["cohen_kappa_std"]
            ),
            "parse_failure_rate": _format_metric(
                row["parse_failure_rate"], digits=4
            ),
            "source_config": row["config"],
        })

    if majority_baseline["macro_f1"] is not None:
        table_rows.append({
            "model": "Majority-class baseline",
            "strategy": "--",
            "macro_f1": _format_metric(majority_baseline["macro_f1"], digits=4),
            "weighted_f1": _format_metric(
                majority_baseline["weighted_f1"], digits=4
            ),
            "macro_precision": "--",
            "macro_recall": "--",
            "cohen_kappa": "--",
            "parse_failure_rate": "--",
            "source_config": str(MANUSCRIPT_DATASET_SUMMARY_PATH),
        })

    fig_iv_sentence = (
        "The highest macro-F1 and highest Cohen's kappa occur in the same configuration."
        if best_f1 and best_kappa and best_f1[0]["config"] == best_kappa[0]["config"]
        else "The highest macro-F1 and highest Cohen's kappa occur in different configurations."
    )

    summary = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Single-source central result values for manuscript text, tables, "
            "plots, captions, abstract, discussion, and conclusion. Generate "
            "after the complete rerun and merge so no stale performance values "
            "remain in the paper."
        ),
        "source_files": {
            "all_results": str(RESULTS_DIR / "all_results.json"),
            "metrics_summary": str(ANALYSIS_DIR / "metrics_summary.json"),
            "performance_details": str(PERFORMANCE_DETAILS_PATH),
            "criterion_level_performance": str(CRITERION_PERFORMANCE_PATH),
            "error_analysis": str(ERROR_ANALYSIS_PATH),
            "manuscript_dataset_summary": str(MANUSCRIPT_DATASET_SUMMARY_PATH),
        },
        "majority_class_baseline": majority_baseline,
        "best_macro_f1_configurations": best_f1,
        "best_cohen_kappa_configurations": best_kappa,
        "table_ii_rows": table_rows,
        "figure_caption_values": {
            "fig_iii": (
                "Macro-F1 by model and prompting strategy. Dashed line: "
                "majority-class baseline "
                f"(macro-F1 = {_format_metric(majority_baseline['macro_f1'], digits=4)})."
            ),
            "fig_iv": f"Macro-F1 versus Cohen's kappa divergence. {fig_iv_sentence}",
        },
        "copy_ready_manuscript_sentences": {
            "main_results": (
                f"The strongest macro-F1 configuration is {f1_config_text}, while "
                f"the strongest chance-corrected agreement is {kappa_config_text}. "
                "The majority-class baseline has macro-F1 = "
                f"{_format_metric(majority_baseline['macro_f1'], digits=4)}, "
                "which provides the reference point for interpreting model gains "
                "beyond class-prior prediction."
            ),
            "capability_gap": (
                "We treat this as the calibration contribution of the paper: "
                "even the strongest configuration achieved macro-F1 = "
                f"{_format_metric(best_f1_value)} and Cohen's kappa = "
                f"{_format_metric(best_kappa_value)}, with the largest remaining "
                f"errors concentrated in {error_pattern}, indicating that current "
                "LLMs remain below the reliability required for autonomous ADR "
                "compliance assessment."
            ),
            "prior_work_comparison": (
                f"Our best macro-F1 of {_format_metric(best_f1_value)} contextualizes "
                "the more optimistic results in the literature because this benchmark "
                "evaluates multi-dimensional compliance on real, heterogeneous ADRs "
                "rather than narrower violation-detection or generation-quality tasks."
            ),
            "conclusion": (
                "Within this benchmark, across the evaluated model and prompting configurations, the "
                f"strongest macro-F1 is {_format_metric(best_f1_value)} and the "
                f"strongest Cohen's kappa is {_format_metric(best_kappa_value)}, "
                "supporting a human-in-the-loop deployment model rather than "
                "autonomous compliance decisions."
            ),
        },
        "claim_scope_notes": {
            "scope_sentence": (
                "Claims should be scoped to the frozen 162-ADR evaluation set, "
                "the C1-C7/Q1-Q7 rubric, the evaluated models, the three prompt "
                "conditions, and the human-reviewed reference labels."
            ),
            "avoid_unqualified_claims": [
                "balanced synthetic datasets systematically overestimate LLM performance",
                "incremental model improvements will not resolve ADR compliance checking",
                "the observed performance ceiling applies to all LLMs or all ADR corpora",
                "frontier or open-weight model conclusions beyond the tested configurations",
            ],
            "safe_claim_pattern": (
                "The results suggest, within the tested benchmark, that evaluations "
                "limited to balanced, synthetic, or highly curated data may miss "
                "error patterns found in heterogeneous real-world ADRs."
            ),
        },
        "manuscript_locations_to_update": [
            "Abstract result claims",
            "Table II performance rows",
            "Criterion-level C1-C7 and Q1-Q7 performance table",
            "SC, DQ, and overall confusion matrices",
            "Fig. III caption and baseline line",
            "Fig. IV caption and kappa/F1 divergence claim",
            "Results main performance paragraph",
            "Best macro-F1 and best kappa observations",
            "Prior-work comparison paragraph",
            "Industry-practitioner implications",
            "Discussion capability-gap sentence",
            "Conclusion result claim",
        ],
        "stale_terms_to_audit": [
            "macro-F1 = 0.31",
            "F1=0.31",
            "F1 = 0.57",
            "Our F1 = 0.57",
            "Claude achieves highest kappa",
            "Claude achieves highest \u03ba",
            "GPT-5.1",
            "mistral-small-latest",
            "systematically overestimate",
            "performance ceiling",
            "incremental model capability improvements alone",
            "balanced synthetic datasets",
        ],
    }

    with open(MANUSCRIPT_RESULTS_SUMMARY_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  Manuscript results summary written to {MANUSCRIPT_RESULTS_SUMMARY_PATH}")
    return summary


def _cost_assumption_row(model_name: str) -> Dict:
    cost = COST_ASSUMPTIONS_USD_PER_MILLION_TOKENS.get(
        model_name,
        {
            "input": 0.0,
            "output": 0.0,
            "basis": "No configured pricing assumption",
        },
    )
    return {
        "input_usd_per_million_tokens": cost.get("input", 0.0),
        "output_usd_per_million_tokens": cost.get("output", 0.0),
        "basis": cost.get("basis"),
        "pricing_source": cost.get("pricing_source"),
        "pricing_date": cost.get("pricing_date"),
        "pricing_verified_date": cost.get("pricing_verified_date"),
        "batch_cost_multiplier": cost.get("batch_multiplier"),
        "batch_input_usd_per_million_tokens": cost.get("batch_input"),
        "batch_output_usd_per_million_tokens": cost.get("batch_output"),
        "batch_pricing_source": cost.get("batch_pricing_source"),
        "subscription_or_tier": cost.get("subscription_or_tier"),
        "cached_token_treatment": cost.get("cached_token_treatment"),
        "local_compute_assumption": cost.get("local_compute_assumption"),
    }


def _token_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    cost = COST_ASSUMPTIONS_USD_PER_MILLION_TOKENS.get(
        model_name, {"input": 0.0, "output": 0.0}
    )
    return (
        input_tokens * float(cost.get("input", 0.0))
        + output_tokens * float(cost.get("output", 0.0))
    ) / 1_000_000


def _row_token_cost_usd(
    model_name: str,
    row: Dict,
    billable_output_tokens: int = None,
) -> float:
    multiplier = float(row.get("cost_multiplier", 1.0) or 1.0)
    return multiplier * _token_cost_usd(
        model_name,
        int(row.get("input_tokens", 0) or 0),
        (
            int(row.get("output_tokens", 0) or 0)
            if billable_output_tokens is None
            else int(billable_output_tokens)
        ),
    )


def _batch_evidence_run_dir() -> Path:
    """Locate retained provider batch outputs for the active analysis run."""
    if BATCH_OUTPUTS_DIR.exists():
        return RUN_DIR
    if RUN_CONFIG_PATH.exists():
        try:
            config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            config = {}
        source_run_id = config.get("source_run_id")
        if source_run_id:
            return RUNS_DIR / str(source_run_id)
    return RUN_DIR


def _load_gemini_batch_usage() -> Dict:
    """Load provider token accounting retained with Gemini batch responses."""
    evidence_dir = _batch_evidence_run_dir() / "batch_jobs" / "outputs"
    usage_by_custom_id = {}
    sources = []
    for path in sorted(evidence_dir.glob("gemini*.jsonl")):
        item_count = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("key")
            usage = (item.get("response") or {}).get("usageMetadata")
            if not custom_id or not isinstance(usage, dict):
                continue
            if custom_id in usage_by_custom_id:
                raise ValueError(
                    f"Duplicate Gemini batch usage key {custom_id!r} in {path} "
                    f"at line {line_number}"
                )
            usage_by_custom_id[custom_id] = usage
            item_count += 1
        sources.append({
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
            "usage_records": item_count,
        })
    return {
        "usage_by_custom_id": usage_by_custom_id,
        "sources": sources,
        "usage_records": len(usage_by_custom_id),
    }


def _load_openai_batch_usage() -> Dict:
    """Load retained OpenAI batch usage, including automatic cache metadata."""
    evidence_dir = _batch_evidence_run_dir() / "batch_jobs" / "outputs"
    usage_by_custom_id = {}
    sources = []
    for path in sorted(evidence_dir.glob("gpt*.jsonl")):
        item_count = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            body = (item.get("response") or {}).get("body") or {}
            usage = body.get("usage")
            if not custom_id or not isinstance(usage, dict):
                continue
            if custom_id in usage_by_custom_id:
                raise ValueError(
                    f"Duplicate OpenAI batch usage key {custom_id!r} in {path} "
                    f"at line {line_number}"
                )
            usage_by_custom_id[custom_id] = usage
            item_count += 1
        sources.append({
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
            "usage_records": item_count,
        })
    return {
        "usage_by_custom_id": usage_by_custom_id,
        "sources": sources,
        "usage_records": len(usage_by_custom_id),
    }


def _summarize_openai_cached_inputs(rows: List[Dict], usage_by_custom_id: Dict) -> Dict:
    """Reconcile retained GPT batch rows and summarize automatic cache use."""
    batch_rows = 0
    rows_with_cached_input = 0
    cached_input_tokens = 0
    for row in rows:
        if row.get("billing_mode") != "batch":
            continue
        custom_id = (row.get("batch_metadata") or {}).get("custom_id")
        if not custom_id or custom_id not in usage_by_custom_id:
            raise ValueError(
                "Missing retained OpenAI provider usage for batch row "
                f"{row.get('adr_id')!r} (custom_id={custom_id!r})"
            )
        usage = usage_by_custom_id[custom_id]
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if (
            int(row.get("input_tokens", 0) or 0) != prompt_tokens
            or int(row.get("output_tokens", 0) or 0) != completion_tokens
        ):
            raise ValueError(
                "OpenAI retained usage does not match saved row token counts for "
                f"{custom_id!r}"
            )
        cached_tokens = int(
            ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            or 0
        )
        batch_rows += 1
        cached_input_tokens += cached_tokens
        rows_with_cached_input += bool(cached_tokens)
    return {
        "retained_batch_rows_reconciled": batch_rows,
        "rows_with_cached_input": rows_with_cached_input,
        "cached_input_tokens": cached_input_tokens,
        "discount_applied_in_estimate": False,
    }


def _gemini_billable_output_tokens(row: Dict, usage_by_custom_id: Dict) -> Dict:
    """Reconcile a Gemini row with provider-reported batch token usage."""
    candidate_tokens = int(row.get("output_tokens", 0) or 0)
    if row.get("billing_mode") != "batch":
        usage_details = row.get("usage_details") or {}
        thinking_tokens = int(usage_details.get("thoughtsTokenCount", 0) or 0)
        return {
            "candidate_output_tokens": candidate_tokens,
            "thinking_tokens": thinking_tokens,
            "billable_output_tokens": candidate_tokens + thinking_tokens,
            "provider_usage_reconciled": bool(usage_details),
        }

    custom_id = (row.get("batch_metadata") or {}).get("custom_id")
    if not custom_id or custom_id not in usage_by_custom_id:
        raise ValueError(
            "Missing retained Gemini provider usage for batch row "
            f"{row.get('adr_id')!r} (custom_id={custom_id!r})"
        )
    usage = usage_by_custom_id[custom_id]
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    provider_candidate_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    thinking_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)
    total_tokens = int(usage.get("totalTokenCount", 0) or 0)
    input_tokens = int(row.get("input_tokens", 0) or 0)

    if input_tokens != prompt_tokens or candidate_tokens != provider_candidate_tokens:
        raise ValueError(
            "Gemini retained usage does not match saved row token counts for "
            f"{custom_id!r}"
        )
    billable_output_tokens = candidate_tokens + thinking_tokens
    if total_tokens and input_tokens + billable_output_tokens != total_tokens:
        raise ValueError(
            f"Gemini totalTokenCount does not reconcile for {custom_id!r}"
        )
    return {
        "candidate_output_tokens": candidate_tokens,
        "thinking_tokens": thinking_tokens,
        "billable_output_tokens": billable_output_tokens,
        "provider_usage_reconciled": True,
    }


def write_cost_analysis_report(all_results: Dict) -> Dict:
    """Write reproducible token and cost accounting for manuscript cost tables."""
    by_model_strategy = {}
    by_model_totals = {}
    observed_access = _summarize_model_access(all_results)
    gemini_batch_evidence = _load_gemini_batch_usage()
    gemini_usage = gemini_batch_evidence["usage_by_custom_id"]
    openai_batch_evidence = _load_openai_batch_usage()
    openai_rows = [
        row
        for row in _iter_result_rows(all_results)
        if (row.get("model_key") or row.get("model")) == "gpt-5.5"
    ]
    openai_cache_summary = _summarize_openai_cached_inputs(
        openai_rows, openai_batch_evidence["usage_by_custom_id"]
    )

    for model_name, strategies in sorted(all_results.items()):
        by_model_totals[model_name] = {
            "model": model_name,
            "pricing": _cost_assumption_row(model_name),
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "parse_failures": 0,
            "retry_count": 0,
            "output_retry_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "billable_output_tokens": 0,
            "provider_usage_reconciled_calls": 0,
            "latency_seconds": 0.0,
            "estimated_cost_usd": 0.0,
        }

        for strategy, reps in sorted(strategies.items()):
            rows = [row for rep in reps for row in rep]
            total_calls = len(rows)
            failed_calls = sum(1 for row in rows if row.get("error"))
            parse_failures = sum(1 for row in rows if not row.get("parse_success"))
            retry_count = sum(int(row.get("retry_count", 0) or 0) for row in rows)
            output_retry_count = sum(int(row.get("output_retry_count", 0) or 0) for row in rows)
            input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in rows)
            output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in rows)
            token_details = []
            for row in rows:
                if model_name == "gemini-2.5-pro":
                    token_details.append(
                        _gemini_billable_output_tokens(row, gemini_usage)
                    )
                else:
                    visible_output = int(row.get("output_tokens", 0) or 0)
                    token_details.append({
                        "candidate_output_tokens": visible_output,
                        "thinking_tokens": 0,
                        "billable_output_tokens": visible_output,
                        "provider_usage_reconciled": False,
                    })
            thinking_tokens = sum(item["thinking_tokens"] for item in token_details)
            billable_output_tokens = sum(
                item["billable_output_tokens"] for item in token_details
            )
            provider_usage_reconciled_calls = sum(
                bool(item["provider_usage_reconciled"]) for item in token_details
            )
            latency_seconds = sum(float(row.get("latency", 0) or 0) for row in rows)
            estimated_cost = sum(
                _row_token_cost_usd(
                    model_name,
                    row,
                    billable_output_tokens=details["billable_output_tokens"],
                )
                for row, details in zip(rows, token_details)
            )
            successful_calls = total_calls - failed_calls
            config_key = f"{model_name}/{strategy}"
            billing_modes = sorted({str(row.get("billing_mode", "sync")) for row in rows})

            by_model_strategy[config_key] = {
                "model": model_name,
                "strategy": strategy,
                "pricing": _cost_assumption_row(model_name),
                "repetitions": len(reps),
                "adrs_per_repetition": len(reps[0]) if reps else 0,
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "parse_failures": parse_failures,
                "retry_count": retry_count,
                "output_retry_count": output_retry_count,
                "billing_modes": billing_modes,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "billable_output_tokens": billable_output_tokens,
                "provider_usage_reconciled_calls": provider_usage_reconciled_calls,
                "avg_input_tokens_per_call": round(float(input_tokens / total_calls), 2) if total_calls else 0.0,
                "avg_output_tokens_per_call": round(float(output_tokens / total_calls), 2) if total_calls else 0.0,
                "avg_billable_output_tokens_per_call": round(float(billable_output_tokens / total_calls), 2) if total_calls else 0.0,
                "latency_seconds_total": round(float(latency_seconds), 2),
                "avg_latency_seconds_per_call": round(float(latency_seconds / total_calls), 2) if total_calls else 0.0,
                "estimated_cost_usd": round(float(estimated_cost), 6),
                "estimated_cost_usd_per_call": round(float(estimated_cost / total_calls), 6) if total_calls else 0.0,
                "estimated_cost_usd_per_adr_per_repetition": round(float(estimated_cost / total_calls), 6) if total_calls else 0.0,
            }

            totals = by_model_totals[model_name]
            totals["total_calls"] += total_calls
            totals["successful_calls"] += successful_calls
            totals["failed_calls"] += failed_calls
            totals["parse_failures"] += parse_failures
            totals["retry_count"] += retry_count
            totals["output_retry_count"] += output_retry_count
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["thinking_tokens"] += thinking_tokens
            totals["billable_output_tokens"] += billable_output_tokens
            totals["provider_usage_reconciled_calls"] += provider_usage_reconciled_calls
            totals["latency_seconds"] += latency_seconds
            totals["estimated_cost_usd"] += estimated_cost

        totals = by_model_totals[model_name]
        total_calls = totals["total_calls"]
        totals["latency_seconds"] = round(float(totals["latency_seconds"]), 2)
        totals["estimated_cost_usd"] = round(float(totals["estimated_cost_usd"]), 6)
        totals["avg_input_tokens_per_call"] = round(float(totals["input_tokens"] / total_calls), 2) if total_calls else 0.0
        totals["avg_output_tokens_per_call"] = round(float(totals["output_tokens"] / total_calls), 2) if total_calls else 0.0
        totals["avg_billable_output_tokens_per_call"] = round(float(totals["billable_output_tokens"] / total_calls), 2) if total_calls else 0.0
        totals["avg_latency_seconds_per_call"] = round(float(totals["latency_seconds"] / total_calls), 2) if total_calls else 0.0
        totals["estimated_cost_usd_per_call"] = round(float(totals["estimated_cost_usd"] / total_calls), 6) if total_calls else 0.0

    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": "Reproducible token and cost accounting for manuscript cost comparisons.",
        "pricing_verified_date": PRICING_VERIFIED_DATE,
        "model_access_summary": observed_access,
        "important_note": (
            "Cost values are estimates from saved token counts, the dated provider "
            "rates recorded in this report, and each row's billing-mode multiplier. "
            "They are not an invoice reconciliation."
        ),
        "cached_token_treatment": CACHED_TOKEN_TREATMENT,
        "openai_cached_input_evidence": {
            **openai_cache_summary,
            "available_provider_usage_records": openai_batch_evidence["usage_records"],
            "sources": openai_batch_evidence["sources"],
        },
        "unsuccessful_call_usage_treatment": UNRETAINED_USAGE_TREATMENT,
        "thinking_token_treatment": (
            "Gemini provider-reported thoughtsTokenCount is added to candidate output "
            "tokens for billing. Retained prompt, candidate, thinking, and total counts "
            "are reconciled by custom_id before cost calculation."
        ),
        "gemini_batch_usage_evidence": {
            "usage_records": gemini_batch_evidence["usage_records"],
            "sources": gemini_batch_evidence["sources"],
        },
        "failed_and_retried_call_treatment": (
            "Saved rows include token totals across validated-call attempts, failed-call "
            "count, parse-failure count, API retry_count, output_retry_count, and attempt "
            "history metadata. Unsuccessful requests are included only when retained "
            "provider usage metadata permits calculation."
        ),
        "batch_pricing_treatment": (
            "Rows collected from provider batch APIs use billing_mode=batch and "
            "cost_multiplier=0.5. Synchronous rows default to cost_multiplier=1.0."
        ),
        "by_model_strategy": by_model_strategy,
        "by_model_totals": by_model_totals,
    }

    with open(COST_ANALYSIS_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Cost analysis report written to {COST_ANALYSIS_PATH}")
    return report


def _repo_cluster_sensitivity(pair_rows: List[Dict]) -> Dict:
    """Exploratory repository-level summary for paired accuracy differences."""
    by_repo = defaultdict(list)
    for row in pair_rows:
        by_repo[row["source_repo"]].append(row)

    repo_rows = []
    for repo, rows in sorted(by_repo.items()):
        b = sum(1 for r in rows if r["a_correct"] and not r["b_correct"])
        c = sum(1 for r in rows if not r["a_correct"] and r["b_correct"])
        n = len(rows)
        repo_rows.append({
            "source_repo": repo,
            "n_pairs": n,
            "b_a_correct_b_incorrect": b,
            "c_a_incorrect_b_correct": c,
            "accuracy_difference_a_minus_b": round(float((b - c) / n), 4) if n else 0.0,
        })

    diffs = [r["accuracy_difference_a_minus_b"] for r in repo_rows]
    return {
        "status": "exploratory_not_a_formal_cluster_adjusted_test",
        "repository_count": len(repo_rows),
        "mean_repo_accuracy_difference_a_minus_b": round(float(np.mean(diffs)), 4) if diffs else 0.0,
        "median_repo_accuracy_difference_a_minus_b": round(float(np.median(diffs)), 4) if diffs else 0.0,
        "repos_favoring_a": sum(1 for d in diffs if d > 0),
        "repos_favoring_b": sum(1 for d in diffs if d < 0),
        "repos_tied": sum(1 for d in diffs if d == 0),
        "per_repository": repo_rows,
    }


def write_pairwise_statistical_tests(all_results: Dict) -> Dict:
    """
    Write a reviewer-auditable pairwise test report.

    ADR-level bootstrap intervals and paired randomization tests are primary.
    All repetitions for an ADR remain together during resampling. Exact
    McNemar tests are retained separately for each repetition as diagnostics.
    """
    configs = _configuration_rows(all_results)
    config_names = sorted(configs)
    family_size = len(config_names) * (len(config_names) - 1) // 2
    alpha = 0.05
    eval_metadata = {}
    if EVAL_SET_PATH.exists():
        with open(EVAL_SET_PATH, encoding="utf-8") as fp:
            eval_set = json.load(fp)
        eval_metadata = {
            row["id"]: {
                "source_repo": row.get("source_repo", "Unknown"),
                "variant": row.get("variant", "OTHER"),
                "domain": row.get("domain", "Other"),
            }
            for row in eval_set.get("adrs", [])
        }

    comparisons = []
    for idx_a in range(len(config_names)):
        for idx_b in range(idx_a + 1, len(config_names)):
            name_a, name_b = config_names[idx_a], config_names[idx_b]
            cfg_a, cfg_b = configs[name_a], configs[name_b]
            reps_to_compare = min(len(cfg_a["reps"]), len(cfg_b["reps"]))
            pair_rows = []
            per_rep = []
            pair_seed = int.from_bytes(
                hashlib.sha256(
                    f"{name_a}|{name_b}|{ACTIVE_PROTOCOL_VERSION}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            primary_inference = _paired_adr_metric_inference(
                cfg_a,
                cfg_b,
                n_resamples=250 if VALIDATION_MODE else 10000,
                seed=pair_seed,
            )

            for rep_idx in range(reps_to_compare):
                rep_a = {
                    row["adr_id"]: row for row in cfg_a["reps"][rep_idx]
                    if row.get("predicted")
                }
                rep_b = {
                    row["adr_id"]: row for row in cfg_b["reps"][rep_idx]
                    if row.get("predicted")
                }
                common_ids = sorted(set(rep_a) & set(rep_b))
                b_count = 0
                c_count = 0
                a_correct_total = 0
                b_correct_total = 0

                for adr_id in common_ids:
                    a_correct = bool(rep_a[adr_id].get("correct"))
                    b_correct = bool(rep_b[adr_id].get("correct"))
                    if a_correct:
                        a_correct_total += 1
                    if b_correct:
                        b_correct_total += 1
                    if a_correct and not b_correct:
                        b_count += 1
                    elif not a_correct and b_correct:
                        c_count += 1

                    meta = eval_metadata.get(adr_id, {})
                    pair_rows.append({
                        "rep": rep_idx + 1,
                        "adr_id": adr_id,
                        "source_repo": meta.get("source_repo", "Unknown"),
                        "a_correct": a_correct,
                        "b_correct": b_correct,
                    })

                n = len(common_ids)
                per_rep.append({
                    "rep": rep_idx + 1,
                    "n_pairs": n,
                    "a_correct": a_correct_total,
                    "b_correct": b_correct_total,
                    "b_a_correct_b_incorrect": b_count,
                    "c_a_incorrect_b_correct": c_count,
                    "accuracy_difference_a_minus_b": round(float((b_count - c_count) / n), 4) if n else 0.0,
                    "mcnemar_exact_p_unadjusted": round(_exact_mcnemar_p_value(b_count, c_count), 6),
                })

            n_total = len(pair_rows)
            b_total = sum(1 for r in pair_rows if r["a_correct"] and not r["b_correct"])
            c_total = sum(1 for r in pair_rows if not r["a_correct"] and r["b_correct"])
            a_total = sum(1 for r in pair_rows if r["a_correct"])
            b_model_total = sum(1 for r in pair_rows if r["b_correct"])

            comparisons.append({
                "config_a": name_a,
                "config_b": name_b,
                "n_repeated_predictions_descriptive": n_total,
                "repetitions_compared": reps_to_compare,
                "primary_inference": primary_inference,
                "pooled_accuracy_descriptive_only": {
                    "warning": (
                        "These totals contain repeated predictions for the same ADR and "
                        "are not used as independent observations for hypothesis testing."
                    ),
                    "binary_outcome": "correct_overall_label_vs_incorrect_overall_label",
                    "a_accuracy": round(float(a_total / n_total), 4) if n_total else 0.0,
                    "b_accuracy": round(float(b_model_total / n_total), 4) if n_total else 0.0,
                    "accuracy_difference_a_minus_b": round(float((b_total - c_total) / n_total), 4) if n_total else 0.0,
                    "b_a_correct_b_incorrect": b_total,
                    "c_a_incorrect_b_correct": c_total,
                    "discordant_predictions": b_total + c_total,
                    "matched_odds_ratio_a_over_b": (
                    None if b_total == 0 and c_total == 0
                    else "Infinity" if c_total == 0
                    else round(float(b_total / c_total), 4)
                    ),
                },
                "per_repetition_mcnemar_secondary": per_rep,
                "repository_cluster_sensitivity": _repo_cluster_sensitivity(pair_rows),
            })

    for comparison in comparisons:
        inference = comparison.get("primary_inference", {})
        for metric_result in inference.get("metrics", {}).values():
            p_unadjusted = metric_result.get("adr_level_paired_randomization_p_unadjusted")
            p_adjusted = (
                min(float(p_unadjusted) * family_size, 1.0)
                if p_unadjusted is not None and family_size
                else p_unadjusted
            )
            metric_result["bonferroni_family_size"] = family_size
            metric_result["paired_randomization_p_bonferroni_adjusted"] = (
                round(float(p_adjusted), 6) if p_adjusted is not None else None
            )
            metric_result["significant_after_bonferroni"] = bool(
                p_adjusted is not None and p_adjusted <= alpha
            )

    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": "Pairwise statistical testing for completed model-strategy configurations.",
        "method": (
            "The ADR is the independent resampling unit. For each pair of model-prompt "
            "configurations, 95% percentile bootstrap intervals are computed for the "
            "difference in mean macro-F1 and mean Cohen's kappa across repetitions. "
            "Paired randomization swaps the two configuration assignments as one block "
            "per ADR across all repetitions. This retains within-ADR dependence and "
            "does not treat repeated model outputs as additional documents."
        ),
        "comparison_family": "all pairwise model-strategy configurations present in all_results.json",
        "configuration_count": len(config_names),
        "family_size": family_size,
        "alpha": alpha,
        "bonferroni_unadjusted_alpha": round(float(alpha / family_size), 6) if family_size else alpha,
        "adjusted_p_alpha": alpha,
        "significance_rule": (
            "For each metric family, compare Bonferroni-adjusted paired-randomization "
            "p-values with adjusted_p_alpha."
        ),
        "configurations": config_names,
        "primary_endpoints": ["macro_f1", "cohen_kappa"],
        "independent_unit": "ADR",
        "repetition_handling": "All available repetitions for a sampled ADR remain together.",
        "effect_estimates": [
            "difference_a_minus_b in mean macro-F1 across repetitions",
            "difference_a_minus_b in mean Cohen's kappa across repetitions",
        ],
        "mcnemar_note": (
            "Exact McNemar tests are reported within each repetition as secondary "
            "diagnostics. No pooled McNemar p-value is calculated across repetitions."
        ),
        "repository_dependence_note": (
            "Repository-level summaries are exploratory sensitivity checks. They "
            "show whether paired accuracy differences are concentrated in a small "
            "number of source repositories, but they are not presented as a formal "
            "cluster-adjusted hypothesis test."
        ),
        "comparisons": comparisons,
    }

    with open(PAIRWISE_STATS_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Pairwise statistical tests written to {PAIRWISE_STATS_PATH}")
    return report


# ============================================================
# CHANGED (2026-08-17): eval-set selection, R2 review comment #6
# ------------------------------------------------------------
# The evaluation set is now selected through proportional stratified sampling
# instead of ranking ADRs by structural/decision-quality scores. Ranking by
# rubric-derived scores conditions on the target label and can remove weak or
# Non-Compliant ADRs from the benchmark. The current n=162 sample is drawn
# across compliance class x template variant x domain, with a per-repository
# cap, and is documented by the run-specific sampling audit.
# ============================================================

# Domain grouping for the source repositories, used as one stratification
# axis in select_eval_adrs(). Repos not listed here (i.e. present in the
# fetched corpus but not yet documented in the paper's repository table)
# fall back to "Other" rather than raising, so sampling never breaks on a
# corpus that has grown since the table was last written.
REPO_DOMAIN = {
    "argoproj/argo-cd": "DevOps",
    "alphagov/govuk-aws": "Government",
    "alphagov/content-publisher": "Government",
    "alphagov/di-authentication-api": "Government",
    "npryce/adr-tools": "Infrastructure",
    "aws/aws-cdk": "Infrastructure",
    "deshpandetanmay/lightweight-architecture-decision-records": "Infrastructure",
    "cortexproject/cortex": "Observability",
    "grafana/grafana": "Observability",
    "backstage/backstage": "Platform",
    "apache/airflow": "Platform",
    "temporalio/temporal": "Platform",
    "openfga/openfga": "Platform",
    "loopdive/js2": "Compiler",
    "MITLibraries/timdex-ui": "Library Discovery",
    "adr/madr": "Tooling",
    "thomvaill/log4brains": "Tooling",
}


def get_domain(source_repo: str) -> str:
    return REPO_DOMAIN.get(source_repo, "Other")


def _count_by(items, key_fn):
    counts = Counter(key_fn(item) for item in items)
    return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))


def _repo_cap_metadata(selected: List[Dict], per_repo_cap: int) -> Dict:
    repo_distribution = Counter(a["source_repo"] for a in selected)
    cap_exceptions = {
        repo: count
        for repo, count in sorted(repo_distribution.items())
        if count > per_repo_cap
    }
    return {
        "per_repo_cap": per_repo_cap,
        "per_repo_cap_type": (
            "soft diversity cap; relaxed only if needed to reach the target "
            "sample size after stratification and few-shot holdout"
        ),
        "soft_cap_exceeded": bool(cap_exceptions),
        "cap_exceptions": cap_exceptions,
    }


def write_sampling_audit(adrs: List[Dict], labels: Dict, selected: List[Dict],
                         n: int, per_repo_cap: int,
                         heldout_ids: set = None) -> None:
    """Write a reviewer-facing audit of corpus filtering and eval-set sampling."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    heldout_ids = heldout_ids or set()

    labeled = [a for a in adrs if a["id"] in labels]
    eligible = [a for a in labeled if a["source_repo"] not in UNLICENSED_REPOS]
    eligible_for_evaluation = [a for a in eligible if a["id"] not in heldout_ids]
    selected_ids = {a["id"] for a in selected}
    selected_labeled = [a for a in selected if a["id"] in labels]

    audit = {
        "written_at": datetime.now().isoformat(),
        "sample_size_target": n,
        "sample_size_actual": len(selected),
        "sample_size_rationale": (
            "Functional validation subset; not used for scientific inference."
            if VALIDATION_MODE
            else "Fixed evaluation-set size used by the reported benchmark; not a population-level precision claim."
        ),
        "sampling_method": (
            "deterministic class-covered functional validation subset"
            if VALIDATION_MODE
            else "proportional stratified sampling"
        ),
        "stratification_axes": [
            "compliance_class",
            "template_variant",
            "application_domain",
        ],
        "repository_diversity": _repo_cap_metadata(selected, per_repo_cap),
        "initial_corpus": {
            "active_adr_json_count": len(adrs),
            "repo_distribution": _count_by(adrs, lambda a: a["source_repo"]),
            "variant_distribution": _count_by(adrs, lambda a: a.get("variant", "OTHER")),
            "filtering_heuristic": (
                "Candidate screening excludes non-substantive files such as indexes, "
                "placeholders, and templates without project-specific decisions. It is "
                "not used to choose only well-formed or compliant ADRs for evaluation."
            ),
            "filtering_threshold": "structural heuristic score >= 3 during corpus construction",
        },
        "label_pool": {
            "labeled_count": len(labeled),
            "eligible_count_after_license_exclusion": len(eligible),
            "eligible_count_after_few_shot_holdout": len(eligible_for_evaluation),
            "excluded_unlicensed_repositories": sorted(UNLICENSED_REPOS),
            "heldout_few_shot_exemplar_ids": sorted(heldout_ids),
            "class_distribution": _count_by(
                eligible, lambda a: labels[a["id"]].get("overall", "Unknown")
            ),
            "evaluation_pool_class_distribution": _count_by(
                eligible_for_evaluation,
                lambda a: labels[a["id"]].get("overall", "Unknown")
            ),
        },
        "evaluation_set": {
            "selected_count": len(selected),
            "class_distribution": _count_by(
                selected_labeled, lambda a: labels[a["id"]].get("overall", "Unknown")
            ),
            "variant_distribution": _count_by(selected, lambda a: a.get("variant", "OTHER")),
            "domain_distribution": _count_by(selected, lambda a: get_domain(a["source_repo"])),
            "repo_distribution": _count_by(selected, lambda a: a["source_repo"]),
            "selected_ids": sorted(selected_ids),
        },
        "bias_control": {
            "quality_ranked_selection_removed": True,
            "target_label_conditioning_avoided": True,
            "few_shot_exemplars_excluded_from_eval_set": not bool(heldout_ids & selected_ids),
            "note": (
                "The evaluation set is not selected by highest structural or decision-quality "
                "score. Class distribution is controlled explicitly so Non-Compliant ADRs are "
                "not filtered out before model evaluation."
            ),
        },
    }

    out_path = ANALYSIS_DIR / "sampling_audit.json"
    with open(out_path, "w") as fp:
        json.dump(audit, fp, indent=2)
    print(f"  Sampling audit written to {out_path}")


# These repositories had no source license evidence during corpus review and
# are excluded from the frozen evaluation pool. Distributed source terms for
# included ADRs are recorded in release/source_licenses.json.
UNLICENSED_REPOS = {
    "alphagov/di-authentication-api",  # now govuk-one-login/authentication-api; still unlicensed
    "deshpandetanmay/lightweight-architecture-decision-records",
}


def _validate_eval_set_payload(saved: Dict, adrs: List[Dict], ground_truth: Dict,
                               n: int) -> tuple[bool, str]:
    """Return whether a saved eval_set.json matches the active corpus and labels."""
    if not isinstance(saved, dict) or not isinstance(saved.get("adrs"), list):
        return False, "eval_set.json must contain an 'adrs' list"
    if saved.get("n") != n or len(saved["adrs"]) != n:
        return False, f"saved set has {len(saved.get('adrs', []))} ADRs but n={n} requested"

    active_ids = {a["id"] for a in adrs}
    duplicate_ids = [
        adr_id for adr_id, count in Counter(e.get("id") for e in saved["adrs"]).items()
        if count > 1
    ]
    if duplicate_ids:
        return False, f"duplicate ADR IDs in eval_set.json: {duplicate_ids[:5]}"

    saved_ids = [e.get("id") for e in saved["adrs"]]
    missing_files = [adr_id for adr_id in saved_ids if adr_id not in active_ids]
    if missing_files:
        return False, f"{len(missing_files)} ADR IDs are not present in results/adrs"

    missing_labels = [adr_id for adr_id in saved_ids if adr_id not in ground_truth]
    if missing_labels:
        return False, f"{len(missing_labels)} ADR IDs are missing from human_ground_truth.json"

    exemplar_overlap = sorted(set(saved_ids) & set(FEW_SHOT_EXEMPLAR_IDS))
    if exemplar_overlap:
        return False, f"few-shot exemplars are present in eval_set.json: {exemplar_overlap}"

    unlicensed = []
    adrs_by_id = {a["id"]: a for a in adrs}
    for adr_id in saved_ids:
        if adrs_by_id[adr_id]["source_repo"] in UNLICENSED_REPOS:
            unlicensed.append(adr_id)
    if unlicensed:
        return False, f"{len(unlicensed)} ADR IDs come from excluded unlicensed repositories"

    return True, "ok"


def select_eval_adrs(adrs: List[Dict], ground_truth: Dict,
                     n: int = 162, per_repo_cap: int = 20,
                     seed: int = None, resample: bool = False) -> List[Dict]:
    """
    Stratified selection of n ADRs for evaluation, proportionally balanced
    across compliance class and, within each class, spread across template
    variant x domain so no single project or format dominates.

    Deliberately NOT ranked by sc_score/dq_met. Sampling on rubric-derived
    scores conditions on the target label and can mechanically exclude
    Non-Compliant ADRs from the eval set. This method samples across the
    labeled pool so the eval set reflects real-world ADR quality rather than
    a curated-toward-compliant subset.

    Persists the selection to eval_set.json so all --run calls use the same
    set. Pass resample=True (or delete eval_set.json) to draw a fresh one.
    """
    select_path = EVAL_SET_PATH

    if select_path.exists() and not resample:
        with open(select_path) as fp:
            saved = json.load(fp)
        valid, reason = _validate_eval_set_payload(saved, adrs, ground_truth, n)
        if not valid:
            print(f"\nERROR: refusing to reuse stale eval_set.json: {reason}")
            print("Run with --phase freeze --resample after human_ground_truth.json is current.")
            sys.exit(1)
        saved_ids = [e["id"] for e in saved["adrs"]]
        adrs_by_id = {a["id"]: a for a in adrs}
        selected = [adrs_by_id[adr_id] for adr_id in saved_ids]
        print(f"\n  Reusing eval_set.json: {len(selected)} ADRs "
              f"(method={saved.get('method', 'unknown')}). Delete or pass --resample to reselect.")
        write_sampling_audit(adrs, ground_truth, selected, n, per_repo_cap,
                             heldout_ids=set(FEW_SHOT_EXEMPLAR_IDS))
        return selected

    heldout_ids = set(FEW_SHOT_EXEMPLAR_IDS)
    annotated = [a for a in adrs if a["id"] in ground_truth
                 and a["source_repo"] not in UNLICENSED_REPOS
                 and a["id"] not in heldout_ids]
    if len(annotated) < n:
        raise ValueError(f"Only {len(annotated)} annotated ADRs available, need n={n}")

    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    rng = random.Random(seed)

    # Stratify by (compliance class, template variant, domain).
    strata = defaultdict(list)
    for a in annotated:
        cls = ground_truth[a["id"]]["overall"]
        variant = a.get("variant", "OTHER")
        domain = get_domain(a["source_repo"])
        strata[(cls, variant, domain)].append(a)
    for group in strata.values():
        rng.shuffle(group)

    # Proportional allocation: each compliance class gets a share of n
    # matching its share of the full annotated pool, so a class that's
    # 46% of the pool (e.g. Non-Compliant) doesn't get filtered down to 2%
    # by construction. Within a class, round-robin across (variant, domain)
    # cells for diversity, respecting a per-repo cap.
    class_counts = Counter(ground_truth[a["id"]]["overall"] for a in annotated)
    total = len(annotated)

    selected, selected_ids = [], set()
    repo_counts = defaultdict(int)

    classes = sorted(class_counts, key=lambda c: class_counts[c], reverse=True)
    quotas = {c: max(1, round(n * class_counts[c] / total)) for c in classes}
    # Rounding can overshoot; trim the largest quota(s) down to hit n exactly.
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=lambda c: quotas[c])
        quotas[biggest] -= 1

    for cls in classes:
        quota = quotas[cls]
        cells = [k for k in strata if k[0] == cls]
        rng.shuffle(cells)
        taken, idx, stalled = 0, 0, 0
        while taken < quota and stalled < len(cells):
            cell = cells[idx % len(cells)]
            idx += 1
            pool = strata[cell]
            placed = False
            while pool:
                candidate = pool.pop(0)
                if candidate["id"] in selected_ids:
                    continue
                if repo_counts[candidate["source_repo"]] >= per_repo_cap:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate["id"])
                repo_counts[candidate["source_repo"]] += 1
                taken += 1
                placed = True
                break
            stalled = 0 if placed else stalled + 1

    # Fallback: if class quotas + diversity caps left us short of n (e.g. a
    # class ran out of eligible cells before its quota was met). Two passes:
    # first try to fill while still respecting per_repo_cap, then only if
    # still short, ignore the cap as an absolute last resort.
    if len(selected) < n:
        remaining = [a for a in annotated if a["id"] not in selected_ids]
        rng.shuffle(remaining)
        for a in remaining:
            if len(selected) == n:
                break
            if repo_counts[a["source_repo"]] >= per_repo_cap:
                continue
            selected.append(a)
            selected_ids.add(a["id"])
            repo_counts[a["source_repo"]] += 1

    if len(selected) < n:
        remaining = [a for a in annotated if a["id"] not in selected_ids]
        rng.shuffle(remaining)
        for a in remaining:
            selected.append(a)
            selected_ids.add(a["id"])
            repo_counts[a["source_repo"]] += 1
            if len(selected) == n:
                break

    selected = selected[:n]

    final_dist = Counter(ground_truth[a["id"]]["overall"] for a in selected)
    print(f"\n  Selected {len(selected)} ADRs via stratified sampling "
          f"(class x variant x domain), seed={seed}")
    print(f"  Class distribution: {dict(final_dist)}  "
          f"(full pool: {dict(class_counts)})")

    payload = {
        "selected_at": datetime.now().isoformat(),
        "n": len(selected),
        "seed": seed,
        "method": "stratified (compliance_class x template_variant x domain)",
        "per_repo_cap": per_repo_cap,
        "repository_diversity": _repo_cap_metadata(selected, per_repo_cap),
        "heldout_few_shot_exemplar_ids": FEW_SHOT_EXEMPLAR_IDS,
        "class_distribution": dict(final_dist),
        "full_pool_class_distribution": dict(class_counts),
        "adrs": [
            {
                "id": a["id"],
                "source_repo": a["source_repo"],
                "variant": a.get("variant", "OTHER"),
                "domain": get_domain(a["source_repo"]),
                "ground_truth": ground_truth[a["id"]]["overall"],
            }
            for a in selected
        ],
    }
    with open(select_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"  Saved to {select_path}")
    write_sampling_audit(adrs, ground_truth, selected, n, per_repo_cap,
                         heldout_ids=heldout_ids)

    return selected


def get_eval_adrs(adrs: List[Dict], ground_truth: Dict,
                  n_eval: int, seed: int = None, resample: bool = False) -> List[Dict]:
    """
    Return the ADRs to evaluate, persisting the choice to eval_sample.json
    so every --run call uses the same set.

    Pass resample=True (or delete eval_sample.json) to draw a fresh sample.
    """
    sample_path = EXPERIMENT_DIR / "eval_sample.json"

    if sample_path.exists() and not resample:
        with open(sample_path) as fp:
            saved = json.load(fp)
        if saved["n_actual"] != n_eval:
            print(f"\n  WARNING: existing sample has {saved['n_actual']} ADRs "
                  f"but --n-eval={n_eval} requested. Drawing a new sample.")
        else:
            sampled_ids = saved["adr_ids"]
            id_index = {aid: i for i, aid in enumerate(sampled_ids)}
            eval_adrs = [a for a in adrs if a["id"] in id_index]
            eval_adrs.sort(key=lambda a: id_index[a["id"]])
            print(f"\n  Reusing existing ADR sample: {len(eval_adrs)} ADRs  (seed={saved['seed']})")
            print(f"  Pass --resample to draw a new sample.")
            return eval_adrs

    eligible = [a for a in adrs if a["id"] in ground_truth]

    if len(eligible) < n_eval:
        raise ValueError(
            f"Not enough ADRs after filtering to meet n_eval={n_eval} "
            f"(only {len(eligible)} have ground truth)"
        )

    # Balanced sampling by variant
    groups = defaultdict(list)

    for adr in eligible:
        groups[adr.get("variant", "OTHER")].append(adr)

    balanced = []
    TARGET_PER_GROUP = max(10, n_eval // max(len(groups), 1))

    for g in groups:
        sample_size = min(len(groups[g]), TARGET_PER_GROUP)
        balanced.extend(random.sample(groups[g], sample_size))

    if len(balanced) >= n_eval:
        eligible = balanced
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    rng = random.Random(seed)
    n = min(n_eval, len(eligible))
    sampled = rng.sample(eligible, n)

    sample_data = {
        "sampled_at": datetime.now().isoformat(),
        "n_requested": n_eval,
        "n_actual": n,
        "seed": seed,
        "adr_ids": [a["id"] for a in sampled],
        "adrs": [
            {
                "id": a["id"],
                "source_repo": a["source_repo"],
                "filename": a["filename"],
                "word_count": a["word_count"],
                "url": a.get("url", ""),
                "ground_truth": ground_truth[a["id"]]["overall"],
            }
            for a in sampled
        ],
    }
    with open(sample_path, "w") as fp:
        json.dump(sample_data, fp, indent=2)

    print(f"\n  Sampled {n} ADRs randomly (seed={seed}) -> {sample_path}")
    return sampled


def _api_key_env_for_model(model_name: str) -> str:
    cfg = MODELS[model_name]
    if cfg["provider"] == "anthropic":
        return "ANTHROPIC_API_KEY"
    if cfg["provider"] == "gemini":
        return cfg.get("api_key_env", "GEMINI_API_KEY")
    return cfg.get("api_key_env", "OPENAI_API_KEY")


def _api_key_present(model_name: str) -> bool:
    cfg = MODELS[model_name]
    env_var = _api_key_env_for_model(model_name)
    return bool(os.environ.get(env_var) or cfg.get("api_key_default"))


def _is_transient_api_error(error: str) -> bool:
    """Return True for provider/network errors that are safe to retry."""
    if not error:
        return False
    err = error.lower()
    transient_markers = [
        "429",
        "rate_limit",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "upstream connect error",
        "disconnect",
        "connection reset",
        "reset before headers",
        "reset reason",
        "overflow",
        "502",
        "503",
        "504",
    ]
    return any(marker in err for marker in transient_markers)


def _is_quota_or_billing_exhausted(error: str) -> bool:
    """Return True for provider quota/billing states that should stop runs."""
    if not error:
        return False
    err = error.lower()
    hard_stop_markers = [
        "insufficient_quota",
        "quota exceeded",
        "resource_exhausted",
        "prepayment credits are depleted",
        "credits are depleted",
        "billing",
        "prepay",
    ]
    return any(marker in err for marker in hard_stop_markers)


def _is_truncated_finish_reason(finish_reason) -> bool:
    if finish_reason is None:
        return False
    return str(finish_reason).lower() in {"length", "max_tokens"}


def _prediction_is_complete(result: Dict, prediction: Dict) -> bool:
    return (
        not result.get("error")
        and prediction.get("overall") is not None
        and prediction.get("criterion_parse_success")
        and not _is_truncated_finish_reason(result.get("finish_reason"))
    )


def call_llm_with_valid_prediction(model_name: str, prompt: str,
                                   max_output_retries: int = 2) -> tuple:
    """
    Call a model until the response contains a complete benchmark prediction.

    API-level transient retries are handled inside call_llm. This wrapper adds
    retries for provider responses that parse partially, omit C1-C7/Q1-Q7
    fields, or end because of output truncation.
    """
    attempt_history = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    total_latency = 0.0
    total_attempts = 0
    last_result = None
    last_prediction = None

    for output_attempt in range(1, max_output_retries + 2):
        result = call_llm(model_name, prompt)
        prediction = (
            extract_prediction(result["raw"])
            if result.get("raw")
            else _empty_prediction()
        )
        usage = result.get("usage", {"input_tokens": 0, "output_tokens": 0})
        total_usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        total_usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        total_latency += float(result.get("latency", 0) or 0)
        total_attempts += int(result.get("attempts", 1) or 1)
        attempt_history.append({
            "output_attempt": output_attempt,
            "api_attempts": result.get("attempts", 1),
            "error": result.get("error"),
            "finish_reason": result.get("finish_reason"),
            "parse_success": prediction.get("overall") is not None,
            "parsed_json": prediction.get("parsed_json", False),
            "criterion_parse_success": prediction.get("criterion_parse_success", False),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })

        last_result = result
        last_prediction = prediction

        if _prediction_is_complete(result, prediction):
            break
        if result.get("quota_exhausted"):
            break
        if output_attempt <= max_output_retries:
            reason = result.get("error") or "incomplete criterion-level JSON"
            print(f"  Retrying incomplete output for {model_name}: {reason}")
            time.sleep(3 * output_attempt)

    last_result = dict(last_result or {})
    last_result["usage"] = total_usage
    last_result["latency"] = round(total_latency, 2)
    last_result["attempts"] = total_attempts
    last_result["retry_count"] = max(0, total_attempts - 1)
    last_result["output_retry_count"] = max(0, len(attempt_history) - 1)
    last_result["attempt_history"] = attempt_history
    return last_result, last_prediction or _empty_prediction()


def _save_rep_results(result_file: Path, rep_results_list: List[List[Dict]]) -> None:
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 6):
        tmp_file = result_file.with_name(
            f".{result_file.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with open(tmp_file, "w", encoding="utf-8") as fp:
                json.dump(rep_results_list, fp, indent=2)
                fp.write("\n")
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_file, result_file)
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except OSError:
                pass
            if attempt == 5:
                break
            time.sleep(0.5 * attempt)
    raise last_error


def _repair_saved_reps(model_name: str, strategy: str,
                       rep_results_list: List[List[Dict]],
                       result_file: Path, eval_adrs: List[Dict],
                       ground_truth: Dict, fs_examples: List[Dict],
                       exhausted_models: set, lock, stop_event) -> bool:
    """
    Repair incomplete rows inside already-saved repetitions.

    This avoids rerunning an entire 162-ADR repetition when a provider returned
    a transient connection error or omitted criterion-level fields for only a
    small subset of rows.
    """
    repaired_any = False
    n_adrs = len(eval_adrs)

    for rep_idx, rep_results in enumerate(rep_results_list, 1):
        bad_indexes = [
            row_idx
            for row_idx, row in enumerate(rep_results, 1)
            if _result_row_validation_reason(row, rep_idx, row_idx)
        ]
        if not bad_indexes:
            continue

        print(
            f"  REPAIR {model_name}/{strategy} rep {rep_idx}: "
            f"{len(bad_indexes)} incomplete rows"
        )
        for adr_idx in bad_indexes:
            if stop_event.is_set():
                return repaired_any

            adr = eval_adrs[adr_idx - 1]
            reason = _result_row_validation_reason(
                rep_results[adr_idx - 1], rep_idx, adr_idx
            )
            print(
                f"  {model_name[:12]:>12} | {strategy[:5]} | r{rep_idx} | "
                f"[{adr_idx}/{n_adrs}] repair | {adr['id'][:20]:>20} | {reason}"
            )
            prompt = make_prompt(adr["text"], strategy, fs_examples)
            result, prediction = call_llm_with_valid_prediction(model_name, prompt)
            attach_run_metadata(result, model_name, strategy, adr["text"], prompt)

            if result.get("quota_exhausted"):
                with lock:
                    exhausted_models.add(model_name)
                print(f"\n  QUOTA EXHAUSTED for {model_name} during repair - stopping")
                _save_rep_results(result_file, rep_results_list)
                return repaired_any

            actual = ground_truth[adr["id"]]["overall"]
            rep_results[adr_idx - 1] = build_result_row(adr, actual, prediction, result)
            repaired_any = True
            _save_rep_results(result_file, rep_results_list)

            valid_prediction = _prediction_is_complete(result, prediction)
            status = "OK" if valid_prediction else "?"
            predicted = prediction.get("overall")
            print(
                f"  {model_name[:12]:>12} | {strategy[:5]} | r{rep_idx} | "
                f"[{adr_idx}/{n_adrs}] repaired | {adr['id'][:20]:>20} | "
                f"{status} {predicted or 'PARSE_FAIL'}"
            )
            time.sleep(RATE_LIMIT_DELAY)

    return repaired_any


def _run_pair(model_name: str, strategy: str,
              eval_adrs: List[Dict], ground_truth: Dict,
              fs_examples: List[Dict], n_reps: int,
              exhausted_models: set, lock,
              stop_event) -> None:
    """Run one model/strategy pair. Called in a thread."""
    with lock:
        if model_name in exhausted_models:
            print(f"  SKIP  {model_name}/{strategy} — quota exhausted")
            return

    result_file = RESULTS_DIR / f"{model_name}_{strategy}.json"
    expected_ids = [adr["id"] for adr in eval_adrs]
    if result_file.exists():
        with open(result_file) as fp:
            rep_results_list = json.load(fp)
        valid_shape, reason = _validate_result_shape(
            rep_results_list, expected_ids, n_reps
        )
        if not valid_shape:
            _archive_stale_result(result_file, reason)
            rep_results_list = []
        elif rep_results_list:
            repaired = _repair_saved_reps(
                model_name, strategy, rep_results_list, result_file,
                eval_adrs, ground_truth, fs_examples, exhausted_models, lock,
                stop_event,
            )
            valid, reason = _validate_result_reps(
                rep_results_list, expected_ids, n_reps
            )
            if not valid:
                if repaired:
                    print(
                        f"  WAIT  {model_name}/{strategy} still incomplete after "
                        f"repair attempt: {reason}"
                    )
                return
        completed_reps = len(rep_results_list)
        if completed_reps:
            print(f"  Resuming {model_name}/{strategy} from rep {completed_reps + 1}")
    else:
        rep_results_list = []
        completed_reps = 0

    if completed_reps >= n_reps:
        print(f"  DONE  {model_name}/{strategy} already complete")
        return

    for rep in range(completed_reps, n_reps):
        if stop_event.is_set():
            break

        rep_results = []
        quota_hit = False

        n_adrs = len(eval_adrs)
        for adr_idx, adr in enumerate(eval_adrs, 1):
            if stop_event.is_set():
                break

            prompt = make_prompt(adr["text"], strategy, fs_examples)
            result, prediction = call_llm_with_valid_prediction(model_name, prompt)
            attach_run_metadata(result, model_name, strategy, adr["text"], prompt)

            if result.get("quota_exhausted"):
                with lock:
                    exhausted_models.add(model_name)
                print(f"\n  QUOTA EXHAUSTED for {model_name} — stopping")
                quota_hit = True
                break

            predicted = prediction.get("overall")
            actual = ground_truth[adr["id"]]["overall"]
            rep_results.append(build_result_row(adr, actual, prediction, result))

            valid_prediction = _prediction_is_complete(result, prediction)
            status = "OK" if valid_prediction and predicted == actual else "FAIL" if valid_prediction else "?"
            pct = adr_idx / n_adrs * 100
            print(f"  {model_name[:12]:>12} | {strategy[:5]} | r{rep+1} | "
                  f"[{adr_idx}/{n_adrs}] {pct:4.0f}% | "
                  f"{adr['id'][:20]:>20} | {status} {predicted or 'PARSE_FAIL'}")

            time.sleep(RATE_LIMIT_DELAY)

        if quota_hit:
            break

        if rep_results:
            rep_results_list.append(rep_results)
            _save_rep_results(result_file, rep_results_list)
            print(f"  Saved rep {rep + 1} -> {result_file}")


def _pending_calls_for_pair(model_name: str, strategy: str,
                            eval_adrs: List[Dict], n_reps: int) -> int:
    """Count only missing repetitions and invalid saved rows for a resumed pair."""
    result_file = RESULTS_DIR / f"{model_name}_{strategy}.json"
    total_calls = n_reps * len(eval_adrs)
    if not result_file.exists():
        return total_calls
    try:
        rep_results = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return total_calls

    expected_ids = [adr["id"] for adr in eval_adrs]
    shape_valid, _ = _validate_result_shape(rep_results, expected_ids, n_reps)
    if not shape_valid:
        return total_calls

    invalid_rows = sum(
        1
        for rep_idx, rep in enumerate(rep_results, 1)
        for row_idx, row in enumerate(rep, 1)
        if _result_row_validation_reason(row, rep_idx, row_idx)
    )
    missing_repetitions = n_reps - len(rep_results)
    return invalid_rows + missing_repetitions * len(eval_adrs)


def run_experiments(adrs: List[Dict], ground_truth: Dict,
                    n_eval: int = N_EVAL, n_reps: int = N_REPS,
                    targets: List[tuple] = None, workers: int = 3,
                    seed: int = None, resample: bool = False):
    """
    Run experiments for the given targets in parallel (one thread per model/strategy pair).

    workers: number of pairs to run simultaneously (default 3).
    Each pair saves its own result file after every rep — safe to kill and resume.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    ensure_run_configuration()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    work = targets if targets is not None else [
        (m, s) for m in MODELS for s in STRATEGIES
    ]

    skipped_models = set()
    valid_work = []
    for model_name, strategy in work:
        if _api_key_present(model_name):
            valid_work.append((model_name, strategy))
        else:
            if model_name not in skipped_models:
                env_var = _api_key_env_for_model(model_name)
                print(f"  SKIP  {model_name} — {env_var} not set")
                skipped_models.add(model_name)
    work = valid_work

    if not work:
        print("\nERROR: No models have API keys set. Nothing to run.")
        return

    eval_adrs = select_eval_adrs(adrs, ground_truth, n=n_eval, seed=seed, resample=resample)
    fs_examples = build_few_shot_examples(adrs, ground_truth, eval_adrs)
    write_rubric_manifest()
    write_threshold_sensitivity(ground_truth, eval_adrs)
    write_manuscript_dataset_summary(adrs, ground_truth, eval_adrs)
    write_prompt_manifest(adrs, ground_truth, eval_adrs)
    run_preflight(adrs, ground_truth, eval_adrs)
    full_design_calls = len(work) * n_reps * len(eval_adrs)
    pending_calls = sum(
        _pending_calls_for_pair(model_name, strategy, eval_adrs, n_reps)
        for model_name, strategy in work
    )

    print(f"\n{'='*60}")
    print(f"PHASE 3: Running experiments")
    print(f"  Targets : {[f'{m}/{s}' for m, s in work]}")
    print(f"  Workers : {workers}  |  Reps: {n_reps}  |  ADRs: {len(eval_adrs)}")
    print(f"  Few-shot exemplars held out: {[ex['id'] for ex in fs_examples]}")
    print(
        f"  Pending calls: {pending_calls} of {full_design_calls} design calls"
        f"  |  Est. time: {pending_calls * 2.5 / 60 / workers:.0f} min"
    )
    print(f"{'='*60}")

    exhausted_models = set()
    lock = threading.Lock()
    stop_event = threading.Event()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_pair,
                model_name, strategy,
                eval_adrs, ground_truth, fs_examples, n_reps,
                exhausted_models, lock, stop_event
            ): (model_name, strategy)
            for model_name, strategy in work
        }
        try:
            for future in as_completed(futures):
                model_name, strategy = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  ERROR in {model_name}/{strategy}: {e}")
        except KeyboardInterrupt:
            print("\n\n  Ctrl+C received — stopping after current API calls finish...")
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            print("  All threads stopped. Progress saved. Resume by re-running the same command.")


def _select_smoke_adr(adrs: List[Dict], ground_truth: Dict,
                      smoke_adr_id: str = None) -> tuple:
    eval_adrs = _load_eval_adrs_from_manifest(adrs)
    eval_ids = {adr["id"] for adr in eval_adrs}
    if smoke_adr_id:
        if smoke_adr_id not in eval_ids:
            raise ValueError(
                f"Smoke ADR '{smoke_adr_id}' is not in the frozen eval_set.json"
            )
        selected = next(adr for adr in eval_adrs if adr["id"] == smoke_adr_id)
    else:
        selected = max(eval_adrs, key=lambda adr: len(adr.get("text", "")))

    if selected["id"] not in ground_truth:
        raise ValueError(f"Smoke ADR '{selected['id']}' has no human ground-truth label")
    return selected, eval_adrs


def run_smoke_tests(adrs: List[Dict], ground_truth: Dict,
                    targets: List[tuple] = None,
                    smoke_adr_id: str = None) -> Dict:
    """Run one API call per selected model/strategy without touching raw_results."""
    ensure_run_configuration()
    SMOKE_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    smoke_adr, eval_adrs = _select_smoke_adr(adrs, ground_truth, smoke_adr_id)
    fs_examples = build_few_shot_examples(adrs, ground_truth, eval_adrs)
    run_preflight(adrs, ground_truth, eval_adrs)

    work = targets if targets is not None else [
        (model_name, strategy)
        for model_name in MODELS
        for strategy in STRATEGIES
    ]

    print(f"\n{'='*60}")
    print("API SMOKE TEST")
    print(f"  Mistral variant: {ACTIVE_MISTRAL_VARIANT}")
    print(f"  ADR: {smoke_adr['id']}")
    print(f"  Targets: {[f'{m}/{s}' for m, s in work]}")
    print(f"  Output directory: {SMOKE_TESTS_DIR}")
    print(f"{'='*60}")

    rows = []
    for model_name, strategy in work:
        prompt = make_prompt(smoke_adr["text"], strategy, fs_examples)
        if not _api_key_present(model_name):
            env_var = _api_key_env_for_model(model_name)
            print(f"  SKIP  {model_name}/{strategy} — {env_var} not set")
            rows.append({
                "model": model_name,
                "strategy": strategy,
                "status": "missing_api_key",
                "required_env_var": env_var,
                "result_schema_version": RESULT_SCHEMA_VERSION,
            })
            continue

        print(f"  CALL  {model_name}/{strategy}")
        result, prediction = call_llm_with_valid_prediction(
            model_name,
            prompt,
            max_output_retries=0,
        )
        attach_run_metadata(
            result, model_name, strategy, smoke_adr["text"], prompt
        )
        row = build_result_row(
            smoke_adr,
            ground_truth[smoke_adr["id"]]["overall"],
            prediction,
            result,
        )
        required_sc_present = all(
            row["predicted_sc_checks"].get(criterion) is not None
            for criterion in SC_WEIGHTS
        ) if isinstance(row.get("predicted_sc_checks"), dict) else False
        required_dq_present = all(
            row["predicted_dq_scores"].get(criterion) is not None
            for criterion in DQ_WEIGHTS
        ) if isinstance(row.get("predicted_dq_scores"), dict) else False
        status = (
            "passed"
            if not row.get("error")
            and row.get("parse_success")
            and row.get("criterion_parse_success")
            and required_sc_present
            and required_dq_present
            else "failed"
        )
        row.update({
            "model": model_name,
            "strategy": strategy,
            "status": status,
            "prompt_chars": len(prompt),
            "raw_response_chars": len(result.get("raw") or ""),
            "raw_response_excerpt": (result.get("raw") or "")[:2000],
            "required_structural_fields_present": required_sc_present,
            "required_decision_quality_fields_present": required_dq_present,
        })
        rows.append(row)
        parsed_label = row.get("predicted") or "PARSE_FAIL"
        print(f"  {'OK' if status == 'passed' else 'FAIL'}   {model_name}/{strategy} -> {parsed_label}")

    passed = sum(1 for row in rows if row.get("status") == "passed")
    skipped = sum(1 for row in rows if row.get("status") == "missing_api_key")
    failed = [row for row in rows if row.get("status") not in {"passed", "missing_api_key"}]
    report = {
        "written_at": datetime.now().isoformat(),
        "purpose": (
            "Pre-flight API smoke test for the active model identifiers, prompt "
            "strategies, JSON schema, parser, token capture, and criterion-level "
            "C1-C7/Q1-Q7 output capture. This file is not used for benchmark scoring."
        ),
        "active_mistral_variant": ACTIVE_MISTRAL_VARIANT,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "active_model_reproducibility": [
            model_reproducibility_row(model_name)
            for model_name in MODELS
        ],
        "smoke_adr": {
            "id": smoke_adr["id"],
            "source_repo": smoke_adr.get("source_repo"),
            "variant": smoke_adr.get("variant", "OTHER"),
            "domain": get_domain(smoke_adr.get("source_repo", "")),
            "human_overall": ground_truth[smoke_adr["id"]]["overall"],
        },
        "summary": {
            "total_targets": len(rows),
            "passed": passed,
            "skipped_missing_api_key": skipped,
            "failed": len(failed),
            "all_passed": len(failed) == 0 and skipped == 0 and bool(rows),
        },
        "results": rows,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SMOKE_TESTS_DIR / f"api_smoke_{timestamp}.json"
    latest_path = SMOKE_TESTS_DIR / "api_smoke_latest.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    with open(latest_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"\n  Smoke report written to {out_path}")
    print(f"  Latest smoke report written to {latest_path}")
    if failed:
        print("  Smoke test had failures. Fix them before the final benchmark run.")
    elif skipped:
        print("  Smoke test skipped one or more targets because API keys were missing.")
    else:
        print("  Smoke test passed for all requested targets.")
    return report


# ============================================================
# PROVIDER BATCH RUNS
# ============================================================

def _to_plain(value: Any):
    """Convert SDK objects into JSON-serializable dictionaries."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return _to_plain(value.dict())
    data = {}
    for attr in (
        "id", "name", "status", "state", "processing_status",
        "output_file_id", "error_file_id", "results_url", "request_counts",
        "dest", "file_name", "inlined_responses", "error", "created_at",
        "ended_at", "expires_at",
    ):
        if hasattr(value, attr):
            data[attr] = _to_plain(getattr(value, attr))
    return data or str(value)


def _plain_get(obj: Any, path: List[str], default=None):
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if hasattr(cur, "name") and not isinstance(cur, str):
            cur = cur.name
    return default if cur is None else cur


def _status_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name") and not isinstance(value, str):
        return str(value.name)
    return str(value)


def _gemini_dest_file_name(obj: Any):
    return (
        _plain_get(obj, ["dest", "file_name"])
        or _plain_get(obj, ["dest", "fileName"])
        or _plain_get(obj, ["dest", "file"])
    )


def _gemini_status_key(status: Any) -> str:
    status_text = _status_name(status)
    return status_text.split(".")[-1]


def _safe_jsonl_write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_batch_manifest() -> Dict:
    if not BATCH_MANIFEST_PATH.exists():
        return {
            "written_at": None,
            "active_mistral_variant": ACTIVE_MISTRAL_VARIANT,
            "jobs": {},
        }
    with open(BATCH_MANIFEST_PATH, encoding="utf-8") as fp:
        return json.load(fp)


def _save_batch_manifest(manifest: Dict) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest["written_at"] = datetime.now().isoformat()
    manifest["active_mistral_variant"] = ACTIVE_MISTRAL_VARIANT
    with open(BATCH_MANIFEST_PATH, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)


def _batch_provider_for_model(model_name: str) -> str:
    cfg = MODELS[model_name]
    if cfg["provider"] == "openai" and "base_url" not in cfg:
        return "openai"
    if cfg["provider"] in {"anthropic", "gemini"}:
        return cfg["provider"]
    return "unsupported"


def _batchable_targets(targets: List[tuple] = None) -> List[tuple]:
    work = targets if targets is not None else [
        (model_name, strategy)
        for model_name in MODELS
        for strategy in STRATEGIES
    ]
    batchable = []
    skipped = []
    for model_name, strategy in work:
        provider = _batch_provider_for_model(model_name)
        if provider == "unsupported":
            skipped.append(f"{model_name}/{strategy}")
        else:
            batchable.append((model_name, strategy))
    if skipped:
        print("  Batch skip unsupported targets:", skipped)
    return batchable


def _batch_model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", model_name).strip("_")[:18]


def _batch_strategy_slug(strategy: str) -> str:
    return {
        "zero_shot": "zs",
        "few_shot": "fs",
        "chain_of_thought": "cot",
    }.get(strategy, re.sub(r"[^A-Za-z0-9_]+", "_", strategy)[:10])


def _batch_job_key(model_name: str, strategy: str) -> str:
    protocol = protocol_fingerprint(model_name, strategy)[:12]
    return f"{model_name}_{strategy}_{protocol}"


def _batch_custom_id(model_name: str, strategy: str, rep: int, adr_idx: int) -> str:
    return f"{_batch_model_slug(model_name)}_{_batch_strategy_slug(strategy)}_r{rep:02d}_{adr_idx:03d}"


def _openai_chat_body(model_name: str, prompt: str) -> Dict:
    cfg = MODELS[model_name]
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if not cfg.get("no_temperature"):
        body["temperature"] = 0
    if cfg.get("use_max_completion_tokens"):
        body["max_completion_tokens"] = cfg.get("max_completion_tokens", 1024)
    else:
        body["max_tokens"] = cfg.get("max_tokens", 1024)
    if cfg.get("reasoning_effort"):
        body["reasoning_effort"] = cfg["reasoning_effort"]
    return body


def _anthropic_message_params(model_name: str, prompt: str) -> Dict:
    cfg = MODELS[model_name]
    params = {
        "model": cfg["model"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.get("max_tokens", 1024),
    }
    if not cfg.get("no_temperature"):
        params["temperature"] = 0
    return params


def _gemini_generate_request(model_name: str, prompt: str) -> Dict:
    cfg = MODELS[model_name]
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "generation_config": {
            "max_output_tokens": cfg.get("max_output_tokens", 1024),
        },
    }


def _batch_request_line(provider: str, custom_id: str,
                        model_name: str, prompt: str) -> Dict:
    if provider == "openai":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _openai_chat_body(model_name, prompt),
        }
    if provider == "anthropic":
        return {
            "custom_id": custom_id,
            "params": _anthropic_message_params(model_name, prompt),
        }
    if provider == "gemini":
        return {
            "key": custom_id,
            "request": _gemini_generate_request(model_name, prompt),
        }
    raise ValueError(f"Unsupported batch provider: {provider}")


def prepare_batch_job_files(model_name: str, strategy: str,
                            eval_adrs: List[Dict], ground_truth: Dict,
                            fs_examples: List[Dict], n_reps: int) -> Dict:
    provider = _batch_provider_for_model(model_name)
    if provider == "unsupported":
        raise ValueError(f"{model_name} does not support this batch path")

    job_key = _batch_job_key(model_name, strategy)
    request_path = BATCH_REQUESTS_DIR / f"{job_key}.jsonl"
    map_path = BATCH_REQUESTS_DIR / f"{job_key}_map.json"
    rows = []
    request_map = {}

    for rep in range(1, n_reps + 1):
        for adr_idx, adr in enumerate(eval_adrs, 1):
            custom_id = _batch_custom_id(model_name, strategy, rep, adr_idx)
            prompt = make_prompt(adr["text"], strategy, fs_examples)
            rows.append(_batch_request_line(provider, custom_id, model_name, prompt))
            request_map[custom_id] = {
                "model": model_name,
                "strategy": strategy,
                "rep": rep,
                "adr_index": adr_idx,
                "adr_id": adr["id"],
                "actual": ground_truth[adr["id"]]["overall"],
                "protocol_fingerprint": protocol_fingerprint(model_name, strategy),
                "input_audit": build_input_audit(adr["text"], prompt),
            }

    _safe_jsonl_write(request_path, rows)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w", encoding="utf-8") as fp:
        json.dump(request_map, fp, indent=2)

    return {
        "job_key": job_key,
        "provider": provider,
        "model": model_name,
        "strategy": strategy,
        "exact_model_id": MODELS[model_name]["model"],
        "protocol_fingerprint": protocol_fingerprint(model_name, strategy),
        "run_id": ACTIVE_RUN_ID,
        "request_file": str(request_path),
        "request_map_file": str(map_path),
        "output_file": str(BATCH_OUTPUTS_DIR / f"{job_key}.jsonl"),
        "raw_result_file": str(RESULTS_DIR / f"{model_name}_{strategy}.json"),
        "request_count": len(rows),
        "repetitions": n_reps,
        "adrs_per_repetition": len(eval_adrs),
        "status": "prepared",
        "prepared_at": datetime.now().isoformat(),
        "billing_mode": "batch",
        "cost_multiplier": 0.5,
    }


def _submit_openai_batch(job: Dict) -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    with open(job["request_file"], "rb") as fp:
        uploaded = client.files.create(file=fp, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "job_key": job["job_key"],
            "model": job["model"],
            "strategy": job["strategy"],
        },
    )
    return {
        "provider_input_file_id": uploaded.id,
        "provider_batch_id": batch.id,
        "provider_status": getattr(batch, "status", None),
        "provider_response": _to_plain(batch),
    }


def _submit_anthropic_batch(job: Dict) -> Dict:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    batch = client.messages.batches.create(
        requests=_read_jsonl(Path(job["request_file"]))
    )
    return {
        "provider_batch_id": batch.id,
        "provider_status": getattr(batch, "processing_status", None),
        "provider_response": _to_plain(batch),
    }


def _submit_gemini_batch(job: Dict) -> Dict:
    from google import genai as google_genai
    from google.genai import types as genai_types
    client = google_genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    uploaded = client.files.upload(
        file=job["request_file"],
        config=genai_types.UploadFileConfig(
            display_name=job["job_key"],
            mime_type="application/jsonl",
        ),
    )
    batch = client.batches.create(
        model=MODELS[job["model"]]["model"],
        src=uploaded.name,
        config={"display_name": job["job_key"]},
    )
    return {
        "provider_input_file_id": uploaded.name,
        "provider_batch_id": batch.name,
        "provider_status": _status_name(_plain_get(batch, ["state"], None)),
        "provider_response": _to_plain(batch),
    }


def submit_batch_job(job: Dict) -> Dict:
    if job["provider"] == "openai":
        update = _submit_openai_batch(job)
    elif job["provider"] == "anthropic":
        update = _submit_anthropic_batch(job)
    elif job["provider"] == "gemini":
        update = _submit_gemini_batch(job)
    else:
        raise ValueError(f"Unsupported batch provider: {job['provider']}")
    job.update(update)
    job["status"] = "submitted"
    job["submitted_at"] = datetime.now().isoformat()
    return job


def create_batch_jobs(adrs: List[Dict], ground_truth: Dict,
                      n_eval: int = N_EVAL, n_reps: int = N_REPS,
                      targets: List[tuple] = None, seed: int = None,
                      resample: bool = False, submit: bool = True) -> Dict:
    ensure_run_configuration()
    eval_adrs = select_eval_adrs(adrs, ground_truth, n=n_eval, seed=seed,
                                 resample=resample)
    fs_examples = build_few_shot_examples(adrs, ground_truth, eval_adrs)
    write_prompt_manifest(adrs, ground_truth, eval_adrs)
    run_preflight(adrs, ground_truth, eval_adrs)
    write_model_reproducibility_manifest()

    work = _batchable_targets(targets)
    if not work:
        print("ERROR: No batch-supported targets selected.")
        return {}

    manifest = _load_batch_manifest()
    manifest.setdefault("jobs", {})
    print(f"\n{'='*60}")
    print("BATCH CREATE")
    print(f"  Submit: {submit}")
    print(f"  Targets: {[f'{m}/{s}' for m, s in work]}")
    print(f"  Reps: {n_reps}  |  ADRs: {len(eval_adrs)}")
    print(f"{'='*60}")

    for model_name, strategy in work:
        provider = _batch_provider_for_model(model_name)
        env_var = _api_key_env_for_model(model_name)
        job = prepare_batch_job_files(
            model_name, strategy, eval_adrs, ground_truth, fs_examples, n_reps
        )
        if submit:
            if not os.environ.get(env_var):
                print(f"  SKIP submit {model_name}/{strategy} - {env_var} not set")
            else:
                print(f"  SUBMIT {model_name}/{strategy} via {provider}")
                job = submit_batch_job(job)
        else:
            print(f"  PREPARED {model_name}/{strategy} -> {job['request_file']}")
        manifest["jobs"][job["job_key"]] = job
        _save_batch_manifest(manifest)

    print(f"\n  Batch manifest written to {BATCH_MANIFEST_PATH}")
    return manifest


def _openai_client_for_batch():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _anthropic_client_for_batch():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _gemini_client_for_batch():
    from google import genai as google_genai
    return google_genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def _poll_batch_job(job: Dict) -> Dict:
    provider = job["provider"]
    if provider == "openai":
        batch = _openai_client_for_batch().batches.retrieve(job["provider_batch_id"])
        status = getattr(batch, "status", None)
        job.update({
            "provider_status": status,
            "provider_output_file_id": getattr(batch, "output_file_id", None),
            "provider_error_file_id": getattr(batch, "error_file_id", None),
            "provider_response": _to_plain(batch),
        })
        if status in {"completed", "failed", "expired", "cancelled", "canceled"}:
            job["status"] = status
            job["completed_at"] = datetime.now().isoformat()
    elif provider == "anthropic":
        batch = _anthropic_client_for_batch().messages.batches.retrieve(
            job["provider_batch_id"]
        )
        status = getattr(batch, "processing_status", None)
        job.update({
            "provider_status": status,
            "provider_response": _to_plain(batch),
            "provider_results_url": getattr(batch, "results_url", None),
        })
        if status in {"ended", "failed", "expired", "canceled", "cancelled"}:
            job["status"] = status
            job["completed_at"] = datetime.now().isoformat()
    elif provider == "gemini":
        client = _gemini_client_for_batch()
        batch = client.batches.get(name=job["provider_batch_id"])
        plain = _to_plain(batch)
        status = _gemini_status_key(_plain_get(batch, ["state"], None))
        job.update({
            "provider_status": status,
            "provider_response": plain,
            "provider_output_file_id": _gemini_dest_file_name(plain),
        })
        if status in {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED",
        }:
            job["status"] = status
            job["completed_at"] = datetime.now().isoformat()
    return job


def poll_batch_jobs(targets: List[tuple] = None) -> Dict:
    manifest = _load_batch_manifest()
    jobs = manifest.get("jobs", {})
    selected = {
        _batch_job_key(model_name, strategy)
        for model_name, strategy in _batchable_targets(targets)
    } if targets else set(jobs)

    print(f"\n{'='*60}")
    print("BATCH POLL")
    print(f"{'='*60}")
    for job_key in sorted(selected):
        job = jobs.get(job_key)
        if not job:
            print(f"  MISSING {job_key} - no batch manifest entry")
            continue
        if not job.get("provider_batch_id"):
            print(f"  PREPARED {job_key} - not submitted")
            continue
        try:
            job = _poll_batch_job(job)
            job["last_poll_error"] = None
            jobs[job_key] = job
            print(f"  {job_key}: {job.get('provider_status')}")
        except Exception as exc:
            job["last_poll_error"] = {
                "at": datetime.now().isoformat(),
                "error": str(exc),
            }
            jobs[job_key] = job
            print(f"  ERROR {job_key}: {exc}")
        _save_batch_manifest(manifest)

    _save_batch_manifest(manifest)
    return manifest


def _download_openai_batch_output(job: Dict) -> str:
    client = _openai_client_for_batch()
    output_file_id = job.get("provider_output_file_id")
    if not output_file_id:
        job = _poll_batch_job(job)
        output_file_id = job.get("provider_output_file_id")
    if not output_file_id:
        raise ValueError(f"{job['job_key']} has no OpenAI output_file_id yet")
    content = client.files.content(output_file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(content)


def _download_anthropic_batch_output(job: Dict) -> str:
    client = _anthropic_client_for_batch()
    lines = []
    for item in client.messages.batches.results(job["provider_batch_id"]):
        lines.append(json.dumps(_to_plain(item), ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _download_gemini_batch_output(job: Dict) -> str:
    client = _gemini_client_for_batch()
    batch = client.batches.get(name=job["provider_batch_id"])
    output_file = _gemini_dest_file_name(batch)
    if not output_file:
        plain = _to_plain(batch)
        output_file = _gemini_dest_file_name(plain)
    if not output_file:
        raise ValueError(f"{job['job_key']} has no Gemini output file yet")
    content = client.files.download(file=output_file)
    return content.decode("utf-8") if isinstance(content, bytes) else str(content)


def download_batch_output(job: Dict) -> Path:
    BATCH_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(job["output_file"])
    if job["provider"] == "openai":
        text = _download_openai_batch_output(job)
    elif job["provider"] == "anthropic":
        text = _download_anthropic_batch_output(job)
    elif job["provider"] == "gemini":
        text = _download_gemini_batch_output(job)
    else:
        raise ValueError(f"Unsupported batch provider: {job['provider']}")
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _extract_openai_batch_result(item: Dict) -> tuple:
    response = item.get("response") or {}
    body = response.get("body") or {}
    error = item.get("error") or response.get("error")
    if response.get("status_code") and response.get("status_code") != 200:
        error = error or body
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    usage = body.get("usage") or {}
    result = {
        "raw": _plain_get(choice, ["message", "content"]),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "latency": None,
        "error": json.dumps(error) if error else None,
        "finish_reason": choice.get("finish_reason"),
    }
    return item.get("custom_id"), result


def _extract_anthropic_batch_result(item: Dict) -> tuple:
    custom_id = item.get("custom_id")
    result_obj = item.get("result") or {}
    result_type = result_obj.get("type")
    message = result_obj.get("message") or {}
    error = None if result_type == "succeeded" else result_obj.get("error")
    content = message.get("content") or []
    raw_parts = []
    for part in content:
        text = part.get("text") if isinstance(part, dict) else None
        if text:
            raw_parts.append(text)
    usage = message.get("usage") or {}
    result = {
        "raw": "\n".join(raw_parts) if raw_parts else None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "latency": None,
        "error": json.dumps(error) if error else None,
        "finish_reason": message.get("stop_reason"),
    }
    return custom_id, result


def _extract_gemini_batch_result(item: Dict) -> tuple:
    custom_id = item.get("key")
    response = item.get("response") or {}
    error = item.get("error") or response.get("error")
    candidates = response.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    parts = _plain_get(candidate, ["content", "parts"], []) or []
    raw_parts = []
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else None
        if text:
            raw_parts.append(text)
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    result = {
        "raw": "\n".join(raw_parts) if raw_parts else None,
        "usage": {
            "input_tokens": usage.get("promptTokenCount", usage.get("prompt_token_count", 0)),
            "output_tokens": usage.get("candidatesTokenCount", usage.get("candidates_token_count", 0)),
        },
        "latency": None,
        "error": json.dumps(error) if error else None,
        "finish_reason": candidate.get("finishReason") or candidate.get("finish_reason"),
    }
    return custom_id, result


def _batch_result_to_row(job: Dict, item: Dict, request_map: Dict,
                         adrs_by_id: Dict[str, Dict]) -> Dict:
    provider = job["provider"]
    if provider == "openai":
        custom_id, result = _extract_openai_batch_result(item)
    elif provider == "anthropic":
        custom_id, result = _extract_anthropic_batch_result(item)
    elif provider == "gemini":
        custom_id, result = _extract_gemini_batch_result(item)
    else:
        raise ValueError(f"Unsupported batch provider: {provider}")

    meta = request_map.get(custom_id)
    if not meta:
        raise ValueError(f"Batch output custom_id not found in request map: {custom_id}")
    result.update({
        "accessed_at": datetime.now().isoformat(),
        "model_metadata": model_reproducibility_row(job["model"]),
        "attempts": 1,
        "retry_count": 0,
        "output_retry_count": 0,
        "billing_mode": "batch",
        "cost_multiplier": 0.5,
        "strategy": meta["strategy"],
        "protocol_fingerprint": meta["protocol_fingerprint"],
        "input_audit": meta["input_audit"],
        "batch_metadata": {
            "job_key": job["job_key"],
            "provider": provider,
            "provider_batch_id": job.get("provider_batch_id"),
            "custom_id": custom_id,
        },
    })
    prediction = (
        extract_prediction(result["raw"])
        if result.get("raw")
        else _empty_prediction()
    )
    return build_result_row(
        adrs_by_id[meta["adr_id"]],
        meta["actual"],
        prediction,
        result,
    )


def collect_batch_job(job: Dict, adrs: List[Dict], expected_ids: List[str],
                      n_reps: int) -> Dict:
    output_path = Path(job["output_file"])
    if not output_path.exists():
        output_path = download_batch_output(job)

    with open(job["request_map_file"], encoding="utf-8") as fp:
        request_map = json.load(fp)
    adrs_by_id = {adr["id"]: adr for adr in adrs}
    rows_by_rep = {rep: [] for rep in range(1, n_reps + 1)}
    missing_ids = set(request_map)
    errors = []

    for item in _read_jsonl(output_path):
        try:
            custom_id = item.get("custom_id") or item.get("key")
            row = _batch_result_to_row(job, item, request_map, adrs_by_id)
            meta = request_map[custom_id]
            rows_by_rep[int(meta["rep"])].append((int(meta["adr_index"]), row))
            missing_ids.discard(custom_id)
        except Exception as exc:
            errors.append({"item": item, "error": str(exc)})

    missing_output_count = len(missing_ids)
    for custom_id in sorted(missing_ids):
        meta = request_map[custom_id]
        adr = adrs_by_id[meta["adr_id"]]
        placeholder_result = {
            "accessed_at": datetime.now().isoformat(),
            "model_metadata": model_reproducibility_row(job["model"]),
            "strategy": meta["strategy"],
            "protocol_fingerprint": meta["protocol_fingerprint"],
            "input_audit": meta["input_audit"],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "attempts": 0,
            "retry_count": 0,
            "output_retry_count": 0,
            "billing_mode": "repair_placeholder",
            "cost_multiplier": 0.0,
            "batch_metadata": {
                "job_key": job["job_key"],
                "provider": job["provider"],
                "provider_batch_id": job.get("provider_batch_id"),
                "custom_id": custom_id,
            },
            "error": "Batch provider returned no output row; pending targeted repair.",
        }
        placeholder = build_result_row(
            adr, meta["actual"], {}, placeholder_result
        )
        rows_by_rep[int(meta["rep"])].append(
            (int(meta["adr_index"]), placeholder)
        )

    rep_results = []
    for rep in range(1, n_reps + 1):
        ordered = [row for _, row in sorted(rows_by_rep[rep], key=lambda pair: pair[0])]
        rep_results.append(ordered)

    result_file = Path(job["raw_result_file"])
    valid, reason = _validate_result_reps(rep_results, expected_ids, n_reps,
                                          require_complete=True)
    shape_valid, shape_reason = _validate_result_shape(
        rep_results, expected_ids, n_reps, require_complete=True
    )
    repair_needed = []
    if shape_valid:
        for rep_idx, rep in enumerate(rep_results, 1):
            for row_idx, row in enumerate(rep, 1):
                row_reason = _result_row_validation_reason(row, rep_idx, row_idx)
                if row_reason:
                    repair_needed.append({
                        "rep": rep_idx,
                        "row": row_idx,
                        "adr_id": row.get("adr_id"),
                        "reason": row_reason,
                    })
    report = {
        "job_key": job["job_key"],
        "provider": job["provider"],
        "model": job["model"],
        "strategy": job["strategy"],
        "output_file": str(output_path),
        "raw_result_file": str(result_file),
        "missing_output_count": missing_output_count,
        "missing_outputs_represented_by_repair_placeholders": missing_output_count,
        "row_conversion_errors": errors,
        "batch_output_valid_for_merge": valid,
        "valid_for_merge": valid,
        "validation_reason": reason,
        "shape_valid_for_repair": shape_valid,
        "shape_validation_reason": shape_reason,
        "repair_needed_count": len(repair_needed),
        "repair_needed_preview": repair_needed[:20],
        "partial_raw_result_written": False,
    }
    report_path = BATCH_OUTPUTS_DIR / f"{job['job_key']}_collect_report.json"

    if not valid:
        if shape_valid:
            if result_file.exists():
                existing_results = json.loads(
                    result_file.read_text(encoding="utf-8")
                )
                existing_valid, existing_reason = _validate_result_reps(
                    existing_results,
                    expected_ids,
                    n_reps,
                    require_complete=True,
                )
                if existing_valid:
                    report["valid_for_merge"] = True
                    report["existing_valid_result_retained"] = True
                    report["validation_reason"] = (
                        "Original batch output contains invalid rows, but a separately "
                        "repaired merge-ready result file already exists and was retained."
                    )
                    print(
                        f"  RETAIN {result_file} - repaired merge-ready result "
                        "already exists"
                    )
                else:
                    existing_shape_valid, _ = _validate_result_shape(
                        existing_results,
                        expected_ids,
                        n_reps,
                        require_complete=True,
                    )
                    if existing_shape_valid:
                        existing_repairs = [
                            {
                                "rep": rep_idx,
                                "row": row_idx,
                                "adr_id": row.get("adr_id"),
                                "reason": row_reason,
                            }
                            for rep_idx, rep in enumerate(existing_results, 1)
                            for row_idx, row in enumerate(rep, 1)
                            if (row_reason := _result_row_validation_reason(
                                row, rep_idx, row_idx
                            ))
                        ]
                        report["existing_partial_result_retained"] = True
                        report["validation_reason"] = existing_reason
                        report["repair_needed_count"] = len(existing_repairs)
                        report["repair_needed_preview"] = existing_repairs[:20]
                        print(
                            f"  RETAIN {result_file} - existing partial result "
                            f"has {len(existing_repairs)} rows to repair"
                        )
                    else:
                        _archive_stale_result(result_file, existing_reason)
                        _save_rep_results(result_file, rep_results)
                        report["partial_raw_result_written"] = True
            else:
                _save_rep_results(result_file, rep_results)
                report["partial_raw_result_written"] = True
        if report["valid_for_merge"]:
            print(
                f"  READY {job['job_key']} using the retained repaired result"
            )
        else:
            print(
                f"  COLLECTED {job['job_key']} but not merge-ready: "
                f"{report['validation_reason']}"
            )
            if report["partial_raw_result_written"]:
                print(
                    f"  WROTE partial {result_file} with "
                    f"{len(repair_needed)} rows to repair"
                )
        print(f"  Review {report_path}")
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        return report

    if result_file.exists():
        existing_valid, existing_reason = _validate_result_reps(
            json.loads(result_file.read_text(encoding="utf-8")),
            expected_ids,
            n_reps,
            require_complete=True,
        )
        if existing_valid:
            report["existing_valid_result_retained"] = True
            print(f"  SKIP write {result_file} - valid result file already exists")
            with open(report_path, "w", encoding="utf-8") as fp:
                json.dump(report, fp, indent=2)
            return report
        _archive_stale_result(result_file, existing_reason)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as fp:
        json.dump(rep_results, fp, indent=2)
    report["partial_raw_result_written"] = True
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"  WROTE {result_file} from batch job {job['job_key']}")
    return report


def collect_batch_jobs(adrs: List[Dict], ground_truth: Dict,
                       n_eval: int = N_EVAL, n_reps: int = N_REPS,
                       targets: List[tuple] = None) -> Dict:
    eval_adrs = _load_eval_adrs_from_manifest(adrs)
    expected_ids = [adr["id"] for adr in eval_adrs]
    manifest = _load_batch_manifest()
    jobs = manifest.get("jobs", {})
    selected = {
        _batch_job_key(model_name, strategy)
        for model_name, strategy in _batchable_targets(targets)
    } if targets else set(jobs)

    print(f"\n{'='*60}")
    print("BATCH COLLECT")
    print(f"{'='*60}")
    reports = {}
    for job_key in sorted(selected):
        job = jobs.get(job_key)
        if not job:
            print(f"  MISSING {job_key} - no batch manifest entry")
            continue
        status = str(job.get("provider_status") or job.get("status") or "")
        terminal_ok = {
            "completed", "ended", "JOB_STATE_SUCCEEDED", "succeeded",
        }
        if not Path(job["output_file"]).exists() and status not in terminal_ok:
            try:
                job = _poll_batch_job(job)
                jobs[job_key] = job
                status = str(job.get("provider_status") or job.get("status") or "")
            except Exception as exc:
                print(f"  SKIP {job_key} - poll failed: {exc}")
                continue
        if not Path(job["output_file"]).exists() and status not in terminal_ok:
            print(f"  SKIP {job_key} - not complete yet: {status}")
            continue
        try:
            reports[job_key] = collect_batch_job(
                job, adrs, expected_ids, n_reps
            )
        except Exception as exc:
            reports[job_key] = {
                "job_key": job_key,
                "provider": job.get("provider"),
                "model": job.get("model"),
                "strategy": job.get("strategy"),
                "status": "collection_failed",
                "error": str(exc),
                "failed_at": datetime.now().isoformat(),
            }
            print(f"  ERROR collecting {job_key}: {exc}")
        _save_batch_manifest(manifest)

    manifest["jobs"] = jobs
    manifest["last_collect_reports"] = reports
    _save_batch_manifest(manifest)
    return reports


def merge_results() -> Dict:
    """
    Scan RESULTS_DIR for individual model/strategy result files,
    build all_results dict, save all_results.json, and return it.
    """
    print(f"\n{'='*60}")
    print(f"MERGE: Combining individual result files")
    print(f"{'='*60}")

    all_results = {}
    found = 0
    incomplete = []
    eval_set_path = EVAL_SET_PATH
    expected_ids = None
    if eval_set_path.exists():
        with open(eval_set_path) as fp:
            eval_set = json.load(fp)
        expected_ids = [a["id"] for a in eval_set.get("adrs", [])]

    for model_name in MODELS:
        for strategy in STRATEGIES:
            result_file = RESULTS_DIR / f"{model_name}_{strategy}.json"
            if not result_file.exists():
                reason = f"missing result file: {result_file}"
                print(f"  MISSING  {model_name}/{strategy} — {reason}")
                incomplete.append({
                    "model": model_name,
                    "strategy": strategy,
                    "reason": reason,
                })
                continue
            with open(result_file) as fp:
                data = json.load(fp)
            if expected_ids is not None:
                valid, reason = _validate_result_reps(
                    data, expected_ids, ACTIVE_N_REPS, require_complete=True
                )
                if not valid:
                    print(f"  SKIP  {model_name}/{strategy} — stale result file: {reason}")
                    incomplete.append({
                        "model": model_name,
                        "strategy": strategy,
                        "reason": reason,
                    })
                    continue
            all_results.setdefault(model_name, {})[strategy] = data
            n_reps = len(data)
            n_adrs = len(data[0]) if data else 0
            print(f"  OK  {model_name}/{strategy}  ({n_reps} reps x {n_adrs} ADRs)")
            found += 1

    expected_count = len(MODELS) * len(STRATEGIES)
    if incomplete or found != expected_count:
        print("\nERROR: Final analysis requires a complete balanced experiment.")
        print(f"Expected {expected_count} model-strategy files, each with "
              f"{ACTIVE_N_REPS} reps x {len(expected_ids or [])} ADRs from {EVAL_SET_PATH}.")
        for item in incomplete:
            print(f"  - {item['model']}/{item['strategy']}: {item['reason']}")
        print("Complete or rerun the listed configurations, then run --phase merge again.")
        sys.exit(1)

    out_path = RESULTS_DIR / "all_results.json"
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n  Merged {found} result files -> {out_path}")
    return all_results


# ============================================================
# PHASE 4: ANALYZE AND GENERATE PAPER TABLES
# ============================================================

def analyze(all_results: Dict, ground_truth: Dict):
    """Compute metrics and generate paper-ready output."""
    from sklearn.metrics import (
        precision_recall_fscore_support,
        cohen_kappa_score, confusion_matrix
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if VALIDATION_MODE:
        print(
            "\nNONPUBLICATION VALIDATION ANALYSIS: functional output only; "
            "do not combine with the reported benchmark."
        )
    write_model_reproducibility_manifest()
    write_rubric_manifest()
    write_threshold_sensitivity(ground_truth, _load_eval_adrs_from_manifest())
    CLASSES = ["Fully_Compliant", "Mostly_Compliant", "Partially_Compliant", "Not_Compliant"]
    STRAT_SHORT = {"zero_shot": "ZS", "few_shot": "FS", "chain_of_thought": "CoT"}
    MODEL_SHORT = {"gpt-5.5": "GPT-5.5", "claude-sonnet-4-6": "Claude Sonnet 4.6",
                   "mistral-7b": "Mistral 7B", "ministral-3-8b": "Ministral 3 8B",
                   "gemini-2.5-pro": "Gemini 2.5P",
                   "llama-3.1-70b": "LLaMA 3.1"}

    print(f"\n{'='*60}")
    print(f"PHASE 4: Analysis")
    print(f"{'='*60}")

    # ---- TABLE II ----
    print(f"\n{'='*60}")
    print("TABLE II: OVERALL COMPLIANCE DETECTION PERFORMANCE")
    print(f"{'='*60}")
    print(f"{'Model':>15} {'Prompt':>6} {'Prec':>12} {'Recall':>12} {'F1-mac':>12} {'F1-wt':>8} {'Kappa':>12}")
    print("-" * 80)

    metrics_summary = {}

    for model_name in all_results:
        metrics_summary[model_name] = {}
        for strategy in all_results[model_name]:
            reps = all_results[model_name][strategy]
            rep_f1s, rep_weighted_f1s, rep_kappas, rep_precs, rep_recs = [], [], [], [], []
            total_responses = sum(len(rep) for rep in reps)
            parse_failures = sum(
                1 for rep in reps for row in rep
                if not row.get("parse_success")
            )
            parse_failure_rate = parse_failures / total_responses if total_responses else 0
            classification_errors_excluding_parse_failures = sum(
                1 for rep in reps for row in rep
                if row.get("parse_success") and not row.get("correct")
            )

            for rep_data in reps:
                y_true = [r["actual"] for r in rep_data if r["predicted"]]
                y_pred = [r["predicted"] for r in rep_data if r["predicted"]]

                if len(y_true) < 10 and not (VALIDATION_MODE and y_true):
                    continue

                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
                _, _, f1w, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
                k = cohen_kappa_score(y_true, y_pred, labels=CLASSES)

                rep_f1s.append(f1)
                rep_weighted_f1s.append(f1w)
                rep_kappas.append(k)
                rep_precs.append(p)
                rep_recs.append(r)

            if rep_f1s:
                ms = MODEL_SHORT.get(model_name, model_name)
                ss = STRAT_SHORT.get(strategy, strategy)
                print(f"{ms:>15} {ss:>6} "
                      f"{np.mean(rep_precs):.2f}+/-{np.std(rep_precs, ddof=1) if len(rep_precs) > 1 else 0.0:.2f} "
                      f"{np.mean(rep_recs):.2f}+/-{np.std(rep_recs, ddof=1) if len(rep_recs) > 1 else 0.0:.2f} "
                      f"{np.mean(rep_f1s):.2f}+/-{np.std(rep_f1s, ddof=1) if len(rep_f1s) > 1 else 0.0:.2f} "
                      f"{np.mean(rep_weighted_f1s):.2f}    "
                      f"{np.mean(rep_kappas):.2f}+/-{np.std(rep_kappas, ddof=1) if len(rep_kappas) > 1 else 0.0:.2f}")

                metrics_summary[model_name][strategy] = {
                    "f1_mean": round(float(np.mean(rep_f1s)), 3),
                    "f1_std": round(float(np.std(rep_f1s, ddof=1)), 3) if len(rep_f1s) > 1 else 0.0,
                    "f1_ci95": {
                        "low": _summary_stats(rep_f1s)["ci95_low"],
                        "high": _summary_stats(rep_f1s)["ci95_high"],
                    },
                    "weighted_f1_mean": round(float(np.mean(rep_weighted_f1s)), 3),
                    "weighted_f1_std": round(
                        float(np.std(rep_weighted_f1s, ddof=1)), 3
                    ) if len(rep_weighted_f1s) > 1 else 0.0,
                    "weighted_f1_ci95": {
                        "low": _summary_stats(rep_weighted_f1s)["ci95_low"],
                        "high": _summary_stats(rep_weighted_f1s)["ci95_high"],
                    },
                    "kappa_mean": round(float(np.mean(rep_kappas)), 3),
                    "kappa_std": round(float(np.std(rep_kappas, ddof=1)), 3) if len(rep_kappas) > 1 else 0.0,
                    "kappa_ci95": {
                        "low": _summary_stats(rep_kappas)["ci95_low"],
                        "high": _summary_stats(rep_kappas)["ci95_high"],
                    },
                    "prec_mean": round(float(np.mean(rep_precs)), 3),
                    "prec_std": round(float(np.std(rep_precs, ddof=1)), 3) if len(rep_precs) > 1 else 0.0,
                    "prec_ci95": {
                        "low": _summary_stats(rep_precs)["ci95_low"],
                        "high": _summary_stats(rep_precs)["ci95_high"],
                    },
                    "rec_mean": round(float(np.mean(rep_recs)), 3),
                    "rec_std": round(float(np.std(rep_recs, ddof=1)), 3) if len(rep_recs) > 1 else 0.0,
                    "rec_ci95": {
                        "low": _summary_stats(rep_recs)["ci95_low"],
                        "high": _summary_stats(rep_recs)["ci95_high"],
                    },
                    "total_responses": total_responses,
                    "parse_failures": parse_failures,
                    "parse_failure_rate": round(float(parse_failure_rate), 4),
                    "classification_errors_excluding_parse_failures": (
                        classification_errors_excluding_parse_failures
                    ),
                }

    # ---- CONFUSION MATRICES ----
    print(f"\n{'='*60}")
    print("CONFUSION MATRICES (CoT)")
    print(f"{'='*60}")

    for model_name in all_results:
        if "chain_of_thought" not in all_results[model_name]:
            continue
        # Use first rep
        rep_data = all_results[model_name]["chain_of_thought"][0]
        y_true = [r["actual"] for r in rep_data if r["predicted"]]
        y_pred = [r["predicted"] for r in rep_data if r["predicted"]]

        if len(y_true) < 10 and not (VALIDATION_MODE and y_true):
            continue

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        ms = MODEL_SHORT.get(model_name, model_name)
        print(f"\n  {ms}:")
        header = " ".join(f"{cls[:8]:>10}" for cls in CLASSES)
        print(f"  {'':>20} {header}")
        for i, cls in enumerate(CLASSES):
            row = " ".join(f"{cm[i][j]:>10}" for j in range(len(CLASSES)))
            print(f"  {'Actual '+cls[:12]:>20} {row}")

    # ---- PAIRWISE STATISTICAL TESTS ----
    print(f"\n{'='*60}")
    print("PAIRWISE STATISTICAL TESTS")
    print(f"{'='*60}")
    pairwise_stats = write_pairwise_statistical_tests(all_results)
    print("  Primary inference uses ADR-level paired bootstrap confidence intervals")
    print("  and paired randomization tests for macro-F1 and Cohen's kappa.")
    print("  Per-repetition exact McNemar tests are reported as secondary diagnostics.")
    print(f"  Family size: {pairwise_stats['family_size']} comparisons; "
          f"Bonferroni unadjusted alpha={pairwise_stats['bonferroni_unadjusted_alpha']}")
    performance_report = write_detailed_performance_report(all_results)
    write_criterion_level_performance_report(all_results, ground_truth)
    error_report = write_error_analysis_report(all_results, ground_truth)

    # ---- COST ANALYSIS ----
    print(f"\n{'='*60}")
    print("TABLE VI: COST-PERFORMANCE")
    print(f"{'='*60}")
    write_cost_analysis_report(all_results)

    for model_name in all_results:
        if "chain_of_thought" not in all_results[model_name]:
            continue
        rep_data = all_results[model_name]["chain_of_thought"][0]
        avg_in = np.mean([int(r.get("input_tokens", 0) or 0) for r in rep_data])
        avg_out = np.mean([int(r.get("output_tokens", 0) or 0) for r in rep_data])
        latencies = [
            float(r["latency"])
            for r in rep_data
            if r.get("latency") is not None
        ]
        avg_lat = np.mean(latencies) if latencies else None
        latency_text = f"{avg_lat:.1f}s" if avg_lat is not None else "n/a"
        cost = np.mean([_row_token_cost_usd(model_name, r) for r in rep_data])
        f1 = metrics_summary.get(model_name, {}).get("chain_of_thought", {}).get("f1_mean", 0)
        ms = MODEL_SHORT.get(model_name, model_name)
        print(f"  {ms:>12} F1={f1:.2f} Cost=${cost:.4f}/ADR "
              f"Lat={latency_text} In={avg_in:.0f} Out={avg_out:.0f}")

    # Save all metrics
    with open(ANALYSIS_DIR / "metrics_summary.json", "w") as fp:
        json.dump(metrics_summary, fp, indent=2)

    write_manuscript_results_summary(
        metrics_summary, performance_report, error_report, ground_truth
    )

    try:
        eval_adrs = _load_eval_adrs_from_manifest()
        write_rule_based_baseline(eval_adrs, ground_truth, all_results)
    except Exception as e:
        print(f"  WARNING: rule-based baseline not written: {e}")

    print(f"\nAnalysis complete. Results in {ANALYSIS_DIR}")
    return metrics_summary


def export_human_annotation_template(adrs: List[Dict], ground_truth: Dict,
                                     n_eval: int = N_EVAL, seed: int = None,
                                     resample: bool = False) -> None:
    """Export CSV/JSON templates for independent human ADR annotation."""
    eval_adrs = select_eval_adrs(adrs, ground_truth, n=n_eval, seed=seed, resample=resample)
    template_dir = EXPERIMENT_DIR / "human_annotation"
    template_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "adr_id", "reviewer_id",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
        "sc_score", "dq_score", "sc_class", "dq_class", "overall", "notes",
    ]
    rows = []
    for adr in eval_adrs:
        rows.append({
            "adr_id": adr["id"],
            "reviewer_id": "",
            "C1": "", "C2": "", "C3": "", "C4": "", "C5": "", "C6": "", "C7": "",
            "Q1": "", "Q2": "", "Q3": "", "Q4": "", "Q5": "", "Q6": "", "Q7": "",
            "sc_score": "", "dq_score": "", "sc_class": "", "dq_class": "",
            "overall": "", "notes": "",
        })

    csv_path = template_dir / "human_annotation_template.csv"
    json_path = template_dir / "human_annotation_template.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)

    print(f"  Exported human annotation CSV:  {csv_path}")
    print(f"  Exported human annotation JSON: {json_path}")


# ============================================================
# MAIN
# ============================================================

def _parse_targets(raw: str) -> List[tuple]:
    """Parse 'model/strategy,...' into validated (model, strategy) pairs."""
    targets = []
    for item in raw.split(","):
        item = item.strip()
        if "/" not in item:
            print(f"ERROR: invalid format {item!r} — expected model/strategy")
            sys.exit(1)
        model_name, strategy = item.rsplit("/", 1)
        if model_name not in MODELS:
            print(f"ERROR: unknown model {model_name!r}")
            print(f"  Known models: {list(MODELS)}")
            sys.exit(1)
        if strategy not in STRATEGIES:
            print(f"ERROR: unknown strategy {strategy!r}")
            print(f"  Known strategies: {STRATEGIES}")
            sys.exit(1)
        targets.append((model_name, strategy))
    return targets


def _load_ground_truth() -> Dict:
    gt_path = GROUND_TRUTH_PATH
    if not gt_path.exists():
        print("ERROR: human reference labels are required for benchmark evaluation.")
        print(f"Expected file: {gt_path}")
        print("Generate reviewer templates with --phase template, then import reviewed labels into human_ground_truth.json.")
        sys.exit(1)
    with open(gt_path) as fp:
        return json.load(fp)


def _load_prelabels() -> Dict:
    prelabel_path = EXPERIMENT_DIR / "prelabels_gpt4o.json"
    if not prelabel_path.exists():
        print("ERROR: GPT-4o prelabels not found. Run --phase annotate first.")
        sys.exit(1)
    with open(prelabel_path) as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser(description="ADR Compliance Benchmark")
    parser.add_argument("--phase", default=None,
                        choices=[
                            "fetch", "annotate", "template", "freeze", "preflight", "repro",
                            "smoke", "run", "prepare_repair", "batch_create", "batch_poll",
                            "batch_collect", "merge", "analyze", "all",
                        ],
                        help="Pipeline phase to execute; omitted with no --run prints help")
    parser.add_argument("--run", metavar="MODEL/STRATEGY[,...]",
        help="Run specific model/strategy pairs (comma-separated); "
                             "requires fetch + human_ground_truth.json to be done already")
    parser.add_argument("--mistral-model",
                        choices=list(MISTRAL_MODEL_VARIANTS.keys()),
                        default=os.environ.get("ADR_MISTRAL_MODEL", ACTIVE_MISTRAL_VARIANT),
                        help=(
                            "Mistral-family configuration to use for smoke, run, "
                            "merge, analyze, and reproducibility manifests "
                            "(default: ministral-3-8b-api; pass "
                            "mistral-7b-local only for a local/vLLM 7B run)."
                        ))
    parser.add_argument(
        "--run-id",
        default=os.environ.get("ADR_RUN_ID", DEFAULT_RUN_ID),
        help=(
            "Isolated output directory name beneath results/runs "
            f"(default: {DEFAULT_RUN_ID})."
        ),
    )
    parser.add_argument(
        "--gpt-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=os.environ.get("ADR_GPT_REASONING_EFFORT", "medium"),
        help="GPT-5.5 reasoning effort pinned for this run (default: medium).",
    )
    parser.add_argument("--smoke-adr-id", default=None,
                        help="Optional eval_set ADR ID to use for --phase smoke")
    parser.add_argument("--batch-dry-run", action="store_true",
                        help="For --phase batch_create, write provider request files without submitting jobs")
    parser.add_argument("--n-eval", type=int, default=162,
                        help="Number of ADRs in the frozen evaluation set (default: 162)")
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help=(
            "Use a deterministic, run-local subset of the frozen evaluation set "
            "for functional validation. Requires a new --run-id and keeps three "
            "repetitions. Results are marked as nonpublication."
        ),
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=3105,
        help="Seed for --validation-size subset selection (default: 3105).",
    )
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel model/strategy pairs to run simultaneously (default: 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for ADR sampling (random if not set)")
    parser.add_argument("--resample", action="store_true",
                        help="Draw a fresh ADR sample even if eval_set.json exists")
    args = parser.parse_args()

    # A bare invocation must never start paid or corpus-changing work.
    if args.phase is None and args.run is None:
        parser.print_help()
        return

    try:
        configure_run_paths(args.run_id)
        configure_mistral_variant(args.mistral_model)
        configure_gpt_reasoning_effort(args.gpt_reasoning_effort)
        if args.validation_size is not None:
            if args.n_eval != N_EVAL:
                raise ValueError(
                    "Use --validation-size by itself; --n-eval remains reserved "
                    "for the frozen publication protocol"
                )
            if args.n_reps != N_REPS:
                raise ValueError(
                    f"Validation mode preserves the publication repetition count ({N_REPS})"
                )
            if args.resample:
                raise ValueError(
                    "Validation subsets are deterministic; do not use --resample"
                )
            unsupported = {"fetch", "annotate", "template", "freeze", "all"}
            if args.phase in unsupported:
                raise ValueError(
                    f"--validation-size does not support --phase {args.phase}"
                )
            configure_validation_mode(args.validation_size, args.validation_seed)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    n_eval = ACTIVE_N_EVAL if VALIDATION_MODE else args.n_eval
    n_reps = ACTIVE_N_REPS if VALIDATION_MODE else args.n_reps

    if RUN_CONFIG_PATH.exists():
        saved = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
        if saved.get("scoring_version") != rubric.SCORING_VERSION:
            parser.error("Archived scoring is read-only. Use scripts/rescore_saved_results.py with a new run ID.")
        if saved.get("analysis_only") and (args.run or args.phase not in {"merge", "analyze"}):
            parser.error("This is an offline analysis revision; only merge and analyze are allowed.")

    protocol_bound = (
        args.phase in {
            "freeze", "preflight", "repro", "smoke", "run", "prepare_repair",
            "batch_create", "batch_poll", "batch_collect", "merge", "analyze", "all",
        }
        or (args.run is not None and args.phase is None)
    )
    if (
        protocol_bound
        and not VALIDATION_MODE
        and (n_eval != N_EVAL or n_reps != N_REPS)
    ):
        print(
            "ERROR: The publication protocol is frozen at "
            f"n_eval={N_EVAL} and n_reps={N_REPS}. Choose a separate protocol "
            "and run ID before changing either value."
        )
        sys.exit(1)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()

    if args.phase == "preflight":
        ensure_run_configuration()
        adrs = load_adrs()
        if not adrs:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)
        ground_truth = _load_ground_truth()
        eval_adrs = _load_eval_adrs_from_manifest(adrs)
        write_prompt_manifest(adrs, ground_truth, eval_adrs)
        write_model_reproducibility_manifest()
        report = run_preflight(adrs, ground_truth, eval_adrs)
        print(
            f"\nPREFLIGHT PASSED for {report['evaluation_set_size']} ADRs. "
            f"Run directory: {RUN_DIR}"
        )
        return

    if args.phase == "smoke":
        targets = _parse_targets(args.run) if args.run else None
        print(f"\n{'='*60}")
        print("SMOKE TEST RUN")
        print(f"Started: {start_time.isoformat()}")
        print(f"Mistral variant: {ACTIVE_MISTRAL_VARIANT}")
        print(f"{'='*60}")
        adrs = load_adrs()
        if not adrs:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)
        ground_truth = _load_ground_truth()
        run_smoke_tests(adrs, ground_truth, targets=targets,
                        smoke_adr_id=args.smoke_adr_id)
        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"SMOKE COMPLETE. Time: {elapsed}")
        print(f"{'='*60}")
        return

    if args.phase == "prepare_repair":
        targets = _parse_targets(args.run) if args.run else None
        print(f"\n{'='*60}")
        print("PREPARE RAW RESULTS FOR MANIFEST REPAIR")
        print(f"Started: {start_time.isoformat()}")
        print(f"Mistral variant: {ACTIVE_MISTRAL_VARIANT}")
        print(f"{'='*60}")
        adrs = load_adrs()
        if not adrs:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)
        ground_truth = _load_ground_truth()
        prepare_raw_results_for_manifest_repair(
            adrs,
            ground_truth,
            targets=targets,
            n_reps=n_reps,
        )
        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"PREPARE REPAIR COMPLETE. Time: {elapsed}")
        print(f"{'='*60}")
        return

    # --run: targeted experiment run — skip fetch/annotate, load from disk
    if args.phase in {"batch_create", "batch_poll", "batch_collect"}:
        targets = _parse_targets(args.run) if args.run else None
        print(f"\n{'='*60}")
        print(f"BATCH PHASE: {args.phase}")
        print(f"Started: {start_time.isoformat()}")
        print(f"Mistral variant: {ACTIVE_MISTRAL_VARIANT}")
        print(f"{'='*60}")
        adrs = load_adrs()
        if not adrs and args.phase in {"batch_create", "batch_collect"}:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)
        ground_truth = _load_ground_truth() if args.phase in {"batch_create", "batch_collect"} else {}
        if args.phase == "batch_create":
            create_batch_jobs(
                adrs, ground_truth, n_eval=n_eval, n_reps=n_reps,
                targets=targets, seed=args.seed, resample=args.resample,
                submit=not args.batch_dry_run,
            )
        elif args.phase == "batch_poll":
            poll_batch_jobs(targets=targets)
        elif args.phase == "batch_collect":
            collection_reports = collect_batch_jobs(
                adrs, ground_truth, n_eval=n_eval, n_reps=n_reps,
                targets=targets,
            )
            failed_reports = [
                report for report in collection_reports.values()
                if report.get("status") == "collection_failed"
                or report.get("valid_for_merge") is False
            ]
            if failed_reports:
                print(
                    f"\nERROR: {len(failed_reports)} completed batch job(s) "
                    "could not be collected into merge-ready results."
                )
                sys.exit(1)
        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"BATCH PHASE COMPLETE. Time: {elapsed}")
        print(f"{'='*60}")
        return

    if args.run and args.phase in (None, "run"):
        targets = _parse_targets(args.run)
        print(f"\n{'='*60}")
        print(f"TARGETED RUN: {[f'{m}/{s}' for m, s in targets]}")
        print(f"Started: {start_time.isoformat()}")
        print(f"{'='*60}")
        adrs = load_adrs()
        if not adrs:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)
        ground_truth = _load_ground_truth()
        run_experiments(adrs, ground_truth, n_eval=n_eval, n_reps=n_reps,
                        targets=targets, workers=args.workers,
                        seed=args.seed, resample=args.resample)
        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"DONE. Time: {elapsed}  —  run --phase merge when all targets are complete.")
        print(f"{'='*60}")
        return

    # --phase based flow
    print(f"\n{'='*60}")
    print(f"ADR COMPLIANCE EXPERIMENT  |  phase={args.phase}")
    print(f"Started: {start_time.isoformat()}")
    print(f"{'='*60}")

    if args.phase in ("fetch", "all"):
        adrs = fetch_adrs_from_github()
    else:
        adrs = load_adrs()

    if args.phase in ("annotate", "template", "freeze", "all", "run"):
        if not adrs:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)

    if args.phase in ("annotate", "all"):
        prelabels = generate_gpt4o_prelabels(adrs)
    else:
        prelabels = None

    if args.phase in ("template",):
        prelabels = _load_prelabels()
        export_human_annotation_template(adrs, prelabels, n_eval=n_eval,
                                         seed=args.seed, resample=args.resample)
    elif args.phase == "freeze":
        ensure_run_configuration()
        ground_truth = _load_ground_truth()
        eval_adrs = select_eval_adrs(adrs, ground_truth, n=n_eval, seed=args.seed,
                                     resample=args.resample)
        build_few_shot_examples(adrs, ground_truth, eval_adrs)
        write_rubric_manifest()
        write_threshold_sensitivity(ground_truth, eval_adrs)
        write_manuscript_dataset_summary(adrs, ground_truth, eval_adrs)
        write_prompt_manifest(adrs, ground_truth, eval_adrs)
        run_preflight(adrs, ground_truth, eval_adrs)
        write_rule_based_baseline(eval_adrs, ground_truth)
        write_model_reproducibility_manifest()
        print(f"\nFreeze complete. Review results/eval_set.json and {ANALYSIS_DIR} before API runs.")
        return
    elif args.phase == "repro":
        ensure_run_configuration()
        write_model_reproducibility_manifest()
        print(f"\nReproducibility manifest complete. Review {MODEL_REPRODUCIBILITY_PATH} before API runs.")
        return
    elif args.phase in ("run", "all"):
        ground_truth = _load_ground_truth()

    if args.phase in ("run", "all"):
        write_model_reproducibility_manifest()
        run_experiments(adrs, ground_truth, n_eval=n_eval, n_reps=n_reps,
                        workers=args.workers,
                        seed=args.seed, resample=args.resample)

    if args.phase in ("merge", "all"):
        ensure_run_configuration()
        ground_truth = _load_ground_truth()
        all_results = merge_results()
        analyze(all_results, ground_truth)

    elif args.phase == "analyze":
        ensure_run_configuration()
        # Legacy: analyze from existing all_results.json
        results_path = RESULTS_DIR / "all_results.json"
        if not results_path.exists():
            print("ERROR: all_results.json not found. Run --phase merge first.")
            sys.exit(1)
        with open(results_path) as fp:
            all_results = json.load(fp)
        ground_truth = _load_ground_truth()
        analyze(all_results, ground_truth)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE. Total time: {elapsed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
