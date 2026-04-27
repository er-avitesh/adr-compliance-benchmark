#!/usr/bin/env python3
"""
Pre-flight config test for the overnight ADR experiment.

Usage:
  python test.py --test_config
  python test.py --run gemini-2.5-pro/zero_shot
  python test.py --run gemini-2.5-pro/zero_shot,gemini-2.5-pro/chain_of_thought
"""

import os
import re
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overnight_experiment import (
    MODELS, STRATEGIES,
    ANNOTATION_SYSTEM_PROMPT, ANNOTATION_PROMPT_TEMPLATE,
    make_prompt, call_llm, extract_classification,
)

# A well-formed synthetic ADR used for all test calls
SAMPLE_ADR = """# ADR-001: Use PostgreSQL for Primary Database

## Status
Accepted

## Context
We need a reliable database for storing user data and financial transactions.
The system requires ACID compliance and support for complex relational queries.

## Decision Drivers
- ACID compliance is mandatory for financial data integrity
- Team has 5+ years of experience with relational databases
- Need to support complex JOIN queries across many tables

## Considered Options
1. PostgreSQL — mature, open-source, full ACID, rich ecosystem
2. MongoDB — flexible document model, horizontal scaling, weaker ACID guarantees

## Decision Outcome
Chosen: PostgreSQL, because it satisfies our ACID requirement and team expertise
minimises operational risk. MongoDB's flexible schema adds no value here.

## Consequences
Positive: Strong consistency guarantees; well-understood tooling; easy to hire for.
Negative: Schema migrations require careful coordination; vertical scaling has limits.
"""

VALID_CLASSES = {"Compliant", "Partially_Compliant", "Non_Compliant"}
FEW_SHOT_EXAMPLES = [{"text": SAMPLE_ADR[:400], "label": "Compliant"}]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_env_key(cfg: dict) -> str:
    if cfg["provider"] == "anthropic":
        return "ANTHROPIC_API_KEY"
    if cfg["provider"] == "gemini":
        return cfg.get("api_key_env", "GEMINI_API_KEY")
    return cfg.get("api_key_env", "OPENAI_API_KEY")


def _mask(val: str) -> str:
    return val[:8] + "..." + val[-4:] if len(val) > 12 else "***"


def _parse_oracle_json(raw: str):
    """Return (ok, detail) for oracle annotation output."""
    if not raw:
        return False, "empty response"
    try:
        clean = raw.strip()
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not match:
            return False, "no JSON object in response"
        parsed = json.loads(match.group(0))
        required = {"overall", "sc_class", "dq_class"}
        missing = required - set(parsed.keys())
        if missing:
            return False, f"missing keys: {missing}"
        if parsed.get("overall") not in VALID_CLASSES:
            return False, f"invalid overall value: {parsed.get('overall')!r}"
        return True, f"overall={parsed['overall']}  sc={parsed['sc_class']}  dq={parsed['dq_class']}"
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e} | raw snippet: {raw[:120]!r}"


# ---------------------------------------------------------------------------
# individual test runners
# ---------------------------------------------------------------------------

def test_env_vars() -> dict[str, bool]:
    print("\n[1/3] ENVIRONMENT VARIABLES")
    print("-" * 50)
    env_map = {
        "ANTHROPIC_API_KEY": "claude-sonnet-4-6",
        "OPENAI_API_KEY":    "gpt-5.1",
        "MISTRAL_API_KEY":   "mistral-7b",
        "GEMINI_API_KEY":    "gemini-2.5-pro",
    }
    results = {}
    for var, model in env_map.items():
        val = os.environ.get(var, "")
        if val:
            print(f"  PASS  {var}: {_mask(val)}  (needed for {model})")
            results[var] = True
        else:
            print(f"  MISS  {var}: not set  (needed for {model})")
            results[var] = False
    return results


def test_oracle(env_ok: bool) -> tuple[bool, str]:
    """Test GPT-4o oracle annotation — different prompt & format from the experiment calls."""
    if not env_ok:
        return None, "OPENAI_API_KEY not set"
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                {"role": "user", "content": ANNOTATION_PROMPT_TEMPLATE.format(
                    adr_text=SAMPLE_ADR[:2000])},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content
        return _parse_oracle_json(raw)
    except Exception as e:
        return False, f"API error: {e}"


def test_model_strategy(model_name: str, strategy: str) -> tuple[bool, str]:
    """One real API call; verify extract_classification() succeeds."""
    prompt = make_prompt(SAMPLE_ADR, strategy, few_shot_examples=FEW_SHOT_EXAMPLES)
    result = call_llm(model_name, prompt)

    if result["error"]:
        return False, f"API error: {result['error']}"

    predicted = extract_classification(result["raw"])
    if predicted is None:
        snippet = repr(result["raw"][:200]) if result["raw"] else "None"
        return False, f"parse failed — raw: {snippet}"
    if predicted not in VALID_CLASSES:
        return False, f"unexpected class: {predicted!r}"

    latency = result["latency"]
    tok_in = result["usage"]["input_tokens"]
    tok_out = result["usage"]["output_tokens"]
    return True, f"{predicted}  ({latency}s  {tok_in}in/{tok_out}out tokens)"


# ---------------------------------------------------------------------------
# main orchestrator
# ---------------------------------------------------------------------------

def run_test_config() -> bool:
    print("=" * 60)
    print("ADR EXPERIMENT — PRE-FLIGHT CONFIG TEST")
    print("=" * 60)

    env = test_env_vars()

    # ---- Oracle test -------------------------------------------------------
    print("\n[2/3] ORACLE ANNOTATION FORMAT (gpt-4o)")
    print("-" * 50)
    ok, detail = test_oracle(env.get("OPENAI_API_KEY"))
    if ok is None:
        print(f"  SKIP  oracle/annotation — {detail}")
        oracle_result = ("oracle/annotation", None)
    else:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  oracle/annotation — {detail}")
        oracle_result = ("oracle/annotation", ok)
    time.sleep(0.5)

    # ---- Experiment model × strategy tests ---------------------------------
    print("\n[3/3] EXPERIMENT FORMAT (all models × strategies)")
    print("-" * 50)
    experiment_results: list[tuple[str, bool | None]] = []

    for model_name, cfg in MODELS.items():
        env_key = _get_env_key(cfg)
        if not env.get(env_key):
            for strategy in STRATEGIES:
                label = f"{model_name}/{strategy}"
                print(f"  SKIP  {label:42s} — {env_key} not set")
                experiment_results.append((label, None))
            continue

        for strategy in STRATEGIES:
            label = f"{model_name}/{strategy}"
            print(f"  ...   {label:42s}", end="", flush=True)
            ok, detail = test_model_strategy(model_name, strategy)
            status = "PASS" if ok else "FAIL"
            print(f"\r  {status}  {label:42s} — {detail}")
            experiment_results.append((label, ok))
            time.sleep(0.5)

    # ---- Summary -----------------------------------------------------------
    all_results = [oracle_result] + experiment_results
    passed  = sum(1 for _, r in all_results if r is True)
    failed  = sum(1 for _, r in all_results if r is False)
    skipped = sum(1 for _, r in all_results if r is None)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    if skipped:
        print(f"  Skipped: {skipped}  (missing API keys)")
    else:
        print(f"  Skipped: 0")

    if passed == 0 and failed == 0:
        print("\n  WARNING: No API keys found — nothing was tested.")
        print("  Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY")
        print("  and re-run before launching the experiment.")
        return False

    if failed:
        print("\n  FAILED tests:")
        for label, r in all_results:
            if r is False:
                print(f"    • {label}")
        print("\n  Fix the above before running the overnight experiment.")
        return False

    if skipped:
        print(f"\n  WARNING: {skipped} test(s) skipped due to missing API keys.")
        print("  Those models will be excluded from the overnight run.")

    print("\n  All reachable models/strategies look good.")
    print("  Safe to launch: python overnight_experiment.py")
    return True


def run_specific(targets: list[tuple[str, str]]) -> bool:
    """Run a specific subset of model/strategy tests."""
    print("=" * 60)
    print("ADR EXPERIMENT — TARGETED TEST")
    print("=" * 60)

    # Check only the env vars needed for the requested models
    needed_keys = {_get_env_key(MODELS[m]) for m, _ in targets}
    print("\nENVIRONMENT VARIABLES")
    print("-" * 50)
    env = {}
    for var in needed_keys:
        val = os.environ.get(var, "")
        if val:
            print(f"  PASS  {var}: {_mask(val)}")
            env[var] = True
        else:
            print(f"  MISS  {var}: not set")
            env[var] = False

    print("\nTESTS")
    print("-" * 50)
    results = []
    for model_name, strategy in targets:
        label = f"{model_name}/{strategy}"
        env_key = _get_env_key(MODELS[model_name])
        if not env.get(env_key):
            print(f"  SKIP  {label:42s} — {env_key} not set")
            results.append((label, None))
            continue
        print(f"  ...   {label:42s}", end="", flush=True)
        ok, detail = test_model_strategy(model_name, strategy)
        status = "PASS" if ok else "FAIL"
        print(f"\r  {status}  {label:42s} — {detail}")
        results.append((label, ok))
        time.sleep(0.5)

    passed  = sum(1 for _, r in results if r is True)
    failed  = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    print("\n" + "=" * 60)
    print(f"  Passed : {passed}  Failed : {failed}  Skipped: {skipped}")
    return failed == 0


def _parse_targets(raw: str) -> list[tuple[str, str]]:
    """Parse 'model/strategy,...' into validated (model, strategy) pairs."""
    targets = []
    for item in raw.split(","):
        item = item.strip()
        if "/" not in item:
            print(f"ERROR: invalid format {item!r} — expected model/strategy")
            sys.exit(1)
        model_name, strategy = item.rsplit("/", 1)
        if model_name not in MODELS:
            print(f"ERROR: unknown model {model_name!r}. Known: {list(MODELS)}")
            sys.exit(1)
        if strategy not in STRATEGIES:
            print(f"ERROR: unknown strategy {strategy!r}. Known: {STRATEGIES}")
            sys.exit(1)
        targets.append((model_name, strategy))
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight config test for the overnight ADR experiment")
    parser.add_argument(
        "--test_config", action="store_true",
        help="Test API connectivity + response format for every model and strategy")
    parser.add_argument(
        "--run", metavar="MODEL/STRATEGY[,...]",
        help="Test specific model/strategy pairs (comma-separated)")
    args = parser.parse_args()

    if args.test_config:
        success = run_test_config()
        sys.exit(0 if success else 1)
    elif args.run:
        targets = _parse_targets(args.run)
        success = run_specific(targets)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
