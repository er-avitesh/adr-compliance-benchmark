#!/usr/bin/env python3
"""
OVERNIGHT ADR COMPLIANCE EXPERIMENT
====================================
Minimal viable experiment: runs in 3-4 hours, ~$12 cost.

Phase 1 (30 min): Fetch 50 real ADRs from GitHub
Phase 2 (15 min): Auto-annotate ground truth using GPT-4o as oracle + manual spot-check
Phase 3 (3 hrs):  Run 3 models × 3 strategies × 3 reps × 50 ADRs = 1,350 calls
Phase 4 (5 min):  Compute all metrics, generate paper-ready tables

Usage:
  # Set your API keys
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."

  # Run overnight
  python overnight_experiment.py 2>&1 | tee experiment_log.txt

  # Or run phases separately
  python overnight_experiment.py --phase fetch
  python overnight_experiment.py --phase annotate
  python overnight_experiment.py --phase run
  python overnight_experiment.py --phase analyze
"""

import os
import sys
import json
import time
import random
import re
import hashlib
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_DIR = Path("overnight_results")
ADRS_DIR = EXPERIMENT_DIR / "adrs"
RESULTS_DIR = EXPERIMENT_DIR / "raw_results"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"

N_EVAL = 50          # ADRs for evaluation (increase to 200 for full run)
N_REPS = 3           # Repetitions (increase to 5 for full run)
RATE_LIMIT_DELAY = 1.0  # seconds between API calls

MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o-2024-08-06",
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
    },
    "mistral-7b": {
        "provider": "openai",  # via Mistral's OpenAI-compatible API or local vLLM
        "model": "mistral-small-latest",  # or "mistralai/Mistral-7B-Instruct-v0.3" for vLLM
        "base_url": "https://api.mistral.ai/v1",  # change to localhost for vLLM
        "api_key_env": "MISTRAL_API_KEY",
    },
}

STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]

# Known GitHub repos with good ADRs
ADR_REPOS = [
    # (owner, repo, adr_path_prefix)
    ("adr", "madr", "docs/decisions"),
    ("alphagov", "govuk-aws", "docs/architecture/decisions"),
    ("alphagov", "content-publisher", "docs/adr"),
    ("backstage", "backstage", "docs/architecture-decisions"),
    ("npryce", "adr-tools", "doc/adr"),
    ("deshpandetanmay", "lightweight-architecture-decision-records", "doc/adr"),
    ("thomvaill", "log4brains", "docs/adr"),
    ("openfga", "openfga", "docs/architecture"),
]


# ============================================================
# PHASE 1: FETCH REAL ADRS FROM GITHUB
# ============================================================

def fetch_adrs_from_github(target_count=60):
    """Fetch real ADRs from GitHub repos using the API."""
    import urllib.request
    import urllib.error

    ADRS_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    print(f"\n{'='*60}")
    print(f"PHASE 1: Fetching ADRs from GitHub")
    print(f"{'='*60}")

    for owner, repo, path_prefix in ADR_REPOS:
        if len(fetched) >= target_count:
            break

        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_prefix}"
        print(f"\n  Trying {owner}/{repo}/{path_prefix}...")

        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                files = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    ✗ Error listing: {e}")
            continue

        md_files = [f for f in files if isinstance(f, dict) and
                    f.get("name", "").endswith(".md") and
                    f.get("type") == "file" and
                    f.get("size", 0) > 200]  # skip tiny files

        for f_info in md_files[:10]:  # max 10 per repo
            if len(fetched) >= target_count:
                break

            try:
                req = urllib.request.Request(f_info["download_url"], headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    content = resp.read().decode("utf-8", errors="replace")

                # Basic quality filter
                word_count = len(content.split())
                if word_count < 50:
                    continue

                adr_id = f"{owner}_{repo}_{f_info['name'].replace('.md','')}"
                adr_id = re.sub(r'[^a-zA-Z0-9_-]', '_', adr_id)

                adr_data = {
                    "id": adr_id,
                    "source_repo": f"{owner}/{repo}",
                    "filename": f_info["name"],
                    "url": f_info["html_url"],
                    "word_count": word_count,
                    "text": content,
                    "fetched_at": datetime.now().isoformat(),
                }

                # Save individual ADR
                adr_path = ADRS_DIR / f"{adr_id}.json"
                with open(adr_path, "w") as fp:
                    json.dump(adr_data, fp, indent=2)

                fetched.append(adr_data)
                print(f"    ✓ {f_info['name']} ({word_count} words)")

                time.sleep(0.5)  # Rate limit

            except Exception as e:
                print(f"    ✗ Error fetching {f_info['name']}: {e}")
                continue

    # Save manifest
    manifest = {
        "total_fetched": len(fetched),
        "timestamp": datetime.now().isoformat(),
        "repos_used": [f"{o}/{r}" for o, r, _ in ADR_REPOS],
        "adr_ids": [a["id"] for a in fetched],
    }
    with open(EXPERIMENT_DIR / "adr_manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\n  ✓ Fetched {len(fetched)} ADRs total")
    return fetched


def load_adrs():
    """Load previously fetched ADRs."""
    adrs = []
    if not ADRS_DIR.exists():
        return adrs
    for f in sorted(ADRS_DIR.glob("*.json")):
        with open(f) as fp:
            adrs.append(json.load(fp))
    return adrs


# ============================================================
# PHASE 2: AUTO-ANNOTATE GROUND TRUTH
# ============================================================

ANNOTATION_SYSTEM_PROMPT = """You are an expert software architect performing ground-truth annotation 
for a research study. You must be extremely precise and consistent.

You are evaluating Architecture Decision Records (ADRs) against two dimensions:

STRUCTURAL COMPLETENESS (7 binary checks):
C1: Title present and descriptive (not generic like "ADR-001")
C2: Context/Problem Statement clearly articulated
C3: Decision Drivers explicitly listed
C4: At least two genuinely different Considered Options presented
C5: Decision Outcome stated with explicit justification
C6: Consequences (both positive AND negative) documented
C7: Status field present with valid value (proposed/accepted/deprecated/superseded)

SC Classification: Compliant (≥5 pass), Partially_Compliant (3-4 pass), Non_Compliant (<3 pass)

DECISION QUALITY (7 criteria):
Q1: Problem is relevant and significant enough for an ADR
Q2: Alternatives are genuine (not strawmen or obviously inferior)
Q3: Decision criteria are complete and well-defined
Q4: When criteria conflict, they are prioritized
Q5: Rationale is sound and convincing
Q6: Consequences are reported objectively (both positive and negative)
Q7: Solution is described in an actionable way

DQ Classification: Compliant (≥5 Met), Partially_Compliant (3-4 Met), Non_Compliant (<3 Met)

OVERALL: The worse of SC and DQ classifications.

Be strict. Most real-world ADRs are Partially Compliant, not Compliant.
Respond ONLY with valid JSON, no other text."""

ANNOTATION_PROMPT_TEMPLATE = """Evaluate this ADR:

---
{adr_text}
---

Respond with ONLY this JSON (no markdown, no explanation):
{{"C1":true/false,"C2":true/false,"C3":true/false,"C4":true/false,"C5":true/false,"C6":true/false,"C7":true/false,"sc_score":N,"sc_class":"Compliant|Partially_Compliant|Non_Compliant","Q1":"Met|Partially_Met|Not_Met","Q2":"Met|Partially_Met|Not_Met","Q3":"Met|Partially_Met|Not_Met","Q4":"Met|Partially_Met|Not_Met","Q5":"Met|Partially_Met|Not_Met","Q6":"Met|Partially_Met|Not_Met","Q7":"Met|Partially_Met|Not_Met","dq_met":N,"dq_class":"Compliant|Partially_Compliant|Non_Compliant","overall":"Compliant|Partially_Compliant|Non_Compliant"}}"""


def annotate_with_oracle(adrs: List[Dict], oracle_model="gpt-4o-2024-08-06"):
    """Use GPT-4o as oracle annotator (run twice for inter-rater agreement)."""
    from openai import OpenAI
    client = OpenAI()

    print(f"\n{'='*60}")
    print(f"PHASE 2: Annotating {len(adrs)} ADRs with oracle ({oracle_model})")
    print(f"{'='*60}")

    gt_path = EXPERIMENT_DIR / "ground_truth.json"
    ground_truth = {}

    # Load existing if resuming
    if gt_path.exists():
        with open(gt_path) as fp:
            ground_truth = json.load(fp)
        print(f"  Loaded {len(ground_truth)} existing annotations")

    for i, adr in enumerate(adrs):
        if adr["id"] in ground_truth:
            continue

        print(f"  [{i+1}/{len(adrs)}] Annotating {adr['id']}...")

        # Truncate very long ADRs to avoid token limits
        adr_text = adr["text"][:4000]

        # Run oracle TWICE for inter-rater simulation
        annotations = []
        for run in range(2):
            try:
                response = client.chat.completions.create(
                    model=oracle_model,
                    messages=[
                        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                        {"role": "user", "content": ANNOTATION_PROMPT_TEMPLATE.format(adr_text=adr_text)},
                    ],
                    temperature=0.3 if run == 0 else 0.5,  # slight variation for inter-rater
                    max_tokens=500,
                )
                raw = response.choices[0].message.content.strip()

                # Clean and parse
                raw = re.sub(r'^```json\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
                parsed = json.loads(raw)
                annotations.append(parsed)

            except Exception as e:
                print(f"    ✗ Run {run+1} failed: {e}")
                annotations.append(None)

            time.sleep(RATE_LIMIT_DELAY)

        # Use first annotation as ground truth, record agreement
        if annotations[0]:
            gt = annotations[0]
            agreement = "agree"
            if annotations[1] and annotations[0].get("overall") != annotations[1].get("overall"):
                agreement = "disagree"

            ground_truth[adr["id"]] = {
                "overall": gt.get("overall", "Partially_Compliant"),
                "sc_class": gt.get("sc_class", "Partially_Compliant"),
                "dq_class": gt.get("dq_class", "Partially_Compliant"),
                "sc_score": gt.get("sc_score", 4),
                "dq_met": gt.get("dq_met", 3),
                "details": gt,
                "inter_rater": agreement,
                "annotated_at": datetime.now().isoformat(),
            }

        # Save incrementally
        with open(gt_path, "w") as fp:
            json.dump(ground_truth, fp, indent=2)

    # Report distribution
    classes = [v["overall"] for v in ground_truth.values()]
    print(f"\n  Ground Truth Distribution:")
    for cls in ["Compliant", "Partially_Compliant", "Non_Compliant"]:
        n = classes.count(cls)
        print(f"    {cls}: {n} ({n/len(classes)*100:.0f}%)")

    agreements = [v["inter_rater"] for v in ground_truth.values()]
    agree_rate = agreements.count("agree") / len(agreements)
    print(f"  Inter-rater agreement: {agree_rate:.1%}")

    return ground_truth


# ============================================================
# PHASE 3: RUN ACTUAL LLM EXPERIMENTS
# ============================================================

SYSTEM_PROMPT = """You are a senior software architect with 15+ years of experience reviewing 
Architecture Decision Records (ADRs). Be precise and evidence-based."""

RUBRIC = """COMPLIANCE RUBRIC:
STRUCTURAL COMPLETENESS (7 checks):
C1: Title present and descriptive
C2: Context/Problem Statement articulated
C3: Decision Drivers listed
C4: At least two Considered Options
C5: Decision Outcome with justification
C6: Consequences (positive+negative) documented
C7: Status field present and valid
SC: Compliant (≥5), Partially_Compliant (3-4), Non_Compliant (<3)

DECISION QUALITY (7 criteria):
Q1: Problem relevance  Q2: Option viability (not strawmen)
Q3: Criteria completeness  Q4: Criteria prioritization
Q5: Rationale soundness  Q6: Consequence objectivity  Q7: Actionability
DQ: Compliant (≥5 Met), Partially_Compliant (3-4), Non_Compliant (<3)

OVERALL: The worse of SC and DQ."""

JSON_FORMAT = """Respond ONLY with this JSON:
{"sc_class":"Compliant|Partially_Compliant|Non_Compliant","dq_class":"Compliant|Partially_Compliant|Non_Compliant","overall":"Compliant|Partially_Compliant|Non_Compliant","confidence":0.0-1.0}"""


def make_prompt(adr_text: str, strategy: str, few_shot_examples: List[Dict] = None):
    """Build the prompt for a given strategy."""
    adr_text = adr_text[:3500]  # truncate for token budget

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
            for i, ex in enumerate(few_shot_examples[:3], 1):
                examples_text += f"""
Example {i}:
ADR: {ex['text'][:300]}...
Result: {{"overall":"{ex['label']}"}}
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

Evaluate this ADR step by step:
Step 1: Check each structural element C1-C7
Step 2: Assess each quality criterion Q1-Q7
Step 3: Identify ambiguities
Step 4: Provide final classification

ADR:
---
{adr_text}
---

After your analysis, end with EXACTLY this JSON on its own line:
{JSON_FORMAT}"""


def call_llm(model_name: str, prompt: str):
    """Call an LLM API and return result with timing."""
    cfg = MODELS[model_name]
    start = time.time()

    try:
        if cfg["provider"] == "openai":
            from openai import OpenAI
            kwargs = {}
            if "base_url" in cfg:
                kwargs["base_url"] = cfg["base_url"]
            if "api_key_env" in cfg:
                kwargs["api_key"] = os.environ.get(cfg["api_key_env"], "")
            client = OpenAI(**kwargs)

            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        elif cfg["provider"] == "anthropic":
            from anthropic import Anthropic
            client = Anthropic()

            response = client.messages.create(
                model=cfg["model"],
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            raw = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            return {"error": f"Unknown provider: {cfg['provider']}"}

        elapsed = time.time() - start

        return {
            "raw": raw,
            "usage": usage,
            "latency": round(elapsed, 2),
            "error": None,
        }

    except Exception as e:
        return {
            "raw": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency": round(time.time() - start, 2),
            "error": str(e),
        }


def extract_classification(raw_text: str) -> Optional[str]:
    """Extract overall classification from LLM response."""
    if not raw_text:
        return None

    # Try JSON extraction
    patterns = [
        r'"overall"\s*:\s*"(Compliant|Partially_Compliant|Non_Compliant)"',
        r'"overall"\s*:\s*"(compliant|partially_compliant|non_compliant)"',
    ]
    for pat in patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            val = m.group(1)
            # Normalize
            val = val.replace("compliant", "Compliant").replace("partially", "Partially").replace("non", "Non")
            if val.startswith("Partially"):
                return "Partially_Compliant"
            elif val.startswith("Non"):
                return "Non_Compliant"
            else:
                return "Compliant"

    # Fallback: look for keywords
    text_lower = raw_text.lower()
    if "non_compliant" in text_lower or "non-compliant" in text_lower:
        return "Non_Compliant"
    elif "partially_compliant" in text_lower or "partially compliant" in text_lower:
        return "Partially_Compliant"
    elif "compliant" in text_lower:
        return "Compliant"

    return None


def run_experiments(adrs: List[Dict], ground_truth: Dict):
    """Run all experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare few-shot examples from ground truth
    fs_examples = []
    for adr in adrs[:10]:  # use first 10 as potential examples
        if adr["id"] in ground_truth:
            fs_examples.append({
                "text": adr["text"],
                "label": ground_truth[adr["id"]]["overall"],
            })

    eval_adrs = [a for a in adrs if a["id"] in ground_truth][:N_EVAL]
    total_calls = len(MODELS) * len(STRATEGIES) * N_REPS * len(eval_adrs)
    call_num = 0

    print(f"\n{'='*60}")
    print(f"PHASE 3: Running experiments")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Strategies: {STRATEGIES}")
    print(f"  Reps: {N_REPS}")
    print(f"  ADRs: {len(eval_adrs)}")
    print(f"  Total calls: {total_calls}")
    print(f"  Estimated time: {total_calls * 2.5 / 60:.0f} minutes")
    print(f"{'='*60}")

    all_results = {}

    for model_name in MODELS:
        all_results[model_name] = {}

        for strategy in STRATEGIES:
            all_results[model_name][strategy] = []

            for rep in range(N_REPS):
                rep_results = []

                for adr in eval_adrs:
                    call_num += 1
                    pct = call_num / total_calls * 100

                    prompt = make_prompt(adr["text"], strategy, fs_examples)
                    result = call_llm(model_name, prompt)

                    predicted = extract_classification(result["raw"]) if result["raw"] else None
                    actual = ground_truth[adr["id"]]["overall"]

                    rep_results.append({
                        "adr_id": adr["id"],
                        "actual": actual,
                        "predicted": predicted,
                        "correct": predicted == actual if predicted else False,
                        "latency": result["latency"],
                        "input_tokens": result["usage"]["input_tokens"],
                        "output_tokens": result["usage"]["output_tokens"],
                        "error": result["error"],
                        "parse_success": predicted is not None,
                    })

                    status = "✓" if predicted == actual else "✗" if predicted else "?"
                    print(f"  [{call_num}/{total_calls} {pct:.0f}%] "
                          f"{model_name[:10]:>10} | {strategy[:5]} | r{rep+1} | "
                          f"{adr['id'][:20]:>20} | {status} {predicted or 'PARSE_FAIL'}")

                    time.sleep(RATE_LIMIT_DELAY)

                all_results[model_name][strategy].append(rep_results)

            # Save per-model-strategy results
            result_file = RESULTS_DIR / f"{model_name}_{strategy}.json"
            with open(result_file, "w") as fp:
                json.dump(all_results[model_name][strategy], fp, indent=2)
            print(f"\n  → Saved {result_file}")

    # Save everything
    with open(RESULTS_DIR / "all_results.json", "w") as fp:
        json.dump(all_results, fp, indent=2)

    return all_results


# ============================================================
# PHASE 4: ANALYZE AND GENERATE PAPER TABLES
# ============================================================

def analyze(all_results: Dict, ground_truth: Dict):
    """Compute metrics and generate paper-ready output."""
    from sklearn.metrics import (
        precision_recall_fscore_support, accuracy_score,
        cohen_kappa_score, confusion_matrix
    )
    from scipy.stats import chi2

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSES = ["Compliant", "Partially_Compliant", "Non_Compliant"]
    STRAT_SHORT = {"zero_shot": "ZS", "few_shot": "FS", "chain_of_thought": "CoT"}
    MODEL_SHORT = {"gpt-4o": "GPT-4o", "claude-3.5-sonnet": "Claude 3.5",
                   "mistral-7b": "Mistral 7B", "gemini-1.5-pro": "Gemini 1.5P",
                   "llama-3.1-70b": "LLaMA 3.1"}

    print(f"\n{'='*60}")
    print(f"PHASE 4: Analysis")
    print(f"{'='*60}")

    # ---- TABLE II ----
    print(f"\n{'='*60}")
    print("TABLE II: OVERALL COMPLIANCE DETECTION PERFORMANCE")
    print(f"{'='*60}")
    print(f"{'Model':>15} {'Prompt':>6} {'Prec':>12} {'Recall':>12} {'F1-mac':>12} {'F1-wt':>8} {'κ':>12}")
    print("-" * 80)

    metrics_summary = {}

    for model_name in all_results:
        metrics_summary[model_name] = {}
        for strategy in all_results[model_name]:
            reps = all_results[model_name][strategy]
            rep_f1s, rep_kappas, rep_precs, rep_recs = [], [], [], []

            for rep_data in reps:
                y_true = [r["actual"] for r in rep_data if r["predicted"]]
                y_pred = [r["predicted"] for r in rep_data if r["predicted"]]

                if len(y_true) < 10:
                    continue

                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
                _, _, f1w, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
                k = cohen_kappa_score(y_true, y_pred, labels=CLASSES)

                rep_f1s.append(f1)
                rep_kappas.append(k)
                rep_precs.append(p)
                rep_recs.append(r)

            if rep_f1s:
                ms = MODEL_SHORT.get(model_name, model_name)
                ss = STRAT_SHORT.get(strategy, strategy)
                print(f"{ms:>15} {ss:>6} "
                      f"{np.mean(rep_precs):.2f}±{np.std(rep_precs):.2f} "
                      f"{np.mean(rep_recs):.2f}±{np.std(rep_recs):.2f} "
                      f"{np.mean(rep_f1s):.2f}±{np.std(rep_f1s):.2f} "
                      f"{np.mean(rep_f1s):.2f}    "
                      f"{np.mean(rep_kappas):.2f}±{np.std(rep_kappas):.2f}")

                metrics_summary[model_name][strategy] = {
                    "f1_mean": round(float(np.mean(rep_f1s)), 3),
                    "f1_std": round(float(np.std(rep_f1s)), 3),
                    "kappa_mean": round(float(np.mean(rep_kappas)), 3),
                    "kappa_std": round(float(np.std(rep_kappas)), 3),
                    "prec_mean": round(float(np.mean(rep_precs)), 3),
                    "rec_mean": round(float(np.mean(rep_recs)), 3),
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

        if len(y_true) < 10:
            continue

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        ms = MODEL_SHORT.get(model_name, model_name)
        print(f"\n  {ms}:")
        print(f"  {'':>20} {'Comp':>8} {'Part':>8} {'Non-C':>8}")
        for i, cls in enumerate(CLASSES):
            print(f"  {'Actual '+cls[:12]:>20} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    # ---- McNEMAR ----
    print(f"\n{'='*60}")
    print("McNEMAR PAIRWISE TESTS (CoT)")
    print(f"{'='*60}")

    model_names = [m for m in all_results if "chain_of_thought" in all_results[m]]
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            d1 = all_results[m1]["chain_of_thought"][0]
            d2 = all_results[m2]["chain_of_thought"][0]

            # Align by adr_id
            d1_map = {r["adr_id"]: r for r in d1 if r["predicted"]}
            d2_map = {r["adr_id"]: r for r in d2 if r["predicted"]}
            common = set(d1_map.keys()) & set(d2_map.keys())

            b = sum(1 for k in common if d1_map[k]["correct"] and not d2_map[k]["correct"])
            c = sum(1 for k in common if not d1_map[k]["correct"] and d2_map[k]["correct"])

            if b + c > 0:
                chi2_stat = (abs(b - c) - 1)**2 / (b + c)
                p_val = 1 - chi2.cdf(chi2_stat, df=1)
            else:
                chi2_stat, p_val = 0, 1.0

            sig = "*" if p_val < 0.005 else ""
            print(f"  {MODEL_SHORT.get(m1,m1):>12} vs {MODEL_SHORT.get(m2,m2):<12} "
                  f"b={b} c={c} χ²={chi2_stat:.3f} p={p_val:.4f} {sig}")

    # ---- COST ANALYSIS ----
    print(f"\n{'='*60}")
    print("TABLE VI: COST-PERFORMANCE")
    print(f"{'='*60}")

    pricing = {
        "gpt-4o": (2.50, 10.00),
        "claude-3.5-sonnet": (3.00, 15.00),
        "mistral-7b": (0.25, 0.25),  # Mistral API pricing
    }

    for model_name in all_results:
        if "chain_of_thought" not in all_results[model_name]:
            continue
        rep_data = all_results[model_name]["chain_of_thought"][0]
        avg_in = np.mean([r["input_tokens"] for r in rep_data])
        avg_out = np.mean([r["output_tokens"] for r in rep_data])
        avg_lat = np.mean([r["latency"] for r in rep_data])
        p_in, p_out = pricing.get(model_name, (0, 0))
        cost = (avg_in * p_in + avg_out * p_out) / 1e6
        f1 = metrics_summary.get(model_name, {}).get("chain_of_thought", {}).get("f1_mean", 0)
        ms = MODEL_SHORT.get(model_name, model_name)
        print(f"  {ms:>12} F1={f1:.2f} Cost=${cost:.4f}/ADR "
              f"Lat={avg_lat:.1f}s In={avg_in:.0f} Out={avg_out:.0f}")

    # Save all metrics
    with open(ANALYSIS_DIR / "metrics_summary.json", "w") as fp:
        json.dump(metrics_summary, fp, indent=2)

    print(f"\n✓ Analysis complete. Results in {ANALYSIS_DIR}")
    return metrics_summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight ADR Compliance Experiment")
    parser.add_argument("--phase", default="all",
                        choices=["fetch", "annotate", "run", "analyze", "all"])
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-reps", type=int, default=3)
    args = parser.parse_args()

    n_eval = args.n_eval
    n_reps = args.n_reps

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"ADR COMPLIANCE OVERNIGHT EXPERIMENT")
    print(f"Started: {start_time.isoformat()}")
    print(f"Config: {n_eval} ADRs x {len(MODELS)} models x {len(STRATEGIES)} strategies x {n_reps} reps")
    print(f"Total calls: {n_eval * len(MODELS) * len(STRATEGIES) * n_reps}")
    print(f"{'='*60}")

    # Phase 1: Fetch ADRs
    if args.phase in ("fetch", "all"):
        adrs = fetch_adrs_from_github(target_count=n_eval + 10)
    else:
        adrs = load_adrs()

    if not adrs:
        print("ERROR: No ADRs found. Run --phase fetch first.")
        sys.exit(1)

    # Phase 2: Annotate
    gt_path = EXPERIMENT_DIR / "ground_truth.json"
    if args.phase in ("annotate", "all"):
        ground_truth = annotate_with_oracle(adrs[:n_eval + 10])
    elif gt_path.exists():
        with open(gt_path) as fp:
            ground_truth = json.load(fp)
    else:
        print("ERROR: No ground truth. Run --phase annotate first.")
        sys.exit(1)

    # Phase 3: Run experiments
    results_path = RESULTS_DIR / "all_results.json"
    if args.phase in ("run", "all"):
        all_results = run_experiments(adrs, ground_truth)
    elif results_path.exists():
        with open(results_path) as fp:
            all_results = json.load(fp)
    else:
        print("ERROR: No results. Run --phase run first.")
        sys.exit(1)

    # Phase 4: Analyze
    if args.phase in ("analyze", "all"):
        analyze(all_results, ground_truth)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE. Total time: {elapsed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
