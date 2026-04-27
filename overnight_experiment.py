#!/usr/bin/env python3
"""
OVERNIGHT ADR COMPLIANCE EXPERIMENT
====================================
4 models × 3 strategies × 3 reps × 200 ADRs = 7,200 calls (~$101, ~10-12 hrs)

Each model/strategy pair saves its own result file independently so runs can be
split across sessions. Use --phase merge to combine and analyze when ready.

Usage:
  export OPENAI_API_KEY="sk-..."   ANTHROPIC_API_KEY="sk-ant-..."
  export MISTRAL_API_KEY="..."     GEMINI_API_KEY="..."

  # Full pipeline
  python overnight_experiment.py --n-eval 200 2>&1 | tee experiment_log.txt

  # Step by step
  python overnight_experiment.py --phase fetch    --n-eval 200
  python overnight_experiment.py --phase annotate
  python overnight_experiment.py --phase run      --n-eval 200   # all models × strategies
  python overnight_experiment.py --phase merge                   # combine files + analyze

  # Targeted run — one or several model/strategy pairs
  python overnight_experiment.py --run gemini-2.5-pro/zero_shot --n-eval 200
  python overnight_experiment.py --run gemini-2.5-pro/zero_shot,gpt-5.1/chain_of_thought
"""

import os
import sys
import json
import time
import re
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

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
    "gpt-5.1": {
        "provider": "openai",
        "model": "gpt-5.1",
        "use_max_completion_tokens": True,
        "no_temperature": True,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "no_temperature": True,
    },
    "mistral-7b": {
        "provider": "openai",  # via Mistral's OpenAI-compatible API or local vLLM
        "model": "mistral-small-latest",  # or "mistralai/Mistral-7B-Instruct-v0.3" for vLLM
        "base_url": "https://api.mistral.ai/v1",  # change to localhost for vLLM
        "api_key_env": "MISTRAL_API_KEY",
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "model": "gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
        "max_output_tokens": 8192,  # thinking model needs headroom for reasoning tokens
    },
}

STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]

with open("adr_dataset.json") as f:
    ADR_DATASET = json.load(f)


# ============================================================
# PHASE 1: FETCH REAL ADRS FROM GITHUB
# ============================================================
from collections import defaultdict

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

    if target_count is None:
        target_count = len(ADR_DATASET)

    ADRS_DIR.mkdir(parents=True, exist_ok=True)

    fetched = []
    repo_counts = defaultdict(int)

    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    print("\n================ FETCH ADRs ================")
    print(f"  Source: adr_dataset.json ({len(ADR_DATASET)} entries)\n")

    for entry in ADR_DATASET:
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
                print(f"    ERROR run {run+1} failed: {e}")
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

JSON_FORMAT = """Respond ONLY with valid JSON.
DO NOT use markdown or code blocks.
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


def call_llm(model_name: str, prompt: str, _retries: int = 3):
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
                create_kwargs["max_completion_tokens"] = 1024
            else:
                create_kwargs["max_tokens"] = 1024
            response = client.chat.completions.create(**create_kwargs)
            raw = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        elif cfg["provider"] == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            create_kwargs = {
                "model": cfg["model"],
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
            }
            if not cfg.get("no_temperature"):
                create_kwargs["temperature"] = 0
            response = client.messages.create(**create_kwargs)
            raw = response.content[0].text
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
            raw = response.text
            if raw is None:
                finish_reason = "unknown"
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)
                raise ValueError(f"empty response from Gemini (finish_reason={finish_reason})")
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
        }

    except Exception as e:
        err = str(e)
        print(f"\nAPI ERROR [{model_name}]:", err)
        quota_exhausted = "insufficient_quota" in err or "quota" in err.lower()

        is_transient_429 = ("429" in err or "rate_limit" in err.lower()) and not quota_exhausted
        if is_transient_429 and _retries > 0:
            wait = 30 if _retries == 1 else 10
            print(f"  Rate limited — waiting {wait}s then retrying ({_retries} left)...")
            time.sleep(wait)
            return call_llm(model_name, prompt, _retries=_retries - 1)

        return {
            "raw": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency": round(time.time() - start, 2),
            "error": err,
            "quota_exhausted": quota_exhausted,
        }


def extract_classification(raw):
    if not raw:
        return None

    try:
        raw = raw.strip()

        # Remove markdown fences like ```json ... ```
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*", "", raw)
            raw = re.sub(r"```$", "", raw)
            raw = raw.strip()

        # Extract JSON (greedy match)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            json_str = match.group(0)

            parsed = json.loads(json_str)

            val = parsed.get("overall", "").lower()

            if "partially" in val:
                return "Partially_Compliant"
            elif "non" in val:
                return "Non_Compliant"
            elif "compliant" in val:
                return "Compliant"

    except Exception as e:
        pass

    raw_lower = raw.lower()

    if "partially_compliant" in raw_lower:
        return "Partially_Compliant"
    elif "non_compliant" in raw_lower:
        return "Non_Compliant"
    elif "compliant" in raw_lower:
        return "Compliant"

    return None


def select_eval_adrs(adrs: List[Dict], ground_truth: Dict,
                     n: int = 100, per_repo_cap: int = 15) -> List[Dict]:
    """
    Filter + rank + select the best n ADRs for the experiment.

    Ranking: sc_score + dq_met (oracle scores from annotation).
    Diversity: at most per_repo_cap ADRs from any single source_repo.
    Persists the selection to eval_set.json so all --run calls use the same set.
    """
    select_path = EXPERIMENT_DIR / "eval_set.json"

    if select_path.exists():
        with open(select_path) as fp:
            saved = json.load(fp)
        saved_ids = {e["id"] for e in saved["adrs"]}
        selected = [a for a in adrs if a["id"] in saved_ids]
        print(f"\n  Reusing eval_set.json: {len(selected)} ADRs. Delete to reselect.")
        return selected

    annotated = [a for a in adrs if a["id"] in ground_truth]

    def rank_score(adr):
        gt = ground_truth[adr["id"]]
        return gt.get("sc_score", 0) + gt.get("dq_met", 0)

    annotated.sort(key=rank_score, reverse=True)

    repo_counts = defaultdict(int)
    selected = []
    selected_ids = set()
    for adr in annotated:
        repo = adr["source_repo"]
        if repo_counts[repo] >= per_repo_cap:
            continue
        selected.append(adr)
        selected_ids.add(adr["id"])
        repo_counts[repo] += 1
        if len(selected) == n:
            break

    # Fallback: fill remaining slots if diversity cap left us short
    if len(selected) < n:
        for adr in annotated:
            if adr["id"] not in selected_ids:
                selected.append(adr)
                selected_ids.add(adr["id"])
            if len(selected) == n:
                break

    print(f"\n  Selected {len(selected)} ADRs by quality rank (top sc+dq score)")
    print(f"  Repo distribution: { {k: v for k, v in repo_counts.items()} }")

    payload = {
        "selected_at": datetime.now().isoformat(),
        "n": len(selected),
        "per_repo_cap": per_repo_cap,
        "adrs": [
            {
                "id": a["id"],
                "source_repo": a["source_repo"],
                "rank_score": rank_score(a),
                "ground_truth": ground_truth[a["id"]]["overall"],
            }
            for a in selected
        ],
    }
    with open(select_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"  Saved to {select_path}")

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
    if result_file.exists():
        with open(result_file) as fp:
            rep_results_list = json.load(fp)
        completed_reps = len(rep_results_list)
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
            result = call_llm(model_name, prompt)

            if result.get("quota_exhausted"):
                with lock:
                    exhausted_models.add(model_name)
                print(f"\n  QUOTA EXHAUSTED for {model_name} — stopping")
                quota_hit = True
                break

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

            status = "OK" if predicted == actual else "FAIL" if predicted else "?"
            pct = adr_idx / n_adrs * 100
            print(f"  {model_name[:12]:>12} | {strategy[:5]} | r{rep+1} | "
                  f"[{adr_idx}/{n_adrs}] {pct:4.0f}% | "
                  f"{adr['id'][:20]:>20} | {status} {predicted or 'PARSE_FAIL'}")

            time.sleep(RATE_LIMIT_DELAY)

        if quota_hit:
            break

        if rep_results:
            rep_results_list.append(rep_results)
            with open(result_file, "w") as fp:
                json.dump(rep_results_list, fp, indent=2)
            print(f"  Saved rep {rep + 1} -> {result_file}")


def run_experiments(adrs: List[Dict], ground_truth: Dict,
                    n_eval: int = N_EVAL, n_reps: int = N_REPS,
                    targets: List[tuple] = None, workers: int = 3):
    """
    Run experiments for the given targets in parallel (one thread per model/strategy pair).

    workers: number of pairs to run simultaneously (default 3).
    Each pair saves its own result file after every rep — safe to kill and resume.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    work = targets if targets is not None else [
        (m, s) for m in MODELS for s in STRATEGIES
    ]

    # Pre-flight: drop any model whose API key is missing
    def _api_key_present(model_name: str) -> bool:
        cfg = MODELS[model_name]
        if cfg["provider"] == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if cfg["provider"] == "gemini":
            return bool(os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY")))
        # openai-compatible
        env_var = cfg.get("api_key_env", "OPENAI_API_KEY")
        return bool(os.environ.get(env_var))

    skipped_models = set()
    valid_work = []
    for model_name, strategy in work:
        if _api_key_present(model_name):
            valid_work.append((model_name, strategy))
        else:
            if model_name not in skipped_models:
                cfg = MODELS[model_name]
                env_var = cfg.get("api_key_env", "ANTHROPIC_API_KEY" if cfg["provider"] == "anthropic" else "OPENAI_API_KEY")
                print(f"  SKIP  {model_name} — {env_var} not set")
                skipped_models.add(model_name)
    work = valid_work

    if not work:
        print("\nERROR: No models have API keys set. Nothing to run.")
        return

    fs_examples = []
    for adr in adrs[:10]:
        if adr["id"] in ground_truth:
            fs_examples.append({
                "text": adr["text"],
                "label": ground_truth[adr["id"]]["overall"],
            })

    eval_adrs = select_eval_adrs(adrs, ground_truth, n=n_eval)
    total_calls = len(work) * n_reps * len(eval_adrs)

    print(f"\n{'='*60}")
    print(f"PHASE 3: Running experiments")
    print(f"  Targets : {[f'{m}/{s}' for m, s in work]}")
    print(f"  Workers : {workers}  |  Reps: {n_reps}  |  ADRs: {len(eval_adrs)}")
    print(f"  Total calls: {total_calls}  |  Est. time: {total_calls * 2.5 / 60 / workers:.0f} min")
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

    for model_name in MODELS:
        for strategy in STRATEGIES:
            result_file = RESULTS_DIR / f"{model_name}_{strategy}.json"
            if not result_file.exists():
                print(f"  MISSING  {model_name}/{strategy}")
                continue
            with open(result_file) as fp:
                data = json.load(fp)
            all_results.setdefault(model_name, {})[strategy] = data
            n_reps = len(data)
            n_adrs = len(data[0]) if data else 0
            print(f"  OK  {model_name}/{strategy}  ({n_reps} reps x {n_adrs} ADRs)")
            found += 1

    if not all_results:
        print("ERROR: No result files found. Run experiments first.")
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
    from scipy.stats import chi2

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSES = ["Compliant", "Partially_Compliant", "Non_Compliant"]
    STRAT_SHORT = {"zero_shot": "ZS", "few_shot": "FS", "chain_of_thought": "CoT"}
    MODEL_SHORT = {"gpt-5.1": "GPT-5.5", "claude-sonnet-4-6": "Claude Sonnet 4.6",
                   "mistral-7b": "Mistral 7B", "gemini-2.5-pro": "Gemini 2.5P",
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
        "gpt-5.1": (5.00, 30.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "mistral-7b": (0.25, 0.25),
        "gemini-2.5-pro": (1.25, 10.00),  # standard tier (<200K ctx)
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

    print(f"\nAnalysis complete. Results in {ANALYSIS_DIR}")
    return metrics_summary


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
    gt_path = EXPERIMENT_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("ERROR: No ground truth found. Run --phase annotate first.")
        sys.exit(1)
    with open(gt_path) as fp:
        return json.load(fp)


def main():
    parser = argparse.ArgumentParser(description="Overnight ADR Compliance Experiment")
    parser.add_argument("--phase", default=None,
                        choices=["fetch", "annotate", "run", "merge", "all"],
                        help="Pipeline phase to execute (default: all)")
    parser.add_argument("--run", metavar="MODEL/STRATEGY[,...]",
                        help="Run specific model/strategy pairs (comma-separated); "
                             "requires fetch + annotate to be done already")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of ADRs to sample for evaluation (default: 200)")
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel model/strategy pairs to run simultaneously (default: 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for ADR sampling (random if not set)")
    parser.add_argument("--resample", action="store_true",
                        help="Draw a fresh random ADR sample even if eval_sample.json exists")
    args = parser.parse_args()

    # Default to 'all' when neither --phase nor --run is given
    if args.phase is None and args.run is None:
        args.phase = "all"

    n_eval = args.n_eval
    n_reps = args.n_reps

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()

    # --run: targeted experiment run — skip fetch/annotate, load from disk
    if args.run:
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
                        targets=targets, workers=args.workers)
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

    if args.phase in ("annotate", "all", "run"):
        if not adrs:
            print("ERROR: No ADRs found. Run --phase fetch first.")
            sys.exit(1)

    if args.phase in ("annotate", "all"):
        ground_truth = annotate_with_oracle(adrs)
    elif args.phase in ("run",):
        ground_truth = _load_ground_truth()

    if args.phase in ("run", "all"):
        run_experiments(adrs, ground_truth, n_eval=n_eval, n_reps=n_reps,
                        workers=args.workers)

    if args.phase in ("merge", "all"):
        ground_truth = _load_ground_truth()
        all_results = merge_results()
        analyze(all_results, ground_truth)

    elif args.phase == "analyze":
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
