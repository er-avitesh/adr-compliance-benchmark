import urllib.request
import json
import time
import os
from collections import defaultdict

# =========================================================
# FINAL CLEAN ADR SOURCES (CURATED)
# =========================================================

ADR_REPOS = [

    # High quality ADRs
    ("alphagov", "govuk-aws", "docs/architecture/decisions"),
    ("alphagov", "content-publisher", "docs/adr"),
    ("backstage", "backstage", "docs/architecture-decisions"),
    ("apache", "airflow", "dev/breeze/doc/adr"),

    # Lightweight ADRs
    ("aws", "aws-cdk", "packages/aws-cdk-lib/core/adr"),
    ("aws", "aws-cdk", "packages/aws-cdk-lib/aws-dynamodb/adr"),
    ("aws", "aws-cdk", "packages/aws-cdk-lib/aws-rds/adr"),

    # ADR-like (for diversity)
    ("argoproj", "argo-cd", "docs/proposals"),
    ("cortexproject", "cortex", "docs/proposals"),
    ("grafana", "grafana", "contribute/architecture"),
    ("ory", "hydra", "internal/httpclient/docs"),

    # Optional fallback
    ("npryce", "adr-tools", "doc/adr"),
    ("thomvaill", "log4brains", "docs/adr"),
]

# =========================================================
# CONFIG
# =========================================================

_token = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if _token:
    HEADERS["Authorization"] = f"token {_token}"

EXCLUDE_PATHS = [
    ".github",
    "vendor",
    "changelog",
    "release",
    "test",
    "examples",
]

KNOWN_ADR_PATTERNS = [
    "architecture/decisions",
    "architecture-decisions",
    "/adr",
    "proposals",
]

MAX_PER_REPO = 25

LOW_VALUE_REPOS = ["madr", "adr-tools", "log4brains"]

# =========================================================
# GitHub API helpers
# =========================================================

def github_api(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"GitHub API error: {e}")

        if "403" in str(e):
            print("Likely rate limited. Waiting 60 seconds...")
            time.sleep(60)

        return None


def get_default_branch(owner, repo):
    data = github_api(f"https://api.github.com/repos/{owner}/{repo}")
    if data:
        return data.get("default_branch", "main")
    return "main"


# =========================================================
# ADR detection logic
# =========================================================

def fetch_preview(owner, repo, path, branch):
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    for _ in range(2):
        try:
            req = urllib.request.Request(raw_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.read(2000).decode("utf-8", errors="ignore")
        except:
            time.sleep(0.5)
    return ""


def is_adr_file(preview):
    t = preview.lower()

    score = 0

    if "decision" in t:
        score += 1

    if any(k in t for k in ["context", "problem", "background"]):
        score += 1

    if any(k in t for k in ["consequence", "impact", "trade-off"]):
        score += 1

    if any(k in t for k in ["alternative", "option"]):
        score += 1

    return score >= 2


# =========================================================
# ADR PATH FETCH
# =========================================================

def find_adr_files(owner, repo, base_path):
    branch = get_default_branch(owner, repo)

    def list_dir(path):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        return github_api(url)

    adr_files = []

    data = list_dir(base_path)
    if not isinstance(data, list):
        print(f"Failed to fetch directory: {owner}/{repo}/{base_path}")
        return []

    def accept_file(raw_path, file_path_lower):
        if not file_path_lower.endswith(".md"):
            return None
        if any(ex in file_path_lower for ex in EXCLUDE_PATHS):
            return None
        is_known = any(p in file_path_lower for p in KNOWN_ADR_PATTERNS)
        preview = fetch_preview(owner, repo, raw_path, branch)
        if not preview.strip():
            print(f"  Preview failed: {raw_path}")
            if is_known:
                return (owner, repo, file_path_lower, 3)
            return None
        score = adr_quality_score(preview)
        quality = classify_quality(score)
        if quality == "LOW":
            return None
        return (owner, repo, file_path_lower, score)

    # Level 1 files
    for item in data:
        if item.get("type") == "file":
            result = accept_file(item["path"], item.get("path", "").lower())
            if result:
                adr_files.append(result)

    # Level 2 (subfolders)
    for item in data:
        if item.get("type") == "dir":
            sub_data = list_dir(item["path"])
            if not isinstance(sub_data, list):
                continue

            for sub in sub_data:
                if sub.get("type") != "file":
                    continue
                result = accept_file(sub["path"], sub.get("path", "").lower())
                if result:
                    adr_files.append(result)

    return adr_files


# =========================================================
# MAIN
# =========================================================

FALLBACK_REPOS = [
    ("adr", "madr", "docs/decisions"),
    ("deshpandetanmay", "lightweight-architecture-decision-records", "doc/adr"),
]

OPTION2_REPOS = [
    ("argoproj", "argo-cd", "docs/operator-manual"),
    ("grafana", "grafana", "docs/sources"),
    ("openfga", "openfga", "docs/architecture"),
    ("temporalio", "temporal", "docs/architecture"),
]

def adr_quality_score(text: str) -> int:
    t = text.lower()
    score = 0

    if "decision" in t:
        score += 2

    if any(k in t for k in ["context", "problem", "background", "motivation"]):
        score += 2

    if any(k in t for k in ["consequence", "impact", "trade-off", "implication"]):
        score += 2

    if any(k in t for k in ["alternative", "option", "considered"]):
        score += 1

    if len(t.split()) > 120:
        score += 1

    return score


def classify_quality(score: int) -> str:
    if score >= 5:
        return "HIGH"
    elif score >= 3:
        return "MEDIUM"
    else:
        return "LOW"
    
FINAL_LIMIT = 200
PER_REPO_CAP = 20


def collect_from(repo_list, all_adrs):
    for owner, repo, path in repo_list:
        print(f"{owner}/{repo} | {path}")
        files = find_adr_files(owner, repo, path)
        all_adrs.extend(files)
        print(f"  Found: {len(files)}  |  running total: {len(all_adrs)}\n")
        time.sleep(0.5)


def main():
    print("\nClean ADR fetch run\n")

    all_adrs = []

    collect_from(ADR_REPOS, all_adrs)

    if len(all_adrs) < FINAL_LIMIT:
        print(f"Total {len(all_adrs)} < {FINAL_LIMIT} -- pulling fallback repos...\n")
        collect_from(FALLBACK_REPOS, all_adrs)

    if len(all_adrs) < FINAL_LIMIT:
        print(f"Still {len(all_adrs)} < {FINAL_LIMIT} -- pulling extended ADR-like sources...\n")
        collect_from(OPTION2_REPOS, all_adrs)

    # Deduplicate — keep highest score per (owner, repo, path)
    unique = {}
    for owner, repo, path, score in all_adrs:
        key = (owner, repo, path)
        if key not in unique or score > unique[key]:
            unique[key] = score
    all_adrs = [(o, r, p, s) for (o, r, p), s in unique.items()]

    # Sort globally by score descending
    all_adrs.sort(key=lambda x: x[3], reverse=True)

    # Diversity cap — per actual repo
    repo_counts = defaultdict(int)
    final_adrs = []
    for owner, repo, path, score in all_adrs:
        repo_key = f"{owner}/{repo}"
        if repo_counts[repo_key] >= PER_REPO_CAP:
            continue
        final_adrs.append((owner, repo, path, score))
        repo_counts[repo_key] += 1
        #if len(final_adrs) == FINAL_LIMIT:
        #    break

    print(f"\nFinal ADR count: {len(final_adrs)}\n")
    for owner, repo, path, score in final_adrs:
        print(f"[{score}] {owner}/{repo} :: {path}")

    export = [
        {"owner": o, "repo": r, "path": p, "score": s}
        for o, r, p, s in final_adrs
    ]
    with open("adr_dataset.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nSaved {len(export)} ADRs to adr_dataset.json")


if __name__ == "__main__":
    main()
