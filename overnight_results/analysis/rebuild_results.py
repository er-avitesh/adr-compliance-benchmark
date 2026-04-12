import json
from pathlib import Path

RESULTS_DIR = Path("overnight_results/raw_results")

all_results = {}

def normalize_runs(data):
    """
    Normalize to: list of reps
    """
    # If already list of lists → return as-is
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return data

    # If flat list → wrap as single rep
    if isinstance(data, list):
        return [data]

    return []

def filter_valid(rep):
    """Keep only valid predictions"""
    return [r for r in rep if isinstance(r, dict) and r.get("predicted") is not None]

for file in RESULTS_DIR.glob("*.json"):
    name = file.stem

    # ❌ skip aggregate file
    if name == "all_results":
        continue

    # ❌ skip broken Claude 3.5 runs
    if "claude-3.5" in name:
        continue

    try:
        model, strategy = name.split("_", 1)
    except:
        print(f"Skipping malformed file: {name}")
        continue

    with open(file) as f:
        data = json.load(f)

    reps = normalize_runs(data)

    cleaned_reps = []
    for rep in reps:
        cleaned = filter_valid(rep)

        # keep only meaningful reps
        if len(cleaned) >= 10:
            cleaned_reps.append(cleaned)

    if not cleaned_reps:
        print(f"Skipping {name} (no valid data)")
        continue

    if model not in all_results:
        all_results[model] = {}

    all_results[model][strategy] = cleaned_reps

# save
output_path = RESULTS_DIR / "all_results.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Clean all_results.json generated at: {output_path}")