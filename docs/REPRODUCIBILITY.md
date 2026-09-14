# Reproducibility Guide

This guide separates verification of the published evidence from a new paid
provider run. A reviewer can complete the offline checks without API keys.

## Command Safety

| Activity | Changes files | Uses provider APIs | Typical cost |
| --- | --- | --- | --- |
| Release verification | No | No | None |
| Unit and offline integration tests | Disposable test files only | No | None |
| Analysis regeneration | Yes, under the selected output directory | No | None |
| Batch dry run | Yes, request files only | No | None |
| Smoke test | Yes | Yes | 12 requests by default |
| Five-ADR validation | Yes, under a new run ID | Yes | 180 evaluations plus any retries |
| Full replication | Yes, under a new run ID | Yes | 5,832 evaluations plus any retries |

A bare `python adr_benchmark.py` command prints help. Paid work requires an
explicit phase or target selection.

## 1. Install the Reference Environment

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The package versions captured during preparation are recorded in
`environment/reference_environment.json`. Provider SDK versions affect a new
API run, but do not change the retained responses.

## 2. Verify the Package Without APIs

```powershell
python -B scripts\verify_release.py
python -B -m unittest discover -s tests -v
python -B scripts\verify_exact_scoring_revision.py --run-id v31_fulltext_gpt_medium_exact_scoring
```

The equivalent POSIX commands use forward slashes:

```bash
python -B scripts/verify_release.py
python -B -m unittest discover -s tests -v
python -B scripts/verify_exact_scoring_revision.py --run-id v31_fulltext_gpt_medium_exact_scoring
```

The exact-scoring verifier rewrites only its verification report so that the
recorded code hashes reflect the checked checkout. Use `verify_release.py` alone
when a strictly read-only audit is required.

Expected core facts:

| Check | Expected value |
| --- | ---: |
| Evaluation ADRs | 162 |
| Source repositories | 14 |
| Model and prompt configurations | 12 |
| Repetitions per configuration | 3 |
| Rows per repetition | 162 |
| Total retained evaluations | 5,832 |
| Truncated retained inputs | 0 |
| Explicit historical text-decoding exceptions | 2 |
| Exact-scoring prediction changes | 40 |
| Second human assessments | 41 |
| Recorded overall-label matches | 37 |
| Criterion-derived overall-class matches | 27 |
| Internally inconsistent second overall labels | 11 |

## 3. Recompute Analysis From Saved Responses

The commands below make no model calls. Run them in a disposable clone if the
checked-in analysis files must remain byte-for-byte unchanged.

```powershell
python -B adr_benchmark.py --phase merge --run-id v31_fulltext_gpt_medium_exact_scoring --gpt-reasoning-effort medium
python -B scripts\analyze_rubric_sensitivity.py --run-id v31_fulltext_gpt_medium_exact_scoring
python -B scripts\analyze_rule_hybrid_comparison.py --run-id v31_fulltext_gpt_medium_exact_scoring
python -B scripts\prepare_manuscript_v31_evidence.py --run-id v31_fulltext_gpt_medium_exact_scoring
```

Regenerate figures and manuscript tables into a separate directory:

```powershell
python -B scripts\generate_publication_assets.py --run-id v31_fulltext_gpt_medium_exact_scoring --output-dir reproduced_assets
```

Compare `reproduced_assets/` with the checked-in
`analysis/manuscript_assets/` directory. PNG files may differ at the binary
level across Matplotlib or font-library versions, so compare plotted values and
the generated CSV files as the primary check.

## 4. Run a Live API Smoke Test

This step is optional, uses paid APIs, and does not validate the historical
availability of a provider model. It checks current connectivity and response
format only.

1. Copy `.env.example` to `.env`.
2. Enter the four provider keys locally.
3. Choose a new disposable run ID.
4. Run the smoke command.

```powershell
python adr_benchmark.py --phase smoke --run-id reviewer_smoke_YYYYMMDD --gpt-reasoning-effort medium
```

The default smoke run covers all 12 model and prompt combinations. To isolate a
provider or reduce cost, use `--run` with comma-separated targets. For example:

```powershell
python adr_benchmark.py --phase smoke --run gpt-5.5/zero_shot,gpt-5.5/few_shot,gpt-5.5/chain_of_thought --run-id reviewer_smoke_gpt_YYYYMMDD --gpt-reasoning-effort medium
```

Smoke outputs are diagnostic only. They must not be merged into the canonical
analysis.

## 5. Validate the Full Lifecycle With Five ADRs

This paid functional check exercises all models, all prompt conditions, and all
three repetitions on a deterministic five-ADR subset. It does not reproduce the
paper's statistics and must use a new run ID. Every command must retain the same
`--validation-size`, `--validation-seed`, Mistral variant, and GPT reasoning
setting.

PowerShell:

```powershell
$RUN_ID = "reviewer_validation_5_YYYYMMDD"
$BATCH_TARGETS = "gpt-5.5/zero_shot,gpt-5.5/few_shot,gpt-5.5/chain_of_thought,claude-sonnet-4-6/zero_shot,claude-sonnet-4-6/few_shot,claude-sonnet-4-6/chain_of_thought,gemini-2.5-pro/zero_shot,gemini-2.5-pro/few_shot,gemini-2.5-pro/chain_of_thought"
$MISTRAL_TARGETS = "ministral-3-8b/zero_shot,ministral-3-8b/few_shot,ministral-3-8b/chain_of_thought"

python adr_benchmark.py --phase preflight --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase smoke --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_create --run $BATCH_TARGETS --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --run $MISTRAL_TARGETS --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_poll --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_collect --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python scripts\verify_validation_run.py --run-id $RUN_ID --expected-adrs 5 --expected-reps 3
python adr_benchmark.py --phase merge --run-id $RUN_ID --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
```

For a no-cost inspection of all batch requests, add `--batch-dry-run` to the
`batch_create` command. It creates nine JSONL files with 15 requests each and
does not submit them.

On Linux or macOS, define the same target strings and use forward slashes:

```bash
RUN_ID="reviewer_validation_5_YYYYMMDD"
BATCH_TARGETS="gpt-5.5/zero_shot,gpt-5.5/few_shot,gpt-5.5/chain_of_thought,claude-sonnet-4-6/zero_shot,claude-sonnet-4-6/few_shot,claude-sonnet-4-6/chain_of_thought,gemini-2.5-pro/zero_shot,gemini-2.5-pro/few_shot,gemini-2.5-pro/chain_of_thought"
MISTRAL_TARGETS="ministral-3-8b/zero_shot,ministral-3-8b/few_shot,ministral-3-8b/chain_of_thought"

python adr_benchmark.py --phase preflight --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase smoke --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_create --run "$BATCH_TARGETS" --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --run "$MISTRAL_TARGETS" --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_poll --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_collect --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
python scripts/verify_validation_run.py --run-id "$RUN_ID" --expected-adrs 5 --expected-reps 3
python adr_benchmark.py --phase merge --run-id "$RUN_ID" --validation-size 5 --validation-seed 3105 --gpt-reasoning-effort medium
```

Repeat polling until all hosted batch jobs complete. If collection reports a
small number of failed rows, run the affected `model/strategy` targets
synchronously with the same validation flags. The runner repairs only invalid
rows. Run the validator again before merge.

## 6. Perform a Complete New Provider Run

A direct replication must use a new run ID, the fixed evaluation manifest,
complete ADR text, hosted Ministral 3 8B, three repetitions, and GPT medium
reasoning effort. Do not use `--resample`.

```powershell
$RUN_ID = "reviewer_fulltext_medium_YYYYMMDD"

python adr_benchmark.py --phase preflight --run-id $RUN_ID --gpt-reasoning-effort medium
python adr_benchmark.py --phase smoke --run-id $RUN_ID --gpt-reasoning-effort medium

python adr_benchmark.py --phase batch_create --run gpt-5.5/zero_shot,gpt-5.5/few_shot,gpt-5.5/chain_of_thought,claude-sonnet-4-6/zero_shot,claude-sonnet-4-6/few_shot,claude-sonnet-4-6/chain_of_thought,gemini-2.5-pro/zero_shot,gemini-2.5-pro/few_shot,gemini-2.5-pro/chain_of_thought --run-id $RUN_ID --gpt-reasoning-effort medium

python adr_benchmark.py --run ministral-3-8b/zero_shot,ministral-3-8b/few_shot,ministral-3-8b/chain_of_thought --run-id $RUN_ID --gpt-reasoning-effort medium

python adr_benchmark.py --phase batch_poll --run-id $RUN_ID --gpt-reasoning-effort medium
python adr_benchmark.py --phase batch_collect --run-id $RUN_ID --gpt-reasoning-effort medium
python adr_benchmark.py --phase merge --run-id $RUN_ID --gpt-reasoning-effort medium
```

Linux and macOS use the same phase order. Replace backslashes in script paths
with forward slashes and define `RUN_ID` with `RUN_ID="..."`.

Repeat `batch_poll` until every submitted job reports a terminal successful
state. If collection identifies individual errored or incomplete rows, run only
the affected targets with the same run ID. The synchronous runner repairs those
rows and leaves valid rows unchanged. Then run `merge` again.

The merge phase intentionally refuses an incomplete or unbalanced experiment.

## 7. Interpret Replication Differences

Provider-hosted models can change availability or implementation behind a
stable product name. A new run is therefore a temporal replication, not a
bitwise reconstruction of the 2026-09-08 provider responses. Compare the new
run at the configuration, class, dimension, and criterion levels. Preserve the
new run ID, timestamps, model metadata, prompt hashes, token accounting, repair
records, and provider access conditions.

The frozen protocol also preserves the two text-decoding exceptions listed in
`release/input_text_exceptions.json`. This is deliberate: it makes the exact
reported inputs reproducible across operating systems. A study that instead
uses the corrected Unicode punctuation is a modified-input replication and
must use a new protocol identifier.

## 8. Evidence Map

| Claim type | Primary artifact |
| --- | --- |
| Frozen sample and repositories | `results/runs/v31_fulltext_gpt_medium/analysis/manuscript_dataset_summary.json` |
| Prompt text and exemplars | `results/runs/v31_fulltext_gpt_medium/analysis/prompt_manifest.json` |
| Input length and hash per call | Canonical `raw_results/*.json` |
| Model and API settings | Canonical `analysis/model_reproducibility.json` |
| Overall and per-class metrics | Canonical `analysis/performance_details.json` |
| Criterion and dimension metrics | Canonical `analysis/criterion_level_performance.json` |
| Pairwise inference | Canonical `analysis/pairwise_statistical_tests.json` |
| Sensitivity analysis | Canonical `analysis/rubric_sensitivity_performance.json` |
| Rule and hybrid baselines | Canonical `analysis/rule_hybrid_comparison.json` |
| Human agreement | Canonical `analysis/manuscript_evidence/evidence.json` |
| Cost assumptions and tokens | Canonical `analysis/cost_analysis.json` |
| Exact-scoring correction | Canonical `analysis/exact_rescoring_report.json` |

Here, "canonical" means
`results/runs/v31_fulltext_gpt_medium_exact_scoring/`.
