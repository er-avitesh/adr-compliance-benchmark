# Protocol Notes

This page records details needed to interpret and reproduce the retained
experiment. These notes do not change the reported protocol or results.

## Frozen Evaluation Manifest

`results/eval_set.json` is the frozen manifest used for all 5,832 provider
evaluations. Its SHA-256 is recorded in each run configuration. The sample size
of 162 is the fixed benchmark size. It is not presented as establishing a
population-level margin of error below five percent.

Historical corpus-acquisition summaries had different counts because they were
produced at different collection and filtering stages. They are not inputs to
the reported evaluation and are excluded from the reviewer release. The release
contains the 162 evaluated ADRs and four source-declared holdout ADR files.

## Few-Shot Holdout Record

The frozen evaluation manifest names four real ADRs as held out. The actual
few-shot prompt used three of those real ADRs and one controlled synthetic
negative exemplar:

1. `alphagov_govuk-aws_0012-security-groups-in-terraform`
2. `adr_madr_0001-use-CC0-or-MIT-as-license`
3. `argoproj_argo-cd_deep-links`
4. `synthetic_not_compliant_v1`

`alphagov_govuk-aws_0001-record-architecture-decisions` was excluded by the
source manifest but was not inserted into any prompt. The retained
`analysis/prompt_manifest.json` is authoritative for prompt contents. None of
the four actual exemplars appears in the 162-ADR evaluation set. The release
verifier checks both the actual prompt set and the unused exclusion.

## Human Assessment Evidence

The primary human file is formula-consistent for the 162 evaluated ADRs. The
second assessment contains 41 records. Recorded overall labels agree with the
primary file for 37 records. When overall classes are derived independently
from each file's C1-C7 and Q1-Q7 values, 27 records agree. Eleven recorded
overall labels in the second file differ from its own criterion-derived class.
Both forms are preserved. The code does not overwrite submitted human labels.

The retained files do not independently establish reviewer assignment,
blinding, or adjudication. Claims about those procedures require the original
study administration records rather than an inference from the JSON files.

## Full-Text Input

Every retained model row records the complete ADR input length and SHA-256. No
retained evaluation is marked as truncated. Two `loopdive/js2` records preserve
the Windows-1252 decoding observed in the provider run. Their stored and model
input hashes are listed in `release/input_text_exceptions.json`.

## Exact Scoring

Models supplied C1-C7 and Q1-Q7 judgments. Overall classes were derived from
those criteria with exact rational arithmetic. Forty retained predictions
changed when a former display-rounding boundary issue was corrected. The
original provider response run remains unchanged; the canonical exact-scoring
analysis is stored under a separate run ID and records zero additional API
calls.

## Validation Runs

`--validation-size` is a functional test mode, not a scientific subsample. It
uses a deterministic class-covered subset, preserves the three repetitions and
all model and prompt settings, writes its own `eval_set.json`, and marks its run
as nonpublication. Results from validation mode must not be pooled with or
reported as part of the 162-ADR study.
