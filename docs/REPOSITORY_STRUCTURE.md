# Repository Structure

The reviewer release is built from an explicit allowlist. Historical working
files may exist in a development checkout, but they are not copied into the
published package.

```text
.
|-- adr_benchmark.py
|-- adr_scoring.py
|-- README.md
|-- LICENSE
|-- THIRD_PARTY_NOTICES.md
|-- CITATION.cff
|-- requirements.txt
|-- requirements-figures.txt
|-- docs/
|   |-- HUMAN_ASSESSMENT.md
|   |-- PROTOCOL_NOTES.md
|   |-- REPRODUCIBILITY.md
|   |-- RELEASE_CHECKLIST.md
|   |-- RELEASE_NOTES.md
|   `-- REVIEWER_FEEDBACK_TRACEABILITY.md
|-- environment/
|-- release/
|   |-- input_text_exceptions.json
|   |-- release_spec.json
|   `-- source_licenses.json
|-- results/
|   |-- adrs/
|   |-- eval_set.json
|   |-- human_ground_truth.json
|   |-- human_ground_truth_interrator.json
|   |-- human_annotation/
|   `-- runs/
|       |-- v31_fulltext_gpt_medium/
|       `-- v31_fulltext_gpt_medium_exact_scoring/
|-- scripts/
|-- tests/
`-- .github/workflows/verify.yml
```

## Ownership Boundaries

| Area | Purpose | May a replication run modify it? |
| --- | --- | --- |
| Root Python modules | Frozen protocol, provider calls, scoring, and analysis | No |
| `results/adrs/` | Frozen ADR texts and source metadata | No |
| Human JSON files | Submitted reference assessments | No |
| `results/eval_set.json` | Publication evaluation order and metadata | No |
| Canonical v31 run directories | Provider evidence and reported exact-scoring analysis | No |
| A new `results/runs/<run-id>/` | Outputs from a smoke, validation, or replication run | Yes |
| `scripts/` | Offline analysis, validation, verification, and release tools | Only through version-controlled code changes |
| `dist/` | Generated reviewer package | Yes; ignored by Git |

## Reviewer Package Allowlist

`scripts/build_release.py` includes only the files needed to inspect the frozen
protocol, rerun it, recompute analyses, verify reported outputs, and attribute
third-party sources. It excludes credentials, dependency caches, manuscript
drafts, rendering workspaces, old acquisition summaries, unscoped results, and
archived intermediate runs. `RELEASE_MANIFEST.json` and `SHA256SUMS.txt` provide
an integrity record for the resulting package.

The allowlist is the publication boundary. Review it whenever a new required
artifact is added.
