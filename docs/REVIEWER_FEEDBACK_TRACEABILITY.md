# Reviewer Feedback Traceability

This matrix maps the principal reviewer concerns to executable checks and
retained evidence. It is a repository quality-control record, not manuscript
text.

| Concern | Repository resolution | Verification evidence | Status |
| --- | --- | --- | --- |
| Human reference validity | Primary and second assessments are retained separately; recorded and criterion-derived agreement are both reported without silent reconciliation. | `docs/HUMAN_ASSESSMENT.md`; `scripts/verify_release.py` | Resolved with documented limitation |
| Sampling clarity | The exact 162-ADR order, class distribution, repository distribution, and source run hash are frozen. The repository does not claim a below-five-percent population margin of error. | `results/eval_set.json`; `docs/PROTOCOL_NOTES.md` | Resolved |
| Source licenses | Every distributed ADR maps to repository-level license evidence and attribution. Unlicensed sources are absent from the reviewer package. | `THIRD_PARTY_NOTICES.md`; `release/source_licenses.json` | Resolved for package preparation |
| Prompt transparency | Full templates, rubric, output schema, parser procedure, hashes, and actual exemplar contents are retained. | Source run `analysis/prompt_manifest.json` | Resolved |
| Few-shot leakage | Exact ID, source-path, and text-similarity checks are retained. The unused manifest holdout is documented separately from actual exemplars. | Source run `analysis/prompt_leakage_audit.json`; `docs/PROTOCOL_NOTES.md` | Resolved with stated scope |
| Model configuration | Exact model IDs, provider settings, output limits, GPT reasoning effort, access date, and pricing date are recorded. | Canonical `analysis/model_reproducibility.json`; `analysis/cost_analysis.json` | Resolved |
| Complete ADR input | Each result row stores source and input lengths and hashes; all 5,832 retained rows report no truncation. | Canonical `raw_results/*.json`; release verifier | Resolved |
| Criterion-derived labels | The parser retains model-declared labels but derives evaluated labels from complete criterion vectors using one ordered rule. | `adr_scoring.py`; prompt manifest; unit tests | Resolved |
| Exact boundary arithmetic | Classification uses fractions and rounds only for display. The 40 affected predictions and all source hashes are retained. | Canonical `analysis/exact_rescoring_report.json`; `analysis/exact_scoring_verification.json` | Resolved |
| Repeated-measures statistics | Primary pairwise inference resamples ADRs as clusters and keeps repetitions together. McNemar tests are per-repetition diagnostics only. | Canonical `analysis/pairwise_statistical_tests.json`; tests | Resolved |
| Sensitivity analysis | Six alternative rubric analyses recompute macro-F1, kappa, and rankings from retained criteria. | Canonical `analysis/rubric_sensitivity_performance.json` | Resolved |
| Rule and hybrid baselines | Rule-only SC, LLM-only, and formula-matched hybrid results are reported together. | Canonical `analysis/rule_hybrid_comparison.json` | Resolved |
| Error and criterion evidence | Confusion matrices, per-class metrics, criterion-level performance, and error examples are retained. | Canonical `analysis/performance_details.json`, `criterion_level_performance.json`, `error_analysis.json` | Resolved |
| Cost and retries | Dated rates, batch multipliers, token treatment, provider usage, and repair records are retained. | Canonical `analysis/cost_analysis.json`; source batch manifest and collection reports | Resolved |
| Executable replication | Offline verification, full paid replication, and five-ADR functional validation are documented and tested. | `docs/REPRODUCIBILITY.md`; CI; unit and integration tests | Resolved |
| Immutable public artifact | Release builder creates an allowlisted package with SHA-256 manifests and refuses dirty publication builds. | `scripts/build_release.py`; `docs/RELEASE_CHECKLIST.md` | Ready; tag and DOI remain release actions |

The final external actions are to build from a clean commit, publish an
immutable Git tag and GitHub release, archive it, record the DOI, and replace
the manuscript's provisional Data Availability identifiers.
