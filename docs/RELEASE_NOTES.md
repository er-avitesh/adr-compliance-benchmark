# ADR Compliance Benchmark Replication Package

This release contains the frozen 162-ADR evaluation manifest, human reference
assessments, retained criterion-level provider responses, exact scoring code,
statistical analyses, sensitivity analyses, baseline comparisons, publication
assets, and integrity hashes used by the manuscript.

Key reproducibility facts:

- 12 model and prompt configurations
- 3 repetitions per configuration
- 5,832 retained evaluations
- complete ADR text for every retained request
- exact criterion-derived classification
- a separate, zero-API-call exact-scoring analysis revision
- source-license and attribution records for distributed ADR texts

Run `python -B scripts/verify_release.py` from the extracted package root before
using the artifacts. See `docs/REPRODUCIBILITY.md` for the offline and live
replication procedures.
