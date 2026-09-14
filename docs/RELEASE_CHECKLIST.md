# Publication Release Checklist

Complete this checklist only after manuscript numbers and figures have been
matched to the canonical exact-scoring run.

## Repository Gate

- [ ] `python -B scripts/verify_release.py` passes.
- [ ] `python -B -m unittest discover -s tests -v` passes.
- [ ] The final live smoke run passes all requested configurations.
- [ ] `git status --short` contains only intended publication changes.
- [ ] No `.env`, credentials, private keys, local dependencies, caches, or
      temporary render directories are tracked.
- [ ] `THIRD_PARTY_NOTICES.md` and `release/source_licenses.json` cover every
      distributed ADR file.
- [ ] README model IDs, output limits, run IDs, input policy, human-review
      wording, class counts, and result locations match the saved artifacts.
- [ ] The manuscript cites the canonical exact-scoring outputs, not archived or
      unscoped `results/analysis` files.

## Build Gate

Choose the final semantic version or manuscript-linked tag, for example
`v1.0.0`.

```powershell
python -B scripts\build_release.py --version v1.0.0
python -B scripts\verify_release.py --root dist\adr-compliance-benchmark-v1.0.0
```

Linux or macOS:

```bash
python -B scripts/build_release.py --version v1.0.0
python -B scripts/verify_release.py --root dist/adr-compliance-benchmark-v1.0.0
```

The builder refuses a dirty worktree unless `--allow-dirty` is supplied. Never
use `--allow-dirty` for the publication artifact. Inspect the generated
`RELEASE_MANIFEST.json` and `SHA256SUMS.txt` before upload.

## GitHub Gate

The following commands are examples. Run them only after the clean package has
passed verification.

```powershell
git tag -a v1.0.0 -m "ADR compliance benchmark replication package v1.0.0"
git push origin master
git push origin v1.0.0
gh release create v1.0.0 dist\adr-compliance-benchmark-v1.0.0.zip --title "ADR compliance benchmark v1.0.0" --notes-file docs\RELEASE_NOTES.md
```

Confirm that the public release resolves to the commit recorded in
`RELEASE_MANIFEST.json`. Do not move or replace the tag after publication.

## Archival Gate

- [ ] Connect the GitHub repository to Zenodo.
- [ ] Archive the final GitHub release.
- [ ] Confirm the Zenodo record includes the authors, title, version, license,
      publication date, and GitHub release URL.
- [ ] Verify the downloaded Zenodo archive against `SHA256SUMS.txt`.
- [ ] Record the issued DOI.

## Manuscript Gate

Replace the provisional Data Availability wording with concrete identifiers:

> The replication package is available in the immutable GitHub release
> `[TAG]` at `[RELEASE URL]` and is archived at Zenodo under DOI `[DOI]`
> (accessed `[DATE]`). The package contains the frozen 162-ADR evaluation
> manifest, human assessments, prompts, retained criterion-level predictions,
> exact scoring and analysis code, generated outputs, and SHA-256 file hashes.
> Benchmark code is MIT licensed; third-party ADR texts remain subject to their
> source repositories' licenses.

Do not leave `[TAG]`, `[RELEASE URL]`, `[DOI]`, or `[DATE]` in the submitted
manuscript.
