#!/usr/bin/env python3
"""Build an allowlisted, hash-manifested reviewer release."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"v?[0-9][A-Za-z0-9._-]{0,39}")
SECRET_PATTERNS = (
    re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{25,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
)
PUBLIC_SCRIPTS = (
    "scripts/analyze_rubric_sensitivity.py",
    "scripts/analyze_rule_hybrid_comparison.py",
    "scripts/build_release.py",
    "scripts/generate_publication_assets.py",
    "scripts/prepare_manuscript_v31_evidence.py",
    "scripts/rescore_saved_results.py",
    "scripts/verify_exact_scoring_revision.py",
    "scripts/verify_release.py",
    "scripts/verify_validation_run.py",
)
ROOT_FILES = (
    ".gitattributes",
    ".github/workflows/verify.yml",
    ".env.example",
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "adr_benchmark.py",
    "adr_scoring.py",
    "requirements-figures.txt",
    "requirements.txt",
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_clean() -> bool:
    return not git_output("status", "--porcelain=v1", "--untracked-files=all")


def ensure_safe_output(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    parent = output_root.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"Refusing to replace path outside output directory: {resolved}")


def referenced_adr_files(spec: dict) -> list[Path]:
    evaluation = read_json(ROOT / "results" / "eval_set.json")
    ids = {item["id"] for item in evaluation["adrs"]}
    ids.update(evaluation.get("heldout_few_shot_exemplar_ids", []))
    ids.discard("synthetic_not_compliant_v1")
    files = [ROOT / "results" / "adrs" / f"{adr_id}.json" for adr_id in sorted(ids)]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen ADR files: {missing[:3]}")
    if len(evaluation["adrs"]) != spec["evaluation_set_size"]:
        raise ValueError("Evaluation set size differs from release specification")
    return files


def collect_files(spec: dict) -> list[Path]:
    files = [ROOT / relative for relative in ROOT_FILES + PUBLIC_SCRIPTS]
    files.extend(path for path in (ROOT / "docs").rglob("*") if path.is_file())
    files.extend(path for path in (ROOT / "environment").rglob("*") if path.is_file())
    files.extend(path for path in (ROOT / "release").rglob("*") if path.is_file())
    files.extend(path for path in (ROOT / "tests").rglob("*.py") if path.is_file())

    shared = (
        "results/eval_set.json",
        "results/human_ground_truth.json",
        "results/human_ground_truth_interrator.json",
    )
    files.extend(ROOT / relative for relative in shared)
    annotation_dir = ROOT / "results" / "human_annotation"
    if annotation_dir.is_dir():
        files.extend(path for path in annotation_dir.rglob("*") if path.is_file())
    files.extend(referenced_adr_files(spec))

    source_run = ROOT / "results" / "runs" / spec["source_run_id"]
    canonical_run = ROOT / "results" / "runs" / spec["canonical_run_id"]
    for run in (source_run, canonical_run):
        if not run.is_dir():
            raise FileNotFoundError(f"Required run directory does not exist: {run}")
    for run in (source_run, canonical_run):
        for path in run.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(run).as_posix()
            if run == canonical_run and relative.startswith("analysis/archive_before_"):
                continue
            files.append(path)

    unique = sorted(set(path.resolve() for path in files), key=lambda path: path.relative_to(ROOT).as_posix())
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release allowlist includes missing files: {missing[:3]}")
    return unique


def validate_paths(files: list[Path], spec: dict) -> None:
    forbidden = set(spec["forbidden_release_path_parts"])
    for path in files:
        relative = path.relative_to(ROOT)
        if set(relative.parts) & forbidden:
            raise ValueError(f"Forbidden path selected for release: {relative}")
        if relative.suffix.lower() in {".pyc", ".pyo"}:
            raise ValueError(f"Python bytecode selected for release: {relative}")


def scan_for_credentials(files: list[Path]) -> None:
    hits = []
    for path in files:
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            hits.append(path.relative_to(ROOT).as_posix())
    if hits:
        raise ValueError(f"Possible API credential found in release file(s): {hits}")


def copy_payload(files: list[Path], staging: Path) -> None:
    for source in files:
        relative = source.relative_to(ROOT)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def anonymize_reviewer_ids(records: dict, prefix: str) -> None:
    identifiers = sorted(
        {
            record.get("reviewer_id")
            for record in records.values()
            if record.get("reviewer_id")
        }
    )
    aliases = {
        identifier: f"{prefix}{index:02d}"
        for index, identifier in enumerate(identifiers, start=1)
    }
    for record in records.values():
        identifier = record.get("reviewer_id")
        if identifier:
            record["reviewer_id"] = aliases[identifier]


def assessment_content_digest(records: dict) -> str:
    content = {
        adr_id: {
            key: value
            for key, value in record.items()
            if key != "reviewer_id"
        }
        for adr_id, record in records.items()
    }
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def public_human_assessments(
    evaluation: dict,
    primary: dict,
    second: dict,
) -> tuple[dict, dict]:
    evaluation_ids = {item["id"] for item in evaluation["adrs"]}
    holdout_ids = set(evaluation.get("heldout_few_shot_exemplar_ids", []))
    public_primary_ids = evaluation_ids | holdout_ids
    primary = {
        adr_id: record
        for adr_id, record in primary.items()
        if adr_id in public_primary_ids
    }
    if set(primary) != public_primary_ids:
        missing = sorted(public_primary_ids - set(primary))
        raise ValueError(f"Public primary assessments are missing ADRs: {missing[:3]}")
    anonymize_reviewer_ids(primary, "P")

    anonymize_reviewer_ids(second, "S")
    return primary, second


def sanitize_human_assessments(staging: Path) -> None:
    evaluation = read_json(staging / "results" / "eval_set.json")
    primary_path = staging / "results" / "human_ground_truth.json"
    second_path = staging / "results" / "human_ground_truth_interrator.json"
    transform_path = staging / "release" / "public_data_transforms.json"
    existing_transforms = {}
    if transform_path.is_file():
        existing_transforms = {
            item["path"]: item
            for item in read_json(transform_path).get("transforms", [])
        }
    source_primary = read_json(primary_path)
    source_second = read_json(second_path)
    source_hashes = {
        "results/human_ground_truth.json": existing_transforms.get(
            "results/human_ground_truth.json", {}
        ).get("source_sha256", digest(primary_path)),
        "results/human_ground_truth_interrator.json": existing_transforms.get(
            "results/human_ground_truth_interrator.json", {}
        ).get("source_sha256", digest(second_path)),
    }
    primary, second = public_human_assessments(
        evaluation,
        source_primary,
        source_second,
    )
    primary_path.write_text(
        json.dumps(primary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(second, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    evaluation_ids = [item["id"] for item in evaluation["adrs"]]
    evidence_path = (
        staging
        / "results"
        / "runs"
        / "v31_fulltext_gpt_medium_exact_scoring"
        / "analysis"
        / "manuscript_evidence"
        / "evidence.json"
    )
    evidence_source_hash = None
    if evidence_path.is_file():
        evidence_source_hash = existing_transforms.get(
            evidence_path.relative_to(staging).as_posix(), {}
        ).get("source_sha256", digest(evidence_path))
        evidence = read_json(evidence_path)
        evidence["reviewer_counts"] = {
            "first": dict(Counter(primary[adr_id]["reviewer_id"] for adr_id in evaluation_ids)),
            "second": dict(Counter(record["reviewer_id"] for record in second.values())),
        }
        evidence_path.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    primary_existing = existing_transforms.get("results/human_ground_truth.json", {})
    second_existing = existing_transforms.get("results/human_ground_truth_interrator.json", {})
    transforms = {
        "schema_version": 1,
        "purpose": "Remove personal reviewer identifiers and unpublished extra labels from public artifacts.",
        "transforms": [
            {
                "path": "results/human_ground_truth.json",
                "source_sha256": source_hashes["results/human_ground_truth.json"],
                "public_sha256": digest(primary_path),
                "source_records": primary_existing.get("source_records", len(source_primary)),
                "public_records": len(primary),
                "removed_records": primary_existing.get("source_records", len(source_primary)) - len(primary),
                "reviewer_codes": sorted({record["reviewer_id"] for record in primary.values()}),
                "preserved_content_sha256_excluding_reviewer_id": assessment_content_digest(primary),
            },
            {
                "path": "results/human_ground_truth_interrator.json",
                "source_sha256": source_hashes["results/human_ground_truth_interrator.json"],
                "public_sha256": digest(second_path),
                "source_records": second_existing.get("source_records", len(source_second)),
                "public_records": len(second),
                "removed_records": second_existing.get("source_records", len(source_second)) - len(second),
                "reviewer_codes": sorted({record["reviewer_id"] for record in second.values()}),
                "preserved_content_sha256_excluding_reviewer_id": assessment_content_digest(second),
            },
        ],
    }
    if evidence_source_hash is not None:
        evidence_relative = evidence_path.relative_to(staging).as_posix()
        transforms["transforms"].append(
            {
                "path": evidence_relative,
                "source_sha256": evidence_source_hash,
                "public_sha256": digest(evidence_path),
                "changed_field": "reviewer_counts object keys",
                "primary_reviewer_codes": sorted(evidence["reviewer_counts"]["first"]),
                "second_reviewer_codes": sorted(evidence["reviewer_counts"]["second"]),
            }
        )
    transform_path.write_text(
        json.dumps(transforms, indent=2) + "\n",
        encoding="utf-8",
    )


def payload_records(staging: Path) -> list[dict]:
    excluded = {"RELEASE_MANIFEST.json", "SHA256SUMS.txt"}
    records = []
    for path in sorted((p for p in staging.rglob("*") if p.is_file()), key=lambda p: p.relative_to(staging).as_posix()):
        relative = path.relative_to(staging).as_posix()
        if relative in excluded:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    return records


def write_metadata(staging: Path, version: str, spec: dict, clean: bool) -> dict:
    records = payload_records(staging)
    try:
        remote = git_output("remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        remote = None
    manifest = {
        "release_manifest_schema_version": 1,
        "package_name": spec["package_name"],
        "package_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_output("rev-parse", "HEAD"),
        "source_remote": remote,
        "working_tree_clean": clean,
        "canonical_run_id": spec["canonical_run_id"],
        "source_run_id": spec["source_run_id"],
        "protocol_version": spec["protocol_version"],
        "scoring_version": spec["scoring_version"],
        "evaluation_set_size": spec["evaluation_set_size"],
        "repetitions": spec["repetitions"],
        "expected_evaluations": spec["expected_evaluations"],
        "input_policy": spec["input_policy"],
        "model_access_date": spec["model_access_date"],
        "pricing_verified_date": spec["pricing_verified_date"],
        "files": records,
    }
    manifest_path = staging / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [f"{item['sha256']}  {item['path']}" for item in records]
    lines.append(f"{digest(manifest_path)}  RELEASE_MANIFEST.json")
    (staging / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_zip(staging: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((p for p in staging.rglob("*") if p.is_file()), key=lambda p: p.relative_to(staging.parent).as_posix()):
            archive.write(path, path.relative_to(staging.parent).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Immutable release version, for example v1.0.0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow a marked local test build; never use for publication")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if not VERSION_RE.fullmatch(args.version):
        parser.error("Version must start with a digit or v plus a digit and contain only letters, numbers, dot, dash, or underscore")

    spec = read_json(ROOT / "release" / "release_spec.json")
    clean = git_is_clean()
    if not clean and not args.allow_dirty:
        parser.error("Refusing to build a publication release from a dirty worktree")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f"{spec['package_name']}-{args.version}"
    ensure_safe_output(staging, output_root)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    files = collect_files(spec)
    validate_paths(files, spec)
    scan_for_credentials(files)
    copy_payload(files, staging)
    sanitize_human_assessments(staging)
    manifest = write_metadata(staging, args.version, spec, clean)

    zip_path = output_root / f"{staging.name}.zip"
    if not args.no_zip:
        if zip_path.exists():
            zip_path.unlink()
        write_zip(staging, zip_path)

    print(f"Release directory: {staging}")
    if not args.no_zip:
        print(f"Release archive: {zip_path}")
    print(f"Payload files: {len(manifest['files'])}")
    print(f"Source commit: {manifest['source_commit']}")
    print(f"Working tree clean: {manifest['working_tree_clean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
