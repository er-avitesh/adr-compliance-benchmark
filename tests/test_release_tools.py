import hashlib
import json
import re
import unittest
from pathlib import Path

import adr_benchmark as benchmark
from scripts import build_release
from scripts import verify_release


ROOT = Path(__file__).resolve().parents[1]


class ReleaseToolTests(unittest.TestCase):
    def test_frozen_text_exceptions_match_retained_model_inputs(self):
        exceptions = verify_release.input_exceptions(ROOT)
        loaded = {item["id"]: item for item in benchmark.load_adrs()}
        self.assertEqual(set(exceptions), set(benchmark.FROZEN_INPUT_TEXT_TRANSFORMS))
        for adr_id, expected in exceptions.items():
            text = loaded[adr_id]["text"]
            self.assertEqual(len(text), expected["model_input_chars"])
            self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), expected["model_input_sha256"])

    def test_release_allowlist_covers_rescoring_sources_and_excludes_debris(self):
        spec = build_release.read_json(ROOT / "release" / "release_spec.json")
        selected = {path.relative_to(ROOT).as_posix() for path in build_release.collect_files(spec)}
        report = verify_release.read_json(
            ROOT
            / "results"
            / "runs"
            / spec["canonical_run_id"]
            / "analysis"
            / "exact_rescoring_report.json"
        )
        sources = {path.replace("\\", "/") for path in report["source_sha256"]}
        self.assertTrue(sources <= selected)
        self.assertFalse(any("node_modules" in Path(path).parts for path in selected))
        self.assertFalse(any("__pycache__" in Path(path).parts for path in selected))
        self.assertFalse(any(path.endswith(".pyc") for path in selected))
        self.assertFalse(any(path.startswith("results/manuscript_v") for path in selected))
        self.assertIn(".gitattributes", selected)
        self.assertIn(".github/workflows/verify.yml", selected)
        self.assertIn("THIRD_PARTY_NOTICES.md", selected)
        self.assertIn("release/source_licenses.json", selected)
        self.assertNotIn("adr_dataset.json", selected)
        self.assertNotIn("dataset_report.json", selected)
        self.assertNotIn("find_adr_paths.py", selected)
        self.assertNotIn("results/adr_manifest.json", selected)
        self.assertNotIn("EXACT_SCORING_GUIDE.md", selected)
        self.assertNotIn("FULLTEXT_RERUN_GUIDE.md", selected)

    def test_public_readme_uses_current_protocol_and_bounded_human_claims(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("v31-fulltext-criterion-derived-1", text)
        self.assertIn("hosted Ministral 3 8B", text)
        self.assertIn("27 of 41", text)
        self.assertIn("5,832", text)
        self.assertNotIn("independent human software-architecture review and adjudication", text)
        self.assertNotIn("blinded to the original labels", text)
        self.assertNotIn("1,024-token", text)
        self.assertNotIn("will accompany the article", text)

    def test_reference_environment_matches_requirement_pins(self):
        environment = json.loads((ROOT / "environment" / "reference_environment.json").read_text(encoding="utf-8"))
        requirements = {}
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.lstrip().startswith("#"):
                name, version = line.split("==", 1)
                requirements[name] = version
        for name, version in requirements.items():
            self.assertEqual(environment["packages"][name], version)

    def test_source_license_inventory_covers_reviewer_release_adrs(self):
        spec = build_release.read_json(ROOT / "release" / "release_spec.json")
        audit = verify_release.Audit()
        evaluation = verify_release.read_json(ROOT / "results" / "eval_set.json")
        verify_release.verify_source_licenses(ROOT, evaluation, audit)
        self.assertEqual(audit.failures, [])
        self.assertGreaterEqual(len(audit.passes), 5)

    def test_public_human_assessments_are_minimal_and_anonymous(self):
        evaluation = json.loads((ROOT / "results" / "eval_set.json").read_text(encoding="utf-8"))
        source_primary = json.loads((ROOT / "results" / "human_ground_truth.json").read_text(encoding="utf-8"))
        source_second = json.loads((ROOT / "results" / "human_ground_truth_interrator.json").read_text(encoding="utf-8"))
        primary, second = build_release.public_human_assessments(
            evaluation,
            source_primary,
            source_second,
        )
        expected = {item["id"] for item in evaluation["adrs"]}
        expected.update(evaluation["heldout_few_shot_exemplar_ids"])

        self.assertEqual(set(primary), expected)
        self.assertEqual(len(primary), 166)
        self.assertEqual(
            {record["reviewer_id"] for record in primary.values()},
            {f"P{index:02d}" for index in range(1, 8)},
        )
        self.assertEqual(
            {record["reviewer_id"] for record in second.values()},
            {f"S{index:02d}" for index in range(1, 4)},
        )

    def test_local_markdown_links_in_public_docs_resolve(self):
        spec = build_release.read_json(ROOT / "release" / "release_spec.json")
        markdown_files = [
            path for path in build_release.collect_files(spec)
            if path.suffix.lower() == ".md"
        ]
        missing = []
        for source in markdown_files:
            text = source.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                path_text = target.split("#", 1)[0]
                destination = (source.parent / path_text).resolve()
                if not destination.exists():
                    missing.append(
                        f"{source.relative_to(ROOT)} -> {target}"
                    )
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
