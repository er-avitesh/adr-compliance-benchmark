import copy
import io
import itertools
import json
import unittest
from fractions import Fraction
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import adr_scoring as rubric
import adr_benchmark as benchmark
from scripts import analyze_rubric_sensitivity as sensitivity
from scripts import analyze_rule_hybrid_comparison as hybrid
from scripts.rescore_saved_results import rescore_row, rescore, run_path
from scripts.prepare_manuscript_v31_evidence import score as evidence_score


ROOT = Path(__file__).resolve().parents[1]


class ExactScoringTests(unittest.TestCase):
    def setUp(self):
        self.checks = {f"C{i}": False for i in range(1, 8)}
        self.scores = {f"Q{i}": 0 for i in range(1, 8)}

    def test_all_binary_vectors_have_exact_integer_sc(self):
        weights = (10, 15, 15, 15, 20, 15, 10)
        for values in itertools.product((0, 1), repeat=7):
            checks = dict(zip(self.checks, values))
            self.assertEqual(rubric.structural_score(checks), sum(a*b for a, b in zip(weights, values)))

    def test_all_dq_vectors_match_integer_numerator(self):
        weights = (10, 15, 15, 10, 25, 15, 10)
        for values in itertools.product(range(4), repeat=7):
            scores = dict(zip(self.scores, values))
            self.assertEqual(rubric.quality_score(scores), Fraction(sum(a*b for a, b in zip(weights, values)), 3))

    def test_full_score_grid_matches_independent_integer_rule(self):
        # SC = 5*s and DQ = 5*q/3 make CS = 2*s + q exactly.
        for s in range(21):
            for q in range(61):
                composite = 2*s + q
                expected = ("Not_Compliant" if composite < 45 or s < 8 else
                            "Fully_Compliant" if composite >= 85 and s >= 16 and q >= 48 else
                            "Mostly_Compliant" if composite >= 70 and s >= 12 and q >= 36 else
                            "Partially_Compliant")
                self.assertEqual(rubric.overall_class(5*s, Fraction(5*q, 3)), expected)

    def test_exact_composite_boundaries_and_neighbors(self):
        for sc, dq, boundary, label in [(40, Fraction(145, 3), 45, "Partially_Compliant"),
                                        (60, Fraction(230, 3), 70, "Mostly_Compliant"),
                                        (80, Fraction(265, 3), 85, "Fully_Compliant")]:
            self.assertEqual(rubric.composite_score(sc, dq), boundary)
            self.assertEqual(rubric.overall_class(sc, dq), label)
            self.assertNotEqual(rubric.overall_class(sc, dq - Fraction(1, 1000000)), label)

    def test_dimension_boundaries(self):
        labels = ["Not_Compliant", "Partially_Compliant", "Mostly_Compliant", "Fully_Compliant"]
        for index, boundary in enumerate((45, 70, 85), 1):
            self.assertEqual(rubric.dimension_class(boundary), labels[index])
            self.assertEqual(rubric.dimension_class(Fraction(boundary) - Fraction(1, 1000)), labels[index-1])

    def test_equal_weights_sum_exactly_to_one(self):
        self.assertEqual(sum(sensitivity.EQUAL_DQ_WEIGHTS.values()), 1)
        scores = dict.fromkeys(self.scores, 3)
        self.assertEqual(rubric.quality_score(scores, sensitivity.EQUAL_DQ_WEIGHTS), 100)

    def test_protocol_weights_match_exact_weights(self):
        for legacy, shared in ((benchmark.SC_WEIGHTS, rubric.SC_WEIGHTS),
                               (benchmark.DQ_WEIGHTS, rubric.DQ_WEIGHTS)):
            self.assertEqual({k: rubric.exact(v) for k, v in legacy.items()}, shared)

    def test_invalid_or_missing_criteria_are_not_silently_scored(self):
        for value in (-1, 4, 1.5, None):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                rubric.quality_score({**self.scores, "Q1": value})
        with self.assertRaises(KeyError):
            rubric.structural_score({})
        with self.assertRaises(ValueError):
            rubric.structural_score({**self.checks, "C1": "false"})

    def test_all_paths_share_exact_classification(self):
        # C1+C2+C3 = 40; Q1+Q2+Q3+Q4+3*Q5+Q6+Q7 = 145.
        checks = {**self.checks, "C1": True, "C2": True, "C3": True}
        scores = {key: 1 for key in self.scores}
        scores.update(Q1=3, Q5=2)
        self.assertEqual(rubric.quality_score(scores), Fraction(145, 3))
        expected = "Partially_Compliant"
        payload = {**checks, **scores, "dq_score": 48.33, "overall": "Not_Compliant"}
        prediction = benchmark.extract_prediction(json.dumps(payload))
        self.assertEqual(prediction["overall"], expected)
        self.assertEqual(prediction["overall_raw"], "Not_Compliant")
        self.assertEqual(prediction["dq_score"], float(Fraction(145, 3)))
        self.assertEqual(prediction["exact_scores"]["composite"], "45")
        self.assertEqual(sensitivity.scenario_label(checks, scores, sensitivity.SCENARIOS[0]), expected)
        self.assertEqual(hybrid.classify(rubric.structural_score(checks), rubric.quality_score(scores)), expected)
        self.assertEqual(evidence_score(payload)["overall"], 1)
        json.dumps(prediction, allow_nan=False)

    def test_rescore_preserves_original_criteria_metadata_and_reference(self):
        original = {"predicted_sc_checks": self.checks, "predicted_dq_scores": self.scores,
                    "actual": "Mostly_Compliant", "predicted": "Mostly_Compliant",
                    "run_id": "original", "protocol_fingerprint": "unchanged",
                    "predicted_overall_raw": "Fully_Compliant", "usage_details": {"thinking": 99}}
        before = copy.deepcopy(original)
        updated = rescore_row(original, "new_analysis")
        self.assertEqual(original, before)
        for key in original.keys() - {"predicted"}:
            self.assertEqual(updated[key], original[key])
        self.assertEqual(updated["predicted"], "Not_Compliant")
        self.assertFalse(updated["correct"])
        self.assertEqual(updated["analysis_run_id"], "new_analysis")
        json.dumps(updated, allow_nan=False)

    def test_metrics_not_rounded_before_aggregation(self):
        values = [0.46502057613168724, 0.5349794238683128, 0.423456789]
        self.assertEqual(sensitivity.rounded(values[0]), values[0])
        self.assertAlmostEqual(hybrid.metric_summary(values)["mean"], sum(values)/3, places=15)
        self.assertEqual(benchmark._safe_metric(values[0]), values[0])
        self.assertEqual(benchmark._summary_stats([values[0]])["mean"], values[0])

    def test_refuses_in_place_and_existing_destinations_without_api(self):
        with mock.patch.object(benchmark, "call_llm", side_effect=AssertionError("No API calls")):
            with self.assertRaisesRegex(ValueError, "nonexistent"):
                rescore("v31_fulltext_gpt_medium", "v31_fulltext_gpt_medium")
        with self.assertRaises(ValueError):
            run_path("../escape")

    def test_archived_and_incomplete_manifests_block_analysis(self):
        for manifest in ({}, {"scoring_version": rubric.SCORING_VERSION,
                               "analysis_only": True, "rescore_complete": False}):
            with mock.patch.object(Path, "read_text", return_value=json.dumps(manifest)):
                with self.assertRaises(ValueError):
                    rubric.require_exact_run(Path("fixture_run"))

    def test_cli_blocks_model_calls_for_analysis_revision(self):
        original_run = benchmark.ACTIVE_RUN_ID
        try:
            with mock.patch("sys.argv", ["adr_benchmark.py", "--phase", "run",
                                           "--run-id", "test-analysis"]), \
                 mock.patch.object(Path, "exists", return_value=True), \
                 mock.patch.object(Path, "read_text", return_value=json.dumps({
                     "scoring_version": rubric.SCORING_VERSION, "analysis_only": True})), \
                 mock.patch.object(benchmark, "call_llm", side_effect=AssertionError("No API calls")), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                benchmark.main()
            self.assertEqual(raised.exception.code, 2)
        finally:
            benchmark.configure_run_paths(original_run)


if __name__ == "__main__":
    unittest.main()
