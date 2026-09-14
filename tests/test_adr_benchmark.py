import hashlib
import json
import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import adr_benchmark as benchmark
from scripts import verify_validation_run


ROOT = Path(__file__).resolve().parents[1]


class FullTextProtocolTests(unittest.TestCase):
    def tearDown(self):
        benchmark.configure_run_paths(benchmark.DEFAULT_RUN_ID)
        benchmark.configure_mistral_variant("ministral-3-8b-api")
        benchmark.configure_gpt_reasoning_effort("medium")

    def test_active_pricing_metadata_is_dated_and_sourced(self):
        expected = {
            "gpt-5.5": (5.00, 30.00, 2.50, 15.00),
            "claude-sonnet-4-6": (3.00, 15.00, 1.50, 7.50),
            "gemini-2.5-pro": (1.25, 10.00, 0.625, 5.00),
        }
        for model_name, rates in expected.items():
            metadata = benchmark.model_reproducibility_row(model_name)
            self.assertEqual(metadata["pricing_verified_date"], "2026-09-10")
            self.assertTrue(metadata["pricing_source"].startswith("https://"))
            self.assertEqual(
                metadata["cost_assumption_usd_per_million_input_tokens"], rates[0]
            )
            self.assertEqual(
                metadata["cost_assumption_usd_per_million_output_tokens"], rates[1]
            )
            self.assertEqual(
                metadata["batch_input_usd_per_million_tokens"], rates[2]
            )
            self.assertEqual(
                metadata["batch_output_usd_per_million_tokens"], rates[3]
            )

        hosted = benchmark.MISTRAL_MODEL_VARIANTS["ministral-3-8b-api"]["cost"]
        self.assertEqual(hosted["input"], 0.15)
        self.assertEqual(hosted["output"], 0.15)
        self.assertEqual(hosted["pricing_verified_date"], "2026-09-10")
        self.assertTrue(hosted["pricing_source"].startswith("https://"))
        self.assertIsNone(hosted["batch_multiplier"])

    def test_model_access_summary_uses_saved_timestamps(self):
        rows = {
            "gpt-5.5": {
                "zero_shot": [[
                    {
                        "adr_id": "adr-1",
                        "model_key": "gpt-5.5",
                        "accessed_at": "2026-09-08T12:42:00.070651",
                    },
                    {
                        "adr_id": "adr-2",
                        "model_key": "gpt-5.5",
                        "accessed_at": "2026-09-08T18:57:44.676207",
                    },
                ]]
            }
        }
        summary = benchmark._summarize_model_access(rows)["gpt-5.5"]
        self.assertEqual(summary["model_access_date"], "2026-09-08")
        self.assertEqual(summary["saved_evaluation_rows"], 2)
        self.assertIn("collection", summary["timestamp_semantics"])

    def test_gemini_batch_thinking_tokens_are_billable_and_reconciled(self):
        row = {
            "adr_id": "adr-1",
            "billing_mode": "batch",
            "batch_metadata": {"custom_id": "gemini_zs_r01_001"},
            "input_tokens": 100,
            "output_tokens": 20,
        }
        usage = {
            "gemini_zs_r01_001": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 30,
                "totalTokenCount": 150,
            }
        }
        details = benchmark._gemini_billable_output_tokens(row, usage)
        self.assertEqual(details["candidate_output_tokens"], 20)
        self.assertEqual(details["thinking_tokens"], 30)
        self.assertEqual(details["billable_output_tokens"], 50)
        self.assertTrue(details["provider_usage_reconciled"])

    def test_gemini_batch_usage_mismatch_is_rejected(self):
        row = {
            "adr_id": "adr-1",
            "billing_mode": "batch",
            "batch_metadata": {"custom_id": "gemini_zs_r01_001"},
            "input_tokens": 100,
            "output_tokens": 20,
        }
        usage = {
            "gemini_zs_r01_001": {
                "promptTokenCount": 101,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 30,
                "totalTokenCount": 151,
            }
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            benchmark._gemini_billable_output_tokens(row, usage)

    def test_cost_uses_billable_output_override(self):
        row = {
            "input_tokens": 100,
            "output_tokens": 20,
            "cost_multiplier": 0.5,
        }
        actual = benchmark._row_token_cost_usd(
            "gemini-2.5-pro", row, billable_output_tokens=50
        )
        expected = 0.5 * ((100 * 1.25 + 50 * 10.00) / 1_000_000)
        self.assertAlmostEqual(actual, expected)

    def test_openai_automatic_cached_inputs_are_reported_but_not_discounted(self):
        rows = [{
            "adr_id": "adr-1",
            "billing_mode": "batch",
            "batch_metadata": {"custom_id": "gpt_zs_r01_001"},
            "input_tokens": 100,
            "output_tokens": 20,
        }]
        usage = {
            "gpt_zs_r01_001": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 64},
            }
        }
        summary = benchmark._summarize_openai_cached_inputs(rows, usage)
        self.assertEqual(summary["retained_batch_rows_reconciled"], 1)
        self.assertEqual(summary["rows_with_cached_input"], 1)
        self.assertEqual(summary["cached_input_tokens"], 64)
        self.assertFalse(summary["discount_applied_in_estimate"])

    def test_local_env_accepts_powershell_variable_syntax(self):
        self.assertEqual(
            benchmark._parse_local_env_line('$env:OPENAI_API_KEY="test-value"'),
            ("OPENAI_API_KEY", "test-value"),
        )
        self.assertEqual(
            benchmark._parse_local_env_line('ANTHROPIC_API_KEY="test-value"'),
            ("ANTHROPIC_API_KEY", "test-value"),
        )

    def test_make_prompt_preserves_complete_adr(self):
        marker = "END-OF-ADR-MARKER"
        adr_text = "A" * 50000 + marker
        prompt = benchmark.make_prompt(adr_text, "zero_shot")
        self.assertIn(marker, prompt)
        self.assertIn(adr_text, prompt)

    def test_ordered_classifier_covers_previous_prompt_gaps(self):
        self.assertEqual(
            benchmark.classify_overall_from_scores(75, 95),
            "Mostly_Compliant",
        )
        self.assertEqual(
            benchmark.classify_overall_from_scores(100, 55),
            "Partially_Compliant",
        )
        self.assertEqual(
            benchmark.classify_overall_from_scores(35, 100),
            "Not_Compliant",
        )
        self.assertEqual(
            benchmark.classify_overall_from_scores(80, 89),
            "Fully_Compliant",
        )

    def test_few_shot_prompt_uses_complete_exemplars(self):
        adrs = benchmark.load_adrs()
        ground_truth = json.loads(
            (ROOT / "results/human_ground_truth.json").read_text(encoding="utf-8")
        )
        eval_adrs = benchmark._load_eval_adrs_from_manifest(adrs)
        examples = benchmark.build_few_shot_examples(
            adrs, ground_truth, eval_adrs
        )
        self.assertEqual(len(examples), 4)
        self.assertIn("synthetic_not_compliant_v1", {ex["id"] for ex in examples})
        prompt = benchmark.make_prompt(eval_adrs[0]["text"], "few_shot", examples)
        for example in examples:
            self.assertIn(example["text"], prompt)

    def test_protocol_fingerprint_changes_with_reasoning_effort(self):
        benchmark.configure_gpt_reasoning_effort("low")
        low = benchmark.protocol_fingerprint("gpt-5.5", "zero_shot")
        benchmark.configure_gpt_reasoning_effort("medium")
        medium = benchmark.protocol_fingerprint("gpt-5.5", "zero_shot")
        self.assertNotEqual(low, medium)

    def test_input_audit_proves_full_text(self):
        text = "# Decision\n\nComplete input."
        prompt = benchmark.make_prompt(text, "zero_shot")
        audit = benchmark.build_input_audit(text, prompt)
        self.assertFalse(audit["input_was_truncated"])
        self.assertEqual(audit["adr_source_chars"], audit["adr_input_chars"])
        self.assertEqual(audit["adr_input_sha256"], benchmark._sha256_text(text))

    def test_result_validation_rejects_another_protocol(self):
        adr = {"id": "test-adr", "text": "# Decision\n\nUse PostgreSQL."}
        prompt = benchmark.make_prompt(adr["text"], "zero_shot")
        raw = json.dumps({
            **{f"C{i}": False for i in range(1, 8)},
            **{f"Q{i}": 0 for i in range(1, 8)},
            "confidence": 0.9,
        })
        prediction = benchmark.extract_prediction(raw)
        result = {
            "raw": raw,
            "model_metadata": benchmark.model_reproducibility_row("gpt-5.5"),
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        benchmark.attach_run_metadata(
            result, "gpt-5.5", "zero_shot", adr["text"], prompt
        )
        row = benchmark.build_result_row(
            adr, "Not_Compliant", prediction, result
        )
        self.assertIsNone(benchmark._result_row_validation_reason(row, 1, 1))
        row["protocol_fingerprint"] = "different-protocol"
        self.assertIn(
            "different protocol",
            benchmark._result_row_validation_reason(row, 1, 1),
        )

    def test_paired_inference_keeps_adr_as_resampling_unit(self):
        labels = benchmark.CLASSES
        rows = [
            {
                "adr_id": f"adr-{idx}",
                "actual": label,
                "predicted": label,
                "correct": True,
            }
            for idx, label in enumerate(labels * 2)
        ]
        config = {"reps": [rows, [dict(row) for row in rows]]}
        result = benchmark._paired_adr_metric_inference(
            config, config, n_resamples=100, seed=7
        )
        self.assertEqual(result["unit_of_resampling"], "ADR")
        self.assertEqual(result["n_adr_clusters"], len(rows))
        self.assertEqual(
            result["metrics"]["macro_f1"]["difference_a_minus_b"], 0.0
        )
        self.assertEqual(
            result["metrics"]["macro_f1"]["adr_level_paired_randomization_p_unadjusted"],
            1.0,
        )

    def test_batch_collection_failure_isolated_to_one_job(self):
        target = ("gpt-5.5", "zero_shot")
        job_key = benchmark._batch_job_key(*target)
        manifest = {
            "jobs": {
                job_key: {
                    "job_key": job_key,
                    "provider": "openai",
                    "model": target[0],
                    "strategy": target[1],
                    "provider_status": "completed",
                    "output_file": "does-not-exist.jsonl",
                }
            }
        }
        with (
            mock.patch.object(
                benchmark, "_load_eval_adrs_from_manifest",
                return_value=[{"id": "adr-1"}],
            ),
            mock.patch.object(benchmark, "_load_batch_manifest", return_value=manifest),
            mock.patch.object(benchmark, "_save_batch_manifest"),
            mock.patch.object(
                benchmark, "collect_batch_job", side_effect=RuntimeError("download failed")
            ),
        ):
            with redirect_stdout(io.StringIO()):
                reports = benchmark.collect_batch_jobs(
                    [{"id": "adr-1"}], {}, targets=[target]
                )
        self.assertEqual(reports[job_key]["status"], "collection_failed")
        self.assertEqual(reports[job_key]["error"], "download failed")

    def test_pending_calls_for_complete_result_is_zero(self):
        eval_adrs = [{"id": "adr-1"}]
        valid_row = {
            "adr_id": "adr-1",
            "result_schema_version": benchmark.RESULT_SCHEMA_VERSION,
            "model_key": "gpt-5.5",
            "strategy": "zero_shot",
            "protocol_fingerprint": benchmark.protocol_fingerprint(
                "gpt-5.5", "zero_shot"
            ),
            "input_was_truncated": False,
            "adr_input_chars": 1,
            "adr_source_chars": 1,
            "adr_input_sha256": "input",
            "prompt_sha256": "prompt",
            "predicted_sc_checks": {f"C{i}": False for i in range(1, 8)},
            "predicted_dq_scores": {f"Q{i}": 0 for i in range(1, 8)},
            "actual": "Not_Compliant",
            "correct": True,
            "predicted": "Not_Compliant",
            "parse_success": True,
            "criterion_parse_success": True,
        }
        valid_row.update(benchmark.rubric.prediction_fields(
            valid_row["predicted_sc_checks"], valid_row["predicted_dq_scores"]))
        with mock.patch.object(Path, "exists", return_value=True), mock.patch.object(
            Path, "read_text", return_value=json.dumps([[valid_row]])
        ):
            self.assertEqual(
                benchmark._pending_calls_for_pair(
                    "gpt-5.5", "zero_shot", eval_adrs, 1
                ),
                0,
            )

    def test_validation_subset_is_deterministic_and_preserves_publication_manifest(self):
        publication_path = ROOT / "results" / "eval_set.json"
        before = hashlib.sha256(publication_path.read_bytes()).hexdigest()
        run_dir = ROOT / "results" / "runs" / "unit_validation_test"
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            benchmark.configure_run_paths("unit_validation")
            benchmark.RUN_DIR = run_dir
            benchmark.RUN_CONFIG_PATH = run_dir / "run_config.json"
            first = benchmark.configure_validation_mode(5, 3105)
            first_ids = [row["id"] for row in first["adrs"]]
            second = benchmark.configure_validation_mode(5, 3105)

            self.assertEqual(first_ids, [row["id"] for row in second["adrs"]])
            self.assertEqual(len(first_ids), 5)
            self.assertEqual(
                set(first["class_distribution"]), set(benchmark.CLASSES)
            )
            self.assertTrue(first["validation_mode"])
            self.assertFalse(first["publication_compatible"])
            self.assertEqual(
                benchmark.EVAL_SET_PATH, run_dir / "eval_set.json"
            )
            config = benchmark.ensure_run_configuration()
            self.assertEqual(config["evaluation_set_size"], 5)
            self.assertEqual(config["repetitions"], 3)
            self.assertFalse(config["publication_compatible"])
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        after = hashlib.sha256(publication_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_bare_cli_prints_help_without_starting_pipeline(self):
        with (
            mock.patch("sys.argv", ["adr_benchmark.py"]),
            mock.patch.object(benchmark, "fetch_adrs_from_github") as fetch,
            redirect_stdout(io.StringIO()) as output,
        ):
            benchmark.main()
        fetch.assert_not_called()
        self.assertIn("--validation-size", output.getvalue())

    def test_five_adr_validation_pipeline_merges_and_analyzes_offline(self):
        run_id = "unit_validation_e2e"
        run_dir = ROOT / "results" / "runs" / run_id
        shutil.rmtree(run_dir, ignore_errors=True)
        try:
            benchmark.configure_run_paths(run_id)
            benchmark.configure_mistral_variant("ministral-3-8b-api")
            benchmark.configure_gpt_reasoning_effort("medium")
            benchmark.configure_validation_mode(5, 3105)
            benchmark.ensure_run_configuration()

            adrs = benchmark.load_adrs()
            eval_adrs = benchmark._load_eval_adrs_from_manifest(adrs)
            ground_truth = json.loads(
                (ROOT / "results" / "human_ground_truth.json").read_text(
                    encoding="utf-8"
                )
            )
            examples = benchmark.build_few_shot_examples(
                adrs, ground_truth, eval_adrs
            )
            benchmark.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            for model_name in benchmark.MODELS:
                for strategy in benchmark.STRATEGIES:
                    repetitions = []
                    for _ in range(3):
                        rows = []
                        for adr in eval_adrs:
                            label = ground_truth[adr["id"]]
                            payload = benchmark._criterion_label_payload(label)
                            prediction = benchmark.extract_prediction(
                                json.dumps(payload)
                            )
                            prompt = benchmark.make_prompt(
                                adr["text"], strategy, examples
                            )
                            result = {
                                "accessed_at": "2026-09-14T00:00:00",
                                "model_metadata": benchmark.model_reproducibility_row(
                                    model_name
                                ),
                                "finish_reason": "stop",
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                                "latency": 0.01,
                            }
                            benchmark.attach_run_metadata(
                                result, model_name, strategy, adr["text"], prompt
                            )
                            rows.append(
                                benchmark.build_result_row(
                                    adr, label["overall"], prediction, result
                                )
                            )
                        repetitions.append(rows)
                    result_path = benchmark.RESULTS_DIR / (
                        f"{model_name}_{strategy}.json"
                    )
                    result_path.write_text(
                        json.dumps(repetitions, indent=2) + "\n",
                        encoding="utf-8",
                    )

            self.assertEqual(
                verify_validation_run.verify(run_id, 5, 3), []
            )
            with redirect_stdout(io.StringIO()):
                merged = benchmark.merge_results()
                metrics = benchmark.analyze(merged, ground_truth)
            self.assertEqual(len(merged), 4)
            self.assertEqual(
                sum(len(strategies) for strategies in merged.values()), 12
            )
            self.assertEqual(
                sum(len(strategies) for strategies in metrics.values()), 12
            )
            self.assertTrue(
                (run_dir / "analysis" / "metrics_summary.json").is_file()
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
