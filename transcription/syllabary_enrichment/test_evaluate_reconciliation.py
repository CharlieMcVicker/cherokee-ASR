# -*- coding: utf-8 -*-
"""
test_evaluate_reconciliation.py

Unit tests for evaluation framework module evaluate_reconciliation.py.
"""

import os
import json
import tempfile
import unittest

from transcription.syllabary_enrichment.evaluate_reconciliation import (
    calculate_cer,
    calculate_relative_improvement,
    run_evaluation,
    print_summary_table,
)


class TestEvaluateReconciliation(unittest.TestCase):

    def test_calculate_cer(self):
        # Exact match
        self.assertEqual(calculate_cer("athaleniskv", "athaleniskv"), 0.0)
        # Empty string handling
        self.assertGreaterEqual(calculate_cer("", "athaleniskv"), 0.0)
        # Differences
        cer_diff = calculate_cer("adalenisgv", "athaleniskv")
        self.assertGreater(cer_diff, 0.0)

    def test_calculate_relative_improvement(self):
        # 0 raw_cer -> 0.0
        self.assertEqual(calculate_relative_improvement(0.0, 0.0), 0.0)
        # Improvement from 0.50 CER to 0.25 CER -> 50% relative improvement
        self.assertAlmostEqual(calculate_relative_improvement(0.50, 0.25), 50.0)
        # Degraded performance from 0.20 CER to 0.30 CER -> -50%
        self.assertAlmostEqual(calculate_relative_improvement(0.20, 0.30), -50.0)

    def test_run_evaluation_splits_and_artifact(self):
        sample_records = [
            {
                "id": "rec_train_1",
                "split": "train",
                "syllabary_text": "ᎠᏓᎴᏂᏍᎬ",
                "base_transliteration": "adalenisgv",
                "emitted_text": "athaleniskv",
                "target_phonetics": "athaleniskv",
                "aligned_pairs": [
                    ("Ꭰ", "a"),
                    ("Ꮣ", "tha"),
                    ("Ꮄ", "le"),
                    ("Ꮒ", "ni"),
                    ("Ꮝ", "s"),
                    ("Ꭼ", "kv"),
                ],
            },
            {
                "id": "rec_val_1",
                "split": "validation",
                "syllabary_text": "ᎣᏏᏲ",
                "base_transliteration": "osiyo",
                "emitted_text": "osiyo",
                "target_phonetics": "osiyo",
                "aligned_pairs": [("Ꭳ", "o"), ("Ꮟ", "si"), ("Ᏺ", "yo")],
            },
            {
                "id": "rec_test_1",
                "split": "test",
                "syllabary_text": "ᏣᎳᎩ",
                "base_transliteration": "tsalagi",
                "emitted_text": "tsalaki",
                "target_phonetics": "tsalaki",
                "aligned_pairs": [("Ꮳ", "tsa"), ("Ꮃ", "la"), ("Ꭹ", "ki")],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "eval_results.json")
            res = run_evaluation(sample_records, output_artifact_path=artifact_path)

            # Check summary structure
            summary = res["summary"]
            self.assertIn("train", summary)
            self.assertIn("validation", summary)
            self.assertIn("test", summary)
            self.assertIn("overall", summary)

            self.assertEqual(summary["train"]["count"], 1)
            self.assertEqual(summary["validation"]["count"], 1)
            self.assertEqual(summary["test"]["count"], 1)
            self.assertEqual(summary["overall"]["count"], 3)

            # Check artifact existence and valid contents
            self.assertTrue(os.path.exists(artifact_path))
            with open(artifact_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            self.assertIn("summary", saved_data)
            self.assertIn("records", saved_data)
            self.assertEqual(len(saved_data["records"]), 3)

            # Verify item key contents
            rec0 = saved_data["records"][0]
            self.assertEqual(rec0["record_id"], "rec_train_1")
            self.assertIn("raw_cer", rec0)
            self.assertIn("reconciled_cer", rec0)
            self.assertIn("delta_cer", rec0)
            self.assertIn("reconciled_phonetics", rec0)

    def test_print_summary_table(self):
        summary = {
            "train": {
                "count": 10,
                "raw_cer": 0.20,
                "reconciled_cer": 0.10,
                "delta_cer": 50.0,
            },
            "validation": {
                "count": 5,
                "raw_cer": 0.25,
                "reconciled_cer": 0.15,
                "delta_cer": 40.0,
            },
            "test": {
                "count": 5,
                "raw_cer": 0.30,
                "reconciled_cer": 0.15,
                "delta_cer": 50.0,
            },
            "overall": {
                "count": 20,
                "raw_cer": 0.24,
                "reconciled_cer": 0.125,
                "delta_cer": 47.92,
            },
        }
        # Smoke test to ensure printing works without throwing errors
        print_summary_table(summary)


if __name__ == "__main__":
    unittest.main()
