# -*- coding: utf-8 -*-
"""
test_batch_inference_aligner.py

Unit and integration tests for batch_inference_aligner.py module.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from transcription.syllabary_enrichment.batch_inference_aligner import (
    load_manifest,
    process_batch_inference_and_alignment,
)


class TestBatchInferenceAligner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)

        self.json_manifest = os.path.join(self.test_dir.name, "test_manifest.json")
        self.jsonl_manifest = os.path.join(self.test_dir.name, "test_manifest.jsonl")
        self.output_cache = os.path.join(self.test_dir.name, "cache.json")

        self.sample_records = [
            {
                "audio_filepath": "/path/to/audio1.wav",
                "syllabary_text": "<ctrl42> Ꭳ",
                "emitted_text": "no o",
            },
            {
                "audio_filepath": "/path/to/audio2.wav",
                "syllabary_text": "ᎠᏓᎴᏂᏍᎬ",
                "emitted_text": "adaleniskv",
            },
        ]

        with open(self.json_manifest, "w", encoding="utf-8") as f:
            json.dump(self.sample_records, f, ensure_ascii=False)

        with open(self.jsonl_manifest, "w", encoding="utf-8") as f:
            for r in self.sample_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_load_manifest_json(self):
        records = load_manifest(self.json_manifest)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["audio_filepath"], "/path/to/audio1.wav")
        self.assertEqual(records[0]["syllabary_text"], "<ctrl42> Ꭳ")
        self.assertEqual(records[0]["emitted_text"], "no o")

    def test_load_manifest_jsonl(self):
        records = load_manifest(self.jsonl_manifest)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["syllabary_text"], "ᎠᏓᎴᏂᏍᎬ")
        self.assertEqual(records[1]["emitted_text"], "adaleniskv")

    def test_process_batch_inference_and_alignment_with_precomputed_emitted(self):
        # All records already have emitted_text, so GPU inference is not triggered
        records = process_batch_inference_and_alignment(
            manifest_path=self.json_manifest,
            output_cache_path=self.output_cache,
            force_recompute=False,
        )

        self.assertEqual(len(records), 2)
        self.assertTrue(os.path.exists(self.output_cache))

        # Check alignment pairs produced
        self.assertIn("aligned_pairs", records[0])
        aligned_pairs_0 = records[0]["aligned_pairs"]
        # Expected tuples [(syllabary_char, asr_text)]
        syllabary_chars = [p[0] for p in aligned_pairs_0]
        self.assertIn("Ꭳ", syllabary_chars)

    def test_caching_and_force_recompute(self):
        # 1. First run creates cache file
        records_first = process_batch_inference_and_alignment(
            manifest_path=self.json_manifest,
            output_cache_path=self.output_cache,
            force_recompute=False,
        )
        self.assertTrue(os.path.exists(self.output_cache))

        # Modify cache file content on disk to test reloading
        with open(self.output_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
        data[0]["test_marker"] = "cached_version"
        with open(self.output_cache, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        # 2. Second run without force_recompute should load from cache file directly
        records_cached = process_batch_inference_and_alignment(
            manifest_path=self.json_manifest,
            output_cache_path=self.output_cache,
            force_recompute=False,
        )
        self.assertEqual(records_cached[0].get("test_marker"), "cached_version")

        # 3. Third run with force_recompute=True should ignore cache file and recompute
        records_recomputed = process_batch_inference_and_alignment(
            manifest_path=self.json_manifest,
            output_cache_path=self.output_cache,
            force_recompute=True,
        )
        self.assertNotIn("test_marker", records_recomputed[0])

    @patch(
        "transcription.syllabary_enrichment.batch_inference_aligner.run_batch_inference"
    )
    def test_gpu_inference_triggered_when_emitted_text_missing(self, mock_run_batch):
        mock_run_batch.return_value = ["no o", "adaleniskv"]

        # Create manifest with missing emitted_text
        no_emitted_manifest = os.path.join(self.test_dir.name, "no_emitted.json")
        records_input = [
            {"audio_filepath": "/path/1.wav", "syllabary_text": "<ctrl42> Ꭳ"},
            {"audio_filepath": "/path/2.wav", "syllabary_text": "ᎠᏓᎴᏂᏍᎬ"},
        ]
        with open(no_emitted_manifest, "w", encoding="utf-8") as f:
            json.dump(records_input, f, ensure_ascii=False)

        records = process_batch_inference_and_alignment(
            manifest_path=no_emitted_manifest,
            output_cache_path=self.output_cache,
            force_recompute=True,
        )

        mock_run_batch.assert_called_once()
        self.assertEqual(records[0]["emitted_text"], "no o")
        self.assertEqual(records[1]["emitted_text"], "adaleniskv")
        self.assertIn("aligned_pairs", records[0])


if __name__ == "__main__":
    unittest.main()
