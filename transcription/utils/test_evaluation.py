import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import torch
import numpy as np
import pandas as pd

from transcription.cherokee.models import CherokeeASRModel, ASRResult
from transcription.utils.evaluation import (
    get_eval_device,
    clean_eval_cache,
    yield_local_checkpoints,
    yield_hf_revisions,
    yield_single_checkpoint,
    run_evaluation,
)


class TestEvaluationUtils(unittest.TestCase):
    def test_get_eval_device(self):
        device = get_eval_device()
        self.assertIn(device, ["mps", "cuda", "cpu"])

    def test_clean_eval_cache(self):
        clean_eval_cache("cpu")
        clean_eval_cache("cuda")
        clean_eval_cache("mps")

    def test_yield_local_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories with config.json and vocab.json
            ckpt1 = os.path.join(tmpdir, "checkpoint-100")
            ckpt2 = os.path.join(tmpdir, "checkpoint-200")
            os.makedirs(ckpt1)
            os.makedirs(ckpt2)
            with open(os.path.join(ckpt1, "config.json"), "w") as f:
                json.dump({}, f)
            with open(os.path.join(ckpt1, "vocab.json"), "w") as f:
                json.dump({}, f)
            with open(os.path.join(ckpt2, "config.json"), "w") as f:
                json.dump({}, f)

            with patch(
                "transcription.cherokee.models.CherokeeASRModel.from_pretrained"
            ) as mock_from_pretrained:
                mock_model = MagicMock(spec=CherokeeASRModel)
                mock_model.processor = MagicMock()
                mock_from_pretrained.return_value = mock_model

                results = list(yield_local_checkpoints(tmpdir))
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0][0], "checkpoint-100")
                self.assertEqual(results[1][0], "checkpoint-200")
                self.assertEqual(results[0][1], mock_model)

    def test_yield_hf_revisions(self):
        revisions = [("rev1", "hash1234567890"), ("rev2", "hash0987654321")]
        with patch(
            "transcription.cherokee.models.CherokeeASRModel.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock(spec=CherokeeASRModel)
            mock_model.processor = MagicMock()
            mock_from_pretrained.return_value = mock_model

            results = list(yield_hf_revisions("test/repo", revisions, token="tok"))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0], "rev1 (hash123)")
            self.assertEqual(results[0][1], mock_model)

    def test_yield_single_checkpoint(self):
        with patch(
            "transcription.cherokee.models.CherokeeASRModel.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock(spec=CherokeeASRModel)
            mock_model.processor = MagicMock()
            mock_from_pretrained.return_value = mock_model

            results = list(yield_single_checkpoint("test/path", token="tok"))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0], "checkpoint")
            self.assertEqual(results[0][1], mock_model)

    def test_run_evaluation(self):
        mock_proc = MagicMock()
        mock_proc.pad.return_value = {
            "input_values": torch.zeros((2, 16000), dtype=torch.float32),
            "attention_mask": torch.ones((2, 16000), dtype=torch.long),
        }

        mock_raw_model = MagicMock()
        # Mock forward pass
        mock_logits = torch.zeros((2, 50, 32), dtype=torch.float32)
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_raw_model.return_value = mock_output
        mock_raw_model._get_feat_extract_output_lengths.return_value = torch.tensor(
            [50, 50]
        )
        mock_raw_model.to.return_value = mock_raw_model

        asr_model = CherokeeASRModel(mock_raw_model, mock_proc, device="cpu")

        # Mock decode
        with patch.object(asr_model, "decode") as mock_decode:
            mock_decode.return_value = ASRResult(
                text="osiyo", transcription="osiyo", confidence=0.99
            )

            test_ds = [
                {
                    "input_values": np.zeros(16000, dtype=np.float32),
                    "sentence": "osiyo",
                },
                {
                    "input_values": np.zeros(16000, dtype=np.float32),
                    "sentence": "osiyo",
                },
            ]

            generator = [("test_ckpt", asr_model, asr_model.processor, "path/test")]

            rows_by_ckpt, ranking_df = run_evaluation(generator, test_ds, batch_size=2)

            self.assertIn("test_ckpt", rows_by_ckpt)
            self.assertEqual(len(rows_by_ckpt["test_ckpt"]), 2)
            self.assertFalse(ranking_df.empty)
            self.assertEqual(ranking_df.iloc[0]["checkpoint"], "test_ckpt")
            self.assertEqual(ranking_df.iloc[0]["median_wer_greedy"], 0.0)
            self.assertEqual(ranking_df.iloc[0]["agg_wer_greedy"], 0.0)
            self.assertEqual(ranking_df.iloc[0]["agg_wer_greedy_masked"], 0.0)


if __name__ == "__main__":
    unittest.main()
