import os
import unittest
from unittest.mock import patch, MagicMock
import torch

from digohwelisgi.utils.model_utils import (
    get_best_model_config,
    get_model,
    get_best_model,
    _MODEL_CACHE,
)


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        _MODEL_CACHE.clear()

    def test_get_best_model_config_default(self):
        config = get_best_model_config()
        self.assertIn("repo", config)
        self.assertIn("revision", config)
        self.assertIsInstance(config["repo"], str)
        self.assertIsInstance(config["revision"], str)

    @patch("builtins.open")
    @patch("os.path.exists")
    def test_get_best_model_config_from_file(self, mock_exists, mock_open):
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = (
            '{"repo": "custom/best-repo", "revision": "rev-best"}'
        )
        mock_open.return_value = mock_file

        with patch(
            "json.load",
            return_value={"repo": "custom/best-repo", "revision": "rev-best"},
        ):
            config = get_best_model_config()
            self.assertEqual(config["repo"], "custom/best-repo")
            self.assertEqual(config["revision"], "rev-best")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_get_model_defaults(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        model, processor, device = get_model(device="cpu")

        self.assertEqual(model, mock_model)
        self.assertEqual(processor, mock_proc)
        self.assertEqual(device, "cpu")
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_get_model_and_get_best_model(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        m1, p1, d1 = get_best_model(device="cpu")
        self.assertEqual(m1, mock_model)
        self.assertEqual(p1, mock_proc)
        self.assertEqual(d1, "cpu")

        m2, p2, d2 = get_model("custom/repo", revision="rev123", device="cpu")
        self.assertEqual(m2, mock_model)
        mock_model_load.assert_called_with("custom/repo", revision="rev123")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_get_model_custom_processor_path(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        m, p, d = get_model(
            "custom/model", processor_path="custom/processor", device="cpu"
        )
        self.assertEqual(m, mock_model)
        self.assertEqual(p, mock_proc)
        mock_proc_load.assert_called_with("custom/processor")
        mock_model_load.assert_called_with("custom/model")

    @patch.dict(os.environ, {"HF_TOKEN": "token_from_env"}, clear=True)
    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_token_resolution_from_env(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        get_model("test/repo", device="cpu")
        mock_model_load.assert_called_with("test/repo", token="token_from_env")
        mock_proc_load.assert_called_with("test/repo", token="token_from_env")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_explicit_token(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        get_model("test/repo", token="explicit_token_123", device="cpu")
        mock_model_load.assert_called_with("test/repo", token="explicit_token_123")
        mock_proc_load.assert_called_with("test/repo", token="explicit_token_123")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_eval_mode_toggle(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        get_model("test/repo", eval_mode=False, device="cpu")
        mock_model.eval.assert_not_called()

        mock_model.reset_mock()
        get_model("test/repo_eval", eval_mode=True, device="cpu")
        mock_model.eval.assert_called_once()

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_caching_behavior(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        m1, p1, d1 = get_model("test/repo", revision="r1", device="cpu", use_cache=True)
        m2, p2, d2 = get_model("test/repo", revision="r1", device="cpu", use_cache=True)

        self.assertEqual(mock_model_load.call_count, 1)
        self.assertEqual(mock_proc_load.call_count, 1)
        self.assertIs(m1, m2)

        # Bypass cache
        m3, p3, d3 = get_model(
            "test/repo", revision="r1", device="cpu", use_cache=False
        )
        self.assertEqual(mock_model_load.call_count, 2)
        self.assertEqual(mock_proc_load.call_count, 2)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_device_auto_resolution_cpu(
        self, mock_proc, mock_model, mock_mps, mock_cuda
    ):
        mock_proc.return_value = MagicMock()
        mock_m = MagicMock()
        mock_model.return_value = mock_m

        _, _, dev = get_model("test/repo", use_cache=False)
        self.assertEqual(dev, "cpu")


if __name__ == "__main__":
    unittest.main()
