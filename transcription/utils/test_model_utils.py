import unittest
from unittest.mock import patch, MagicMock
from transcription.utils.model_utils import (
    get_best_model_config,
    get_model,
    get_best_model,
    _MODEL_CACHE,
)


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        _MODEL_CACHE.clear()

    def test_get_best_model_config(self):
        config = get_best_model_config()
        self.assertIn("repo", config)
        self.assertIn("revision", config)
        self.assertIsInstance(config["repo"], str)
        self.assertIsInstance(config["revision"], str)

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

        m2, p2, d2 = get_model("custom/repo", revision="rev123", device="cpu")
        self.assertEqual(m2, mock_model)
        mock_model_load.assert_called_with("custom/repo", revision="rev123")

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


if __name__ == "__main__":
    unittest.main()
