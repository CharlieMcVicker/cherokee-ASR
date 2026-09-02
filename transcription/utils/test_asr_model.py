import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from transcription.models.asr_model import CherokeeASRModel, ASRResult, WordConfidence
from transcription.utils.model_utils import _MODEL_CACHE
from transcription.inference.infer import (
    greedy_inference,
    infer_pcm_array,
    infer_single_audio,
    calculate_word_confidences,
    transcribe_audio_batch,
)


class TestDataStructures(unittest.TestCase):
    def test_word_confidence_dataclass_and_serialization(self):
        wc = WordConfidence(
            word="osiyo",
            confidence=0.98,
            start_time=0.1,
            end_time=0.5,
            chars=[
                {"char": "o", "confidence": 0.99, "start_time": 0.1, "alternatives": []}
            ],
        )
        self.assertEqual(wc.word, "osiyo")
        self.assertEqual(wc.confidence, 0.98)
        self.assertEqual(wc.start_time, 0.1)
        self.assertEqual(wc.end_time, 0.5)
        self.assertEqual(len(wc.chars), 1)

        d = wc.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["word"], "osiyo")
        self.assertEqual(d["confidence"], 0.98)
        self.assertEqual(d["start_time"], 0.1)
        self.assertEqual(d["end_time"], 0.5)
        self.assertEqual(len(d["chars"]), 1)

    def test_asr_result_dataclass_and_dict_access(self):
        wc = WordConfidence(
            word="osiyo",
            confidence=0.98,
            start_time=0.1,
            end_time=0.5,
        )
        result = ASRResult(
            text="osiyo",
            transcription="osiyo",
            confidence=0.98,
            words=[wc],
        )

        # Attribute access
        self.assertEqual(result.text, "osiyo")
        self.assertEqual(result.transcription, "osiyo")
        self.assertEqual(result.confidence, 0.98)
        self.assertEqual(len(result.words), 1)

        # Dict item access
        self.assertEqual(result["text"], "osiyo")
        self.assertEqual(result["transcription"], "osiyo")
        self.assertEqual(result["confidence"], 0.98)
        self.assertEqual(result["words"], [wc])

        # Dict get method
        self.assertEqual(result.get("text"), "osiyo")
        self.assertEqual(result.get("nonexistent_key", "default_val"), "default_val")

        # to_dict serialization
        res_dict = result.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertEqual(res_dict["text"], "osiyo")
        self.assertEqual(res_dict["words"][0]["word"], "osiyo")


class TestCherokeeASRModel(unittest.TestCase):
    def setUp(self):
        _MODEL_CACHE.clear()

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_pretrained_and_options(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        model = CherokeeASRModel.from_pretrained(
            "test/repo",
            revision="r1",
            processor_path="test/proc",
            device="cpu",
            token="secret_token",
            eval_mode=True,
            use_cache=True,
        )
        self.assertIsInstance(model, CherokeeASRModel)
        self.assertEqual(model.model, mock_model)
        self.assertEqual(model.processor, mock_proc)
        self.assertEqual(model.device, "cpu")
        mock_proc_load.assert_called_with(
            "test/proc", revision="r1", token="secret_token"
        )
        mock_model_load.assert_called_with(
            "test/repo", revision="r1", token="secret_token"
        )

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_config(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        # Valid config
        model = CherokeeASRModel.from_config(
            {"repo": "test/repo", "revision": "v1", "processor_path": "test/proc"},
            device="cpu",
        )
        self.assertIsInstance(model, CherokeeASRModel)

        # Invalid config missing repo
        with self.assertRaises(ValueError):
            CherokeeASRModel.from_config({}, device="cpu")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_get_best_model(self, mock_proc_load, mock_model_load):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        model = CherokeeASRModel.get_best_model(device="cpu")
        self.assertIsInstance(model, CherokeeASRModel)

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_pretrained_or_best_explicit_repo(
        self, mock_proc_load, mock_model_load
    ):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo="test/custom-repo", revision="rev1", device="cpu"
        )
        self.assertIsInstance(model, CherokeeASRModel)
        self.assertEqual(model.model, mock_model)
        self.assertEqual(model.processor, mock_proc)
        mock_model_load.assert_called_with("test/custom-repo", revision="rev1")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_pretrained_or_best_none_repo(
        self, mock_proc_load, mock_model_load
    ):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.return_value = mock_model

        model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=None, device="cpu"
        )
        self.assertIsInstance(model, CherokeeASRModel)
        self.assertEqual(model.model, mock_model)
        self.assertEqual(model.processor, mock_proc)

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_pretrained_or_best_fallback_on_explicit_failure(
        self, mock_proc_load, mock_model_load
    ):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.side_effect = [
            RuntimeError("Network or checkpoint error"),
            mock_model,
        ]

        model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo="invalid/missing-repo",
            fallback_repo="facebook/wav2vec2-base-960h",
            device="cpu",
        )
        self.assertIsInstance(model, CherokeeASRModel)
        self.assertEqual(mock_model_load.call_count, 2)
        mock_model_load.assert_called_with("facebook/wav2vec2-base-960h")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_pretrained_or_best_fallback_on_none_failure(
        self, mock_proc_load, mock_model_load
    ):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.side_effect = [
            RuntimeError("Best model unreachable"),
            mock_model,
        ]

        model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=None,
            fallback_repo="facebook/wav2vec2-base-960h",
            device="cpu",
        )
        self.assertIsInstance(model, CherokeeASRModel)
        self.assertEqual(mock_model_load.call_count, 2)
        mock_model_load.assert_called_with("facebook/wav2vec2-base-960h")

    @patch("transformers.Wav2Vec2ForCTC.from_pretrained")
    @patch("transformers.Wav2Vec2Processor.from_pretrained")
    def test_factory_from_pretrained_or_best_fallback_failure_raises(
        self, mock_proc_load, mock_model_load
    ):
        mock_proc = MagicMock()
        mock_proc_load.return_value = mock_proc
        mock_model_load.side_effect = [
            RuntimeError("Primary load failed"),
            RuntimeError("Fallback load failed"),
        ]

        with self.assertRaises(RuntimeError):
            CherokeeASRModel.from_pretrained_or_best(
                path_or_repo="invalid/repo",
                fallback_repo="facebook/wav2vec2-base-960h",
                device="cpu",
            )

    def test_preprocess_audio_types(self):
        # 1D numpy array
        np_1d = np.zeros(16000, dtype=np.float32)
        proc_np = CherokeeASRModel.preprocess_audio(np_1d)
        self.assertEqual(proc_np.shape, (16000,))
        self.assertEqual(proc_np.dtype, np.float32)

        # 2D stereo numpy array (converts to mono)
        np_2d = np.ones((16000, 2), dtype=np.float32)
        proc_stereo = CherokeeASRModel.preprocess_audio(np_2d)
        self.assertEqual(proc_stereo.shape, (16000,))

        # List of floats
        float_list = [0.0] * 16000
        proc_list = CherokeeASRModel.preprocess_audio(float_list)
        self.assertEqual(proc_list.shape, (16000,))

        # Bytes
        pcm_bytes = np_1d.tobytes()
        proc_bytes = CherokeeASRModel.preprocess_audio(pcm_bytes)
        self.assertEqual(proc_bytes.shape, (16000,))

        # 1D & 2D torch tensor
        t_1d = torch.zeros(16000, dtype=torch.float32)
        proc_t = CherokeeASRModel.preprocess_audio(t_1d)
        self.assertEqual(proc_t.shape, (16000,))

        t_2d = torch.zeros((16000, 2), dtype=torch.float32)
        proc_t2 = CherokeeASRModel.preprocess_audio(t_2d)
        self.assertEqual(proc_t2.shape, (16000,))

        # Resampling non-16kHz
        proc_resampled = CherokeeASRModel.preprocess_audio(np_1d, sample_rate=8000)
        self.assertEqual(proc_resampled.shape, (32000,))

        # Unsupported type
        with self.assertRaises(TypeError):
            CherokeeASRModel.preprocess_audio(12345)  # type: ignore

    @patch("soundfile.read")
    @patch("os.path.exists")
    def test_preprocess_audio_file(self, mock_exists, mock_sf_read):
        mock_exists.return_value = True
        # Mock 1D audio read at 16kHz
        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        proc_file = CherokeeASRModel.preprocess_audio("audio.wav")
        self.assertEqual(proc_file.shape, (16000,))

        # Mock stereo audio read at 44.1kHz (resamples)
        mock_sf_read.return_value = (np.zeros((44100, 2), dtype=np.float32), 44100)
        proc_resample = CherokeeASRModel.preprocess_audio("stereo.wav")
        self.assertEqual(proc_resample.shape, (16000,))

        # Non-existent file
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            CherokeeASRModel.preprocess_audio("nonexistent.wav")

    def test_procedural_inference_pipeline(self):
        mock_model = MagicMock()
        mock_proc = MagicMock()

        # Setup mock processor tokenizer and vocab
        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.word_delimiter_token_id = 4
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}

        # Mock processor __call__
        mock_inputs = MagicMock()
        mock_inputs.input_values = [np.zeros(16000, dtype=np.float32)]
        mock_proc.return_value = mock_inputs

        # Mock decode and batch_decode
        def mock_decode(ids):
            mapping = {0: "", 1: "a", 2: "d", 3: "l", 4: " "}
            return "".join([mapping.get(int(i), "") for i in ids])

        mock_proc.decode.side_effect = mock_decode
        mock_proc.batch_decode.side_effect = lambda sequences: [
            mock_decode(seq) for seq in sequences
        ]

        # Mock model forward logits [1, sequence_length=5, vocab_size=5]
        # Token ids sequence: [1 ('a'), 2 ('d'), 4 (' '), 3 ('l'), 1 ('a')] -> text "ad la"
        logits_tensor = torch.zeros((1, 5, 5))
        logits_tensor[0, 0, 1] = 10.0  # 'a'
        logits_tensor[0, 1, 2] = 10.0  # 'd'
        logits_tensor[0, 2, 4] = 10.0  # ' '
        logits_tensor[0, 3, 3] = 10.0  # 'l'
        logits_tensor[0, 4, 1] = 10.0  # 'a'

        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output

        asr_model = CherokeeASRModel(mock_model, mock_proc, device="cpu")

        # Test Layer 1: get_logits
        pcm = np.zeros(16000, dtype=np.float32)
        logits = asr_model.get_logits(pcm)
        self.assertEqual(logits.shape, (5, 5))

        # Test Layer 2: get_probabilities
        probs = asr_model.get_probabilities(logits)
        self.assertEqual(probs.shape, (5, 5))
        self.assertTrue(
            torch.allclose(torch.sum(probs, dim=-1), torch.ones(5), atol=1e-3)
        )

        # Test get_probabilities from audio input directly
        probs_from_audio = asr_model.get_probabilities(pcm)
        self.assertEqual(probs_from_audio.shape, (5, 5))

        # Test Layer 3: get_word_confidences (using probs and logits)
        words = asr_model.get_word_confidences(probs)
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0].word, "ad")
        self.assertEqual(words[1].word, "la")
        self.assertGreater(words[0].confidence, 0.9)

        # Test Layer 4: decode
        result = asr_model.decode(logits, compute_word_confidences=True)
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.text, "ad la")
        self.assertEqual(result["transcription"], "ad la")
        self.assertGreater(result.confidence, 0.9)
        self.assertEqual(len(result.words), 2)

        # Test decode without word confidences
        result_no_words = asr_model.decode(logits, compute_word_confidences=False)
        self.assertEqual(result_no_words.text, "ad la")
        self.assertEqual(len(result_no_words.words), 0)

        # Test transcribe end-to-end
        res_transcribe = asr_model.transcribe(pcm)
        self.assertIsInstance(res_transcribe, ASRResult)
        self.assertEqual(res_transcribe.text, "ad la")

    def test_mps_fallback_in_get_logits(self):
        mock_model = MagicMock()
        mock_proc = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.input_values = [np.zeros(16000, dtype=np.float32)]
        mock_proc.return_value = mock_inputs

        logits_tensor = torch.zeros((1, 5, 5))
        mock_output = MagicMock()
        mock_output.logits = logits_tensor

        # First call on MPS raises NotImplementedError, second call on CPU succeeds
        mock_model.side_effect = [NotImplementedError("MPS not supported"), mock_output]

        asr_model = CherokeeASRModel(mock_model, mock_proc, device="mps")
        logits = asr_model.get_logits(np.zeros(16000, dtype=np.float32))
        self.assertEqual(asr_model.device, "cpu")
        self.assertEqual(logits.shape, (5, 5))

        # Test .to(device) method
        asr_model.to("cpu")
        self.assertEqual(asr_model.device, "cpu")
        mock_model.to.assert_called_with("cpu")

    def test_transcribe_batch(
        self,
    ):
        mock_model = MagicMock()
        mock_proc = MagicMock()

        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.word_delimiter_token_id = 4
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}

        def mock_decode(ids):
            mapping = {0: "", 1: "a", 2: "d", 3: "l", 4: " "}
            res = []
            prev = None
            for i in ids:
                val = int(i)
                if val != prev:
                    res.append(mapping.get(val, ""))
                    prev = val
            return "".join(res)

        mock_proc.decode.side_effect = mock_decode
        mock_proc.batch_decode.side_effect = lambda sequences: [
            mock_decode(seq) for seq in sequences
        ]

        # Mock batch processor
        mock_batch_inputs = MagicMock()
        mock_batch_inputs.input_values = torch.zeros((2, 16000))
        mock_batch_inputs.attention_mask = torch.ones((2, 16000))
        mock_proc.return_value = mock_batch_inputs

        logits_tensor = torch.zeros((2, 5, 5))
        logits_tensor[:, :, 1] = 10.0  # all 'a'
        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output
        mock_model._get_feat_extract_output_lengths.return_value = torch.tensor([5, 5])

        asr_model = CherokeeASRModel(mock_model, mock_proc, device="cpu")

        batch_pcm = [
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ]
        results = asr_model.transcribe_batch(batch_pcm, batch_size=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ASRResult)
        self.assertEqual(results[0].text, "a")

    def test_transcribe_batch_oom_fallback(self):
        mock_model = MagicMock()
        mock_proc = MagicMock()

        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.word_delimiter_token_id = 4
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}
        mock_proc.decode.side_effect = lambda ids: "a"
        mock_proc.batch_decode.side_effect = lambda sequences: ["a" for _ in sequences]

        mock_batch_inputs = MagicMock()
        mock_batch_inputs.input_values = torch.zeros((2, 16000))
        mock_proc.return_value = mock_batch_inputs

        # Batch forward raises CUDA OOM error
        mock_output = MagicMock()
        logits_tensor = torch.zeros((1, 5, 5))
        logits_tensor[:, :, 1] = 10.0
        mock_output.logits = logits_tensor

        mock_model.side_effect = [
            RuntimeError("CUDA out of memory"),
            mock_output,
            mock_output,
        ]

        asr_model = CherokeeASRModel(mock_model, mock_proc, device="cpu")
        batch_pcm = [
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ]
        results = asr_model.transcribe_batch(batch_pcm, batch_size=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "a")


class TestInferBackwardsCompatibility(unittest.TestCase):
    """
    Verify backwards compatibility of infer.py functions.
    """

    def test_greedy_inference_2d_and_3d(self):
        mock_proc = MagicMock()
        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1}
        mock_proc.batch_decode.return_value = ["osiyo"]

        # 2D logits [seq_len, vocab_size]
        logits_2d = torch.zeros((5, 2))
        logits_2d[:, 1] = 10.0
        res_2d = greedy_inference(logits_2d, mock_proc)
        self.assertIsInstance(res_2d, dict)
        assert isinstance(res_2d, dict)
        self.assertEqual(res_2d["transcription"], "osiyo")
        self.assertIn("syllabary", res_2d)
        self.assertIn("confidence", res_2d)

        # 3D logits [batch_size, seq_len, vocab_size]
        mock_proc.batch_decode.return_value = ["osiyo", "wado"]
        logits_3d = torch.zeros((2, 5, 2))
        logits_3d[:, :, 1] = 10.0
        res_3d = greedy_inference(logits_3d, mock_proc)
        self.assertIsInstance(res_3d, list)
        assert isinstance(res_3d, list)
        self.assertEqual(len(res_3d), 2)
        self.assertEqual(res_3d[0]["transcription"], "osiyo")
        self.assertEqual(res_3d[1]["transcription"], "wado")

    def test_calculate_word_confidences_compat(self):
        mock_proc = MagicMock()
        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.word_delimiter_token_id = 4
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "|": 4}
        mock_proc.decode.side_effect = lambda ids: (
            "a" if ids == [1] else ("d" if ids == [2] else "")
        )

        probs = np.zeros((3, 5))
        probs[0, 1] = 0.99
        probs[1, 2] = 0.95
        probs[2, 0] = 0.99  # PAD

        pred_ids = np.array([1, 2, 0])
        word_details = calculate_word_confidences(probs, pred_ids, mock_proc)
        self.assertIsInstance(word_details, list)
        self.assertEqual(len(word_details), 1)
        self.assertEqual(word_details[0]["word"], "ad")
        self.assertIn("start_time", word_details[0])
        self.assertIn("end_time", word_details[0])
        self.assertIn("chars", word_details[0])

    def test_infer_pcm_array_compat(self):
        mock_model = MagicMock()
        mock_proc = MagicMock()
        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1}
        mock_proc.batch_decode.return_value = ["osiyo"]

        mock_inputs = MagicMock()
        mock_inputs.input_values = [np.zeros(16000, dtype=np.float32)]
        mock_proc.return_value = mock_inputs

        logits_tensor = torch.zeros((1, 5, 2))
        logits_tensor[:, :, 1] = 10.0
        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output

        pcm = np.zeros(16000, dtype=np.float32)
        res = infer_pcm_array(mock_model, mock_proc, pcm, device="cpu")
        self.assertIsInstance(res, list)
        self.assertEqual(res[0]["transcription"], "osiyo")

    @patch("soundfile.read")
    @patch("os.path.exists")
    def test_infer_single_audio_compat(self, mock_exists, mock_sf_read):
        mock_exists.return_value = True
        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_model = MagicMock()
        mock_proc = MagicMock()
        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1}
        mock_proc.batch_decode.return_value = ["osiyo"]

        mock_inputs = MagicMock()
        mock_inputs.input_values = [np.zeros(16000, dtype=np.float32)]
        mock_proc.return_value = mock_inputs

        logits_tensor = torch.zeros((1, 5, 2))
        logits_tensor[:, :, 1] = 10.0
        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output

        res = infer_single_audio(mock_model, mock_proc, "audio.wav", device="cpu")
        self.assertIsInstance(res, list)
        self.assertEqual(res[0]["transcription"], "osiyo")

    @patch("soundfile.read")
    @patch("os.path.exists")
    def test_transcribe_audio_batch_compat(self, mock_exists, mock_sf_read):
        mock_exists.return_value = True
        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_model = MagicMock()
        mock_proc = MagicMock()
        mock_proc.tokenizer.pad_token_id = 0
        mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1}
        mock_proc.batch_decode.return_value = ["osiyo", "osiyo"]

        mock_inputs = MagicMock()
        mock_inputs.input_values = torch.zeros((2, 16000))
        mock_inputs.attention_mask = torch.ones((2, 16000))
        mock_proc.return_value = mock_inputs

        logits_tensor = torch.zeros((2, 5, 2))
        logits_tensor[:, :, 1] = 10.0
        mock_output = MagicMock()
        mock_output.logits = logits_tensor
        mock_model.return_value = mock_output
        mock_model._get_feat_extract_output_lengths.return_value = torch.tensor([5, 5])

        results = transcribe_audio_batch(
            mock_model, mock_proc, ["a1.wav", "a2.wav"], device="cpu", batch_size=2
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["transcription"], "osiyo")


if __name__ == "__main__":
    unittest.main()
