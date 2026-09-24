import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from transcription.core.models.output import ModelOutput
from transcription.core.models.inference import (
    compute_audio_cache_key,
    infer_emissions,
    infer_emissions_batch,
    infer_emissions_sliding_window,
    preprocess_audio,
)

from transcription.core.models.model import ASRModel
from transcription.cherokee.models import CherokeeASRModel


class TestModelOutput(unittest.TestCase):
    def setUp(self):
        self.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}
        # 5 frames, 5 vocab items
        # Frame 0: 1 ('a')
        # Frame 1: 1 ('a') - repeat collapsed
        # Frame 2: 2 ('d')
        # Frame 3: 4 ('|') - space
        # Frame 4: 3 ('l')
        lpz = np.full((5, 5), -10.0, dtype=np.float32)
        lpz[0, 1] = 0.0
        lpz[1, 1] = 0.0
        lpz[2, 2] = 0.0
        lpz[3, 4] = 0.0
        lpz[4, 3] = 0.0
        self.lpz = lpz

    def test_decode_tokens_and_greedy(self):
        output = ModelOutput(
            lpz=self.lpz,
            vocab=self.vocab,
            frame_duration_sec=0.02,
        )
        tokens = output.decode_tokens(collapse_repeats=True, remove_pad=True)
        self.assertEqual(tokens, ["a", "d", " ", "l"])

        greedy_text = output.decode_greedy()
        self.assertEqual(greedy_text, "ad l")

    def test_save_and_load_npz(self, tmp_path_str=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "output.npz"
            output = ModelOutput(
                lpz=self.lpz,
                vocab=self.vocab,
                frame_duration_sec=0.02,
                metadata={"test_key": "test_val"},
            )
            output.save(dest)
            self.assertTrue(dest.exists())

            loaded = ModelOutput.load(dest)
            np.testing.assert_allclose(loaded.lpz, output.lpz)
            self.assertEqual(loaded.token_to_id, output.token_to_id)
            self.assertEqual(loaded.frame_duration_sec, 0.02)
            self.assertEqual(loaded.metadata.get("test_key"), "test_val")
            self.assertEqual(loaded.decode_greedy(), "ad l")


class TestStandaloneInference(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_proc = MagicMock()

        self.mock_proc.tokenizer.pad_token_id = 0
        self.mock_proc.tokenizer.word_delimiter_token_id = 4
        self.mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}

        def mock_proc_call(speech, **kwargs):
            inputs = MagicMock()
            if (
                isinstance(speech, list)
                and len(speech) > 0
                and isinstance(speech[0], np.ndarray)
            ):
                # batched
                inputs.input_values = torch.zeros(
                    (len(speech), 16000), dtype=torch.float32
                )
                inputs.attention_mask = torch.ones(
                    (len(speech), 16000), dtype=torch.long
                )
            else:
                inputs.input_values = [np.zeros(16000, dtype=np.float32)]
            return inputs

        self.mock_proc.side_effect = mock_proc_call

        logits_tensor = torch.full((1, 4, 5), -10.0)
        logits_tensor[0, 0, 1] = 10.0  # 'a'
        logits_tensor[0, 1, 2] = 10.0  # 'd'
        logits_tensor[0, 2, 4] = 10.0  # '|'
        logits_tensor[0, 3, 3] = 10.0  # 'l'

        mock_out = MagicMock()
        mock_out.logits = logits_tensor
        self.mock_model.return_value = mock_out

    def test_infer_emissions_single_and_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            pcm = np.zeros(16000, dtype=np.float32)
            out1 = infer_emissions(
                model=self.mock_model,
                processor=self.mock_proc,
                audio_input=pcm,
                cache_dir=tmp_dir,
                model_identifier="test_model",
            )
            self.assertIsInstance(out1, ModelOutput)
            self.assertEqual(out1.decode_greedy(), "ad l")
            self.assertEqual(self.mock_model.call_count, 1)

            # Check cache hit bypasses model forward pass
            out2 = infer_emissions(
                model=self.mock_model,
                processor=self.mock_proc,
                audio_input=pcm,
                cache_dir=tmp_dir,
                model_identifier="test_model",
            )
            self.assertEqual(
                self.mock_model.call_count, 1
            )  # No additional forward pass!
            self.assertEqual(out2.decode_greedy(), "ad l")
            np.testing.assert_allclose(out1.lpz, out2.lpz)

    def test_infer_emissions_batch(self):
        # Set up batch forward pass
        batch_logits = torch.full((2, 4, 5), -10.0)
        batch_logits[:, 0, 1] = 10.0  # 'a'
        batch_logits[:, 1, 2] = 10.0  # 'd'
        batch_logits[:, 2, 4] = 10.0  # '|'
        batch_logits[:, 3, 3] = 10.0  # 'l'
        mock_out = MagicMock()
        mock_out.logits = batch_logits
        self.mock_model.return_value = mock_out
        self.mock_model._get_feat_extract_output_lengths.return_value = torch.tensor(
            [4, 4]
        )

        pcm_list = [
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ]
        outputs = infer_emissions_batch(
            model=self.mock_model,
            processor=self.mock_proc,
            audio_inputs=pcm_list,
            batch_size=2,
            model_identifier="test_batch_model",
        )
        self.assertEqual(len(outputs), 2)
        self.assertIsInstance(outputs[0], ModelOutput)
        self.assertEqual(outputs[0].decode_greedy(), "ad l")
        self.assertEqual(outputs[1].decode_greedy(), "ad l")


class TestASRModelWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_proc = MagicMock()
        self.mock_proc.tokenizer.pad_token_id = 0
        self.mock_proc.tokenizer.word_delimiter_token_id = 4
        self.mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}

        def mock_proc_call(speech, **kwargs):
            inputs = MagicMock()
            if (
                isinstance(speech, list)
                and len(speech) > 0
                and isinstance(speech[0], np.ndarray)
            ):
                inputs.input_values = torch.zeros(
                    (len(speech), 16000), dtype=torch.float32
                )
                inputs.attention_mask = torch.ones(
                    (len(speech), 16000), dtype=torch.long
                )
            else:
                inputs.input_values = [np.zeros(16000, dtype=np.float32)]
            return inputs

        self.mock_proc.side_effect = mock_proc_call

        logits_tensor = torch.full((1, 4, 5), -10.0)
        logits_tensor[0, 0, 1] = 10.0
        logits_tensor[0, 1, 2] = 10.0
        logits_tensor[0, 2, 4] = 10.0
        logits_tensor[0, 3, 3] = 10.0

        mock_out = MagicMock()
        mock_out.logits = logits_tensor
        self.mock_model.return_value = mock_out

    def test_clean_infer_interface(self):
        asr_model = ASRModel(
            model=self.mock_model,
            processor=self.mock_proc,
            device="cpu",
            model_name="clean_model",
        )
        pcm = np.zeros(16000, dtype=np.float32)
        out = asr_model.infer(pcm)
        self.assertIsInstance(out, ModelOutput)
        self.assertEqual(out.decode_greedy(), "ad l")

    def test_cherokee_asr_model_inherits_infer(self):
        cherokee_model = CherokeeASRModel(
            model=self.mock_model,
            processor=self.mock_proc,
            device="cpu",
            model_name="cherokee_test",
        )
        pcm = np.zeros(16000, dtype=np.float32)
        out = cherokee_model.infer(pcm)
        self.assertIsInstance(out, ModelOutput)
        self.assertEqual(out.decode_greedy(), "ad l")

    def test_infer_sliding_window_delegation(self):
        asr_model = ASRModel(
            model=self.mock_model,
            processor=self.mock_proc,
            device="cpu",
            model_name="clean_model",
        )
        pcm = np.zeros(16000 * 2, dtype=np.float32)
        out = asr_model.infer_sliding_window(pcm, chunk_seconds=5.0, margin_seconds=0.5)
        self.assertIsInstance(out, ModelOutput)
        self.assertEqual(out.decode_greedy(), "ad l")


class TestSlidingWindowInference(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_proc = MagicMock()
        self.mock_proc.tokenizer.pad_token_id = 0
        self.mock_proc.tokenizer.word_delimiter_token_id = 4
        self.mock_proc.tokenizer.vocab = {"[PAD]": 0, "a": 1, "d": 2, "l": 3, "|": 4}

        def mock_proc_call(speech, **kwargs):
            inputs = MagicMock()
            inputs.input_values = [np.array(speech, dtype=np.float32)]
            return inputs

        self.mock_proc.side_effect = mock_proc_call

        def mock_model_call(input_tensor):
            # 16000 samples -> 50 frames (320 samples per frame)
            n_samples = input_tensor.shape[1]
            n_frames = max(1, n_samples // 320)
            logits = torch.full((1, n_frames, 5), -10.0)
            logits[0, :, 1] = 10.0  # 'a' on all frames
            out = MagicMock()
            out.logits = logits
            return out

        self.mock_model.side_effect = mock_model_call

    def test_short_audio_single_pass(self):
        # Audio length 2s <= chunk_seconds 5s
        pcm = np.zeros(16000 * 2, dtype=np.float32)
        out = infer_emissions_sliding_window(
            model=self.mock_model,
            processor=self.mock_proc,
            audio_input=pcm,
            chunk_seconds=5.0,
            margin_seconds=1.0,
            model_identifier="test_sw_short",
        )
        self.assertIsInstance(out, ModelOutput)
        self.assertEqual(self.mock_model.call_count, 1)
        self.assertEqual(out.decode_greedy(), "a")
        self.assertEqual(out.lpz.shape[0], 100)  # 2s * 50 fps = 100 frames

    def test_long_audio_multi_chunk_margin_trimming(self):
        # Audio length 6s with chunk_seconds=3s, margin_seconds=0.5s
        # step = 3 - 2*0.5 = 2s
        # Chunk 1: [0s, 3s] -> 150 frames. is_first: left_trim=0, right_trim=25 -> frames [0, 125] (125 frames, covers 0s-2.5s)
        # Chunk 2: [2s, 5s] -> 150 frames. is_mid: left_trim=25, right_trim=25 -> frames [25, 125] (100 frames, covers 2.5s-4.5s)
        # Chunk 3: [4s, 6s] -> 100 frames. is_last: left_trim=25, right_trim=0 -> frames [25, 100] (75 frames, covers 4.5s-6.0s)
        # Total frames = 125 + 100 + 75 = 300 frames (6s * 50 fps)
        pcm = np.zeros(16000 * 6, dtype=np.float32)
        out = infer_emissions_sliding_window(
            model=self.mock_model,
            processor=self.mock_proc,
            audio_input=pcm,
            chunk_seconds=3.0,
            margin_seconds=0.5,
            model_identifier="test_sw_long",
        )
        self.assertIsInstance(out, ModelOutput)
        self.assertEqual(self.mock_model.call_count, 3)
        self.assertEqual(out.lpz.shape[0], 300)
        self.assertEqual(out.metadata["duration_sec"], 6.0)

    def test_invalid_chunk_margin_raises(self):
        pcm = np.zeros(16000 * 10, dtype=np.float32)
        with self.assertRaises(ValueError):
            infer_emissions_sliding_window(
                model=self.mock_model,
                processor=self.mock_proc,
                audio_input=pcm,
                chunk_seconds=2.0,
                margin_seconds=1.0,  # 2.0 <= 2 * 1.0 -> raises ValueError
            )

        with self.assertRaises(ValueError):
            infer_emissions_sliding_window(
                model=self.mock_model,
                processor=self.mock_proc,
                audio_input=pcm,
                chunk_seconds=2.0,
                margin_seconds=1.5,  # 2.0 < 2 * 1.5 -> raises ValueError
            )

    def test_sliding_window_caching(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            pcm = np.zeros(16000 * 4, dtype=np.float32)
            out1 = infer_emissions_sliding_window(
                model=self.mock_model,
                processor=self.mock_proc,
                audio_input=pcm,
                chunk_seconds=2.0,
                margin_seconds=0.5,
                cache_dir=tmp_dir,
                model_identifier="test_sw_cache",
            )
            self.assertIsInstance(out1, ModelOutput)
            call_count_after_first = self.mock_model.call_count
            self.assertGreater(call_count_after_first, 0)

            # Second call should load from cache and make 0 new model calls
            out2 = infer_emissions_sliding_window(
                model=self.mock_model,
                processor=self.mock_proc,
                audio_input=pcm,
                chunk_seconds=2.0,
                margin_seconds=0.5,
                cache_dir=tmp_dir,
                model_identifier="test_sw_cache",
            )
            self.assertEqual(self.mock_model.call_count, call_count_after_first)
            np.testing.assert_allclose(out1.lpz, out2.lpz)


if __name__ == "__main__":
    unittest.main()
