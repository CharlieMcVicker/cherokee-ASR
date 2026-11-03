import os
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from digohwelisgi.core.audio import AudioAugmentationConfig, AudioAugmenter
from digohwelisgi.training.train import (
    CONFIG,
    CTCTrainer,
    DataCollatorCTCWithPadding,
    _resolve_audio_path,
)


def test_resolve_audio_path_known_paths():
    dataset_path = "data/training/processed/sentence_audio"

    # Test bible split audio path resolution
    bible_rel = "cherokee_new_testament/split_audio/mark_01_02.wav"
    resolved_bible = _resolve_audio_path(bible_rel, dataset_path)
    assert os.path.exists(resolved_bible)
    assert "data/projects" in resolved_bible

    # Test original sentence audio path resolution (legacy prefix)
    legacy_orig = (
        "training_data/processed/sentence_audio/Sentence_for_entry_1368_01.wav"
    )
    resolved_orig = _resolve_audio_path(legacy_orig, dataset_path)
    assert os.path.exists(resolved_orig)
    assert "data/training" in resolved_orig

    # Test direct basename inside dataset_path
    base_file = "Sentence_for_entry_1368_01.wav"
    resolved_base = _resolve_audio_path(base_file, dataset_path)
    assert os.path.exists(resolved_base)


def test_data_collator_with_augmentation():
    class DummyTokenizer:
        pad_token_id = 0

    class DummyProcessor:
        tokenizer = DummyTokenizer()

        def pad(self, features=None, labels=None, padding=True, return_tensors="pt"):
            if features is not None:
                max_len = max(len(f["input_values"]) for f in features)
                batch = torch.zeros((len(features), max_len), dtype=torch.float32)
                for i, f in enumerate(features):
                    batch[i, : len(f["input_values"])] = torch.tensor(
                        f["input_values"], dtype=torch.float32
                    )
                return {"input_values": batch}
            elif labels is not None:
                max_len = max(len(f["input_ids"]) for f in labels)
                batch = torch.zeros((len(labels), max_len), dtype=torch.long)
                attention = torch.zeros((len(labels), max_len), dtype=torch.long)
                for i, f in enumerate(labels):
                    batch[i, : len(f["input_ids"])] = torch.tensor(
                        f["input_ids"], dtype=torch.long
                    )
                    attention[i, : len(f["input_ids"])] = 1
                return {"input_ids": batch, "attention_mask": attention}
            raise ValueError("Must provide features or labels")

    processor = DummyProcessor()
    aug_config = AudioAugmentationConfig(
        time_stretch_p=0.0,
        pitch_shift_p=0.0,
        gaussian_noise_p=1.0,
        gaussian_noise_min_amplitude=0.02,
        gaussian_noise_max_amplitude=0.05,
    )
    augmenter = AudioAugmenter(config=aug_config)

    collator = DataCollatorCTCWithPadding(
        processor=processor,  # type: ignore
        padding=True,
        augmenter=augmenter,
        augment=False,
    )

    raw_features = [
        {"input_values": [0.0] * 1000, "labels": [1, 2, 3]},
        {"input_values": [0.0] * 800, "labels": [4, 5]},
    ]

    # In eval/no-augment mode:
    batch_no_aug = collator(raw_features)
    assert torch.all(batch_no_aug["input_values"] == 0.0)

    # In train/augment mode:
    collator.augment = True
    batch_aug = collator(raw_features)
    assert not torch.all(batch_aug["input_values"] == 0.0)
    assert batch_aug["labels"].shape[0] == 2


def test_ctc_trainer_dataloader_toggle():
    mock_collator = MagicMock()
    mock_collator.augment = False

    trainer = CTCTrainer.__new__(CTCTrainer)
    trainer.data_collator = mock_collator

    # Mock super().get_train_dataloader and super().get_eval_dataloader
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "transformers.Trainer.get_train_dataloader",
            lambda self: "train_loader",
        )
        mp.setattr(
            "transformers.Trainer.get_eval_dataloader",
            lambda self, eval_dataset=None: "eval_loader",
        )

        res_train = trainer.get_train_dataloader()
        assert res_train == "train_loader"
        assert mock_collator.augment is True

        res_eval = trainer.get_eval_dataloader()
        assert res_eval == "eval_loader"
        assert mock_collator.augment is False


def test_parse_args_augmentation_flags():
    from digohwelisgi.training.train import parse_args

    test_args = [
        "train.py",
        "--no-augment",
        "--time-stretch-p",
        "0.25",
        "--pitch-shift-p",
        "0.1",
        "--gaussian-noise-p",
        "0.05",
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", test_args)
        args = parse_args()
        assert args.augment is False
        assert CONFIG["augment"] is False
        assert CONFIG["time_stretch_p"] == 0.25
        assert CONFIG["pitch_shift_p"] == 0.1
        assert CONFIG["gaussian_noise_p"] == 0.05


def test_prepare_datasets_merges_bible_and_orig():
    import pandas as pd
    from digohwelisgi.training.train import prepare_datasets

    # Mock audio read so prepare_batch does not touch disk
    dummy_audio = np.zeros(16000, dtype=np.float32)

    class DummyBatchProcessor:
        def __call__(self, *args, **kwargs):
            if "text" in kwargs or (len(args) > 0 and isinstance(args[0], str)):
                mock_text = MagicMock()
                mock_text.input_ids = [1, 2, 3]
                return mock_text
            res = MagicMock()
            audio_val = (
                args[0]
                if args
                else kwargs.get("audio", np.zeros(16000, dtype=np.float32))
            )
            res.input_values = [audio_val]
            return res

    processor = DummyBatchProcessor()

    dfs = {
        "train_orig": pd.DataFrame(
            {"path": ["f1.wav", "f2.wav"], "sentence": ["s1", "s2"]}
        ),
        "train_bible": pd.DataFrame({"path": ["b1.wav"], "sentence": ["b1"]}),
        "valid_orig": pd.DataFrame(
            {"path": ["v1.wav", "v2.wav"], "sentence": ["vs1", "vs2"]}
        ),
        "valid_bible": pd.DataFrame({"path": ["vb1.wav"], "sentence": ["vbs1"]}),
        "test_orig": pd.DataFrame({"path": ["t1.wav"], "sentence": ["ts1"]}),
        "test_bible": pd.DataFrame({"path": ["tb1.wav"], "sentence": ["tbs1"]}),
    }

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("soundfile.read", lambda path: (dummy_audio, 16000))
        train_ds, valid_ds, test_orig, test_bible = prepare_datasets(
            dfs, "path", "sentence", processor
        )

        # Force evaluation of lazy interleaved generator while soundfile.read is patched
        _ = list(train_ds)

        # valid_orig (2) + valid_bible (1) = 3 total merged validation samples
        assert len(valid_ds) == 3
        # test splits are kept separate for disaggregated evaluation
        assert len(test_orig) == 1
        assert len(test_bible) == 1


def test_parse_args_gradient_checkpointing(monkeypatch):
    import sys
    from digohwelisgi.training.train import parse_args

    # Default is False
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    assert args.gradient_checkpointing is False
    assert CONFIG["gradient_checkpointing"] is False

    # Explicit flag turns it on
    monkeypatch.setattr(sys, "argv", ["train.py", "--gradient-checkpointing"])
    args = parse_args()
    assert args.gradient_checkpointing is True
    assert CONFIG["gradient_checkpointing"] is True


def test_evaluate_checkpoints_dual_ranking(monkeypatch):
    import pandas as pd
    from digohwelisgi.training.train import evaluate_checkpoints_dual

    ranking_df_orig = pd.DataFrame(
        [
            {
                "checkpoint": "ckpt-100",
                "path": "/models/ckpt-100",
                "wer": 0.25,
                "cer": 0.10,
                "median_wer": 0.20,
                "median_cer": 0.08,
            },
            {
                "checkpoint": "ckpt-200",
                "path": "/models/ckpt-200",
                "wer": 0.15,
                "cer": 0.05,
                "median_wer": 0.10,
                "median_cer": 0.04,
            },
        ]
    )
    ranking_df_bible = pd.DataFrame(
        [
            {
                "checkpoint": "ckpt-100",
                "path": "/models/ckpt-100",
                "wer": 0.30,
                "cer": 0.15,
                "median_wer": 0.28,
                "median_cer": 0.12,
            },
            {
                "checkpoint": "ckpt-200",
                "path": "/models/ckpt-200",
                "wer": 0.20,
                "cer": 0.08,
                "median_wer": 0.18,
                "median_cer": 0.07,
            },
        ]
    )

    rows_orig_mock = {"ckpt-100": [], "ckpt-200": []}
    rows_bible_mock = {"ckpt-100": [], "ckpt-200": []}

    call_count = 0

    def mock_run_evaluation(gen, ds, data_collator=None, batch_size=16):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return rows_orig_mock, ranking_df_orig.copy()
        return rows_bible_mock, ranking_df_bible.copy()

    monkeypatch.setattr(
        "digohwelisgi.training.train.yield_local_checkpoints",
        lambda *args, **kwargs: [("mock", None, None, "/path")],
    )
    monkeypatch.setattr(
        "digohwelisgi.training.train.run_evaluation", mock_run_evaluation
    )

    dummy_collator = MagicMock()
    dummy_proc = MagicMock()

    (
        best_ckpt_label,
        best_wer,
        best_cer,
        best_ckpt_path,
        rows_orig,
        ranking_orig,
        rows_bible,
        ranking_bible,
    ) = evaluate_checkpoints_dual("/models", [], [], dummy_collator, dummy_proc)

    # ckpt-200 had lower WER (0.15 vs 0.25)
    assert best_ckpt_label == "ckpt-200"
    assert best_wer == 0.15
    assert best_cer == 0.05
    assert best_ckpt_path == "/models/ckpt-200"
    assert ranking_orig.iloc[0]["checkpoint"] == "ckpt-200"
    assert ranking_bible.iloc[0]["checkpoint"] == "ckpt-200"
