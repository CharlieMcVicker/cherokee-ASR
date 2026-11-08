import os
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from digohwelisgi.core.audio import AudioAugmentationConfig, AudioAugmenter
from digohwelisgi.cherokee.training.train import (
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

    # Verify zero-mean and unit-variance re-normalization after augmentation
    # Note: Pad zeros outside the valid slice are padding, check the active slice (length 800)
    feat_0 = batch_aug["input_values"][0, :1000].numpy()
    feat_1 = batch_aug["input_values"][1, :800].numpy()
    assert np.isclose(np.mean(feat_0), 0.0, atol=1e-5)
    assert np.isclose(np.var(feat_0), 1.0, atol=1e-3)
    assert np.isclose(np.mean(feat_1), 0.0, atol=1e-5)
    assert np.isclose(np.var(feat_1), 1.0, atol=1e-3)


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
    from digohwelisgi.cherokee.training.train import parse_args

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
    from digohwelisgi.cherokee.training.train import prepare_datasets

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
    from digohwelisgi.cherokee.training.train import parse_args

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
    from digohwelisgi.cherokee.training.train import evaluate_checkpoints_dual

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
        "digohwelisgi.cherokee.training.train.yield_local_checkpoints",
        lambda *args, **kwargs: [("mock", None, None, "/path")],
    )
    monkeypatch.setattr(
        "digohwelisgi.cherokee.training.train.run_evaluation", mock_run_evaluation
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


def test_parse_args_hub_and_step_defaults(monkeypatch):
    import sys
    from digohwelisgi.cherokee.training.train import parse_args

    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    assert args.epochs == 30
    assert args.eval_steps == 200
    assert args.save_steps == 400
    assert args.save_total_limit == 2
    assert args.save_only_model is False
    assert args.hub_strategy == "all_checkpoints"
    assert args.hub_prefix == "dual-tone-moraic"
    assert CONFIG["epochs"] == 30
    assert CONFIG["eval_steps"] == 200
    assert CONFIG["save_steps"] == 400
    assert CONFIG["save_total_limit"] == 2
    assert CONFIG["save_only_model"] is False
    assert CONFIG["hub_strategy"] == "all_checkpoints"
    assert CONFIG["hub_prefix"] == "dual-tone-moraic"

    # Custom override
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--epochs",
            "25",
            "--eval-steps",
            "300",
            "--save-steps",
            "500",
            "--save-total-limit",
            "1",
            "--save-only-model",
            "--hub-strategy",
            "every_save",
            "--hub-prefix",
            "custom-exp",
            "--push-to-hub",
            "--hub-model-id",
            "user/my-model",
        ],
    )
    args = parse_args()
    assert args.epochs == 25
    assert args.eval_steps == 300
    assert args.save_steps == 500
    assert args.save_total_limit == 1
    assert args.save_only_model is True
    assert args.hub_strategy == "every_save"
    assert args.hub_prefix == "custom-exp"
    assert CONFIG["save_total_limit"] == 1
    assert CONFIG["save_only_model"] is True
    assert "user/custom-exp-" in CONFIG["hub_model_id"]


def test_promote_best_checkpoint_guards_same_file(tmp_path):
    from digohwelisgi.cherokee.training.train import promote_best_checkpoint

    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    dummy_file = model_dir / "pytorch_model.bin"
    dummy_file.write_text("weights")

    mock_proc = MagicMock()
    # Promoting "final" where best_ckpt_path == folder_model_files
    promote_best_checkpoint("final", str(model_dir), str(model_dir), mock_proc)
    mock_proc.save_pretrained.assert_called_once_with(str(model_dir))
    assert dummy_file.read_text() == "weights"


def test_train_model_pushes_and_tags_hub(monkeypatch):
    from digohwelisgi.cherokee.training.train import train_model

    mock_trainer = MagicMock()
    mock_processor = MagicMock()
    mock_api_instance = MagicMock()

    monkeypatch.setitem(CONFIG, "push_to_hub", True)
    monkeypatch.setitem(CONFIG, "hub_model_id", "test-user/test-repo")
    monkeypatch.setitem(CONFIG, "hub_token", "fake_token")

    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: mock_api_instance)

    train_model(mock_trainer, None, "/dummy/folder", mock_processor)

    mock_trainer.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_trainer.save_model.assert_called_once_with("/dummy/folder")
    mock_processor.save_pretrained.assert_called_once_with("/dummy/folder")
    mock_trainer.push_to_hub.assert_called_once_with(
        commit_message="Best checkpoint on combined validation set (load_best_model_at_end)"
    )
    mock_api_instance.create_tag.assert_called_once_with(
        repo_id="test-user/test-repo",
        tag="best-combined-valid",
        tag_message="Best checkpoint on combined validation set (load_best_model_at_end)",
        repo_type="model",
        exist_ok=True,
    )


def test_upload_checkpoint_resilient_filters_and_retries(tmp_path, monkeypatch):
    from digohwelisgi.cherokee.training.train import upload_checkpoint_resilient

    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_text("weights")
    (checkpoint_dir / "config.json").write_text("{}")
    (checkpoint_dir / "optimizer.pt").write_text("heavy_optimizer")
    (checkpoint_dir / "scheduler.pt").write_text("heavy_scheduler")

    uploaded = []
    mock_api_instance = MagicMock()

    def mock_upload_file(path_or_fileobj, path_in_repo, repo_id, repo_type):
        uploaded.append(path_in_repo)

    mock_api_instance.upload_file = mock_upload_file
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: mock_api_instance)

    upload_checkpoint_resilient(
        repo_id="test-user/test-model",
        folder_path=str(checkpoint_dir),
        token="test-token",
    )

    mock_api_instance.create_repo.assert_called_once_with(
        repo_id="test-user/test-model", repo_type="model", exist_ok=True
    )
    # optimizer.pt and scheduler.pt must be excluded
    assert "optimizer.pt" not in uploaded
    assert "scheduler.pt" not in uploaded
    assert "config.json" in uploaded
    assert "model.safetensors" in uploaded
