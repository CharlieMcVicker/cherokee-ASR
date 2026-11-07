# -*- coding: utf-8 -*-
"""
Unit and integration tests for `Wav2Vec2ForDualCTC` and `DualDataCollatorCTCWithPadding` (Phase 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from digohwelisgi.cherokee.models.dual_head import (
    DualDataCollatorCTCWithPadding,
    Wav2Vec2ForDualCTC,
)
from digohwelisgi.cherokee.orthography.tones import (
    MORAIC_TONE_VOCAB,
    SEGMENTAL_CV_VOCAB,
)


def _create_tiny_dual_model(
    num_hidden_layers: int = 4, num_branch_layers: int = 2
) -> Wav2Vec2ForDualCTC:
    config = Wav2Vec2Config(
        hidden_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(32, 32, 32, 32, 32, 32, 32),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        vocab_size=len(SEGMENTAL_CV_VOCAB),
        pad_token_id=SEGMENTAL_CV_VOCAB["[PAD]"],
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        final_dropout=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layerdrop=0.0,
    )
    config.num_branch_layers = num_branch_layers
    config.tone_vocab_size = len(MORAIC_TONE_VOCAB)
    config.tone_pad_token_id = MORAIC_TONE_VOCAB["[PAD]"]
    config.tone_loss_weight = 1.0
    model = Wav2Vec2ForDualCTC(config)
    model.sync_tone_branch_from_trunk()
    return model


def test_dual_head_forward_backward_synthetic() -> None:
    torch.manual_seed(42)
    model = _create_tiny_dual_model(num_hidden_layers=4, num_branch_layers=2)
    model.train()

    batch_size = 4
    seq_len = 8000  # ~0.5s at 16kHz -> ~24 frames
    input_values = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # 4 samples have CV labels; only first 2 samples have valid tone labels (last 2 are masked with -100)
    labels = torch.tensor(
        [
            [2, 5, 0, 7, 2],
            [11, 2, 9, -100, -100],
            [6, 2, 7, 10, -100],
            [12, 5, 11, 2, -100],
        ],
        dtype=torch.long,
    )
    tone_labels = torch.tensor(
        [
            [1, 2, 0, 1, 6],
            [1, 3, 1, -100, -100],
            [-100, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100],
        ],
        dtype=torch.long,
    )

    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
        tone_labels=tone_labels,
    )

    assert outputs.loss is not None
    assert not torch.isnan(outputs.loss)
    assert outputs.logits is not None
    assert outputs.logits_tone is not None
    assert outputs.logits.shape[0] == batch_size
    assert outputs.logits.shape[2] == len(SEGMENTAL_CV_VOCAB)
    assert outputs.logits_tone.shape[0] == batch_size
    assert outputs.logits_tone.shape[2] == len(MORAIC_TONE_VOCAB)

    outputs.loss.backward()

    # Shared Trunk (layer 0, 1), Branch A (layer 2, 3 + lm_head), and Branch B (tone_branch_layers + tone_lm_head)
    # must all receive non-zero gradients
    trunk_layer_0 = dict(model.wav2vec2.encoder.layers[0].named_parameters())[
        "feed_forward.output_dense.weight"
    ]
    branch_a_layer_3 = dict(model.wav2vec2.encoder.layers[3].named_parameters())[
        "feed_forward.output_dense.weight"
    ]
    branch_b_layer_0 = dict(model.tone_branch_layers[0].named_parameters())[
        "feed_forward.output_dense.weight"
    ]

    assert (
        trunk_layer_0.grad is not None and trunk_layer_0.grad.abs().sum().item() > 0.0
    )
    assert (
        branch_a_layer_3.grad is not None
        and branch_a_layer_3.grad.abs().sum().item() > 0.0
    )
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum().item() > 0.0

    assert (
        branch_b_layer_0.grad is not None
        and branch_b_layer_0.grad.abs().sum().item() > 0.0
    )
    assert model.tone_lm_head.weight.grad is not None
    assert model.tone_lm_head.weight.grad.abs().sum().item() > 0.0

    # Verify that when ALL samples have tone_labels = -100 (e.g. a Bible-only batch),
    # Branch B receives strictly ZERO gradient while Branch A and Shared Trunk receive non-zero gradients.
    model.zero_grad()
    all_masked_tone_labels = torch.full_like(tone_labels, -100)
    outputs_bible_only = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
        tone_labels=all_masked_tone_labels,
    )
    assert outputs_bible_only.loss is not None
    assert model.last_loss_tone == 0.0
    outputs_bible_only.loss.backward()

    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum().item() > 0.0
    assert (
        model.tone_lm_head.weight.grad is None
        or model.tone_lm_head.weight.grad.abs().sum().item() == 0.0
    )
    assert (
        branch_b_layer_0.grad is None or branch_b_layer_0.grad.abs().sum().item() == 0.0
    )


def test_dual_collator_padding_and_shapes(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(SEGMENTAL_CV_VOCAB), encoding="utf-8")
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    collator = DualDataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
        augment=False,
    )

    features = [
        {
            "input_values": np.random.randn(4000).astype(np.float32).tolist(),
            "labels": [12, 2, 0, 6, 2],
            "tone_labels": [1, 2, 0, 1, 3],
        },
        {
            "input_values": np.random.randn(6400).astype(np.float32).tolist(),
            "labels": [11, 5, 9],
            "tone_labels": [-100],
        },
    ]

    batch = collator(features)
    assert "input_values" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert "tone_labels" in batch

    assert batch["input_values"].shape == (2, 6400)
    assert batch["labels"].shape == (2, 5)
    assert batch["tone_labels"].shape == (2, 5)
    assert (batch["labels"][1, 3:] == -100).all()
    assert (batch["tone_labels"][1, :] == -100).all()


def test_integration_gate_4_five_step_optimization(tmp_path: Path) -> None:
    """
    Gate 4: Save/reload via `from_pretrained` (verifying `tone_branch_layers` sync from base checkpoint)
    and run 5 steps of AdamW optimization on a 4-sample batch, verifying total loss decreases.
    """
    torch.manual_seed(123)
    base_model = _create_tiny_dual_model(num_hidden_layers=4, num_branch_layers=2)
    ckpt_dir = tmp_path / "tiny_ckpt"
    base_model.save_pretrained(ckpt_dir)

    loaded_model = Wav2Vec2ForDualCTC.from_pretrained(str(ckpt_dir))
    loaded_model.freeze_feature_encoder()
    loaded_model.train()

    # Verify CNN feature extractor is frozen while trunk & both branches are trainable
    for param in loaded_model.wav2vec2.feature_extractor.parameters():
        assert not param.requires_grad
    assert loaded_model.lm_head.weight.requires_grad
    assert loaded_model.tone_lm_head.weight.requires_grad

    batch_size = 4
    seq_len = 8000
    input_values = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.tensor(
        [
            [2, 5, 0, 7, 2],
            [11, 2, 9, -100, -100],
            [6, 2, 7, 10, -100],
            [12, 5, 11, 2, -100],
        ],
        dtype=torch.long,
    )
    tone_labels = torch.tensor(
        [
            [1, 2, 0, 1, 6],
            [1, 3, 1, -100, -100],
            [-100, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100],
        ],
        dtype=torch.long,
    )

    optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=3e-3)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = loaded_model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            tone_labels=tone_labels,
        )
        assert out.loss is not None
        assert not torch.isnan(out.loss)
        losses.append(float(out.loss.detach().cpu().item()))
        out.loss.backward()
        optimizer.step()

    assert (
        losses[-1] < losses[0]
    ), f"Expected loss to decrease over 5 steps, got {losses}"
