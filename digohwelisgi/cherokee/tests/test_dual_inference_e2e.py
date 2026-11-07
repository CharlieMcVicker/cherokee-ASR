# -*- coding: utf-8 -*-
"""
End-to-end unit and integration tests for Dual-Head ModelOutput currency,
CherokeeASRModel dual checkpoint loading, and Praat TextGrid Moraic Tone tier export (Phase 6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from digohwelisgi.cherokee.models import CherokeeASRModel, Wav2Vec2ForDualCTC
from digohwelisgi.cherokee.orthography.tones import (
    MORAIC_TONE_VOCAB,
    SEGMENTAL_CV_VOCAB,
    encode_cv_text,
    encode_tone_tokens,
    save_dual_vocabularies,
    to_moraic_tone_tokens,
)
from digohwelisgi.core.alignment.models import (
    AlignedChunk,
    AlignmentOutput,
    WordInterval,
)
from digohwelisgi.core.exporters.textgrid import export_textgrid
from digohwelisgi.core.models.output import ModelOutput


def _make_one_hot_lpz(token_ids: list[int], vocab_size: int, pad_id: int) -> np.ndarray:
    num_frames = len(token_ids) * 2 + 2
    lpz = np.full((num_frames, vocab_size), -12.0, dtype=np.float32)
    lpz[:, pad_id] = 0.0
    for idx, tid in enumerate(token_ids):
        frame = idx * 2
        lpz[frame, :] = -12.0
        lpz[frame, tid] = 0.0
    return lpz


def test_model_output_reconciled_projection(tmp_path: Path) -> None:
    canonical = "na3 a2ni22tshu34tsa2 skwa21lhe34hsti2"
    cv_text, tone_str = to_moraic_tone_tokens(canonical)

    cv_ids = encode_cv_text(cv_text)
    tone_ids = encode_tone_tokens(tone_str)

    lpz_cv = _make_one_hot_lpz(
        cv_ids, len(SEGMENTAL_CV_VOCAB), SEGMENTAL_CV_VOCAB["[PAD]"]
    )
    lpz_tone = _make_one_hot_lpz(
        tone_ids, len(MORAIC_TONE_VOCAB), MORAIC_TONE_VOCAB["[PAD]"]
    )

    output = ModelOutput(
        lpz=lpz_cv,
        vocab=SEGMENTAL_CV_VOCAB,
        frame_duration_sec=0.02,
        lpz_tone=lpz_tone,
        vocab_tone=MORAIC_TONE_VOCAB,
    )

    assert output.decode_greedy() == cv_text
    assert output.decode_tone_tokens() == tone_str
    assert output.decode_reconciled(mode="dp") == canonical
    assert output.decode_reconciled(mode="fast") == canonical

    # Verify .npz save/load round-trip preserves lpz_tone and vocab_tone
    npz_path = tmp_path / "dual_emissions.npz"
    output.save(npz_path)
    loaded = ModelOutput.load(npz_path)

    assert loaded.lpz_tone is not None
    assert loaded.vocab_tone is not None
    assert np.allclose(loaded.lpz_tone, lpz_tone)
    assert loaded.decode_tone_tokens() == tone_str
    assert loaded.decode_reconciled(mode="dp") == canonical


def test_cherokee_asr_model_dual_checkpoint_loading(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "dual_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_path, _ = save_dual_vocabularies(ckpt_dir)

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
    processor.save_pretrained(str(ckpt_dir))

    config = Wav2Vec2Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(32, 32, 32, 32, 32, 32, 32),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        vocab_size=len(SEGMENTAL_CV_VOCAB),
        pad_token_id=SEGMENTAL_CV_VOCAB["[PAD]"],
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
    )
    config.num_branch_layers = 2
    config.tone_vocab_size = len(MORAIC_TONE_VOCAB)
    config.tone_pad_token_id = MORAIC_TONE_VOCAB["[PAD]"]
    model = Wav2Vec2ForDualCTC(config)
    model.save_pretrained(str(ckpt_dir))

    asr = CherokeeASRModel.from_pretrained(str(ckpt_dir), device="cpu", use_cache=False)
    assert isinstance(asr.model, Wav2Vec2ForDualCTC)

    dummy_audio = np.zeros(8000, dtype=np.float32)
    out = asr.infer(dummy_audio)
    assert out.lpz_tone is not None
    assert out.vocab_tone is not None
    assert out.lpz.shape[0] == out.lpz_tone.shape[0]
    assert out.lpz.shape[1] == len(SEGMENTAL_CV_VOCAB)
    assert out.lpz_tone.shape[1] == len(MORAIC_TONE_VOCAB)
    assert isinstance(out.decode_reconciled(), str)


def test_textgrid_exporter_with_tones(tmp_path: Path) -> None:
    words = [
        WordInterval(word="na", start_sec=0.10, end_sec=0.40, confidence=0.95),
        WordInterval(word="anitshutsa", start_sec=0.45, end_sec=1.20, confidence=0.92),
    ]
    tone_intervals = [
        WordInterval(word="C 3_1", start_sec=0.10, end_sec=0.40, confidence=0.91),
        WordInterval(
            word="2_1 C 2_2 C 3_2 C 2_1", start_sec=0.45, end_sec=1.20, confidence=0.89
        ),
    ]
    reconciled_intervals = [
        WordInterval(word="na3", start_sec=0.10, end_sec=0.40, confidence=0.93),
        WordInterval(
            word="a2ni22tshu34tsa2", start_sec=0.45, end_sec=1.20, confidence=0.90
        ),
    ]

    chunk = AlignedChunk(
        chunk_id="utt_01",
        start_sec=0.0,
        end_sec=1.50,
        words=words,
    )
    alignment = AlignmentOutput(aligned_chunks=[chunk])

    tg_path = export_textgrid(
        alignment=alignment,
        output_dir=tmp_path,
        filename="dual_test.TextGrid",
        additional_word_tiers={"Reconciled Toned TTH": reconciled_intervals},
        tone_intervals=tone_intervals,
    )

    content = Path(tg_path).read_text(encoding="utf-8")
    assert 'name = "Words"' in content
    assert 'name = "Reconciled Toned TTH"' in content
    assert 'name = "Moraic Tones"' in content
    assert 'text = "C 3_1"' in content
    assert 'text = "2_1 C 2_2 C 3_2 C 2_1"' in content
    assert 'text = "a2ni22tshu34tsa2"' in content
