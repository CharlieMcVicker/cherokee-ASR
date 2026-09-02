# -*- coding: utf-8 -*-
"""
infer.py

Centralized inference helper module for Wav2Vec2 CTC greedy decoding.
Provides consistent inference functions used across training, evaluation, and serving.
"""

import logging
import os
import re
import struct
import unicodedata
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transcription.utils.syllabary_map import phonetics_to_syllabary

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def normalize_text(text):
    """Normalize transcriptions consistently."""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    apostrophe_variants = r"[’‘ʼʻ`´‛]"
    chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]"
    text = re.sub(apostrophe_variants, "'", text)
    text = re.sub(chars_to_remove_regex, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_tones(text):
    """Remove tone numbers (0-9)."""
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_length(text):
    """Collapse any sequence of 2 or more of the same vowel into a single vowel."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"([aeiouv])\1+", r"\1", text)


def strip_both(text):
    """Remove both tones and collapse vowel lengths."""
    return strip_length(strip_tones(text))


def load_and_preprocess_audio(audio_path, target_sr=TARGET_SAMPLE_RATE):
    """Load an audio file, resample to 16kHz, convert to mono, and return as 1D numpy array."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    speech_array, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(speech_array, dtype=torch.float32)

    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def calculate_word_confidences(probs, pred_ids, processor):
    """Calculate character and word level confidences and alternative predictions."""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.cpu().numpy()

    token_probs = probs[np.arange(len(pred_ids)), pred_ids]
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.vocab.get("[PAD]", 0)

    chars = []
    char_probs = []
    char_alts = []
    char_times = []

    top_k = 5
    top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]
    top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)

    word_delimiter_token_id = getattr(
        processor.tokenizer, "word_delimiter_token_id", None
    )

    prev_id = -1
    for i, token_id in enumerate(pred_ids):
        if token_id != pad_id and token_id != prev_id:
            if token_id == word_delimiter_token_id:
                token_str = " "
            else:
                token_str = processor.decode([token_id])

            if token_str:
                chars.append(token_str)
                char_probs.append(float(token_probs[i]))
                char_times.append(round(i * 0.02, 3))

                alts = []
                for k in range(top_k):
                    alt_id = top_k_indices[i, k]
                    alt_prob = float(top_k_probs[i, k])
                    if alt_id != token_id:
                        if alt_id == word_delimiter_token_id:
                            alt_str = " "
                        else:
                            alt_str = processor.decode([alt_id])
                        if alt_str:
                            alts.append({"char": alt_str, "confidence": alt_prob})
                char_alts.append(alts)

        prev_id = token_id

    words_details = []
    current_word = ""
    current_chars = []
    for char, prob, alts, time in zip(chars, char_probs, char_alts, char_times):
        if char == " ":
            if current_word:
                word_conf = float(np.mean([c["confidence"] for c in current_chars]))
                words_details.append(
                    {
                        "word": current_word,
                        "confidence": word_conf,
                        "start_time": current_chars[0]["start_time"],
                        "end_time": round(current_chars[-1]["start_time"] + 0.02, 3),
                        "chars": current_chars,
                    }
                )
                current_word = ""
                current_chars = []
        else:
            current_word += char
            current_chars.append(
                {
                    "char": char,
                    "confidence": prob,
                    "start_time": time,
                    "alternatives": alts,
                }
            )
    if current_word:
        word_conf = float(np.mean([c["confidence"] for c in current_chars]))
        words_details.append(
            {
                "word": current_word,
                "confidence": word_conf,
                "start_time": current_chars[0]["start_time"],
                "end_time": round(current_chars[-1]["start_time"] + 0.02, 3),
                "chars": current_chars,
            }
        )

    return words_details


def greedy_inference(logits, processor):
    """
    Perform greedy decoding on logits and return texts with confidence scores.
    Accepts 2D (sequence_len, vocab_size) or 3D (batch_size, sequence_len, vocab_size) logits.

    Note on architecture:
        The core CTC acoustic model outputs phonetic text hypotheses.
        The 'syllabary' key in the returned dictionary is provided for legacy compatibility
        with older callers. Downstream Cherokee Syllabary transliteration and phonetic
        rule reconciliation logically belong to `transcription.syllabary_enrichment` and
        `transcription.utils.syllabary_map.phonetics_to_syllabary`.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    is_batched = len(logits.shape) == 3
    if not is_batched:
        logits = logits.unsqueeze(0)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1)

    decoded_texts = processor.batch_decode(pred_ids)

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.vocab.get("[PAD]", 0)

    results = []
    for i in range(logits.shape[0]):
        seq_probs = probs[i].cpu().numpy()
        seq_ids = pred_ids[i].cpu().numpy()
        text = decoded_texts[i].strip()

        token_probs = seq_probs[np.arange(len(seq_ids)), seq_ids]
        non_pad_mask = seq_ids != pad_id
        if non_pad_mask.any():
            confidence = float(np.mean(token_probs[non_pad_mask]))
        else:
            confidence = float(np.mean(token_probs))

        results.append(
            {
                "text": text,
                "transcription": text,
                "syllabary": phonetics_to_syllabary(text),
                "confidence": confidence,
            }
        )

    return results if is_batched else results[0]


def infer_single_audio(model, processor, audio_path, device=None):
    """Transcribe a single audio file."""
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    speech = load_and_preprocess_audio(audio_path)
    input_values = processor(speech, sampling_rate=TARGET_SAMPLE_RATE).input_values[0]
    input_tensor = torch.tensor([input_values]).to(device)

    try:
        with torch.no_grad():
            logits = model(input_tensor).logits
    except NotImplementedError as e:
        if device == "mps":
            # Fallback to CPU if MPS operations fail
            device = "cpu"
            model.to(device)
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                logits = model(input_tensor).logits
        else:
            raise e

    return greedy_inference(logits, processor)


from transcription.utils.model_utils import (
    get_best_model,
    get_model,
)


def infer_pcm_array(
    model, processor, pcm_data, sample_rate=TARGET_SAMPLE_RATE, device=None
):
    """Transcribe raw 1D float32 PCM numpy array, bytes, or float list directly in memory."""
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    if isinstance(pcm_data, bytes):
        speech = np.frombuffer(pcm_data, dtype=np.float32)
    elif isinstance(pcm_data, list):
        speech = np.array(pcm_data, dtype=np.float32)
    elif isinstance(pcm_data, np.ndarray):
        speech = pcm_data.astype(np.float32)
    else:
        raise TypeError(f"Unsupported pcm_data type: {type(pcm_data)}")

    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=-1)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        speech = resampler(waveform).squeeze(0).numpy()

    duration = len(speech) / TARGET_SAMPLE_RATE
    logger.info(
        "Running ASR inference on PCM audio: samples=%d, duration=%.2fs, sr=%d, device=%s",
        len(speech),
        duration,
        sample_rate,
        device,
    )

    input_values = processor(speech, sampling_rate=TARGET_SAMPLE_RATE).input_values[0]
    input_tensor = torch.tensor(np.array([input_values]), dtype=torch.float32).to(
        device
    )

    try:
        with torch.no_grad():
            logits = model(input_tensor).logits
    except NotImplementedError as e:
        if device == "mps":
            device = "cpu"
            model.to(device)
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                logits = model(input_tensor).logits
        else:
            raise e

    result = greedy_inference(logits, processor)
    logger.info("ASR inference complete: result=%s", result)
    return result


def transcribe_audio_batch(model, processor, audio_paths, device=None, batch_size=16):
    """Transcribe a list of audio files using batching, proper padding, and OOM fallbacks."""
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        batch_speech = []
        for path in batch_paths:
            try:
                speech = load_and_preprocess_audio(path)
                batch_speech.append(speech)
            except Exception as e:
                print(f"Error loading audio file {path}: {e}")
                batch_speech.append(np.zeros((TARGET_SAMPLE_RATE,), dtype=np.float32))

        # Pad and build inputs
        inputs = processor(
            batch_speech,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device)
        attention_mask = (
            inputs.attention_mask.to(device) if "attention_mask" in inputs else None
        )

        try:
            with torch.no_grad():
                outputs = model(
                    input_values=input_values, attention_mask=attention_mask
                )
                logits = outputs.logits
        except Exception as e:
            err_str = str(e).lower()
            is_oom = "out of memory" in err_str or (
                hasattr(torch.cuda, "OutOfMemoryError")
                and isinstance(e, torch.cuda.OutOfMemoryError)
            )
            is_cudnn_err = "unable to find an engine" in err_str or "cudnn" in err_str

            if is_oom or is_cudnn_err:
                reason = "OOM" if is_oom else "cuDNN error"
                print(
                    f"  {reason} on batch. Falling back to sequential inference for this batch...",
                    flush=True,
                )

                # Free large tensors to ensure empty_cache succeeds
                if "input_values" in locals():
                    del input_values
                if "attention_mask" in locals():
                    del attention_mask

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # Sequential fallback
                for speech_item in batch_speech:
                    single_inputs = processor(
                        [speech_item],
                        sampling_rate=TARGET_SAMPLE_RATE,
                        padding=True,
                        return_tensors="pt",
                    )
                    single_input_values = single_inputs.input_values.to(device)
                    single_attention_mask = (
                        single_inputs.attention_mask.to(device)
                        if "attention_mask" in single_inputs
                        else None
                    )

                    try:
                        with torch.no_grad():
                            with torch.backends.cudnn.flags(enabled=False):
                                single_outputs = model(
                                    input_values=single_input_values,
                                    attention_mask=single_attention_mask,
                                )
                                single_logits = single_outputs.logits

                        input_len = len(speech_item)
                        logit_len = int(
                            model._get_feat_extract_output_lengths(input_len)
                        )
                        sliced_logits = single_logits[0, :logit_len, :]
                        res = greedy_inference(sliced_logits, processor)
                        results.append(res)
                    except Exception as seq_e:
                        print(
                            f"  Fatal OOM even with batch_size=1. Skipping item...",
                            flush=True,
                        )
                        results.append({"text": "", "confidence": 0.0})
                    finally:
                        for var_name in (
                            "single_logits",
                            "single_outputs",
                            "single_inputs",
                            "single_input_values",
                            "single_attention_mask",
                        ):
                            locals().pop(var_name, None)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                continue
            elif isinstance(e, NotImplementedError) and device == "mps":
                print(
                    "  MPS execution failed. Falling back to CPU backend for this batch...",
                    flush=True,
                )
                device = "cpu"
                model.to(device)
                input_values = input_values.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.no_grad():
                    outputs = model(
                        input_values=input_values, attention_mask=attention_mask
                    )
                    logits = outputs.logits
            else:
                raise e

        # Squeeze/slice logits by attention mask lengths to avoid decoding padding
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1)
            output_lengths = model._get_feat_extract_output_lengths(input_lengths)
            output_lengths = output_lengths.cpu().numpy()
        else:
            output_lengths = [logits.shape[1]] * logits.shape[0]

        for idx in range(logits.shape[0]):
            actual_len = int(output_lengths[idx])
            sliced_logits = logits[idx, :actual_len, :]
            res = greedy_inference(sliced_logits, processor)
            results.append(res)

        # Free batch tensors to ensure we don't peak VRAM on next batch allocation
        for var_name in (
            "logits",
            "outputs",
            "inputs",
            "input_values",
            "attention_mask",
        ):
            locals().pop(var_name, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results
