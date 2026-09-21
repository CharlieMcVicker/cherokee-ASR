# -*- coding: utf-8 -*-
"""
scripts/realign_gs_mm_ctc.py

Realigns the saving-the-voices/gs_mm interview audio and transcript using the
syncope- and intrusion-aware CTC segmentation aligner with calibrated code-switching
English projection and phonotactic provenance isolation.

CRITICAL MODEL SELECTION:
- Toneless Pre-Bible Model:
  - Repo: charliemcvicker/length-only-20260704-155307-asr-cherokee-colon
  - Revision: 76e62140955f4738abdab345ea34068b02d8d2a2
  - Device: mps (or cpu fallback)
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict

import torch

from transcription.alignment.arpabet.projector import get_default_projector
from transcription.alignment.models import CTCAlignerConfig
from transcription.alignment.pipeline import align_syllabary_ctc
from transcription.models.asr_model import CherokeeASRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PRE_BIBLE_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
PRE_BIBLE_MODEL_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"

AUDIO_PATH = Path("saving-the-voices/gs_mm.wav")
TRANSCRIPT_PATH = Path("saving-the-voices/gs_mm.txt")
OUTPUT_DIR = Path("saving-the-voices/output_ctc")

TEXTGRID_FILENAME = "gs_mm_ctc.TextGrid"
MANIFEST_FILENAME = "gs_mm_ctc_manifest.json"


def run_interview_ctc_realignment() -> Dict[str, Any]:
    start_time = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading toneless pre-Bible model on device: {device}...")

    model = CherokeeASRModel.from_pretrained(
        path_or_repo=PRE_BIBLE_MODEL_REPO,
        revision=PRE_BIBLE_MODEL_REVISION,
        device=device,
    )
    logger.info("Model loaded successfully.")

    logger.info("Initializing calibrated synthetic target projector...")
    projector = get_default_projector()
    logger.info("Projector initialized.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = CTCAlignerConfig(
        enforce_phonotactics=True,
        cache=True,
        enable_vad_soft_masking=True,
        vad_p_low=0.30,
        vad_p_high=0.75,
        vad_pad_ms=20,
    )

    logger.info(
        f"Starting code-switched CTC segmentation alignment for {AUDIO_PATH} -> {OUTPUT_DIR}..."
    )
    alignment_output = align_syllabary_ctc(
        audio=AUDIO_PATH,
        transcript=TRANSCRIPT_PATH,
        output_dir=OUTPUT_DIR,
        model=model,
        config=config,
        projector=projector,
        code_switched=True,
        strip_speaker=True,
        reconcile=True,
        export_praat=True,
        export_manifest=True,
        textgrid_filename=TEXTGRID_FILENAME,
        manifest_filename=MANIFEST_FILENAME,
    )
    elapsed_sec = time.time() - start_time
    logger.info(f"CTC Alignment completed in {elapsed_sec:.2f}s.")

    manifest_path = OUTPUT_DIR / MANIFEST_FILENAME
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    metrics = manifest_data.get("metrics", {})
    aligned_lines = manifest_data.get("lines", [])
    logger.info(f"Aligned {len(aligned_lines)} turns.")
    logger.info(f"Metrics: {metrics}")

    return manifest_data


if __name__ == "__main__":
    run_interview_ctc_realignment()
