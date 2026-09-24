#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/realign_gs_mm_ctc.py

Thin driver: realigns saving-the-voices/gs_mm interview audio and transcript
using DialogueAlignmentPipeline with calibrated code-switching English projection.
"""

from pathlib import Path
from digohwelisgi.alignment.models import CTCAlignerConfig
from digohwelisgi.pipelines.dialogue import align_dialogue

AUDIO_PATH = Path("data/projects/saving-the-voices/gs_mm.wav")
TRANSCRIPT_PATH = Path("data/projects/saving-the-voices/gs_mm.txt")
OUTPUT_DIR = Path("data/projects/saving-the-voices/output_ctc_no_hs")

if __name__ == "__main__":
    config = CTCAlignerConfig(
        enforce_phonotactics=True,
        cache=True,
        enable_vad_soft_masking=True,
        vad_p_low=0.30,
        vad_p_high=0.75,
        vad_pad_ms=20,
    )
    align_dialogue(
        audio=AUDIO_PATH,
        transcript=TRANSCRIPT_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
        code_switched=True,
        strip_speaker=True,
        reconcile=True,
        export_praat=True,
        export_manifest=True,
        textgrid_filename="gs_mm_ctc.TextGrid",
        manifest_filename="gs_mm_ctc_manifest.json",
    )
