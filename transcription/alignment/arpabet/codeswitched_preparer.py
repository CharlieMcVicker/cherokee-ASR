# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.codeswitched_preparer module.

Compatibility shim forwarding to transcription.cherokee.codeswitching.codeswitched_preparer.
"""

from transcription.cherokee.codeswitching.codeswitched_preparer import (
    CodeSwitchedLineResult,
    CodeSwitchedPreparer,
    CodeSwitchedToken,
    TokenType,
    classify_token,
    create_groundtruth_for_code_switched_syllabary,
    extract_speaker_prefix,
    prepare_code_switched_token,
    split_compound_clitic,
    strip_boundary_punctuation,
)

__all__ = [
    "TokenType",
    "CodeSwitchedToken",
    "CodeSwitchedLineResult",
    "CodeSwitchedPreparer",
    "strip_boundary_punctuation",
    "split_compound_clitic",
    "classify_token",
    "prepare_code_switched_token",
    "extract_speaker_prefix",
    "create_groundtruth_for_code_switched_syllabary",
]
