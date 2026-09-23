# -*- coding: utf-8 -*-
"""
transcription.cherokee.codeswitching

Cherokee code-switching module: ARPAbet loanword projection, compound clitic parsing,
and script discrimination for mixed English and Cherokee Syllabary texts.
"""

from transcription.cherokee.codeswitching.projector import (
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_STATIC_DICTIONARY_PATH,
    SyntheticTargetProjector,
    generate_static_dictionary,
    get_default_projector,
    get_english_loanwords_tth_dict,
    is_english_word,
    load_default_confusion_matrix,
    normalize_code_switched_text,
    project_english_text,
    project_english_word,
)
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
    # Projector
    "DEFAULT_CONFUSION_MATRIX_PATH",
    "DEFAULT_STATIC_DICTIONARY_PATH",
    "SyntheticTargetProjector",
    "generate_static_dictionary",
    "get_default_projector",
    "get_english_loanwords_tth_dict",
    "is_english_word",
    "load_default_confusion_matrix",
    "normalize_code_switched_text",
    "project_english_text",
    "project_english_word",
    # Preparer & Tokens
    "CodeSwitchedLineResult",
    "CodeSwitchedPreparer",
    "CodeSwitchedToken",
    "TokenType",
    "classify_token",
    "create_groundtruth_for_code_switched_syllabary",
    "extract_speaker_prefix",
    "prepare_code_switched_token",
    "split_compound_clitic",
    "strip_boundary_punctuation",
]
