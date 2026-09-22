# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.projector module.

Compatibility shim forwarding to transcription.cherokee.codeswitching.projector.
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

__all__ = [
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
]
