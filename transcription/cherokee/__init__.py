# -*- coding: utf-8 -*-
"""
transcription.cherokee

Domain-specific Cherokee phonetic, linguistic, orthographic representations, and models.
"""

from transcription.cherokee.orthography import (
    CHEROKEE_SYLLABARY_BASE_MAP,
    CHEROKEE_SYLLABARY_MAP,
    CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP,
    PHONETIC_TO_SYLLABARY_MAP,
    Orthography,
    clean_punctuation_and_whitespace,
    convert_orthography,
    phonetics_to_syllabary,
    remove_tones_and_double_vowels,
    replace_tones,
    respell_consonants,
    strip_tones_and_colons,
    syllabary_to_phonetics,
)
from transcription.cherokee.phonotactics import (
    PhonemeCategory,
    PhonotacticAnalysis,
    PhonotacticToken,
    analyze_phonotactics,
    get_intrusion_site_mask,
    get_syncope_mask,
    is_valid_phonotactic_sequence,
    prepare_cherokee_text,
    tokenize_phonemes,
)
from transcription.cherokee.models import (
    CherokeeASRModel,
    load_cherokee_asr_model,
)
from transcription.cherokee.distance import (
    ConfusionMatrixCostMetric,
    PhonologicalConfusionCostMetric,
)
from transcription.cherokee.enrichment import (
    SyllableAlignment,
    SyllableAlignmentEngine,
    align_character_syllable,
    align_character_syllable_detailed,
    get_base_transliteration,
    is_cherokee_syllable,
    reconcile_alignment_by_chunk,
    reconcile_alignment_words,
    reconcile_phonetics,
    reconcile_word_intervals,
)
from transcription.cherokee.codeswitching import (
    CodeSwitchedLineResult,
    CodeSwitchedPreparer,
    CodeSwitchedToken,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_STATIC_DICTIONARY_PATH,
    SyntheticTargetProjector,
    TokenType,
    classify_token,
    create_groundtruth_for_code_switched_syllabary,
    extract_speaker_prefix,
    generate_static_dictionary,
    get_default_projector,
    get_english_loanwords_tth_dict,
    is_english_word,
    load_default_confusion_matrix,
    normalize_code_switched_text,
    prepare_code_switched_token,
    project_english_text,
    project_english_word,
    split_compound_clitic,
    strip_boundary_punctuation,
)

__all__ = [
    # Orthography
    "Orthography",
    "convert_orthography",
    "clean_punctuation_and_whitespace",
    "strip_tones_and_colons",
    # Syllabary Map
    "CHEROKEE_SYLLABARY_BASE_MAP",
    "CHEROKEE_SYLLABARY_UNCONDITIONAL_MAP",
    "CHEROKEE_SYLLABARY_MAP",
    "PHONETIC_TO_SYLLABARY_MAP",
    "syllabary_to_phonetics",
    "phonetics_to_syllabary",
    # Tones
    "respell_consonants",
    "replace_tones",
    "remove_tones_and_double_vowels",
    # Models
    "CherokeeASRModel",
    "load_cherokee_asr_model",
    # Distance Metrics
    "ConfusionMatrixCostMetric",
    "PhonologicalConfusionCostMetric",
    # Phonotactics
    "PhonemeCategory",
    "PhonotacticToken",
    "PhonotacticAnalysis",
    "tokenize_phonemes",
    "get_syncope_mask",
    "get_intrusion_site_mask",
    "is_valid_phonotactic_sequence",
    "analyze_phonotactics",
    "prepare_cherokee_text",
    # Codeswitching
    "CodeSwitchedLineResult",
    "CodeSwitchedPreparer",
    "CodeSwitchedToken",
    "DEFAULT_CONFUSION_MATRIX_PATH",
    "DEFAULT_STATIC_DICTIONARY_PATH",
    "SyntheticTargetProjector",
    "TokenType",
    "classify_token",
    "create_groundtruth_for_code_switched_syllabary",
    "extract_speaker_prefix",
    "generate_static_dictionary",
    "get_default_projector",
    "get_english_loanwords_tth_dict",
    "is_english_word",
    "load_default_confusion_matrix",
    "normalize_code_switched_text",
    "prepare_code_switched_token",
    "project_english_text",
    "project_english_word",
    "split_compound_clitic",
    "strip_boundary_punctuation",
    # Enrichment
    "SyllableAlignment",
    "SyllableAlignmentEngine",
    "align_character_syllable",
    "align_character_syllable_detailed",
    "get_base_transliteration",
    "is_cherokee_syllable",
    "reconcile_alignment_by_chunk",
    "reconcile_alignment_words",
    "reconcile_phonetics",
    "reconcile_word_intervals",
]
