# -*- coding: utf-8 -*-
"""
test_cherokee_codeswitching.py

Unit tests for digohwelisgi.cherokee.codeswitching module:
- SyntheticTargetProjector with static dictionary and dynamic fallback
- make_cherokee_projector factory and default path resolution
- CodeSwitchedPreparer, CodeSwitchedToken, TokenType, TokenClassification
- Compound clitic segmentation and zero double-conversion verification (e.g. coffee -> khasi, JayᎢ -> tsei)
- Dictionary path resolution relative to repo root data/arpabet_alignment/dictionaries/english_loanwords_tth.json
"""

import json
from pathlib import Path
import tempfile
import pytest

from digohwelisgi.cherokee.codeswitching.types import (
    ArpabetToken,
    CherokeeToken,
    SyntheticCherokeeTarget,
    SyntheticTargetProjectorProtocol,
)
from digohwelisgi.cherokee.codeswitching import (
    CHEROKEE_TTH_TARGET_PHONEMES,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_STATIC_DICTIONARY_PATH,
    CherokeeSyntheticTargetProjector,
    CodeSwitchedLineResult,
    CodeSwitchedPreparer,
    CodeSwitchedToken,
    SyntheticTargetProjector,
    TokenClassification,
    TokenType,
    classify_token,
    create_groundtruth_for_code_switched_syllabary,
    extract_speaker_prefix,
    generate_static_dictionary,
    get_default_projector,
    get_english_loanwords_tth_dict,
    is_english_word,
    load_default_confusion_matrix,
    make_cherokee_projector,
    normalize_code_switched_text,
    prepare_code_switched_token,
    project_english_text,
    project_english_word,
    split_compound_clitic,
    strip_boundary_punctuation,
)


@pytest.fixture(scope="module")
def default_matrix():
    return load_default_confusion_matrix(DEFAULT_CONFUSION_MATRIX_PATH)


@pytest.fixture(scope="module")
def default_projector(default_matrix):
    return get_default_projector()


def test_dictionary_resolution():
    """Verify dictionary path resolves relative to data/arpabet_alignment/dictionaries/english_loanwords_tth.json."""
    assert DEFAULT_STATIC_DICTIONARY_PATH.exists()
    assert DEFAULT_STATIC_DICTIONARY_PATH.name == "english_loanwords_tth.json"
    data = get_english_loanwords_tth_dict()
    assert isinstance(data, dict)
    assert len(data) > 0
    assert "coffee" in data


def test_projector_protocol_compliance(default_projector):
    assert isinstance(default_projector, SyntheticTargetProjectorProtocol)


def test_make_cherokee_projector_factory():
    projector = make_cherokee_projector()
    assert isinstance(projector, CherokeeSyntheticTargetProjector)
    assert isinstance(projector, SyntheticTargetProjector)
    assert len(projector.dictionary) > 0
    assert "coffee" in projector.dictionary
    assert CHEROKEE_TTH_TARGET_PHONEMES is not None
    assert len(CHEROKEE_TTH_TARGET_PHONEMES) > 0


def test_is_english_word():
    assert not is_english_word("ᏣᎳᎩ")
    assert not is_english_word("ᎯᎠ")
    assert not is_english_word("ᎠᎩᏚᎵ")
    assert is_english_word("coffee")
    assert is_english_word("hospital")
    assert not is_english_word("")
    assert not is_english_word("1234")


def test_project_word_static_and_dynamic(default_projector):
    # 'coffee' is in static dictionary -> 'khasi'
    target_coffee = default_projector.project_word("coffee")
    assert isinstance(target_coffee, SyntheticCherokeeTarget)
    assert target_coffee.source_word == "coffee"
    assert target_coffee.projected_tth == "khasi"
    assert target_coffee.confidence_score > 0.0
    assert target_coffee.syllabary == "ᎧᏏ"

    # Dynamic fallback word
    target_unknown = default_projector.project_word("superconductor")
    assert isinstance(target_unknown, SyntheticCherokeeTarget)
    assert len(target_unknown.projected_tth) > 0
    # Consonant inventory check: strictly no d and no g
    assert "d" not in target_unknown.projected_tth
    assert "g" not in target_unknown.projected_tth


def test_split_compound_clitic():
    assert split_compound_clitic("JayᎢ") == ("Jay", "Ꭲ")
    assert split_compound_clitic("WellingᏛ") == ("Welling", "Ꮫ")
    assert split_compound_clitic("CooksonᎢ,") == ("Cookson", "Ꭲ")
    assert split_compound_clitic("Soldier") is None
    assert split_compound_clitic("ᏣᎳᎩ") is None


def test_classify_token():
    assert classify_token("ᎯᎠ") == TokenType.CHEROKEE_SYLLABARY
    assert classify_token("Soldier") == TokenType.ENGLISH
    assert classify_token("JayᎢ") == TokenType.COMPOUND_CLITIC
    assert classify_token("...") == TokenType.PUNCTUATION
    assert TokenClassification == TokenType


def test_prepare_code_switched_token_zero_double_conversion(default_projector):
    """Ensure English tokens are NEVER passed through Cherokee DG-to-TTH consonant mutation."""
    tok_soldier = prepare_code_switched_token("Soldier", projector=default_projector)
    assert tok_soldier.token_type == TokenType.ENGLISH
    assert tok_soldier.english_stem == "Soldier"
    assert "hsow" in tok_soldier.canonical_tth

    tok_jay = prepare_code_switched_token("JayᎢ", projector=default_projector)
    assert tok_jay.token_type == TokenType.COMPOUND_CLITIC
    assert tok_jay.english_stem == "Jay"
    assert tok_jay.syllabary_clitic == "Ꭲ"
    assert tok_jay.canonical_tth == "tsei"


def test_codeswitched_preparer_class(default_projector):
    preparer = CodeSwitchedPreparer(projector=default_projector)
    line_res = preparer.prepare_line("Guy Soldier: ᎯᏅ ᎣᏏᏍ ᎭᏛᎩ?", strip_speaker=True)
    assert isinstance(line_res, CodeSwitchedLineResult)
    assert line_res.speaker == "Guy Soldier"
    assert line_res.unified_tth == "hinv ohsihs hatvki"


def test_normalize_code_switched_text(default_projector):
    mixed = "ᎯᎠ coffee ᎠᎩᏚᎵ"
    norm = normalize_code_switched_text(mixed, projector=default_projector)
    assert norm == "hi'a khasi akituli"
