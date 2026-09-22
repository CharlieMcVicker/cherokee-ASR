# -*- coding: utf-8 -*-
"""
phonotactics.py

Cherokee Phonotactic Engine, Rule Parser, and CTC Text Preparer.
Provides domain types, phoneme tokenization, syncope masking, intrusive
laryngeal (/h/, /'/) candidate site masking, surface constraints (*HH, *C', *ChR),
and prepare_cherokee_text conforming to TextPreparerProtocol.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


class PhonemeCategory(str, Enum):
    """Phonotactic categories for Cherokee phonemes and clusters."""

    VOWEL = "vowel"
    PLAIN_STOP = "plain_stop"  # t, k, kw, ts, tl
    ASPIRATED_STOP = "aspirated_stop"  # th, kh, kwh, tsh, tlh
    SIBILANT = "sibilant"  # s, hs
    PLAIN_SONORANT = "plain_sonorant"  # l, m, n, w, y
    VOICELESS_SONORANT = "voiceless_sonorant"  # lh, nh, wh, yh
    LARYNGEAL_FRICATIVE = "laryngeal_fricative"  # h
    GLOTTAL_STOP = "glottal_stop"  # '
    SIBILANT_CLUSTER = "sibilant_cluster"  # hsk, hst, hskw, hsl, hsts, slh
    OTHER = "other"


# Canonical token inventories
VOWEL_SET: Set[str] = set("aeiouvAEIOUV")
PLAIN_STOP_SET: Set[str] = {"t", "k", "kw", "ts", "tl"}
ASPIRATED_STOP_SET: Set[str] = {"th", "kh", "kwh", "tsh", "tlh"}
SIBILANT_SET: Set[str] = {"s", "hs"}
PLAIN_SONORANT_SET: Set[str] = {"l", "m", "n", "w", "y"}
VOICELESS_SONORANT_SET: Set[str] = {"lh", "nh", "wh", "yh"}
LARYNGEAL_SET: Set[str] = {"h", "'"}
SIBILANT_CLUSTER_SET: Set[str] = {"hskw", "hsts", "hsk", "hst", "hsl", "slh"}

# Multi-character units ordered longest-to-shortest for longest-prefix tokenization
_CANONICAL_TOKEN_SPECS: List[Tuple[str, PhonemeCategory]] = [
    # 2-character digraphs
    ("hskw", PhonemeCategory.SIBILANT_CLUSTER),
    ("hsts", PhonemeCategory.SIBILANT_CLUSTER),
    ("hsk", PhonemeCategory.SIBILANT_CLUSTER),
    ("hst", PhonemeCategory.SIBILANT_CLUSTER),
    ("hsl", PhonemeCategory.SIBILANT_CLUSTER),
    ("slh", PhonemeCategory.SIBILANT_CLUSTER),
    ("kwh", PhonemeCategory.ASPIRATED_STOP),
    ("tlh", PhonemeCategory.ASPIRATED_STOP),
    ("tsh", PhonemeCategory.ASPIRATED_STOP),
    ("ch", PhonemeCategory.ASPIRATED_STOP),
    ("kw", PhonemeCategory.PLAIN_STOP),
    ("gw", PhonemeCategory.PLAIN_STOP),
    ("qu", PhonemeCategory.PLAIN_STOP),
    ("th", PhonemeCategory.ASPIRATED_STOP),
    ("kh", PhonemeCategory.ASPIRATED_STOP),
    ("ts", PhonemeCategory.PLAIN_STOP),
    ("tl", PhonemeCategory.PLAIN_STOP),
    ("hs", PhonemeCategory.SIBILANT),
    ("lh", PhonemeCategory.VOICELESS_SONORANT),
    ("nh", PhonemeCategory.VOICELESS_SONORANT),
    ("wh", PhonemeCategory.VOICELESS_SONORANT),
    ("yh", PhonemeCategory.VOICELESS_SONORANT),
    # Single characters
    ("t", PhonemeCategory.PLAIN_STOP),
    ("d", PhonemeCategory.PLAIN_STOP),
    ("k", PhonemeCategory.PLAIN_STOP),
    ("g", PhonemeCategory.PLAIN_STOP),
    ("j", PhonemeCategory.PLAIN_STOP),
    ("s", PhonemeCategory.SIBILANT),
    ("l", PhonemeCategory.PLAIN_SONORANT),
    ("m", PhonemeCategory.PLAIN_SONORANT),
    ("n", PhonemeCategory.PLAIN_SONORANT),
    ("w", PhonemeCategory.PLAIN_SONORANT),
    ("y", PhonemeCategory.PLAIN_SONORANT),
    ("h", PhonemeCategory.LARYNGEAL_FRICATIVE),
    ("'", PhonemeCategory.GLOTTAL_STOP),
]


@dataclass(frozen=True)
class PhonotacticToken:
    """Immutable phonetic token representation with category and slice coordinates."""

    symbol: str
    category: PhonemeCategory
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class PhonotacticAnalysis:
    """Immutable full phonotactic analysis result for a given text."""

    text: str
    tokens: Tuple[PhonotacticToken, ...]
    char_syncope_mask: Tuple[bool, ...]
    char_intrusion_mask: Tuple[bool, ...]
    token_syncope_mask: Tuple[bool, ...]
    token_intrusion_mask: Tuple[bool, ...]


def tokenize_phonemes(text: str) -> List[PhonotacticToken]:
    """
    Tokenizes Cherokee phonetic text into classified phonetic tokens,
    resolving multi-character digraphs, trigraphs, and clusters using longest-prefix matching.

    Args:
        text: Transliterated/phonetic Cherokee text.

    Returns:
        List of classified PhonotacticToken objects with span indices.
    """
    tokens: List[PhonotacticToken] = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        # Check vowels first
        if char in VOWEL_SET:
            tokens.append(
                PhonotacticToken(
                    symbol=char,
                    category=PhonemeCategory.VOWEL,
                    start_idx=i,
                    end_idx=i + 1,
                )
            )
            i += 1
            continue

        # Check multi-character and single consonant/laryngeal specs (case-insensitive prefix match)
        matched = False
        lower_sub = text[i:].lower()

        for symbol, cat in _CANONICAL_TOKEN_SPECS:
            if lower_sub.startswith(symbol):
                matched_len = len(symbol)
                actual_str = text[i : i + matched_len]
                tokens.append(
                    PhonotacticToken(
                        symbol=actual_str,
                        category=cat,
                        start_idx=i,
                        end_idx=i + matched_len,
                    )
                )
                i += matched_len
                matched = True
                break

        if not matched:
            # Pass through spaces, punctuation, digits, or unknown characters
            tokens.append(
                PhonotacticToken(
                    symbol=char,
                    category=PhonemeCategory.OTHER,
                    start_idx=i,
                    end_idx=i + 1,
                )
            )
            i += 1

    return tokens


def get_syncope_mask(
    text_or_tokens: Union[str, Sequence[PhonotacticToken]],
    return_char_mask: bool = True,
) -> List[bool]:
    """
    Generates a boolean mask indicating positions eligible for vocalic syncope (vowel deletion).

    In Oklahoma Cherokee:
    - Short vowels in non-initial syllables or open syllables following an onset consonant
      are phonotactically permitted to delete under fast speech / syncopation.
    - Syncope is blocked if deletion would create an illegal non-laryngeal cluster
      or violate forbidden cluster constraints (*ChR).

    Args:
        text_or_tokens: Input Cherokee phonetic text or pre-tokenized sequence of PhonotacticToken.
        return_char_mask: If True, returns mask of length equal to character count.
                          If False, returns mask aligned to token count.

    Returns:
        List[bool] where True marks an eligible syncope site.
    """
    if isinstance(text_or_tokens, str):
        if not text_or_tokens:
            return []
        tokens = tokenize_phonemes(text_or_tokens)
        num_chars = len(text_or_tokens)
    else:
        tokens = list(text_or_tokens)
        if not tokens:
            return []
        num_chars = max((t.end_idx for t in tokens), default=0)

    num_tokens = len(tokens)
    token_mask: List[bool] = [False] * num_tokens

    for idx, tok in enumerate(tokens):
        if tok.category == PhonemeCategory.VOWEL:
            prev_tok = tokens[idx - 1] if idx > 0 else None
            next_tok = tokens[idx + 1] if idx < num_tokens - 1 else None

            # Vowel preceded by an onset consonant is candidate for syncope
            if prev_tok is not None and prev_tok.category in (
                PhonemeCategory.PLAIN_STOP,
                PhonemeCategory.ASPIRATED_STOP,
                PhonemeCategory.SIBILANT,
                PhonemeCategory.PLAIN_SONORANT,
                PhonemeCategory.VOICELESS_SONORANT,
                PhonemeCategory.SIBILANT_CLUSTER,
            ):
                # Check constraint: *ChR (stop + voiceless sonorant is forbidden)
                if (
                    prev_tok.category
                    in (PhonemeCategory.PLAIN_STOP, PhonemeCategory.ASPIRATED_STOP)
                    and next_tok is not None
                    and next_tok.category == PhonemeCategory.VOICELESS_SONORANT
                ):
                    continue

                token_mask[idx] = True

        # Lateral deaffrication: leading 't' in lateral affricate clusters ('tl', 'tlh') can delete in spoken Cherokee
        elif tok.symbol in ("tl", "tlh"):
            token_mask[idx] = True

    if not return_char_mask:
        return token_mask

    char_mask: List[bool] = [False] * num_chars
    for tok, is_syncope in zip(tokens, token_mask):
        if is_syncope:
            if tok.symbol in ("tl", "tlh"):
                # Mark only the initial 't' stop occlusion of the lateral cluster
                if tok.start_idx < num_chars:
                    char_mask[tok.start_idx] = True
            else:
                for c_idx in range(tok.start_idx, tok.end_idx):
                    if c_idx < num_chars:
                        char_mask[c_idx] = True

    return char_mask


def get_intrusion_site_mask(
    text_or_tokens: Union[str, Sequence[PhonotacticToken]],
    return_char_mask: bool = True,
) -> List[bool]:
    """
    Generates a boolean mask indicating positions eligible for intrusive laryngeals
    (/h/ coda aspiration, post-consonantal aspiration, or /'/ glottalization).

    Enforces Cherokee surface phonotactic constraints:
    - *HH constraint: Laryngeals can NEVER occur adjacent to another laryngeal or 'hs'.
    - *C' constraint: Post-consonantal glottal stops are prohibited.
    - *ChR constraint: Stop + voiceless sonorant clusters are prohibited.

    Eligible environments:
    1. Coda laryngeals / Pre-consonantal: V + _ + C (where C is plain stop, sibilant, or sonorant).
    2. Post-consonantal aspiration: C + _ + V (where C is plain stop or plain sonorant).

    Args:
        text_or_tokens: Input Cherokee phonetic text or pre-tokenized sequence of PhonotacticToken.
        return_char_mask: If True, returns mask of length equal to character count.
                          If False, returns mask aligned to token count.

    Returns:
        List[bool] where True marks an eligible intrusive laryngeal site.
    """
    if isinstance(text_or_tokens, str):
        if not text_or_tokens:
            return []
        tokens = tokenize_phonemes(text_or_tokens)
        num_chars = len(text_or_tokens)
    else:
        tokens = list(text_or_tokens)
        if not tokens:
            return []
        num_chars = max((t.end_idx for t in tokens), default=0)

    num_tokens = len(tokens)
    token_mask: List[bool] = [False] * num_tokens

    for idx, tok in enumerate(tokens):
        prev_tok = tokens[idx - 1] if idx > 0 else None

        # 1. Coda laryngeals / Pre-consonantal intrusions (before consonants):
        if (
            tok.category
            in (
                PhonemeCategory.PLAIN_STOP,
                PhonemeCategory.ASPIRATED_STOP,
                PhonemeCategory.PLAIN_SONORANT,
                PhonemeCategory.VOICELESS_SONORANT,
                PhonemeCategory.SIBILANT,
            )
            and not tok.symbol.startswith("h")
            and not tok.symbol.startswith("'")
        ):
            if prev_tok is not None and (
                prev_tok.category
                not in (
                    PhonemeCategory.LARYNGEAL_FRICATIVE,
                    PhonemeCategory.GLOTTAL_STOP,
                    PhonemeCategory.SIBILANT_CLUSTER,
                )
                and not prev_tok.symbol.endswith("h")
                and not prev_tok.symbol.endswith("'")
                and prev_tok.symbol != "hs"
            ):
                token_mask[idx] = True

        # 2. Post-consonantal aspiration (on vowels following plain consonants) and word-initial vowel onsets:
        elif tok.category == PhonemeCategory.VOWEL:
            if prev_tok is None:
                token_mask[idx] = True
            elif (
                prev_tok.category
                in (
                    PhonemeCategory.PLAIN_SONORANT,
                    PhonemeCategory.PLAIN_STOP,
                )
                and not prev_tok.symbol.endswith("h")
                and not prev_tok.symbol.startswith("h")
            ):
                token_mask[idx] = True

    if not return_char_mask:
        return token_mask

    char_mask: List[bool] = [False] * num_chars
    for tok, is_intrusion in zip(tokens, token_mask):
        if is_intrusion:
            for c_idx in range(tok.start_idx, tok.end_idx):
                if c_idx < num_chars:
                    char_mask[c_idx] = True

    return char_mask


def is_valid_phonotactic_sequence(text: str) -> bool:
    """
    Checks if a Cherokee phonetic string satisfies Cherokee surface phonotactic constraints:
    1. *HH Constraint: No adjacent laryngeals ('h', ''', 'hs' doublets like 'hh', ''h', 'h'', '''').
    2. *C' Constraint: No post-consonantal glottal stop (e.g. 'k'', 't'', 'n'').
    3. *ChR Constraint: No stop + voiceless sonorant (e.g. 'thlh', 'khnh').

    Args:
        text: Cherokee phonetic string.

    Returns:
        True if text satisfies constraints, False otherwise.
    """
    if not text:
        return True

    tokens = tokenize_phonemes(text)
    num_tokens = len(tokens)

    for idx in range(num_tokens - 1):
        tok = tokens[idx]
        nxt = tokens[idx + 1]

        # Check *HH constraint (adjacent laryngeals / glottals)
        curr_ends_with_laryngeal = (
            tok.category
            in (PhonemeCategory.LARYNGEAL_FRICATIVE, PhonemeCategory.GLOTTAL_STOP)
            or tok.symbol.endswith("h")
            or tok.symbol.endswith("'")
        )
        nxt_starts_with_laryngeal = (
            nxt.category
            in (PhonemeCategory.LARYNGEAL_FRICATIVE, PhonemeCategory.GLOTTAL_STOP)
            or nxt.symbol.startswith("h")
            or nxt.symbol.startswith("'")
        )

        if curr_ends_with_laryngeal and nxt_starts_with_laryngeal:
            return False

        # Check *C' constraint (post-consonantal glottal stop)
        if (
            tok.category
            in (
                PhonemeCategory.PLAIN_STOP,
                PhonemeCategory.ASPIRATED_STOP,
                PhonemeCategory.PLAIN_SONORANT,
                PhonemeCategory.VOICELESS_SONORANT,
                PhonemeCategory.SIBILANT,
                PhonemeCategory.SIBILANT_CLUSTER,
            )
            and nxt_starts_with_laryngeal
            and nxt.symbol.startswith("'")
        ):
            return False

        # Check *ChR constraint (stop + voiceless sonorant)
        if (
            tok.category in (PhonemeCategory.PLAIN_STOP, PhonemeCategory.ASPIRATED_STOP)
            and nxt.category == PhonemeCategory.VOICELESS_SONORANT
        ):
            return False

    return True


def analyze_phonotactics(text: str) -> PhonotacticAnalysis:
    """
    Performs full phonotactic analysis on Cherokee phonetic text.

    Args:
        text: Input Cherokee phonetic text.

    Returns:
        Immutable PhonotacticAnalysis object with tokens, syncope masks, and intrusion site masks.
    """
    tokens = tuple(tokenize_phonemes(text))
    char_syncope = tuple(get_syncope_mask(tokens, return_char_mask=True))
    char_intrusion = tuple(get_intrusion_site_mask(tokens, return_char_mask=True))
    token_syncope = tuple(get_syncope_mask(tokens, return_char_mask=False))
    token_intrusion = tuple(get_intrusion_site_mask(tokens, return_char_mask=False))

    return PhonotacticAnalysis(
        text=text,
        tokens=tokens,
        char_syncope_mask=char_syncope,
        char_intrusion_mask=char_intrusion,
        token_syncope_mask=token_syncope,
        token_intrusion_mask=token_intrusion,
    )


def prepare_cherokee_text(
    config: Any,
    text: Union[str, Sequence[str]],
    char_list: Optional[Sequence[str]] = None,
    enforce_phonotactics: bool = True,
    token_masks: Optional[Sequence[Tuple[Sequence[bool], Sequence[bool]]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Prepares Cherokee phonetic text for syncope- and intrusion-aware CTC segmentation.

    Constructs the 2D ground truth label matrix (ground_truth_mat) and utterance boundary
    indices, and automatically generates config.is_syncope_token and config.is_intrusive_site
    masks derived from Cherokee surface phonotactics. Implements TextPreparerProtocol.

    Args:
        config: An instance of CtcSegmentationParameters (or compatible configuration object).
        text: Input text as a single string, list of words, or list of utterances.
        char_list: Sequence of vocabulary characters/tokens from the ASR model.
        enforce_phonotactics: If True, generates phonotactically accurate syncope and intrusive masks.
        token_masks: Optional pre-computed sequence of (syncope_mask, intrusion_mask) tuples per word
                     (e.g., from CodeSwitchedToken to strictly isolate English loanwords from Cherokee phonotactics).

    Returns:
        Tuple[np.ndarray, List[int]]:
            - ground_truth_mat: 2D numpy array of shape (L, 1) with token integer indices.
            - utt_begin_indices: List of start row indices for each utterance/word in ground_truth_mat.
    """
    raw_char_list = char_list or getattr(config, "char_list", None)
    if raw_char_list is None:
        raise ValueError(
            "char_list must be provided explicitly or via config.char_list to prepare_cherokee_text"
        )
    c_list = list(raw_char_list)

    space_symbol = getattr(config, "space", " ")
    blank_symbol = int(getattr(config, "blank", 0))

    if isinstance(text, str):
        utterances = [text]
    else:
        utterances = list(text)

    ground_truth: List[str] = [""]
    utt_begin_indices: List[int] = []
    gt_char_coords: List[Tuple[int, bool, bool]] = [
        (0, False, False)
    ]  # (gt_idx, is_syncope, is_intrusion)

    word_mask_idx = 0
    for utt in utterances:
        if ground_truth[-1] != space_symbol:
            ground_truth.append(space_symbol)
            gt_char_coords.append((len(ground_truth) - 1, False, False))

        utt_begin_indices.append(len(ground_truth) - 1)

        words = utt.split() if isinstance(utt, str) else [str(utt)]
        for w_idx, word in enumerate(words):
            if w_idx > 0 and ground_truth[-1] != space_symbol:
                ground_truth.append(space_symbol)
                gt_char_coords.append((len(ground_truth) - 1, False, False))

            if token_masks is not None and word_mask_idx < len(token_masks):
                custom_sync, custom_intrus = token_masks[word_mask_idx]
                word_mask_idx += 1
                word_syncope = (
                    list(custom_sync) if enforce_phonotactics else [False] * len(word)
                )
                word_intrusion = (
                    list(custom_intrus) if enforce_phonotactics else [False] * len(word)
                )
            else:
                word_syncope = (
                    get_syncope_mask(word, return_char_mask=True)
                    if enforce_phonotactics
                    else [False] * len(word)
                )
                word_intrusion = (
                    get_intrusion_site_mask(word, return_char_mask=True)
                    if enforce_phonotactics
                    else [False] * len(word)
                )

            for ch_idx, ch in enumerate(word):
                if ch in c_list:
                    ground_truth.append(ch)
                    is_sync = (
                        word_syncope[ch_idx] if ch_idx < len(word_syncope) else False
                    )
                    is_intrusive = (
                        word_intrusion[ch_idx]
                        if ch_idx < len(word_intrusion)
                        else False
                    )
                    gt_char_coords.append(
                        (len(ground_truth) - 1, is_sync, is_intrusive)
                    )

    if ground_truth[-1] != space_symbol:
        ground_truth.append(space_symbol)
        gt_char_coords.append((len(ground_truth) - 1, False, False))

    utt_begin_indices.append(len(ground_truth) - 1)

    L = len(ground_truth)
    ground_truth_mat = np.ones((L, 1), dtype=np.int64) * -1
    syncope_mask = np.zeros(L, dtype=np.int8)
    intrusion_mask = np.zeros(L, dtype=np.int8)

    for i in range(1, L):
        sym = ground_truth[i]
        if sym == space_symbol:
            ground_truth_mat[i, 0] = blank_symbol
        else:
            ground_truth_mat[i, 0] = c_list.index(sym)

    for gt_idx, is_sync, is_intrusive in gt_char_coords:
        if 0 <= gt_idx < L:
            if is_sync:
                syncope_mask[gt_idx] = 1
            if is_intrusive:
                intrusion_mask[gt_idx] = 1

    if hasattr(config, "is_syncope_token"):
        config.is_syncope_token = syncope_mask
    if hasattr(config, "is_intrusive_site"):
        config.is_intrusive_site = intrusion_mask

    return ground_truth_mat, utt_begin_indices


__all__ = [
    "PhonemeCategory",
    "PhonotacticToken",
    "PhonotacticAnalysis",
    "VOWEL_SET",
    "PLAIN_STOP_SET",
    "ASPIRATED_STOP_SET",
    "SIBILANT_SET",
    "PLAIN_SONORANT_SET",
    "VOICELESS_SONORANT_SET",
    "LARYNGEAL_SET",
    "SIBILANT_CLUSTER_SET",
    "tokenize_phonemes",
    "get_syncope_mask",
    "get_intrusion_site_mask",
    "is_valid_phonotactic_sequence",
    "analyze_phonotactics",
    "prepare_cherokee_text",
]
