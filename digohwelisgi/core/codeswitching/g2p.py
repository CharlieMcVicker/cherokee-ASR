# -*- coding: utf-8 -*-
"""
digohwelisgi.core.codeswitching.g2p

Grapheme-to-Phoneme (G2P) extraction for English text using g2p_en.
Normalizes text, strips punctuation, and maps words to standardized stress-stripped
or stress-annotated ARPAbet phoneme sequences conforming to EnglishToArpabetProtocol.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple
import g2p_en

from digohwelisgi.core.codeswitching.types import (
    STANDARD_ARPABET_PHONEMES,
    ARPAbetPhone,
    EnglishToArpabetProtocol,
)

# Regex matching a valid ARPAbet phone token with optional trailing stress digit (0, 1, 2)
_ARPABET_PHONE_RE = re.compile(r"^([A-Za-z]+)(\d)?$")


class G2PEngine:
    """
    G2P engine implementing EnglishToArpabetProtocol using g2p_en.G2p.

    Normalizes input text, filters whitespace and punctuation tokens, and returns
    immutable ARPAbetPhone instances conforming to the standard 39-phoneme inventory.
    """

    def __init__(self, g2p: Optional[g2p_en.G2p] = None) -> None:
        self._g2p = g2p if g2p is not None else g2p_en.G2p()

    def extract(self, text: str, strip_stress: bool = True) -> Tuple[ARPAbetPhone, ...]:
        """
        Extract standardized ARPAbet tokens from English text.

        Args:
            text: Input English word, phrase, or sentence.
            strip_stress: If True, stress markers (0, 1, 2) are stripped (stress=None).
                          If False, stress markers are retained on the ARPAbetPhone.

        Returns:
            Tuple of ARPAbetPhone instances.
        """
        cleaned = text.strip()
        if not cleaned:
            return ()

        raw_tokens: Sequence[str] = self._g2p(cleaned)
        extracted: list[ARPAbetPhone] = []

        for raw in raw_tokens:
            token_str = str(raw).strip()
            if not token_str:
                continue

            match = _ARPABET_PHONE_RE.match(token_str)
            if not match:
                # Punctuation or non-phonemic symbol
                continue

            phone = match.group(1).upper()
            stress_digit = int(match.group(2)) if match.group(2) is not None else None

            # Ensure phone is part of standard ARPAbet inventory
            if phone not in STANDARD_ARPABET_PHONEMES:
                continue

            if strip_stress:
                extracted.append(ARPAbetPhone(phone=phone, stress=None))
            else:
                extracted.append(ARPAbetPhone(phone=phone, stress=stress_digit))

        return tuple(extracted)

    def __call__(
        self, text: str, strip_stress: bool = True
    ) -> Tuple[ARPAbetPhone, ...]:
        """Callable alias satisfying EnglishToArpabetProtocol."""
        return self.extract(text, strip_stress=strip_stress)

    extract_arpabet = extract


# Alias for compatibility
G2pExtractor = G2PEngine

# Global singleton instance for efficient reuse across the pipeline
_DEFAULT_EXTRACTOR: Optional[G2PEngine] = None


def get_default_g2p() -> G2PEngine:
    """Return or lazily initialize the default global G2PEngine instance."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = G2PEngine()
    return _DEFAULT_EXTRACTOR


def extract_arpabet(
    text: str,
    strip_stress: bool = True,
    extractor: Optional[EnglishToArpabetProtocol] = None,
) -> Tuple[ARPAbetPhone, ...]:
    """
    Convenience function to extract ARPAbet tokens from text.

    Args:
        text: Input English text.
        strip_stress: Whether to strip numeric stress digits.
        extractor: Optional custom extractor implementing EnglishToArpabetProtocol.
    """
    g2p = extractor if extractor is not None else get_default_g2p()
    return g2p.extract(text, strip_stress=strip_stress)


__all__ = [
    "G2PEngine",
    "G2pExtractor",
    "extract_arpabet",
    "get_default_g2p",
]
