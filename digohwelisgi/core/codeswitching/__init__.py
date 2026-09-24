# -*- coding: utf-8 -*-
"""
digohwelisgi.core.codeswitching

Language-agnostic cross-lingual codeswitching, ARPAbet phonetic tokenization,
G2P extraction, acoustic confusion matrix modeling, EM training, and Viterbi target projection.
"""

from digohwelisgi.core.codeswitching.types import (
    EPSILON_TOKEN,
    STANDARD_ARPABET_CONSONANTS,
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
    AlignmentPath,
    ARPAbetPhone,
    ArpabetToken,
    BeamHypothesis,
    ConfusionEntry,
    CrossEntropyLoss,
    EnglishToArpabetProtocol,
    GenericConfusionMatrixProtocol,
    GenericSyntheticTargetProjectorProtocol,
    JointNgram,
    ProjectedTarget,
    StressPattern,
    SubstitutionMapping,
    TargetPhone,
    TracebackResult,
    WordManifestEntry,
)
from digohwelisgi.core.codeswitching.g2p import (
    G2PEngine,
    G2pExtractor,
    extract_arpabet,
    get_default_g2p,
)
from digohwelisgi.core.codeswitching.matrix import (
    AcousticConfusionMatrix,
    normalize_source_key,
    normalize_target_key,
)
from digohwelisgi.core.codeswitching.trainer import (
    ConfusionMatrixTrainer,
    align_word_pair_generic,
    bucket_by_duration,
    ctc_prefix_beam_search,
    sanitize_model_id,
)
from digohwelisgi.core.codeswitching.projector import (
    SyntheticTargetProjector,
    generate_static_dictionary,
)

__all__ = [
    # Types
    "EPSILON_TOKEN",
    "STANDARD_ARPABET_CONSONANTS",
    "STANDARD_ARPABET_PHONEMES",
    "STANDARD_ARPABET_VOWELS",
    "AlignmentPath",
    "ARPAbetPhone",
    "ArpabetToken",
    "BeamHypothesis",
    "ConfusionEntry",
    "CrossEntropyLoss",
    "EnglishToArpabetProtocol",
    "GenericConfusionMatrixProtocol",
    "GenericSyntheticTargetProjectorProtocol",
    "JointNgram",
    "ProjectedTarget",
    "StressPattern",
    "SubstitutionMapping",
    "TargetPhone",
    "TracebackResult",
    # G2P
    "G2PEngine",
    "G2pExtractor",
    "extract_arpabet",
    "get_default_g2p",
    # Matrix
    "AcousticConfusionMatrix",
    "normalize_source_key",
    "normalize_target_key",
    # Trainer
    "ConfusionMatrixTrainer",
    "align_word_pair_generic",
    "bucket_by_duration",
    "ctc_prefix_beam_search",
    "sanitize_model_id",
    # Projector
    "SyntheticTargetProjector",
    "generate_static_dictionary",
]
