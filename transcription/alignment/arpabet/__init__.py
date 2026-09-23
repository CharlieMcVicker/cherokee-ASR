# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet package.

Acoustic alignment and empirical statistical mapping between ARPAbet
phonetic sequences and Cherokee ASR emissions.
"""

from transcription.alignment.arpabet.types import (
    CANONICAL_CHEROKEE_CONSONANTS,
    CANONICAL_CHEROKEE_PHONEMES,
    CANONICAL_CHEROKEE_TTH_PHONEMES,
    CANONICAL_CHEROKEE_VOWELS,
    EPSILON_TOKEN,
    STANDARD_ARPABET_CONSONANTS,
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
    AcousticConfusionMatrix,
    AlignedTokenPair,
    ArpabetToken,
    CherokeeToken,
    DPTracebackAccumulatorProtocol,
    EnglishToArpabetProtocol,
    G2PExtractorProtocol,
    InferenceCacheManifest,
    SyntheticCherokeeTarget,
    SyntheticTargetProjectorProtocol,
    TopKHypothesis,
    TracebackAlignerProtocol,
    TracebackAlignmentResult,
    WordInferenceCacheEntry,
    WordManifestEntry,
)
from transcription.alignment.arpabet.dataset import (
    PhoneticWordBalancer,
    WordCandidate,
    extract_balanced_dataset,
    extract_word_clip,
    iter_librispeech_utterances,
    load_words_manifest,
    save_words_manifest,
)
from transcription.alignment.arpabet.forced_aligner import (
    AlignedWordSpan,
    ForcedAlignerProtocol,
    MMSForcedAligner,
    get_default_forced_aligner,
)
from transcription.alignment.arpabet.g2p import (
    G2pExtractor,
    extract_arpabet,
    get_default_g2p,
)
from transcription.alignment.arpabet.inference import (
    bucket_by_duration,
    ctc_prefix_beam_search,
    decode_greedy_with_confidences,
    extract_top_k_hypotheses,
    group_digraphs_with_confidences,
    run_model_inference_on_manifest,
    sanitize_model_id,
)
from transcription.alignment.arpabet.matrix import (
    ARTICULATORY_FEATURE_DISTANCES,
    WagnerFischerAligner,
    align_word_pair,
    build_articulatory_seed_matrix,
    get_articulatory_distance,
    train_acoustic_confusion_matrix,
)

__all__ = [
    # Constants
    "EPSILON_TOKEN",
    "STANDARD_ARPABET_VOWELS",
    "STANDARD_ARPABET_CONSONANTS",
    "STANDARD_ARPABET_PHONEMES",
    "CANONICAL_CHEROKEE_VOWELS",
    "CANONICAL_CHEROKEE_CONSONANTS",
    "CANONICAL_CHEROKEE_TTH_PHONEMES",
    "CANONICAL_CHEROKEE_PHONEMES",
    # Phonetic Tokens
    "ArpabetToken",
    "CherokeeToken",
    # Manifest & Cache Models
    "WordManifestEntry",
    "TopKHypothesis",
    "WordInferenceCacheEntry",
    "InferenceCacheManifest",
    # Matrix & Target Models
    "AcousticConfusionMatrix",
    "SyntheticCherokeeTarget",
    # DP Alignment Models
    "AlignedTokenPair",
    "TracebackAlignmentResult",
    # Pure Mapping Protocols
    "EnglishToArpabetProtocol",
    "G2PExtractorProtocol",
    "TracebackAlignerProtocol",
    "DPTracebackAccumulatorProtocol",
    "SyntheticTargetProjectorProtocol",
    # G2P Extractor
    "G2pExtractor",
    "extract_arpabet",
    "get_default_g2p",
    # Forced Aligner
    "AlignedWordSpan",
    "ForcedAlignerProtocol",
    "MMSForcedAligner",
    "get_default_forced_aligner",
    # Dataset & Balancer
    "WordCandidate",
    "PhoneticWordBalancer",
    "extract_word_clip",
    "save_words_manifest",
    "load_words_manifest",
    "iter_librispeech_utterances",
    "extract_balanced_dataset",
    # Inference & Caching
    "sanitize_model_id",
    "bucket_by_duration",
    "group_digraphs_with_confidences",
    "decode_greedy_with_confidences",
    "ctc_prefix_beam_search",
    "extract_top_k_hypotheses",
    "run_model_inference_on_manifest",
    # Matrix, Seed & Alignment
    "ARTICULATORY_FEATURE_DISTANCES",
    "get_articulatory_distance",
    "build_articulatory_seed_matrix",
    "align_word_pair",
    "WagnerFischerAligner",
    "train_acoustic_confusion_matrix",
]
