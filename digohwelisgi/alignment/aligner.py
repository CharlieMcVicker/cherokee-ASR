# -*- coding: utf-8 -*-
"""
aligner.py

Chunk-level Sliding Window DTW and Word-level Needleman-Wunsch DP alignment engine.
Re-exported from digohwelisgi.core.alignment.dp for backward compatibility.
"""

from digohwelisgi.core.alignment.dp import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)

__all__ = [
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
]
