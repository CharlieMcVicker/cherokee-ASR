#!/usr/bin/env python3
"""
CLI entry point for finding optimal alignment cost threshold T*
via interactive binary search or automated quantile thresholding.
"""

import sys
from transcription.alignment.threshold_finder import main

if __name__ == "__main__":
    sys.exit(main())
