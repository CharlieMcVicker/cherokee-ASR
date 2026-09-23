# -*- coding: utf-8 -*-
"""
cli.py

Backwards-compatible forwarding shim for align-cherokee CLI.
Relocated to transcription.apps.cli.
"""

from transcription.apps.cli import main, run_alignment_cli, run_alignment_pipeline

__all__ = [
    "main",
    "run_alignment_cli",
    "run_alignment_pipeline",
]

if __name__ == "__main__":
    main()
