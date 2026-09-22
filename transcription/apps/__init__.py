# -*- coding: utf-8 -*-
"""
transcription.apps package.

Application and CLI entrypoints.
"""

from transcription.apps.cli import main, run_alignment_cli

__all__ = [
    "main",
    "run_alignment_cli",
]
