# -*- coding: utf-8 -*-
"""
transcription.apps package.

Application and CLI entrypoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcription.apps.cli import main, run_alignment_cli  # noqa: F401


def __getattr__(name: str):  # type: ignore[misc]
    if name in ("main", "run_alignment_cli"):
        from transcription.apps import cli as _cli

        return getattr(_cli, name)
    raise AttributeError(f"module 'transcription.apps' has no attribute {name!r}")
