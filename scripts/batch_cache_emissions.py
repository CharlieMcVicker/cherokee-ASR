#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_cache_emissions.py

Bulk batch ASR inference script to pre-populate CachedASREmissionsExtractor disk cache.
Processes audio files in batched GPU/CPU passes with dynamic padding, saving TokenEmission
sequences directly to data/runs/cache/emissions/.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from digohwelisgi.core.models.inference import (
    compute_audio_cache_key,
    infer_emissions_batch,
)
from digohwelisgi.cherokee.models import CherokeeASRModel

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_DIR = BASE_DIR / "data/projects/cherokee_new_testament" / "audio_source"
DEFAULT_CACHE_DIR = BASE_DIR / "runs" / "cache" / "emissions"
DEFAULT_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
DEFAULT_MODEL_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def discover_audio_files(
    sources: Sequence[Union[str, Path]],
) -> List[Path]:
    """
    Recursively discovers audio files from a list of paths or directories.
    """
    audio_files: List[Path] = []
    seen = set()

    for src in sources:
        p = Path(src)
        if not p.exists():
            logger.warning("Source path does not exist: %s", p)
            continue
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_AUDIO_EXTS and p.resolve() not in seen:
                audio_files.append(p)
                seen.add(p.resolve())
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                    if f.resolve() not in seen:
                        audio_files.append(f)
                        seen.add(f.resolve())

    return sorted(audio_files)


def batch_cache_emissions(
    audio_paths: Sequence[Union[str, Path]],
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    batch_size: int = 16,
    skip_vad: bool = False,
    force_reload: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs bulk batch inference across audio files and populates ModelOutput .npz disk cache.
    """
    resolved_cache_dir = Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    file_list = [Path(p) for p in audio_paths]
    if not file_list:
        print("No audio files provided to batch_cache_emissions.")
        return {
            "total_files": 0,
            "cache_hits": 0,
            "newly_cached": 0,
            "elapsed_sec": 0.0,
        }

    token = os.environ.get("HF_TOKEN", None)
    prefix_slug = (
        f"{model_repo.replace('/', '_')}_{model_revision}"
        if model_revision
        else model_repo.replace("/", "_")
    )

    print(f"Loading ASR model: {model_repo} (revision: {model_revision}) ...")
    t0_load = time.time()
    asr_model = CherokeeASRModel.from_pretrained_or_best(
        path_or_repo=model_repo,
        revision=model_revision,
        token=token,
        device=device,
    )
    print(f"Model loaded on {asr_model.device} in {time.time() - t0_load:.2f}s")

    # Check cache status upfront
    cached_count = 0
    uncached_count = 0
    for f in file_list:
        k = compute_audio_cache_key(f, model_identifier=prefix_slug)
        c_path = resolved_cache_dir / f"{k}.npz"
        if c_path.exists() and not force_reload:
            cached_count += 1
        else:
            uncached_count += 1

    print(
        f"Found {len(file_list)} files: {cached_count} already cached, {uncached_count} to process (force_reload={force_reload})"
    )

    t0_infer = time.time()
    outputs = infer_emissions_batch(
        model=asr_model.model,
        processor=asr_model.processor,
        audio_inputs=file_list,
        batch_size=batch_size,
        cache_dir=resolved_cache_dir,
        model_identifier=prefix_slug,
        device=asr_model.device,
    )
    elapsed_infer = time.time() - t0_infer

    print("\n" + "=" * 60)
    print("BULK EMISSIONS CACHE SUMMARY")
    print("=" * 60)
    print(f"Total audio files      : {len(file_list)}")
    print(f"Existing cache hits    : {cached_count}")
    print(f"Newly computed/cached  : {uncached_count}")
    print(f"Total ModelOutputs     : {len(outputs)}")
    print(f"Inference / cache time : {elapsed_infer:.2f}s")
    if uncached_count > 0 and elapsed_infer > 0:
        print(f"Average time per file  : {elapsed_infer / uncached_count:.2f}s")
    print(f"Cache location         : {resolved_cache_dir.resolve()}")
    print("=" * 60)

    return {
        "total_files": len(file_list),
        "cache_hits": cached_count,
        "newly_cached": uncached_count,
        "total_outputs": len(outputs),
        "elapsed_sec": elapsed_infer,
        "cache_dir": str(resolved_cache_dir.resolve()),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-populate CachedASREmissionsExtractor disk cache using high-throughput batch inference."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        default=[str(DEFAULT_AUDIO_DIR)],
        help=f"Audio files or directories to process (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help=f"HuggingFace model repository (default: {DEFAULT_MODEL_REPO})",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help=f"HuggingFace model revision (default: {DEFAULT_MODEL_REVISION})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Disk cache output directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for audio tensor collation (default: 16)",
    )
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        default=False,
        help="Bypass VAD segmentation and treat whole files as single chunks",
    )
    parser.add_argument(
        "--force",
        "--force-reload",
        action="store_true",
        dest="force",
        default=False,
        help="Force recomputation even if cached entries exist",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (e.g., 'cuda', 'mps', 'cpu'). Auto-detected by default.",
    )

    args = parser.parse_args()

    audio_files = discover_audio_files(args.sources)
    if not audio_files:
        print(f"No audio files found in: {args.sources}")
        return

    batch_cache_emissions(
        audio_paths=audio_files,
        model_repo=args.model_repo,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        skip_vad=args.skip_vad,
        force_reload=args.force,
        device=args.device,
    )


if __name__ == "__main__":
    main()
