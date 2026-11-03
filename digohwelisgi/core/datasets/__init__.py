# -*- coding: utf-8 -*-
"""
digohwelisgi.core.datasets

Language-agnostic dataset specifications, packaging, and staging engine.
"""

from digohwelisgi.core.datasets.types import (
    AudioResolver,
    DatasetSplit,
    PackagedAudioRecord,
    PackagingConfig,
    PackagingManifest,
    PackagingResult,
)
from digohwelisgi.core.datasets.packaging import (
    compute_file_sha256,
    default_audio_resolver,
    inspect_audio_file,
    package_audio_dataset,
    upload_to_huggingface_hub,
)

__all__ = [
    "AudioResolver",
    "DatasetSplit",
    "PackagedAudioRecord",
    "PackagingConfig",
    "PackagingManifest",
    "PackagingResult",
    "compute_file_sha256",
    "default_audio_resolver",
    "inspect_audio_file",
    "package_audio_dataset",
    "upload_to_huggingface_hub",
]
