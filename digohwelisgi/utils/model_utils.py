import json
import os
from typing import Any, Optional, Tuple, Union

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

_MODEL_CACHE: dict[tuple[Any, ...], tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, str]] = {}


def get_best_model_config() -> dict[str, str]:
    """
    Finds and loads the best_model.json configuration file.
    Traverses up from the directory containing this file until it finds best_model.json,
    or falls back to checking the current working directory, then the parent directories.
    If not found, returns the default fallback.
    """
    # Start traversing up from the current file's directory
    start_dirs = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]

    for start_dir in start_dirs:
        current_dir = start_dir
        while current_dir and current_dir != os.path.dirname(current_dir):
            config_path = os.path.join(current_dir, "best_model.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        if "repo" in config and "revision" in config:
                            return config
                except Exception as e:
                    print(
                        f"Warning: Failed to load best_model.json from {config_path}: {e}"
                    )
            current_dir = os.path.dirname(current_dir)

    # Fallback default values
    return {"repo": "charliemcvicker/asr-cherokee", "revision": "5464d15"}


def get_model(
    path_or_repo: Optional[str] = None,
    revision: Optional[str] = None,
    processor_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    token: Optional[str] = None,
    eval_mode: bool = True,
    use_cache: bool = True,
) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, str]:
    """
    Unified entrypoint to load a Wav2Vec2 model and processor based on repository/path and revision.
    If path_or_repo is None, automatically loads the configuration from best_model.json.
    """
    if path_or_repo is None:
        config = get_best_model_config()
        path_or_repo = config["repo"]
        if revision is None:
            revision = config.get("revision")

    token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    dev_str = (
        str(device)
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    )

    proc_path = processor_path or path_or_repo
    cache_key = (path_or_repo, revision, proc_path, dev_str, eval_mode)
    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    load_kwargs: dict[str, Any] = {}
    if token:
        load_kwargs["token"] = token
    if revision:
        load_kwargs["revision"] = revision

    processor: Any = Wav2Vec2Processor.from_pretrained(proc_path, **load_kwargs)
    model: Any = Wav2Vec2ForCTC.from_pretrained(path_or_repo, **load_kwargs)

    if eval_mode:
        model.eval()
    model.to(dev_str)

    result = (model, processor, dev_str)
    if use_cache:
        _MODEL_CACHE[cache_key] = result
    return result


def get_best_model(
    device: Optional[Union[str, torch.device]] = None,
    token: Optional[str] = None,
    eval_mode: bool = True,
    use_cache: bool = True,
) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, str]:
    """
    Loads the best model and processor as specified in best_model.json.
    """
    return get_model(
        path_or_repo=None,
        revision=None,
        device=device,
        token=token,
        eval_mode=eval_mode,
        use_cache=use_cache,
    )
