# -*- coding: utf-8 -*-
"""
test_models.py

Unit tests for transcription.cherokee.models.
Tests CherokeeASRModel factory loading, defaults, and methods.
"""

from unittest.mock import MagicMock, patch
import pytest
import torch

from transcription.cherokee.models import (
    DEFAULT_CHEROKEE_FALLBACK_REPO,
    CherokeeASRModel,
    load_cherokee_asr_model,
)


@patch("transcription.cherokee.models.loader.get_model")
def test_cherokee_asr_model_from_pretrained(mock_get_model):
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_get_model.return_value = (mock_model, mock_processor, "cpu")

    model = CherokeeASRModel.from_pretrained("test/repo", revision="rev1")
    assert isinstance(model, CherokeeASRModel)
    assert model.model_name == "test/repo_rev1"
    assert model.device == "cpu"
    mock_get_model.assert_called_once_with(
        path_or_repo="test/repo",
        revision="rev1",
        processor_path=None,
        device=None,
        token=None,
        eval_mode=True,
        use_cache=True,
    )


@patch("transcription.cherokee.models.loader.get_best_model_config")
@patch("transcription.cherokee.models.loader.get_model")
def test_cherokee_asr_model_get_best_model(mock_get_model, mock_get_config):
    mock_get_config.return_value = {"repo": "best/repo", "revision": "best_rev"}
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_get_model.return_value = (mock_model, mock_processor, "cpu")

    model = CherokeeASRModel.get_best_model()
    assert isinstance(model, CherokeeASRModel)
    assert model.model_name == "best/repo_best_rev"
    mock_get_model.assert_called_once_with(
        path_or_repo="best/repo",
        revision="best_rev",
        processor_path=None,
        device=None,
        token=None,
        eval_mode=True,
        use_cache=True,
    )


@patch("transcription.cherokee.models.loader.CherokeeASRModel.get_best_model")
def test_load_cherokee_asr_model_convenience(mock_get_best):
    mock_instance = MagicMock(spec=CherokeeASRModel)
    mock_get_best.return_value = mock_instance

    result = load_cherokee_asr_model()
    assert result == mock_instance
    mock_get_best.assert_called_once()
