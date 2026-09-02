# -*- coding: utf-8 -*-
"""
app.py

FastAPI backend server and PyWebView API bridge for Syllabary Transcriber.
"""

import base64
import os
import struct
from typing import Any, List, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from transcription.models.asr_model import CherokeeASRModel


import logging

logger = logging.getLogger(__name__)


class SyllabaryApi:
    """PyWebView API bridge class for webview integration."""

    def __init__(self, model_dir: str | None = None) -> None:
        self.asr_model: CherokeeASRModel | None = None
        self.model_dir = model_dir

    def _ensure_model_loaded(self) -> None:
        if self.asr_model is None:
            self.asr_model = CherokeeASRModel.from_pretrained_or_best(
                path_or_repo=self.model_dir
            )

    def transcribe_pcm(
        self, pcm_data: Union[List[float], str], sample_rate: int = 16000
    ) -> dict[str, Any]:
        """
        Transcribe audio PCM data passed either as a list of float samples or a base64 encoded string.
        """
        self._ensure_model_loaded()

        if isinstance(pcm_data, str):
            # Decode base64 audio data
            raw_bytes = base64.b64decode(pcm_data)
            # Check if float32 or int16
            if len(raw_bytes) % 4 == 0:
                float_samples = [
                    struct.unpack("<f", raw_bytes[i : i + 4])[0]
                    for i in range(0, len(raw_bytes), 4)
                ]
            elif len(raw_bytes) % 2 == 0:
                int_samples = [
                    struct.unpack("<h", raw_bytes[i : i + 2])[0]
                    for i in range(0, len(raw_bytes), 2)
                ]
                float_samples = [s / 32768.0 for s in int_samples]
            else:
                raise ValueError("Invalid base64 PCM byte length")
            pcm_array = float_samples
        else:
            pcm_array = pcm_data

        logger.info(
            "transcribe_pcm called with %d samples at %d Hz",
            len(pcm_array),
            sample_rate,
        )

        assert self.asr_model is not None
        result = self.asr_model.transcribe(
            audio_input=pcm_array, sample_rate=sample_rate
        )
        res_dict = result.to_dict()
        logger.info("transcribe_pcm result: %s", res_dict)
        return res_dict


app = FastAPI(title="Syllabary Transcriber API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_wasm_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


# Global API instance for server routes
api_instance = SyllabaryApi()


class TranscribeRequest(BaseModel):
    pcm_data: Union[List[float], str] = Field(
        ..., description="List of float PCM samples or base64 encoded string"
    )
    sample_rate: int = Field(default=16000, description="Sampling rate of PCM audio")


@app.post("/api/transcribe-pcm")
def transcribe_pcm_endpoint(request: TranscribeRequest) -> dict[str, Any]:
    try:
        res = api_instance.transcribe_pcm(
            pcm_data=request.pcm_data, sample_rate=request.sample_rate
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import mimetypes

mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")

dist_dir = os.path.join(os.path.dirname(__file__), "ui", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
