from typing import Any
import argparse
import os
import glob
import numpy as np
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings

from transcription.inference.infer import greedy_inference

warnings.filterwarnings("ignore")


def parse_args():
    from transcription.utils.model_utils import get_best_model_config

    model_config = get_best_model_config()
    default_repo = model_config["repo"]

    parser = argparse.ArgumentParser(description="Run Inference on Audio")
    parser.add_argument("--mode", choices=["short", "long"], required=True)
    parser.add_argument("--audio_file", required=True)
    parser.add_argument(
        "--model_dir",
        default=default_repo,
        help="Path to model directory or Hugging Face repo ID.",
    )
    parser.add_argument("--output_tsv", type=str, default="")
    return parser.parse_args()


def convert_to_human_orthography(input_string):
    output_string = input_string
    orth_origin = ["ax", "ex", "ix", "ox", "ux", "q"]
    orth_target = ["ā", "ē", "ī", "ō", "ū", "ꞌ"]
    for o, t in zip(orth_origin, orth_target):
        output_string = output_string.replace(o, t)
    return output_string


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model and processor...")
    from transcription.utils.model_utils import get_best_model_config

    model_config = get_best_model_config()

    revision_str: str | None = None
    if args.model_dir == model_config["repo"]:
        revision_str = str(model_config["revision"])
        print(
            f"Using model configuration from best_model.json: {args.model_dir} (revision: {revision_str})"
        )

    model_kwargs = {}
    if revision_str is not None:
        model_kwargs["revision"] = revision_str

    model_obj = Wav2Vec2ForCTC.from_pretrained(args.model_dir, **model_kwargs)
    model = model_obj.to(device)  # type: ignore
    model.eval()
    processor: Any = Wav2Vec2Processor.from_pretrained(args.model_dir, **model_kwargs)

    print(f"Loading audio: {args.audio_file}")
    speech, sr = librosa.load(args.audio_file, sr=16000)

    if args.mode == "short":
        # Process the entire file
        inputs = processor(  # type: ignore
            speech, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_values = getattr(inputs, "input_values")
        with torch.no_grad():
            logits = model(input_values.to(device)).logits

        res = greedy_inference(logits[0], processor)
        if isinstance(res, list):
            res = res[0]
        pred_text = str(res["text"])

        final_text = convert_to_human_orthography(pred_text)
        print(f"TRANSCRIPTION: {final_text}")

    elif args.mode == "long":
        # Load VAD model
        VAD_class = None
        try:
            from speechbrain.inference.VAD import VAD as VAD_class  # type: ignore
        except ImportError:
            try:
                from speechbrain.pretrained import VAD as VAD_class  # type: ignore
            except ImportError:
                pass

        if VAD_class is None:
            print("SpeechBrain VAD is not available.")
            return

        print("Loading VAD model...")
        vad = VAD_class.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
        )
        print("Running VAD...")
        boundaries = vad.get_speech_segments(args.audio_file)  # type: ignore

        # Chop into max 15s segments
        segments = []
        for start, end in boundaries:
            start_val, end_val = float(start), float(end)
            while end_val - start_val > 15.0:
                segments.append((start_val, start_val + 15.0))
                start_val += 15.0
            segments.append((start_val, end_val))

        print(f"Found {len(segments)} segments to transcribe.")
        results = []
        for start_val, end_val in segments:
            start_idx = int(start_val * 16000)
            end_idx = int(end_val * 16000)
            chunk = speech[start_idx:end_idx]

            if len(chunk) < 1600:  # skip < 100ms
                continue

            inputs = processor(  # type: ignore
                chunk, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = getattr(inputs, "input_values")
            with torch.no_grad():
                logits = model(input_values.to(device)).logits

            res = greedy_inference(logits[0], processor)
            if isinstance(res, list):
                res = res[0]
            pred_text = str(res["text"])

            final_text = convert_to_human_orthography(pred_text)
            results.append((start_val, end_val, final_text))

        out_tsv = (
            args.output_tsv
            if args.output_tsv
            else args.audio_file.rsplit(".", 1)[0] + ".tsv"
        )
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("start\tend\ttranscription\n")
            for start, end, text in results:
                f.write(f"{start:.3f}\t{end:.3f}\t{text}\n")
        print(f"Saved transcriptions to {out_tsv}")


if __name__ == "__main__":
    main()
