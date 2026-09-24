#!/usr/bin/env python3
import os
import sys
import time
import argparse
import tempfile
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
from scipy.signal import stft, istft

# Add root directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digohwelisgi.core.audio import (
    get_energy_profile,
    segment_audio_from_profile,
    compute_metrics,
    get_best_parameters,
    split_long_segments_smart,
)


def find_noise_profile(audio, duration_ms=2000):
    """
    Scans the audio in 100ms steps to find the quietest duration_ms window.
    Returns the noise AudioSegment, start time in ms, and end time in ms.
    """
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sr = audio.frame_rate
    win_size = int((sr * duration_ms) / 1000)

    if len(samples) <= win_size:
        return audio, 0, len(audio)

    step_samples = int(sr * 0.1)  # 100ms step
    min_energy = float("inf")
    best_start = 0

    for start in range(0, len(samples) - win_size, step_samples):
        window = samples[start : start + win_size]
        energy = np.mean(window**2)
        if energy < min_energy:
            min_energy = energy
            best_start = start

    start_ms = int((best_start / sr) * 1000)
    end_ms = start_ms + duration_ms
    return audio[start_ms:end_ms], start_ms, end_ms


def denoise_audio(audio, noise_profile, alpha=2.0, beta=0.02, nperseg=512):
    """
    Denoises the AudioSegment using spectral subtraction.
    """
    sr = audio.frame_rate
    max_amp = (
        float(audio.max_possible_amplitude)
        if audio.max_possible_amplitude > 0
        else 32768.0
    )

    # Convert audio to float32 normalized to [-1.0, 1.0]
    sig = np.array(audio.get_array_of_samples(), dtype=np.float32) / max_amp
    noise = np.array(noise_profile.get_array_of_samples(), dtype=np.float32) / max_amp

    # Compute STFT
    _, _, N = stft(noise, fs=sr, nperseg=nperseg)
    noise_mu = np.mean(np.abs(N), axis=1, keepdims=True)

    f, t, S = stft(sig, fs=sr, nperseg=nperseg)
    S_amp = np.abs(S)
    S_phase = np.angle(S)

    # Subtract noise amplitude
    S_clean_amp = S_amp - alpha * noise_mu
    S_clean_amp = np.maximum(S_clean_amp, beta * S_amp)

    # Reconstruct Complex Spectrum & ISTFT
    S_clean = S_clean_amp * np.exp(1j * S_phase)
    _, sig_clean = istft(S_clean, fs=sr, nperseg=nperseg)

    # Align lengths
    if len(sig_clean) > len(sig):
        sig_clean = sig_clean[: len(sig)]
    elif len(sig_clean) < len(sig):
        sig_clean = np.pad(sig_clean, (0, len(sig) - len(sig_clean)))

    # Convert back to 16-bit PCM AudioSegment
    sig_int = np.clip(sig_clean * max_amp, -32768.0, 32767.0).astype(np.int16)

    return AudioSegment(
        sig_int.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )


def get_vad_model():
    """
    Loads SpeechBrain VAD model, forcing CPU mode for stability.
    """
    try:
        from speechbrain.inference.VAD import VAD
    except ImportError:
        from speechbrain.pretrained import VAD
    import torch

    device = "cpu"
    return VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty", run_opts={"device": device}
    )


def segment_audio_via_vad(audio, vad_model, keep_silence=200):
    """
    Segments an AudioSegment using the SpeechBrain VAD model.
    Applies keep_silence padding (ms) around the boundaries without overlap.
    """
    total_duration_ms = len(audio)

    # Save AudioSegment to a temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)
    try:
        audio.export(temp_path, format="wav")
        # SpeechBrain VAD expects 16kHz mono. `audio` is already set to 16kHz mono.
        boundaries = vad_model.get_speech_segments(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Convert boundaries to milliseconds
    raw_segments = []
    for seg in boundaries:
        start_ms = int(float(seg[0]) * 1000)
        end_ms = int(float(seg[1]) * 1000)
        raw_segments.append((start_ms, end_ms))

    # Apply keep_silence padding without overlap
    segments = []
    num_segs = len(raw_segments)
    for idx, (start_ms, end_ms) in enumerate(raw_segments):
        # Start padding
        if idx == 0:
            pad_start = max(0, start_ms - keep_silence)
        else:
            prev_end_ms = raw_segments[idx - 1][1]
            gap_before = max(0, start_ms - prev_end_ms)
            pad_start = start_ms - min(keep_silence, gap_before // 2)

        # End padding
        if idx == num_segs - 1:
            pad_end = min(total_duration_ms, end_ms + keep_silence)
        else:
            next_start_ms = raw_segments[idx + 1][0]
            gap_after = max(0, next_start_ms - end_ms)
            pad_end = end_ms + min(keep_silence, gap_after // 2)

        segments.append(
            {"start": pad_start, "end": pad_end, "duration": pad_end - pad_start}
        )

    return segments


def is_noise_segment(chunk, noise_profile):
    """
    Classifies if a chunk is background noise or speech based on average dBFS,
    maximum dBFS, and RMS energy ratio to the reference noise floor.
    """
    chunk_dbfs = chunk.dBFS
    chunk_max_dbfs = chunk.max_dBFS

    samples = np.array(chunk.get_array_of_samples(), dtype=np.float32)
    rms = np.sqrt(np.mean(samples**2)) if len(samples) > 0 else 0

    noise_samples = np.array(noise_profile.get_array_of_samples(), dtype=np.float32)
    noise_rms = np.sqrt(np.mean(noise_samples**2)) if len(noise_samples) > 0 else 1e-5

    rms_ratio = rms / (noise_rms + 1e-8)

    if chunk_max_dbfs < -42:
        return True, f"Max volume too quiet ({chunk_max_dbfs:.1f} dBFS)"
    if rms_ratio < 1.8:
        return True, f"RMS ratio to noise floor too low ({rms_ratio:.2f}x)"
    if chunk_dbfs < -55:
        return True, f"Average volume too quiet ({chunk_dbfs:.1f} dBFS)"

    return False, "Speech detected"


def process_single_file(
    mp3_path,
    out_base_dir,
    alpha=2.0,
    beta=0.02,
    keep_noise=False,
    vad_model=None,
    use_energy=False,
    vad_keep_silence=200,
    overlap_ms=250,
):
    t_start = time.time()
    base_name = os.path.splitext(os.path.basename(mp3_path))[0]
    out_dir = os.path.join(out_base_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{base_name}] Loading audio...")
    # Load and immediately downsample to 16kHz mono to optimize memory and processing
    audio = AudioSegment.from_file(mp3_path).set_frame_rate(16000).set_channels(1)

    print(f"[{base_name}] Finding quietest 2-second noise window...")
    noise_prof, n_start, n_end = find_noise_profile(audio, duration_ms=2000)
    print(
        f"[{base_name}] Noise profile found at {n_start/1000.0:.2f}s - {n_end/1000.0:.2f}s"
    )

    print(f"[{base_name}] Applying spectral subtraction denoising...")
    denoised_audio = denoise_audio(audio, noise_prof, alpha=alpha, beta=beta)

    if not use_energy and vad_model is not None:
        print(f"[{base_name}] Segmenting audio using SpeechBrain VAD...")
        segments = segment_audio_via_vad(
            denoised_audio, vad_model, keep_silence=vad_keep_silence
        )
        print(f"[{base_name}] VAD found {len(segments)} initial segments.")
    else:
        print(f"[{base_name}] Computing energy profile for segmentation...")
        dbfs_profile = get_energy_profile(denoised_audio, step_ms=10)

        print(f"[{base_name}] Dynamically tuning segmentation parameters...")
        best_config = get_best_parameters(dbfs_profile, len(denoised_audio))

        print(
            f"[{base_name}] Selected params: thresh={best_config['silence_thresh']} dBFS, "
            f"min_silence_len={best_config['min_silence_len']} ms, keep_silence={best_config['keep_silence']} ms. "
            f"Initial segment count: {best_config.get('count', 0)}"
        )

        segments = best_config.get("segments", [])

    # Split any remaining segment larger than 10 seconds using silence-aware method
    final_segments = split_long_segments_smart(
        segments, denoised_audio, max_duration_ms=10000, overlap_ms=overlap_ms
    )
    print(f"[{base_name}] Total candidate segments: {len(final_segments)}")

    speech_segments = []
    noise_count = 0
    for seg in final_segments:
        start_ms = seg["start"]
        end_ms = seg["end"]
        chunk = denoised_audio[start_ms:end_ms]

        is_noise, reason = is_noise_segment(chunk, noise_prof)
        if is_noise:
            noise_count += 1
            if keep_noise:
                seg["is_noise"] = True
                speech_segments.append(seg)
        else:
            seg["is_noise"] = False
            speech_segments.append(seg)

    print(
        f"[{base_name}] Noise classification: {noise_count} noise / {len(final_segments) - noise_count} speech."
    )
    if keep_noise:
        print(
            f"[{base_name}] Keeping all segments (including noise). Exporting {len(speech_segments)} segments..."
        )
    else:
        print(
            f"[{base_name}] Filtering out noise. Exporting {len(speech_segments)} speech-only segments..."
        )

    for idx, seg in enumerate(speech_segments):
        start_ms = seg["start"]
        end_ms = seg["end"]

        chunk = denoised_audio[start_ms:end_ms]

        # Normalize non-noise segments for volume
        if not seg.get("is_noise", False):
            chunk = normalize(chunk)

        suffix = "_noise" if seg.get("is_noise", False) else ""
        out_filename = f"{base_name}_segment_{idx:04d}_{start_ms}_{end_ms}{suffix}.wav"
        out_filepath = os.path.join(out_dir, out_filename)

        chunk.export(out_filepath, format="wav")

    print(f"[{base_name}] Completed processing in {time.time() - t_start:.2f} seconds!")


def main():
    parser = argparse.ArgumentParser(
        description="Process interview MP3s with Audacity-like denoising and segmentation."
    )
    parser.add_argument(
        "--input",
        default="cvcs-mp3s",
        help="Path to input MP3 file or directory of MP3 files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/denoised_segments",
        help="Directory to save output segments.",
    )
    parser.add_argument(
        "--single-file",
        help="Name of a single MP3 file in the input directory to process (for debugging/testing).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Noise subtraction coefficient (default: 2.0).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="Spectral floor coefficient (default: 0.02).",
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Keep classified noise segments instead of filtering them out.",
    )
    parser.add_argument(
        "--use-energy",
        action="store_true",
        help="Force energy-profile based segmentation instead of VAD.",
    )
    parser.add_argument(
        "--vad-keep-silence",
        type=int,
        default=200,
        help="Keep silence padding for VAD segments in ms (default: 200).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=250,
        help="Overlap padding at fallback split boundaries in ms (default: 250).",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input path '{input_path}' does not exist.")
        sys.exit(1)

    # Gather target files
    if os.path.isdir(input_path):
        if args.single_file:
            target_path = os.path.join(input_path, args.single_file)
            if not os.path.exists(target_path):
                print(f"Error: file '{target_path}' not found.")
                sys.exit(1)
            mp3_files = [target_path]
        else:
            mp3_files = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(".mp3")
            ]
            mp3_files.sort()
    else:
        mp3_files = [input_path]

    if not mp3_files:
        print(f"No MP3 files found to process at '{input_path}'.")
        sys.exit(0)

    vad_model = None
    if not args.use_energy:
        print("Initializing SpeechBrain VAD model...")
        try:
            vad_model = get_vad_model()
        except Exception as e:
            print(
                f"Warning: Failed to load VAD model ({e}). Falling back to energy-based segmentation."
            )
            args.use_energy = True

    print(f"Found {len(mp3_files)} MP3 file(s) to process.")
    for idx, f in enumerate(mp3_files):
        print(
            f"\n--- Processing file {idx+1}/{len(mp3_files)}: {os.path.basename(f)} ---"
        )
        try:
            process_single_file(
                f,
                args.out_dir,
                alpha=args.alpha,
                beta=args.beta,
                keep_noise=args.keep_noise,
                vad_model=vad_model,
                use_energy=args.use_energy,
                vad_keep_silence=args.vad_keep_silence,
                overlap_ms=args.overlap,
            )
        except Exception as e:
            print(f"Error processing '{f}': {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
