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

from transcription.audio.segment import (
    get_energy_profile,
    segment_audio_from_profile,
    compute_metrics,
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


def get_best_parameters(dbfs_profile, total_len_ms):
    """
    Finds the best segmentation parameters matching maximum segment duration <= 10s.
    """
    thresholds = [-55, -50, -45, -40, -35, -30, -25, -20]
    min_silence_lens = [100, 200, 300, 500, 800, 1000]
    keep_silences = [0, 50, 100, 150, 200]

    results = []
    for thresh in thresholds:
        for min_sil in min_silence_lens:
            for keep_sil in keep_silences:
                segments = segment_audio_from_profile(
                    dbfs_profile,
                    total_len_ms,
                    step_ms=10,
                    min_silence_len=min_sil,
                    silence_thresh=thresh,
                    keep_silence=keep_sil,
                )
                metrics = compute_metrics(segments, total_len_ms)

                # Overlap calculation
                sorted_segs = sorted(segments, key=lambda s: s["start"])
                overlap_ms = 0
                for i in range(len(sorted_segs) - 1):
                    cur_end = sorted_segs[i]["end"]
                    nxt_start = sorted_segs[i + 1]["start"]
                    if nxt_start < cur_end:
                        overlap_ms += (
                            min(cur_end, sorted_segs[i + 1]["end"]) - nxt_start
                        )

                overlap_percent = (
                    (overlap_ms / total_len_ms * 100.0) if total_len_ms > 0 else 0.0
                )

                results.append(
                    {
                        "silence_thresh": thresh,
                        "min_silence_len": min_sil,
                        "keep_silence": keep_sil,
                        "percent_segmented": metrics["percent_segmented"],
                        "overlap_percent": overlap_percent,
                        "max_len": metrics["max_len"],
                        "avg_len": metrics["avg_len"],
                        "count": metrics["count"],
                        "segments": segments,
                    }
                )

    # Filter for max_len <= 10.0 and at least 1 segment
    valid_results = [r for r in results if r["max_len"] <= 10.0 and r["count"] > 0]

    def score_config(r):
        net_coverage = r["percent_segmented"] - r["overlap_percent"]
        silence_penalty = (r["keep_silence"] / 100.0) * 0.5
        fragment_penalty = 5.0 if (r["avg_len"] < 1.0 and r["count"] > 5) else 0.0
        return net_coverage - silence_penalty - fragment_penalty

    if valid_results:
        best = max(valid_results, key=score_config)
    else:
        # Fallback: pick the one with minimum max_len among those with > 0 segments
        with_segments = [r for r in results if r["count"] > 0]
        if with_segments:
            best = min(with_segments, key=lambda x: x["max_len"])
        else:
            best = {
                "silence_thresh": -40,
                "min_silence_len": 500,
                "keep_silence": 100,
                "segments": [],
            }

    return best


def split_long_segments_smart(segments, audio, max_duration_ms=10000, overlap_ms=250):
    """
    Splits any segments longer than max_duration_ms by scanning for internal silence/pauses,
    or falling back to the quietest point within the segment to avoid cutting in the middle of words.
    Adds overlap_ms padding at boundaries where a fallback split occurred.
    """
    final_segs = []

    def process_segment(start_ms, end_ms):
        duration = end_ms - start_ms
        if duration <= max_duration_ms:
            final_segs.append({"start": start_ms, "end": end_ms, "duration": duration})
            return

        # Extract the long sub-audio
        sub_audio = audio[start_ms:end_ms]
        dbfs_profile = get_energy_profile(sub_audio, step_ms=10)

        # Grid parameters to find brief pauses/silences inside active speech
        thresholds = [-45, -40, -35, -30, -25, -20, -15]
        min_silence_lens = [500, 400, 300, 200, 100, 50]
        keep_silences = [100, 50, 0]

        for min_sil in min_silence_lens:
            for thresh in thresholds:
                for keep_sil in keep_silences:
                    sub_segs = segment_audio_from_profile(
                        dbfs_profile,
                        duration,
                        step_ms=10,
                        min_silence_len=min_sil,
                        silence_thresh=thresh,
                        keep_silence=keep_sil,
                    )
                    # Filter out empty or trivial splits that don't subdivide the duration
                    if not sub_segs or len(sub_segs) <= 1:
                        continue

                    max_sub_len = max(s["duration"] for s in sub_segs)
                    if max_sub_len <= max_duration_ms:
                        # Found a configuration that splits all parts to <= 10s!
                        for s in sub_segs:
                            process_segment(start_ms + s["start"], start_ms + s["end"])
                        return

        # Fallback: Find the quietest 100ms window in the middle 40% (30% to 70%) of the segment to split
        samples = np.array(sub_audio.get_array_of_samples(), dtype=np.float32)
        sr = sub_audio.frame_rate
        win_size = int(sr * 0.1)  # 100ms

        start_search = int(len(samples) * 0.3)
        end_search = int(len(samples) * 0.7)

        if end_search - start_search > win_size:
            min_energy = float("inf")
            best_split_idx = (start_search + end_search) // 2
            step = int(sr * 0.05)  # 50ms step

            for idx in range(start_search, end_search - win_size, step):
                win = samples[idx : idx + win_size]
                energy = np.mean(win**2)
                if energy < min_energy:
                    min_energy = energy
                    best_split_idx = idx + (win_size // 2)

            split_ms = int((best_split_idx / sr) * 1000)
        else:
            split_ms = duration // 2

        # Recursively process the two halves with overlap padding to prevent phrase/word clipping
        left_end = min(end_ms, start_ms + split_ms + overlap_ms)
        right_start = max(start_ms, start_ms + split_ms - overlap_ms)

        process_segment(start_ms, left_end)
        process_segment(right_start, end_ms)

    for seg in segments:
        process_segment(seg["start"], seg["end"])

    return sorted(final_segs, key=lambda x: x["start"])


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
