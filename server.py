import os
import sys
import random
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import subprocess
import wave
import contextlib
import pandas as pd
from datetime import date
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import tempfile
import uuid
import csv

# Refresh PATH from Registry (Windows) so the server picks up new winget installations (like ffmpeg) without restarting
try:
    import winreg
    for hkey, subkey in [
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment")
    ]:
        try:
            with winreg.OpenKey(hkey, subkey, 0, winreg.KEY_READ) as key:
                path_val, _ = winreg.QueryValueEx(key, "Path")
                for path_dir in path_val.split(os.path.pathsep):
                    path_dir_expanded = os.path.expandvars(path_dir)
                    if path_dir_expanded and path_dir_expanded not in os.environ["PATH"]:
                        os.environ["PATH"] += os.path.pathsep + path_dir_expanded
        except Exception:
            pass
except Exception:
    pass


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppConfig:
    SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_elan_dir(cls): return os.path.join(cls.SANDBOX_DIR, "processed-elan-files")

    @classmethod
    def get_wav_dir(cls): return os.path.join(cls.SANDBOX_DIR, "wav")

    @classmethod
    def get_metadata_csv(cls): return os.path.join(cls.SANDBOX_DIR, "wav-metadata.csv")

    @classmethod
    def get_inf_dir(cls): return os.path.join(cls.SANDBOX_DIR, "audiofiles-to-transcribe")

    @classmethod
    def get_model_dir(cls): return os.path.join(cls.SANDBOX_DIR, "wav2vec2-model")

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.get_elan_dir(), exist_ok=True)
        os.makedirs(cls.get_wav_dir(), exist_ok=True)
        os.makedirs(cls.get_inf_dir(), exist_ok=True)
        os.makedirs(cls.get_model_dir(), exist_ok=True)

AppConfig.ensure_dirs()

class ProcessElanRequest(BaseModel):
    txt_file: str
    wav_file: str
    gender: str

class GenerateSplitsRequest(BaseModel):
    train_pct: int
    valid_pct: int
    test_pct: int
    max_duration: int
    file_prefix: str
    use_code_switched: bool
    use_doubtful: bool

vad_model = None

def get_vad_model():
    global vad_model
    if vad_model is None:
        try:
            from speechbrain.inference.VAD import VAD
        except ImportError:
            from speechbrain.pretrained import VAD
        
        import torch
        device = "cpu" # Force CPU for VAD to avoid CUDA kernel arch mismatches
        
        vad_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            run_opts={"device": device}
        )
    return vad_model

@app.get("/api/audio/{filename:path}")
def get_audio(filename: str):
    path = os.path.join(AppConfig.SANDBOX_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path)

class VADRequest(BaseModel):
    filename: str

@app.post("/api/vad_segments")
def get_vad_segments(req: VADRequest):
    try:
        path = os.path.join(AppConfig.SANDBOX_DIR, req.filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        vad = get_vad_model()
        
        temp_id = str(uuid.uuid4())
        temp_wav = f"temp_vad_{temp_id}.wav"
        
        cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", temp_wav]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {process.stderr}")
            
        boundaries = vad.get_speech_segments(temp_wav)
        
        segments = []
        for seg in boundaries:
            segments.append({"start": round(float(seg[0]), 3), "end": round(float(seg[1]), 3)})
            
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return {"segments": segments}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class JulieVADRequest(BaseModel):
    filename: str
    silence_thresh: int = -40
    min_silence_len: int = 500
    keep_silence: int = 100

audio_profile_cache = {}

@app.post("/api/julie_segments")
def get_julie_segments(req: JulieVADRequest):
    try:
        from transcription.audio.segment import get_energy_profile, segment_audio_from_profile
        from pydub import AudioSegment
        
        path = os.path.join(AppConfig.SANDBOX_DIR, req.filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        cache_key = req.filename
        mtime = os.path.getmtime(path)
        
        if cache_key not in audio_profile_cache or audio_profile_cache[cache_key]['mtime'] != mtime:
            print(f"Computing energy profile for {req.filename}...")
            audio = AudioSegment.from_file(path)
            total_len = len(audio)
            dbfs_profile = get_energy_profile(audio, step_ms=10)
            audio_profile_cache[cache_key] = {
                'profile': dbfs_profile,
                'total_len': total_len,
                'mtime': mtime
            }
        
        cached = audio_profile_cache[cache_key]
        
        segments = segment_audio_from_profile(
            cached['profile'],
            cached['total_len'],
            step_ms=10,
            min_silence_len=req.min_silence_len,
            silence_thresh=req.silence_thresh,
            keep_silence=req.keep_silence
        )
        
        sec_segments = []
        for seg in segments:
            sec_segments.append({"start": round(seg['start'] / 1000.0, 3), "end": round(seg['end'] / 1000.0, 3)})
            
        return {"segments": sec_segments}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class Region(BaseModel):
    start: float
    end: float
    text: str = ""

class SaveElanRequest(BaseModel):
    regions: list[Region]
    filename: str
    
@app.post("/api/save_elan")
def save_elan(req: SaveElanRequest):
    try:
        out_path = os.path.join(AppConfig.get_elan_dir(), req.filename)
        lines = []
        for r in req.regions:
            text = r.text if r.text else "TBD"
            lines.append(f"default\tspeaker\t{r.start:.3f}\t{r.end:.3f}\t{text}")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
        return {"message": "Success", "filepath": out_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FileSetting(BaseModel):
    silence_thresh: int = -40
    min_silence_len: int = 500
    keep_silence: int = 100

class PreviewSegmentRequest(BaseModel):
    target_path: str
    silence_thresh: int = -40
    min_silence_len: int = 500
    keep_silence: int = 100
    file_settings: dict[str, FileSetting] = {}

def process_file_preview(rel_path, default_thresh, default_min_silence, default_keep_silence, custom_setting=None):
    from transcription.audio.segment import get_energy_profile, segment_audio_from_profile
    from pydub import AudioSegment
    
    thresh = custom_setting.silence_thresh if custom_setting else default_thresh
    min_silence = custom_setting.min_silence_len if custom_setting else default_min_silence
    keep_silence = custom_setting.keep_silence if custom_setting else default_keep_silence

    full_audio_path = os.path.join(AppConfig.SANDBOX_DIR, rel_path)
    cache_key = rel_path
    mtime = os.path.getmtime(full_audio_path)

    if cache_key not in audio_profile_cache or audio_profile_cache[cache_key]['mtime'] != mtime:
        audio = AudioSegment.from_file(full_audio_path)
        total_len = len(audio)
        dbfs_profile = get_energy_profile(audio, step_ms=10)
        audio_profile_cache[cache_key] = {
            'profile': dbfs_profile,
            'total_len': total_len,
            'mtime': mtime
        }

    cached = audio_profile_cache[cache_key]
    total_len_ms = cached['total_len']
    dbfs_profile = cached['profile']

    segments = segment_audio_from_profile(
        dbfs_profile,
        total_len_ms,
        step_ms=10,
        min_silence_len=min_silence,
        silence_thresh=thresh,
        keep_silence=keep_silence
    )

    total_duration = round(total_len_ms / 1000.0, 2)
    segment_count = len(segments)
    
    # Calculate non-overlapping result duration and overlap
    sorted_segs = sorted(segments, key=lambda s: s['start'])
    result_duration_ms = sum(s['duration'] for s in sorted_segs)
    
    overlap_ms = 0
    for i in range(len(sorted_segs) - 1):
        cur_end = sorted_segs[i]['end']
        nxt_start = sorted_segs[i+1]['start']
        if nxt_start < cur_end:
            overlap_ms += min(cur_end, sorted_segs[i+1]['end']) - nxt_start

    result_duration = round(result_duration_ms / 1000.0, 2)
    coverage_percent = round((result_duration / total_duration * 100.0), 1) if total_duration > 0 else 0.0
    overlap_duration = round(overlap_ms / 1000.0, 2)
    overlap_percent = round((overlap_duration / total_duration * 100.0), 1) if total_duration > 0 else 0.0

    # Duration Histogram bins: 20 bins between min and max duration
    all_lengths = [s['duration'] / 1000.0 for s in sorted_segs]
    if not all_lengths:
        histogram = []
        min_val = 0
        max_val = 0
    else:
        all_lengths.sort()
        min_val = all_lengths[0]
        max_val = all_lengths[-1]
        count = len(all_lengths)
        
        bins = 20
        histogram = []
        if max_val > min_val:
            bin_size = (max_val - min_val) / bins
            bin_counts = [0] * bins
            
            for length in all_lengths:
                idx = int((length - min_val) / bin_size)
                if idx >= bins:
                    idx = bins - 1
                bin_counts[idx] += 1
                
            for i in range(bins):
                start = min_val + (i * bin_size)
                end = start + bin_size
                histogram.append({
                    "start": start,
                    "end": end,
                    "count": bin_counts[i]
                })
        else:
            histogram = [{"start": min_val, "end": max_val, "count": count}]

    return {
        "file": rel_path,
        "filename": os.path.basename(rel_path),
        "total_duration": total_duration,
        "segment_count": segment_count,
        "result_duration": result_duration,
        "coverage_percent": coverage_percent,
        "overlap_duration": overlap_duration,
        "overlap_percent": overlap_percent,
        "min": min_val,
        "max": max_val,
        "histogram": histogram,
        "settings": {
            "silence_thresh": thresh,
            "min_silence_len": min_silence,
            "keep_silence": keep_silence
        }
    }

@app.post("/api/preview_segments")
def preview_segments(req: PreviewSegmentRequest):
    try:
        import os
        from concurrent.futures import ThreadPoolExecutor

        target_full_path = os.path.join(AppConfig.SANDBOX_DIR, req.target_path)
        if not os.path.exists(target_full_path):
            raise HTTPException(status_code=404, detail=f"Path not found: {req.target_path}")

        files_to_process = []
        if os.path.isdir(target_full_path):
            for f in sorted(os.listdir(target_full_path)):
                if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    files_to_process.append(os.path.join(req.target_path, f))
        else:
            files_to_process.append(req.target_path)

        def worker(rel_path):
            custom = req.file_settings.get(rel_path)
            return process_file_preview(rel_path, req.silence_thresh, req.min_silence_len, req.keep_silence, custom)

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(files_to_process)))) as executor:
            results = list(executor.map(worker, files_to_process))

        total_files = len(results)
        total_batch_duration = round(sum(r['total_duration'] for r in results), 2)
        total_segments = sum(r['segment_count'] for r in results)
        total_result_duration = round(sum(r['result_duration'] for r in results), 2)
        overall_coverage_percent = round((total_result_duration / total_batch_duration * 100.0), 1) if total_batch_duration > 0 else 0.0
        total_overlap_duration = round(sum(r['overlap_duration'] for r in results), 2)

        summary = {
            "total_files": total_files,
            "total_batch_duration": total_batch_duration,
            "total_segments": total_segments,
            "total_result_duration": total_result_duration,
            "overall_coverage_percent": overall_coverage_percent,
            "total_overlap_duration": total_overlap_duration
        }

        return {"previews": results, "summary": summary}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class SmartSegmentRequest(BaseModel):
    file_path: str

@app.post("/api/smart_segment")
def smart_segment(req: SmartSegmentRequest):
    try:
        from transcription.audio.segment import get_energy_profile, segment_audio_from_profile, compute_metrics
        from pydub import AudioSegment
        import os

        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_audio_path = os.path.normpath(os.path.join(base_dir, req.file_path))

        if not os.path.exists(full_audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        audio = AudioSegment.from_file(full_audio_path)
        total_len = len(audio)
        dbfs_profile = get_energy_profile(audio, step_ms=10)

        thresholds = [-55, -50, -45, -40, -35, -30, -25, -20]
        min_silence_lens = [100, 200, 300, 500, 800, 1000]
        keep_silences = [0, 50, 100, 150, 200]

        results = []
        for thresh in thresholds:
            for min_sil in min_silence_lens:
                for keep_sil in keep_silences:
                    segments = segment_audio_from_profile(
                        dbfs_profile,
                        total_len,
                        step_ms=10,
                        min_silence_len=min_sil,
                        silence_thresh=thresh,
                        keep_silence=keep_sil
                    )
                    metrics = compute_metrics(segments, total_len)

                    # Compute overlap ms
                    sorted_segs = sorted(segments, key=lambda s: s['start'])
                    overlap_ms = 0
                    for i in range(len(sorted_segs) - 1):
                        cur_end = sorted_segs[i]['end']
                        nxt_start = sorted_segs[i+1]['start']
                        if nxt_start < cur_end:
                            overlap_ms += min(cur_end, sorted_segs[i+1]['end']) - nxt_start

                    overlap_percent = (overlap_ms / total_len * 100.0) if total_len > 0 else 0.0

                    results.append({
                        'silence_thresh': thresh,
                        'min_silence_len': min_sil,
                        'keep_silence': keep_sil,
                        'percent_segmented': metrics['percent_segmented'],
                        'overlap_percent': overlap_percent,
                        'max_len': metrics['max_len'],
                        'avg_len': metrics['avg_len'],
                        'count': metrics['count']
                    })
        
        if not results:
            return {"silence_thresh": -40, "min_silence_len": 500, "keep_silence": 100}

        def score_config(r):
            # Net coverage = percent_segmented - overlap_percent
            net_coverage = r['percent_segmented'] - r['overlap_percent']
            # Small penalty for excessive keep_silence when net_coverage is equal
            silence_penalty = (r['keep_silence'] / 100.0) * 0.5
            # Penalty for micro-fragmented segments
            fragment_penalty = 5.0 if (r['avg_len'] < 1.0 and r['count'] > 5) else 0.0

            return net_coverage - silence_penalty - fragment_penalty

        # Filter for max_len <= 25.0 and at least 1 segment
        valid_results = [r for r in results if r['max_len'] <= 25.0 and r['count'] > 0]

        if valid_results:
            best = max(valid_results, key=score_config)
        else:
            # Fallback: pick the one with min max_len among those with > 0 coverage
            with_segments = [r for r in results if r['count'] > 0]
            if with_segments:
                best = min(with_segments, key=lambda x: x['max_len'])
            else:
                best = {"silence_thresh": -40, "min_silence_len": 500, "keep_silence": 100}

        return {
            "silence_thresh": best.get('silence_thresh', -40),
            "min_silence_len": best.get('min_silence_len', 500),
            "keep_silence": best.get('keep_silence', 100)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class BatchSegmentRequest(BaseModel):
    target_path: str
    silence_thresh: int = -40
    min_silence_len: int = 500
    keep_silence: int = 100
    file_settings: dict[str, FileSetting] = {}

@app.post("/api/batch_segment")
def batch_segment(req: BatchSegmentRequest):
    try:
        from transcription.audio.segment import get_energy_profile, segment_audio_from_profile
        from pydub import AudioSegment
        import csv
        import os
        
        target_full_path = os.path.join(AppConfig.SANDBOX_DIR, req.target_path)
        if not os.path.exists(target_full_path):
            raise HTTPException(status_code=404, detail=f"Path not found: {req.target_path}")

        files_to_process = []
        if os.path.isdir(target_full_path):
            for f in sorted(os.listdir(target_full_path)):
                if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    files_to_process.append(os.path.join(req.target_path, f))
        else:
            files_to_process.append(req.target_path)
            
        out_dir = AppConfig.get_inf_dir()
        os.makedirs(out_dir, exist_ok=True)
        
        segmented_count = 0
        total_chunks = 0
        
        manifest_path = os.path.join(out_dir, "segmentation_manifest.csv")
        manifest_exists = os.path.exists(manifest_path)
        
        with open(manifest_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not manifest_exists:
                writer.writerow(["original_filename", "segmented_filepath", "start_ms", "end_ms"])
            
            for rel_path in files_to_process:
                full_audio_path = os.path.join(AppConfig.SANDBOX_DIR, rel_path)
                print(f"Segmenting {rel_path}...")

                custom = req.file_settings.get(rel_path)
                thresh = custom.silence_thresh if custom else req.silence_thresh
                min_silence = custom.min_silence_len if custom else req.min_silence_len
                keep_silence = custom.keep_silence if custom else req.keep_silence

                cache_key = rel_path
                mtime = os.path.getmtime(full_audio_path)

                if cache_key in audio_profile_cache and audio_profile_cache[cache_key]['mtime'] == mtime:
                    cached = audio_profile_cache[cache_key]
                    total_len = cached['total_len']
                    dbfs_profile = cached['profile']
                    audio = AudioSegment.from_file(full_audio_path)
                else:
                    audio = AudioSegment.from_file(full_audio_path)
                    total_len = len(audio)
                    dbfs_profile = get_energy_profile(audio, step_ms=10)

                segments = segment_audio_from_profile(
                    dbfs_profile,
                    total_len,
                    step_ms=10,
                    min_silence_len=min_silence,
                    silence_thresh=thresh,
                    keep_silence=keep_silence
                )
                
                original_filename = os.path.basename(rel_path)
                base_name = os.path.splitext(original_filename)[0]
                
                subfolder_path = os.path.join(out_dir, base_name)
                os.makedirs(subfolder_path, exist_ok=True)
                
                for i, seg in enumerate(segments):
                    chunk = audio[seg['start']:seg['end']]
                    chunk_name = f"{base_name}_{i:04d}.wav"
                    chunk_out_path = os.path.join(subfolder_path, chunk_name)
                    chunk.export(chunk_out_path, format="wav")
                    
                    writer.writerow([original_filename, f"{base_name}/{chunk_name}", seg['start'], seg['end']])
                    total_chunks += 1
                segmented_count += 1
            
        return {"message": f"Processed {segmented_count} files into {total_chunks} segments.", "output_dir": out_dir}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class BatchStatsRequest(BaseModel):
    target_folders: list[str]

@app.post("/api/batch_stats")
def get_batch_stats(req: BatchStatsRequest):
    try:
        import soundfile as sf
        
        all_lengths = []
        for folder in req.target_folders:
            if not os.path.exists(folder):
                continue
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.wav'):
                        audio_path = os.path.join(root, file)
                        try:
                            info = sf.info(audio_path)
                            duration = info.frames / info.samplerate
                            all_lengths.append(duration)
                        except Exception:
                            pass
        
        if not all_lengths:
            return {"count": 0, "total_duration": 0, "min": 0, "max": 0, "histogram": []}
            
        all_lengths.sort()
        count = len(all_lengths)
        total = sum(all_lengths)
        min_val = all_lengths[0]
        max_val = all_lengths[-1]
        
        # Create histogram (20 bins)
        bins = 20
        histogram = []
        if max_val > min_val:
            bin_size = (max_val - min_val) / bins
            bin_counts = [0] * bins
            
            for length in all_lengths:
                idx = int((length - min_val) / bin_size)
                if idx >= bins:
                    idx = bins - 1
                bin_counts[idx] += 1
                
            for i in range(bins):
                start = min_val + (i * bin_size)
                end = start + bin_size
                histogram.append({
                    "start": start,
                    "end": end,
                    "count": bin_counts[i]
                })
        else:
            histogram = [{"start": min_val, "end": max_val, "count": count}]
            
        return {
            "count": count,
            "total_duration": total,
            "min": min_val,
            "max": max_val,
            "histogram": histogram
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class BatchInferenceRequest(BaseModel):
    target_folders: list[str]
    checkpoint: str = ""
    output_csv_name: str = "batch_inference_results.csv"

@app.post("/api/batch_inference")
def run_batch_inference(req: BatchInferenceRequest):
    try:
        import os
        import sys
        import subprocess
        import shutil
        import uuid
        import tempfile
        import pandas as pd
        
        if not req.target_folders:
            raise HTTPException(status_code=400, detail="No folders selected.")
            
        out_name = req.output_csv_name.strip()
        if not out_name.endswith('.csv'):
            out_name += '.csv'
            
        output_csv = os.path.join(AppConfig.SANDBOX_DIR, "data/results", out_name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        with tempfile.TemporaryDirectory() as temp_batch_dir:
            mapping = {}
            for rel_folder in req.target_folders:
                folder = os.path.join(AppConfig.SANDBOX_DIR, rel_folder)
                if not os.path.exists(folder):
                    continue
                
                if os.path.isfile(folder):
                    candidate_files = [folder]
                else:
                    candidate_files = []
                    for root, dirs, files in os.walk(folder):
                        for f in files:
                            candidate_files.append(os.path.join(root, f))

                for orig_path in candidate_files:
                    if orig_path.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                        rel_orig_path = os.path.relpath(orig_path, AppConfig.SANDBOX_DIR).replace("\\", "/")
                        if rel_orig_path not in mapping.values():
                            ext = os.path.splitext(orig_path)[1]
                            temp_name = f"{uuid.uuid4().hex}{ext}"
                            temp_path = os.path.join(temp_batch_dir, temp_name)
                            shutil.copy2(orig_path, temp_path)
                            mapping[temp_name] = rel_orig_path

            if not mapping:
                raise Exception("No supported audio files (.wav, .mp3, etc.) found in selected folders.")
                
            temp_csv = os.path.join(temp_batch_dir, "inference_temp.csv")
            python_exe = os.path.join(AppConfig.SANDBOX_DIR, "venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = sys.executable

            cmd = [
                python_exe,
                "-m", "transcription.inference.batch",
                temp_batch_dir,
                "--output", temp_csv
            ]
            if req.checkpoint:
                cmd.extend([
                    "--checkpoint", req.checkpoint,
                    "--processor", req.checkpoint
                ])
            print(f"Running batched inference on {len(mapping)} files...", flush=True)
            result = subprocess.run(cmd, cwd=AppConfig.SANDBOX_DIR)
            if result.returncode != 0:
                raise Exception(f"Batch inference failed with exit code {result.returncode}. Check terminal for details.")
                
            if not os.path.exists(temp_csv):
                raise Exception("Batch inference failed to generate CSV.")
                
            df = pd.read_csv(temp_csv)
            df["file_path"] = df["file_path"].apply(lambda x: mapping[os.path.basename(x)])
            df["filename"] = df["file_path"].apply(lambda x: os.path.basename(x))
            
            df.to_csv(output_csv, index=False)
            
        return {"message": f"Batch inference complete for {len(mapping)} files across {len(req.target_folders)} folders.", "csv_path": output_csv}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/config/best_model")
def get_best_model():
    try:
        from transcription.utils.model_utils import get_best_model_config
        return get_best_model_config()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
def get_files():
    base_dir = AppConfig.SANDBOX_DIR
    txt_files = []
    wav_files = []
    csv_files = []
    if not os.path.exists(base_dir):
        return {"txt_files": [], "wav_files": [], "csv_files": []}
        
    for root, dirs, files in os.walk(base_dir):
        # Prune massive ignored directories in place to prevent os.walk from even entering them
        dirs[:] = [d for d in dirs if d not in ["venv", "node_modules", ".git", "__pycache__"] and not d.startswith("logs-")]
        

        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), base_dir).replace("\\", "/")
            lower_f = f.lower()
            if lower_f.endswith(".txt"):
                txt_files.append(rel_path)
            elif lower_f.endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".webm", ".opus")):
                wav_files.append(rel_path)
            elif lower_f.endswith(".csv"):
                csv_files.append(rel_path)
                
    return {"txt_files": txt_files, "wav_files": wav_files, "csv_files": csv_files}

@app.get("/api/inference_files")
def get_inference_files():
    base_dir = AppConfig.SANDBOX_DIR
    wav_files = []
    if not os.path.exists(base_dir):
        return {"wav_files": []}
        
    for root, dirs, files in os.walk(base_dir):
        # Prune massive ignored directories in place to prevent os.walk from even entering them
        dirs[:] = [d for d in dirs if d not in ["venv", "node_modules", ".git", "__pycache__"] and not d.startswith("logs-")]
        

        for f in files:
            if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".webm", ".opus")):
                rel_path = os.path.relpath(os.path.join(root, f), base_dir).replace("\\", "/")
                wav_files.append(rel_path)
                
    return {"wav_files": wav_files}

def countDigits(num): return len(str(num))
def addZeros(num, max_num): return str(num).zfill(countDigits(max_num))
def reformatTranscription(text):
    punctuation = [ "[", "]", "\"", "(", ")", ".", "\u0f7b", "_", "|", "》", "?", "!", "/", ',', '-', '?', '<', '…', '>' ]
    for p in punctuation: text = text.replace(p, " ")
    for _ in range(5): text = text.replace("  "," ")
    return text.strip().lower()

@app.post("/api/process_elan")
def process_elan(req: ProcessElanRequest):
    try:
        tab_path = os.path.join(AppConfig.SANDBOX_DIR, req.txt_file)
        wav_path = os.path.join(AppConfig.SANDBOX_DIR, req.wav_file)
        
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=404, detail="wav file not found")
        
        if not os.path.exists(tab_path):
            raise HTTPException(status_code=404, detail="txt file not found")
            
        stem = os.path.splitext(req.txt_file)[0]
        speaker_code = stem.rsplit("-", 1)[1].upper() if "-" in stem else stem.upper()
        audio_prefix = os.path.splitext(req.wav_file)[0]

        try:
            with open(tab_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(tab_path, 'r', encoding='utf-16') as f:
                lines = f.readlines()
            
        timeStart, timeEnd, transcriptions = [], [], []
        
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    start = float(parts[2])
                    end = float(parts[3])
                except ValueError:
                    continue
                trans = parts[-1]
                dur = end - start
                if dur > 0 and reformatTranscription(trans):
                    timeStart.append(start)
                    timeEnd.append(end)
                    transcriptions.append(trans)
                    
        if not timeStart:
            raise HTTPException(status_code=400, detail="No valid annotations found.")
            
        points = []
        filenames = []
        
        for i, (start, end) in enumerate(zip(timeStart, timeEnd)):
            tempName1 = f"{speaker_code}-{audio_prefix}-{addZeros(i+1, len(timeStart))}.wav"
            tempName2 = f"{speaker_code}-{audio_prefix}-{addZeros(i+2, len(timeStart))}.wav"
            if points and points[-1] == start:
                points.append(end)
                filenames.append(tempName1)
            else:
                points.append(start)
                points.append(end)
                filenames.append(tempName2)
                
        pointSeq = ','.join(str(p) for p in points)
        zeros = countDigits(len(timeStart))
        
        outFileName = os.path.join(AppConfig.get_wav_dir(), f"{speaker_code}-{audio_prefix}-%0{zeros}d.wav")
        
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-f", "segment", "-ac", "1", "-ar", "16000", "-async", "1", "-segment_times", pointSeq, outFileName]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"FFmpeg error: {stderr}")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="FFmpeg is not installed or not in PATH.")
            
        todaysDate = date.today().strftime("%Y%m%d")
        
        try:
            if os.path.exists(AppConfig.get_metadata_csv()):
                df = pd.read_csv(AppConfig.get_metadata_csv())
            else:
                columns = ["wav_filename", "sandbox", "date", "speakerCode", "speakerGender", "wav_filesize", "duration_seconds", "codeSwitch", "needsFurtherCheck", "transcript_clean", "transcript"]
                df = pd.DataFrame(columns=columns)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading metadata CSV: {str(e)}")
            
        new_rows = []
        for i in range(len(timeStart)):
            fname = filenames[i]
            full_path = os.path.join(AppConfig.get_wav_dir(), fname)
            
            duration, size = 0, 0
            if os.path.exists(full_path):
                try:
                    with contextlib.closing(wave.open(full_path, 'r')) as f:
                        duration = f.getnframes() / float(f.getframerate())
                    size = os.path.getsize(full_path)
                except Exception:
                    pass
                
            new_rows.append({
                "wav_filename": fname,
                "sandbox": os.path.basename(AppConfig.SANDBOX_DIR).replace("sandbox-", ""),
                "date": todaysDate,
                "speakerCode": speaker_code,
                "speakerGender": req.gender,
                "wav_filesize": size,
                "duration_seconds": duration,
                "codeSwitch": "",
                "needsFurtherCheck": "",
                "transcript_clean": reformatTranscription(transcriptions[i]),
                "transcript": transcriptions[i]
            })
            
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(AppConfig.get_metadata_csv(), index=False)
        
        return {"message": "Success", "rows_added": len(new_rows)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/generate_splits")
def generate_splits(req: GenerateSplitsRequest):
    if req.train_pct + req.valid_pct + req.test_pct != 100:
        raise HTTPException(status_code=400, detail="Percentages must sum to 100")
        
    if not os.path.exists(AppConfig.get_metadata_csv()):
        raise HTTPException(status_code=404, detail="Metadata CSV not found. Process ELAN first.")
        
    df = pd.read_csv(AppConfig.get_metadata_csv())
    
    if not req.use_code_switched and 'codeSwitch' in df.columns:
        df = df[df['codeSwitch'] != "1"]
        df = df[df['codeSwitch'] != 1]
    if not req.use_doubtful and 'needsFurtherCheck' in df.columns:
        df = df[df['needsFurtherCheck'] != "1"]
        df = df[df['needsFurtherCheck'] != 1]
        
    if 'transcript' in df.columns:
        df = df[df['transcript'].notna() & (df['transcript'] != "")]
    
    if 'duration_seconds' in df.columns:
        df['duration_seconds'] = df['duration_seconds'].astype(str).str.replace(',', '.').astype(float)
        df = df[df['duration_seconds'] <= req.max_duration]
        
    if 'wav_filename' not in df.columns or 'transcript' not in df.columns:
         raise HTTPException(status_code=400, detail="CSV missing required columns")

    paths = [os.path.join(AppConfig.get_wav_dir(), row['wav_filename']).replace("\\", "/") for _, row in df.iterrows()]
    sentences = df['transcript'].tolist()
    
    combined = list(zip(paths, sentences))
    random.shuffle(combined)
    
    n_total = len(combined)
    n_train = int(round(n_total * (req.train_pct / 100.0)))
    n_valid = int(round(n_total * (req.valid_pct / 100.0)))
    
    train_data = combined[:n_train]
    valid_data = combined[n_train:n_train+n_valid]
    test_data = combined[n_train+n_valid:]
    
    def save_split(data, filename):
        out_df = pd.DataFrame(data, columns=["path", "sentence"])
        out_path = os.path.join(AppConfig.SANDBOX_DIR, filename)
        out_df.to_csv(out_path, index=False)
        return len(data)
        
    t_cnt = save_split(train_data, f"{req.file_prefix}-train.csv")
    v_cnt = save_split(valid_data, f"{req.file_prefix}-valid.csv")
    te_cnt = save_split(test_data, f"{req.file_prefix}-test.csv")
    
    return {"message": "Splits generated successfully", "train": t_cnt, "valid": v_cnt, "test": te_cnt}

@app.get("/api/devices")
def get_devices():
    try:
        import torch
        devices = [{"id": "cpu", "name": "CPU"}]
        if torch.cuda.is_available():
            devices.append({"id": "all", "name": "All GPUs"})
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append({"id": f"cuda:{i}", "name": f"GPU {i}: {name}"})
        return {"devices": devices}
    except Exception:
        return {"devices": [{"id": "cpu", "name": "CPU (Default)"}]}

@app.get("/api/checkpoints")
def get_checkpoints():
    # WARNING: This logic expects checkpoints to be directly under AppConfig.get_model_dir() (wav2vec2-model/).
    # However, transcription.training.train writes checkpoints to output_w2v2/wav2vec2-large-xlsr/.
    # Consequently, these checkpoint paths are going to be messed up / out of sync.
    model_dir = AppConfig.get_model_dir()
    checkpoints = []
    if os.path.exists(model_dir):
        for d in os.listdir(model_dir):
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d)):
                checkpoints.append(d)
    
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1, reverse=True)
    return {"checkpoints": ["charliemcvicker/asr-cherokee"] + checkpoints}

class TrainRequest(BaseModel):
    train_csv: str
    valid_csv: str
    test_csv: str
    epochs: int
    ngrams: int
    run_id: str
    lang_prefix: str
    lmplz_path: str = None
    device: str = "all"

@app.post("/api/train")
def train_model(req: TrainRequest):
    # Run the new training script entrypoint instead of scripts/run_training.py
    cmd = [
        sys.executable, "-m", "transcription.training.train",
        "--train-csv", req.train_csv,
        "--valid-csv", req.valid_csv,
        "--test-csv", req.test_csv,
        "--epochs", str(req.epochs),
    ]
    if req.lmplz_path:
        cmd.extend(["--lmplz-path", req.lmplz_path])
        
    env = os.environ.copy()
    if req.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif req.device.startswith("cuda:"):
        idx = req.device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = idx

    try:
        # Run training in a separate process, non-blocking
        subprocess.Popen(cmd, env=env, cwd=AppConfig.SANDBOX_DIR)
        return {"message": "Training started in the background. Check terminal for logs."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TranscribeLongRequest(BaseModel):
    audio_file: str
    checkpoint: str

@app.post("/api/transcribe_long")
def transcribe_long(req: TranscribeLongRequest):
    # Assuming audio files are uploaded or present in audiofiles-to-transcribe
    audio_path = os.path.join(AppConfig.SANDBOX_DIR, req.audio_file)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    out_tsv = os.path.join(AppConfig.SANDBOX_DIR, req.audio_file.rsplit('.', 1)[0] + ".tsv")
    
    # WARNING: This resolves the checkpoint path against AppConfig.get_model_dir().
    # Because transcription.training.train outputs to output_w2v2/wav2vec2-large-xlsr, this will be messed up.
    checkpoint_val = req.checkpoint
    if checkpoint_val != "charliemcvicker/asr-cherokee":
        checkpoint_val = os.path.join(AppConfig.get_model_dir(), checkpoint_val)
    
    cmd = [
        sys.executable, os.path.join(AppConfig.SANDBOX_DIR, "scripts", "run_inference_julie.py"),
        audio_path,
        "--checkpoint", checkpoint_val,
        "--processor", checkpoint_val
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=AppConfig.SANDBOX_DIR)
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {res.stderr}")
            
        transcription = ""
        lines = res.stdout.splitlines()
        for i, line in enumerate(lines):
            if "KENLM DECODING PREDICTIONS:" in line:
                if i + 1 < len(lines):
                    transcription = lines[i+1].replace("Transcription:", "").strip()
                break
        
        if not transcription:
            for i, line in enumerate(lines):
                if "GREEDY DECODING PREDICTIONS:" in line:
                    if i + 1 < len(lines):
                        transcription = lines[i+1].replace("Transcription:", "").strip()
                    break

        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("start\tend\ttranscription\n")
            f.write(f"0.000\t0.000\t{transcription}\n")

        return {"message": "Transcription complete", "tsv_file": out_tsv, "transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File, Form

@app.post("/api/transcribe_mic")
def transcribe_mic(
    checkpoint: str = Form(...),
    audio: UploadFile = File(...)
):
    # Save uploaded audio to temp file
    temp_id = str(uuid.uuid4())
    temp_wav = os.path.join(tempfile.gettempdir(), f"mic_{temp_id}.wav")
    
    with open(temp_wav, "wb") as f:
        f.write(audio.file.read())
        
    # WARNING: This resolves the checkpoint path against AppConfig.get_model_dir().
    # Because transcription.training.train outputs to output_w2v2/wav2vec2-large-xlsr, this will be messed up.
    checkpoint_val = checkpoint
    if checkpoint_val != "charliemcvicker/asr-cherokee":
        checkpoint_val = os.path.join(AppConfig.get_model_dir(), checkpoint_val)
        
    cmd = [
        sys.executable, os.path.join(AppConfig.SANDBOX_DIR, "scripts", "run_inference_julie.py"),
        temp_wav,
        "--checkpoint", checkpoint_val,
        "--processor", checkpoint_val
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=AppConfig.SANDBOX_DIR)
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {res.stderr}")
            
        transcription = ""
        lines = res.stdout.splitlines()
        for i, line in enumerate(lines):
            if "KENLM DECODING PREDICTIONS:" in line:
                if i + 1 < len(lines):
                    transcription = lines[i+1].replace("Transcription:", "").strip()
                break
        
        if not transcription:
            for i, line in enumerate(lines):
                if "GREEDY DECODING PREDICTIONS:" in line:
                    if i + 1 < len(lines):
                        transcription = lines[i+1].replace("Transcription:", "").strip()
                    break
                
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

@app.get("/api/config/folder")
def get_folder():
    return {"folder": AppConfig.SANDBOX_DIR}

@app.post("/api/config/select_folder")
def select_folder():
    cmd = [
        sys.executable, "-c",
        "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); folder = filedialog.askdirectory(title='Select Base Folder'); print(folder);"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    selected_folder = res.stdout.strip()
    if selected_folder:
        AppConfig.SANDBOX_DIR = os.path.abspath(selected_folder)
        AppConfig.ensure_dirs()
    return {"folder": AppConfig.SANDBOX_DIR}

@app.get("/api/labeler/data")
def get_labeler_data(file: str = "data/results/batch_inference_results.csv"):
    csv_file = os.path.join(AppConfig.SANDBOX_DIR, file)
    train_file = os.path.join(AppConfig.SANDBOX_DIR, "data", "processed", "train_labeled.csv")
    
    if not os.path.exists(csv_file):
        raise HTTPException(status_code=404, detail=f"File not found: {file}")
        
    try:
        data = []
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            path_col = "file_path" if "file_path" in reader.fieldnames else "path" if "path" in reader.fieldnames else None
            txt_col = "greedy_transcription" if "greedy_transcription" in reader.fieldnames else "sentence" if "sentence" in reader.fieldnames else None
            
            for row in reader:
                audio_rel_path = row.get(path_col, "") if path_col else ""
                
                # If path is relative like 'sentence_audio/...', fix it so get_audio can find it
                if audio_rel_path.startswith("sentence_audio/"):
                    audio_rel_path = f"data/processed/{audio_rel_path}"
                    
                greedy_txt = row.get(txt_col, "")
                import json
                word_confs = []
                if "word_confidences" in row and row["word_confidences"]:
                    try:
                        word_confs = json.loads(row["word_confidences"])
                        greedy_words = [w for w in greedy_txt.strip().split() if w]
                        if len(word_confs) == 1 and len(greedy_words) > 1:
                            all_chars = word_confs[0].get("chars", [])
                            new_word_confs = []
                            char_idx = 0
                            for tw in greedy_words:
                                w_chars = []
                                for i in range(len(tw)):
                                    if char_idx < len(all_chars):
                                        w_chars.append(all_chars[char_idx])
                                        char_idx += 1
                                if w_chars:
                                    avg_c = float(np.mean([c.get("confidence", 0.0) for c in w_chars])) if w_chars else 0.0
                                    new_word = {
                                        "word": tw,
                                        "confidence": avg_c,
                                        "chars": w_chars
                                    }
                                    if "start_time" in w_chars[0]:
                                        new_word["start_time"] = w_chars[0]["start_time"]
                                        new_word["end_time"] = round(w_chars[-1].get("start_time", 0) + 0.02, 3)
                                    new_word_confs.append(new_word)
                            if char_idx < len(all_chars):
                                rem = all_chars[char_idx:]
                                avg_c = float(np.mean([c.get("confidence", 0.0) for c in rem])) if rem else 0.0
                                new_word = {
                                    "word": "".join([c.get("char", "") for c in rem]),
                                    "confidence": avg_c,
                                    "chars": rem
                                }
                                if "start_time" in rem[0]:
                                    new_word["start_time"] = rem[0]["start_time"]
                                    new_word["end_time"] = round(rem[-1].get("start_time", 0) + 0.02, 3)
                                new_word_confs.append(new_word)
                            word_confs = new_word_confs
                    except:
                        pass
                
                data.append({
                    "file_path": audio_rel_path,
                    "filename": row.get("filename", os.path.basename(audio_rel_path)),
                    "greedy_transcription": greedy_txt,
                    "greedy_confidence": float(row.get("greedy_confidence", 0.0)) if row.get("greedy_confidence") else 0.0,
                    "word_confidences": word_confs
                })
        
        # Sort segments by confidence ascending
        data.sort(key=lambda x: x["greedy_confidence"])
        
        labeled_map = {}
        if os.path.exists(train_file):
            with open(train_file, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labeled_map[row.get("path", "")] = row.get("sentence", "")
                    
        for row in data:
            row["labeled_sentence"] = labeled_map.get(row["file_path"], "")
            
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LabelItem(BaseModel):
    path: str
    sentence: str

class SaveLabelsRequest(BaseModel):
    labels: list[LabelItem]

@app.post("/api/labeler/save")
def save_labels(req: SaveLabelsRequest):
    train_file = os.path.join(AppConfig.SANDBOX_DIR, "data", "processed", "train_labeled.csv")
    try:
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "sentence"])
            for label in req.labels:
                writer.writerow([label.path, label.sentence])
        return {"status": "success", "message": f"Successfully saved {len(req.labels)} labels"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
