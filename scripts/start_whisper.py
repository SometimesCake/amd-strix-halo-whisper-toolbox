#!/usr/bin/env python3
"""
OpenAI-compatible Whisper transcription server with async job queue.

Endpoints:
  POST /v1/audio/transcriptions         — synchronous, OpenAI-compatible (unchanged)
  POST /v1/audio/transcriptions/async   — async job queue, returns job ID immediately
  GET  /jobs/{job_id}                   — poll job status
  GET  /jobs/{job_id}/result            — download completed JSON transcript
  GET  /health                          — server health
"""

import argparse
import asyncio
import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import torch
import whisper
import whisperx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn


def parse_args():
    p = argparse.ArgumentParser(description="Whisper transcription server (OpenAI-compatible)")
    p.add_argument("--model",      default="large-v3",
                   help="Whisper model (tiny/base/small/medium/large-v2/large-v3/turbo)")
    p.add_argument("--host",       default="0.0.0.0",   help="Bind host")
    p.add_argument("--port",       type=int, default=8000, help="Bind port")
    p.add_argument("--device",     default=None,        help="Device (cuda/cpu). Auto-detects ROCm.")
    p.add_argument("--language",   default=None,        help="Default language (e.g. 'en'). None = auto-detect.")
    p.add_argument("--jobs-dir",   default="/tmp/whisper-jobs",
                   help="Directory for async job output files")
    p.add_argument("--batch-size", type=int, default=None,
                   help="WhisperX batch size for async jobs (default: 16 on GPU, 4 on CPU)")
    return p.parse_args()


args = parse_args()

device       = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
compute_type = "float16" if device != "cpu" else "float32"
batch_size   = args.batch_size or (16 if device != "cpu" else 4)
jobs_dir     = Path(args.jobs_dir)
hf_token     = os.environ.get("HF_TOKEN")

# Sync endpoint: openai-whisper via PyTorch — full ROCm GPU support on gfx1151
print(f"Loading openai-whisper model '{args.model}' on device '{device}' ...", flush=True)
ow_model = whisper.load_model(args.model, device=device)
print("openai-whisper ready.", flush=True)

# Async endpoint: whisperx (ctranslate2) — word timestamps + speaker diarization
# Note: ctranslate2 ROCm support is experimental; falls back to CPU if GPU init fails
print(f"Loading whisperx model '{args.model}' ({compute_type}) ...", flush=True)
wx_model = whisperx.load_model(args.model, device=device, compute_type=compute_type)
print("whisperx ready.", flush=True)

if not hf_token:
    print("Note: HF_TOKEN not set — speaker diarization disabled.", flush=True)


# ── Global state (initialized in lifespan) ────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock: asyncio.Lock
_transcription_sem: asyncio.Semaphore
_thread_pool: ThreadPoolExecutor


# ── WhisperX pipeline (runs in thread, shares model with main process) ─────────

def _transcribe(audio_path: str, language: str | None) -> dict:
    """Transcribe → word-align → speaker diarize. Returns whisperx result dict."""
    audio  = whisperx.load_audio(audio_path)
    result = wx_model.transcribe(audio, batch_size=batch_size, language=language)
    lang   = result.get("language", "en")

    try:
        model_a, meta = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(
            result["segments"], model_a, meta, audio, device,
            return_char_alignments=False,
        )
        del model_a
    except Exception as exc:
        print(f"  Warning: word alignment failed ({exc}). Continuing.", flush=True)

    if hf_token:
        try:
            diarize      = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segs = diarize(audio)
            result       = whisperx.assign_word_speakers(diarize_segs, result)
            del diarize
        except Exception as exc:
            print(f"  Warning: diarization failed ({exc}). Continuing.", flush=True)

    for seg in result.get("segments", []):
        seg.setdefault("speaker", "UNKNOWN")
        for w in seg.get("words", []):
            w.setdefault("speaker", seg["speaker"])

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _stream_to_disk(upload: UploadFile, dest: Path) -> None:
    """Write an upload to disk in chunks — never reads the whole file into RAM."""
    async with aiofiles.open(dest, "wb") as f:
        async for chunk in upload.stream():
            await f.write(chunk)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _jobs_lock, _transcription_sem, _thread_pool
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _jobs_lock        = asyncio.Lock()
    _transcription_sem = asyncio.Semaphore(1)   # one transcription at a time (GPU memory)
    _thread_pool      = ThreadPoolExecutor(max_workers=1)
    yield
    _thread_pool.shutdown(wait=False)


app = FastAPI(title="Whisper Transcription API", lifespan=lifespan)


# ── Sync transcription (openai-whisper, PyTorch — full ROCm GPU) ──────────────

def _transcribe_sync(audio_path: str, language: str | None, temperature: float,
                     prompt: str | None) -> dict:
    kwargs = dict(language=language, temperature=temperature, fp16=(device != "cpu"))
    if prompt:
        kwargs["initial_prompt"] = prompt
    return whisper.transcribe(ow_model, audio_path, **kwargs)


# ── Sync endpoint — OpenAI-compatible, unchanged contract ─────────────────────

@app.post("/v1/audio/transcriptions")
async def transcribe_sync(
    file:            UploadFile = File(...),
    model:           str   = Form(default=None),
    language:        str   = Form(default=None),
    response_format: str   = Form(default="json"),
    temperature:     float = Form(default=0.0),
    prompt:          str   = Form(default=None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await _stream_to_disk(file, tmp_path)
        loop = asyncio.get_running_loop()
        async with _transcription_sem:
            result = await loop.run_in_executor(
                _thread_pool, _transcribe_sync,
                str(tmp_path), language or args.language, temperature, prompt,
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    if response_format == "text":
        return result["text"]
    if response_format == "verbose_json":
        return JSONResponse(result)
    return JSONResponse({"text": result["text"]})


# ── Async endpoint — job queue for large files ─────────────────────────────────

@app.post("/v1/audio/transcriptions/async", status_code=202)
async def transcribe_async(
    file:         UploadFile = File(...),
    model:        str = Form(default=None),
    language:     str = Form(default=None),
    speaker_name: str = Form(default=None),
):
    job_id  = str(uuid.uuid4())
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)

    suffix     = Path(file.filename or "audio.mp3").suffix or ".mp3"
    audio_path = job_dir / f"audio{suffix}"
    await _stream_to_disk(file, audio_path)

    async with _jobs_lock:
        _jobs[job_id] = {
            "job_id":      job_id,
            "status":      "queued",
            "filename":    file.filename,
            "created_at":  _now_iso(),
            "started_at":  None,
            "finished_at": None,
            "error":       None,
        }

    asyncio.create_task(_run_job(job_id, audio_path, language or args.language))
    return {"job_id": job_id}


async def _run_job(job_id: str, audio_path: Path, language: str | None) -> None:
    async with _jobs_lock:
        _jobs[job_id]["status"]     = "processing"
        _jobs[job_id]["started_at"] = _now_iso()

    result_path = audio_path.parent / "result.json"
    loop = asyncio.get_running_loop()
    try:
        async with _transcription_sem:
            result = await loop.run_in_executor(
                _thread_pool, _transcribe, str(audio_path), language,
            )
        async with aiofiles.open(result_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(result, ensure_ascii=False))
        async with _jobs_lock:
            _jobs[job_id]["status"]      = "done"
            _jobs[job_id]["finished_at"] = _now_iso()
    except Exception as exc:
        async with _jobs_lock:
            _jobs[job_id]["status"]      = "failed"
            _jobs[job_id]["finished_at"] = _now_iso()
            _jobs[job_id]["error"]       = str(exc)


# ── Job status / result endpoints ─────────────────────────────────────────────

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(job)


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    if job["status"] in ("queued", "processing"):
        return JSONResponse({"status": job["status"]}, status_code=202)
    if job["status"] == "failed":
        return JSONResponse({"error": job["error"]}, status_code=422)

    result_path = jobs_dir / job_id / "result.json"
    stem = Path(job["filename"]).stem if job["filename"] else job_id
    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename=f"{stem}.json",
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": args.model, "device": device}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting server on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
