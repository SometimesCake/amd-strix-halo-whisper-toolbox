#!/usr/bin/env python3
"""
OpenAI-compatible Whisper transcription server with async job queue.

Both endpoints use openai-whisper (PyTorch) for transcription — full ROCm GPU support.
ctranslate2 (the whisperx transcription backend) is not used because it has no ROCm
support and crashes on CPU on this system.

When HF_TOKEN is set the diarization server runs a three-step pipeline:
  1. openai-whisper  — transcription          (PyTorch / ROCm GPU)
  2. whisperx align  — word-level timestamps  (PyTorch / ROCm GPU)
  3. pyannote        — speaker diarization    (PyTorch / ROCm GPU)

Endpoints:
  POST /v1/audio/transcriptions         — synchronous, OpenAI-compatible
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
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Suppress ROCm runtime noise — must be set before torch is imported
# since the ROCm runtime initialises at import time.
os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")  # silences all MIOpen output including DB warnings
os.environ.setdefault("HSA_XNACK", "0")         # xnack unsupported warning

# Suppress Python-level warnings from pyannote (torchcodec, TF32, std() dof)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\..*")

import aiofiles
import torch
import whisper
import whisperx
import whisperx.diarize
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn


def parse_args():
    p = argparse.ArgumentParser(description="Whisper transcription server (OpenAI-compatible)")
    p.add_argument("--model",    default="large-v3",
                   help="Whisper model (tiny/base/small/medium/large-v2/large-v3/turbo)")
    p.add_argument("--host",     default="0.0.0.0",  help="Bind host")
    p.add_argument("--port",     type=int, default=8000, help="Bind port")
    p.add_argument("--device",   default=None,        help="Device (cuda/cpu). Auto-detects ROCm.")
    p.add_argument("--language", default=None,        help="Default language (e.g. 'en'). None = auto-detect.")
    p.add_argument("--jobs-dir", default="/tmp/whisper-jobs",
                   help="Directory for async job output files")
    return p.parse_args()


args      = parse_args()
device    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
jobs_dir  = Path(args.jobs_dir)
hf_token  = os.environ.get("HF_TOKEN")

# Always load openai-whisper — used by both transcription and diarization servers.
# The diarization pipeline then passes the openai-whisper segments into whisperx
# alignment and pyannote diarization, both of which use PyTorch and have full GPU support.
print(f"Loading openai-whisper model '{args.model}' on device '{device}' ...", flush=True)
ow_model = whisper.load_model(args.model, device=device)

if hf_token:
    print("openai-whisper ready. HF_TOKEN set — diarization pipeline enabled.", flush=True)
else:
    print("openai-whisper ready. No HF_TOKEN — transcription only.", flush=True)


# ── Global state (initialized in lifespan) ────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock: asyncio.Lock
_transcription_sem: asyncio.Semaphore
_thread_pool: ThreadPoolExecutor


# ── Full diarization pipeline: openai-whisper → align → diarize ───────────────

def _transcribe(audio_path: str, language: str | None) -> dict:
    """
    Three-step pipeline — all PyTorch, full ROCm GPU support:
      1. openai-whisper transcription
      2. whisperx word alignment (wav2vec2)
      3. pyannote speaker diarization
    """
    audio = whisperx.load_audio(audio_path)

    # Step 1 — transcribe with openai-whisper (GPU)
    ow_result = whisper.transcribe(ow_model, audio_path, language=language, fp16=(device != "cpu"))
    lang      = ow_result.get("language", language or "en")
    result    = {"segments": ow_result["segments"], "language": lang}

    # Step 2 — word-level alignment (GPU)
    try:
        model_a, meta = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(
            result["segments"], model_a, meta, audio, device,
            return_char_alignments=False,
        )
        del model_a
    except Exception as exc:
        print(f"  Warning: word alignment failed ({exc}). Continuing.", flush=True)

    # Step 3 — speaker diarization (GPU)
    try:
        diarize      = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
        diarize_segs = diarize(audio)
        result       = whisperx.diarize.assign_word_speakers(diarize_segs, result)
        del diarize
    except Exception as exc:
        print(f"  Warning: diarization failed ({exc}). Continuing.", flush=True)

    for seg in result.get("segments", []):
        seg.setdefault("speaker", "UNKNOWN")
        seg.pop("words", None)  # word-level data not needed in final output

    return result


# ── Fast transcription-only path (no diarization) ─────────────────────────────

def _transcribe_sync(audio_path: str, language: str | None, temperature: float,
                     prompt: str | None) -> dict:
    kwargs = dict(language=language, temperature=temperature, fp16=(device != "cpu"))
    if prompt:
        kwargs["initial_prompt"] = prompt
    return whisper.transcribe(ow_model, audio_path, **kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _stream_to_disk(upload: UploadFile, dest: Path) -> None:
    """Write an upload to disk in chunks — never reads the whole file into RAM."""
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(65536):
            await f.write(chunk)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _jobs_lock, _transcription_sem, _thread_pool
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _jobs_lock         = asyncio.Lock()
    _transcription_sem = asyncio.Semaphore(1)   # one transcription at a time (GPU memory)
    _thread_pool       = ThreadPoolExecutor(max_workers=1)
    yield
    _thread_pool.shutdown(wait=False)


app = FastAPI(title="Whisper Transcription API", lifespan=lifespan)


# ── Sync endpoint ─────────────────────────────────────────────────────────────
# Diarization server (HF_TOKEN set): full three-step pipeline, returns speaker labels.
# Transcription server (no HF_TOKEN):  fast openai-whisper only, returns text.

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
            if hf_token:
                result = await loop.run_in_executor(
                    _thread_pool, _transcribe,
                    str(tmp_path), language or args.language,
                )
                text = " ".join(seg.get("text", "").strip() for seg in result.get("segments", []))
            else:
                result = await loop.run_in_executor(
                    _thread_pool, _transcribe_sync,
                    str(tmp_path), language or args.language, temperature, prompt,
                )
                text = result["text"]
    finally:
        tmp_path.unlink(missing_ok=True)

    if response_format == "text":
        return text
    if response_format == "verbose_json":
        return JSONResponse(result)
    return JSONResponse({"text": text})


# ── Async endpoint — job queue for large files ────────────────────────────────

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
            if hf_token:
                result = await loop.run_in_executor(
                    _thread_pool, _transcribe, str(audio_path), language,
                )
            else:
                result = await loop.run_in_executor(
                    _thread_pool, _transcribe_sync, str(audio_path), language, 0.0, None,
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
