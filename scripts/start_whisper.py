#!/usr/bin/env python3
"""
OpenAI-compatible Whisper transcription server.
Endpoint: POST /v1/audio/transcriptions
"""
import argparse
import io
import tempfile
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Whisper transcription server (OpenAI-compatible)")
    p.add_argument("--model",    default="turbo",     help="Whisper model name (tiny/base/small/medium/large-v3/turbo)")
    p.add_argument("--host",     default="0.0.0.0",   help="Bind host")
    p.add_argument("--port",     type=int, default=8000, help="Bind port")
    p.add_argument("--device",   default=None,        help="Device override (cuda/cpu). Auto-detects ROCm by default.")
    p.add_argument("--language", default=None,        help="Default language (e.g. 'en'). None = auto-detect.")
    return p.parse_args()

args = parse_args()

import torch
import whisper
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading whisper model '{args.model}' on device '{device}' ...", flush=True)
model = whisper.load_model(args.model, device=device)
print("Model ready.", flush=True)

app = FastAPI(title="Whisper Transcription API")

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=args.model),
    language: str = Form(default=args.language),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    prompt: str = Form(default=None),
):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or ".wav")[1] or ".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        kwargs = dict(
            language=language or args.language,
            temperature=temperature,
            fp16=(device != "cpu"),
        )
        if prompt:
            kwargs["initial_prompt"] = prompt

        result = whisper.transcribe(model, tmp_path, **kwargs)
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return result["text"]
    if response_format == "verbose_json":
        return JSONResponse(result)
    # default: json (OpenAI-compatible)
    return JSONResponse({"text": result["text"]})

@app.get("/health")
def health():
    return {"status": "ok", "model": args.model, "device": device}

if __name__ == "__main__":
    print(f"Starting server on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
