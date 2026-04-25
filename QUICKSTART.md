# Whisper API — Quick Start

## 1. HuggingFace Setup (one-time, required for speaker labels)

Speaker diarization (who is speaking) requires a free HuggingFace account and token.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the licence at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the licence at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Skip this step if you don't need speaker labels — transcription still works without it.

---

## 2. Start the Server

### Option A — Fedora Toolbx (recommended)

```bash
# Pull the latest image and (re)create the toolbox
./refresh_toolbox.sh

# Enter the toolbox
toolbox enter whisper

# Start the server — with speaker diarization
HF_TOKEN=hf_xxxxxxxxxxxx start-whisper --model large-v3

# Start the server — transcription only (no speaker labels)
start-whisper --model large-v3
```

### Option B — Detached background server (persists after terminal closes)

Runs the server as a detached Podman container. The container keeps running after you close the terminal.

```bash
# Start both (default) — diarization on :8000, transcription-only on :8001
./start-server.sh

# Start one mode only
./start-server.sh --mode diarization
./start-server.sh --mode transcription
```

Useful commands once running:

```bash
podman logs -f whisper-diarization      # tail diarization logs
podman logs -f whisper-transcription    # tail transcription logs
curl http://localhost:8000/health       # diarization ready check
curl http://localhost:8001/health       # transcription ready check
podman stop whisper-diarization
podman stop whisper-transcription
```

The server won't be ready to accept requests until model loading is complete — watch `podman logs` for `Starting server on 0.0.0.0:8000`.

### Option C — Podman (interactive, foreground)

```bash
podman run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v ~/.cache/whisper:/root/.cache/whisper \
  -p 8000:8000 \
  localhost/whisper-therock-gfx1151:latest \
  start-whisper --model large-v3
```

The server is ready when you see:
```
openai-whisper ready.
whisperx ready.
Starting server on 0.0.0.0:8000
```

Models are downloaded on first run and cached in `~/.cache/whisper/`.

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `large-v3` | Model to use — see model table below |
| `--port` | `8000` | TCP port |
| `--language` | auto | Fix language, e.g. `en` (faster than auto-detect) |
| `--jobs-dir` | `/tmp/whisper-jobs` | Where async job files are stored |
| `--device` | auto | `cuda` for GPU, `cpu` to force CPU |

### Model reference

| Model | VRAM | Speed | Notes |
|---|---|---|---|
| `turbo` | ~3 GB | fastest | Near large-v3 quality, recommended for quick jobs |
| `large-v3` | ~6 GB | slower | Best accuracy |
| `small` | ~1 GB | very fast | Good for clear English audio |

---

## 3. Verify the Server is Running

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "large-v3", "device": "cuda"}
```

---

## 4. Transcribe a File

### Short files — synchronous

Returns the transcript directly. Use for files where you're happy to wait for the response (up to a few minutes of audio).

```bash
# Returns {"text": "..."}
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3"

# Plain text
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "response_format=text"

# Force English (skips language detection)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "language=en"
```

---

### Large files — async (recommended for audiobooks, long recordings)

Upload returns immediately with a job ID. Poll for status, then download the result when done.
Output is a rich JSON with word-level timestamps and speaker labels.

#### Step 1 — Upload

```bash
JOB=$(curl -s -X POST http://localhost:8000/v1/audio/transcriptions/async \
  -F "file=@audiobook.mp3" \
  -F "language=en" \
  | jq -r .job_id)

echo "Job ID: $JOB"
```

#### Step 2 — Poll for status

```bash
curl http://localhost:8000/jobs/$JOB
```
```json
{
  "job_id": "550e8400-...",
  "status": "processing",
  "filename": "audiobook.mp3",
  "created_at": "2026-04-24T10:00:00+00:00",
  "started_at": "2026-04-24T10:00:05+00:00",
  "finished_at": null,
  "error": null
}
```

Status values: `queued` → `processing` → `done` or `failed`

#### Step 3 — Download result

```bash
curl -O http://localhost:8000/jobs/$JOB/result
# Saves as audiobook.json
```

#### All-in-one polling script

```bash
# Upload
JOB=$(curl -s -X POST http://localhost:8000/v1/audio/transcriptions/async \
  -F "file=@audiobook.mp3" \
  -F "language=en" \
  | jq -r .job_id)
echo "Submitted: $JOB"

# Poll until done
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB | jq -r .status)
  echo "$(date '+%H:%M:%S')  $STATUS"
  [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ] && break
  sleep 30
done

# Download
[ "$STATUS" = "done" ] && curl -O http://localhost:8000/jobs/$JOB/result
```

#### Output format

```json
{
  "segments": [
    {
      "start": 0.031,
      "end": 1.733,
      "text": " This is Audible.",
      "words": [
        {"word": "This", "start": 0.031, "end": 0.892, "score": 0.93, "speaker": "SPEAKER_00"},
        {"word": "is",   "start": 0.912, "end": 0.952, "score": 0.50, "speaker": "SPEAKER_00"},
        {"word": "Audible.", "start": 1.253, "end": 1.733, "score": 0.61, "speaker": "SPEAKER_00"}
      ],
      "speaker": "SPEAKER_00"
    }
  ]
}
```

Speaker labels (`SPEAKER_00`, `SPEAKER_01`, etc.) are included when `HF_TOKEN` is set.
Without it, all `speaker` fields are `"UNKNOWN"`.

---

## 5. Async endpoint — result status codes

| Response | Meaning |
|---|---|
| `404` | Job ID not found |
| `202` | Job still queued or processing — try again later |
| `422` | Job failed — check the `error` field in `GET /jobs/{id}` |
| `200` | Done — JSON file is attached |
