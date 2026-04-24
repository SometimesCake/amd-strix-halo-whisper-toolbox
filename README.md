# AMD Strix Halo (gfx1151) — Whisper Toolbox

A **Fedora 43** Docker/Podman container (Toolbx-compatible) for GPU-accelerated speech-to-text using **OpenAI Whisper** on **AMD Ryzen AI Max "Strix Halo" (gfx1151)**. Built on **TheRock nightly ROCm builds** — the same ROCm source used by the vLLM toolbox in this project family.

---

## Table of Contents

- [Overview](#overview)
- [Repository File Guide](#repository-file-guide)
- [Design Decisions](#design-decisions)
- [How to Build](#how-to-build)
- [How to Use](#how-to-use)
  - [Python Image — Toolbx (Recommended)](#python-image--toolbx-recommended)
  - [Python Image — Docker/Podman](#python-image--dockerpodman)
  - [whisper.cpp Image](#whispercpp-image)
- [API Reference](#api-reference)
- [Changing Models](#changing-models)
- [Host Configuration](#host-configuration)

---

## Overview

This toolbox provides two separate images targeting different use cases:

| Image | File | Stack | Use Case |
|---|---|---|---|
| **Python / REST** | `Dockerfile` | Fedora 43 + ROCm nightly + PyTorch nightly + `openai-whisper` + FastAPI | OpenAI-compatible HTTP API, scriptable, flexible |
| **whisper.cpp** | `Dockerfile.whispercpp` | Fedora 43 + ROCm nightly + whisper.cpp (HIP) | Lightweight C++ binary, web UI, command line |

Both images target **gfx1151** (Strix Halo iGPU) and pull the latest TheRock nightly ROCm SDK at build time.

---

## Repository File Guide

This section describes every file and what to edit if you need to change something.

```
amd-strix-halo-wisper/
├── Dockerfile                   # Python image build instructions
├── Dockerfile.whispercpp        # whisper.cpp image build instructions
├── README.md                    # This file
├── refresh_toolbox.sh           # Helper: pull latest image, recreate Toolbx
└── scripts/
    ├── install_deps.sh          # System packages installed via dnf (both images)
    ├── install_rocm_sdk.sh      # Downloads and installs TheRock ROCm from S3
    ├── 01-rocm-env.sh           # Extra ROCm env vars loaded at shell login
    ├── 99-toolbox-banner.sh     # Welcome banner shown on interactive shell open
    ├── zz-venv-last.sh          # Ensures /opt/venv/bin stays first in PATH
    └── start_whisper.py         # The FastAPI transcription server (Python image)
```

### `Dockerfile`

The main Python image. Steps in order:

1. **Install system deps** — runs `scripts/install_deps.sh` via dnf
2. **Install ROCm SDK** — runs `scripts/install_rocm_sdk.sh`; controlled by two build ARGs:
   - `ROCM_MAJOR_VER=7` — major version of TheRock to pull (change to `8` when ROCm 8 releases)
   - `GFX=gfx1151` — GPU architecture target (change for other AMD GPUs, e.g. `gfx1100` for RX 7900)
3. **Python venv** — creates `/opt/venv` with Python 3.12; activated globally via `/etc/profile.d/venv.sh`
4. **PyTorch nightly** — pulled from `rocm.nightlies.amd.com/v2-staging/gfx1151/` with `--pre` flag; includes `torchaudio` and `torchvision`. A one-line `sed` patch fixes a JSON serialization bug present in recent nightlies.
5. **Whisper + API deps** — installs `openai-whisper`, `transformers`, `accelerate`, `fastapi`, `uvicorn`, `python-multipart`, `soundfile`, `numpy`
6. **Cleanup** — strips pip cache, `__pycache__`, dnf cache to reduce image size
7. **Profile scripts** — copies `01-rocm-env.sh`, `99-toolbox-banner.sh`, `zz-venv-last.sh` into `/etc/profile.d/`
8. **start-whisper** — copies `start_whisper.py` to `/opt/start-whisper` and symlinks it to `/usr/local/bin/start-whisper`

**To change the ROCm version or GPU target**, edit the ARG lines near the top:
```dockerfile
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
```

**To add a Python package**, add it to the `pip install` block in step 5.

---

### `Dockerfile.whispercpp`

The lightweight C++ image. Steps in order:

1. **Install C++ build deps** — inline `dnf install` (no Python, lighter than the Python image); includes `ffmpeg-free` for audio decoding
2. **Install ROCm SDK** — same `scripts/install_rocm_sdk.sh` as the Python image
3. **ROCm ENV vars** — set inline as Docker `ENV` statements (required for CMake to find HIP at build time)
4. **Clone and build whisper.cpp** — clones `github.com/ggerganov/whisper.cpp` at HEAD, builds with:
   - `-DGGML_HIP=ON` — enables the HIP (ROCm) GPU backend
   - `-DAMDGPU_TARGETS=gfx1151` — compiles GPU kernels specifically for Strix Halo
   - ROCm Clang (`/opt/rocm/llvm/bin/clang++`) as both C and C++ compiler
   - Targets built: `whisper-server` and `whisper-cli`
5. **Symlinks** — `whisper-server` and `whisper-cli` linked into `/usr/local/bin/`
6. **Entrypoint** — defaults to `whisper-server --help`; override with your own args at `podman run` time

**To build a specific whisper.cpp commit** instead of HEAD, add after the `git clone` line:
```dockerfile
RUN git -C /opt/whisper.cpp checkout <commit-hash>
```

---

### `scripts/install_deps.sh`

Installs system packages via `dnf` for the **Python image**. Edit this file to add or remove system-level packages.

Current packages and why each is included:

| Package | Reason |
|---|---|
| `python3.12`, `python3.12-devel` | Runtime and headers for building Python extensions |
| `git` | Needed by some pip packages that build from source |
| `libatomic` | Required by ROCm/HIP libraries |
| `bash`, `ca-certificates`, `curl` | Shell and HTTPS downloads |
| `gcc`, `gcc-c++`, `binutils`, `make` | C/C++ toolchain for building pip wheels |
| `cmake`, `ninja-build` | Build system used by some pip packages |
| `aria2c` | Multi-connection downloader used by `install_rocm_sdk.sh` |
| `tar`, `xz` | Archive tools for ROCm tarball extraction |
| `vim`, `nano` | Text editors for convenience inside the container |
| `zlib-devel`, `openssl-devel` | Common C library headers |
| `ffmpeg-free`, `ffmpeg-free-devel` | Audio decoding — required by `openai-whisper` to read mp3/m4a/flac/etc. |
| `gperftools-libs` | Provides `libtcmalloc_minimal.so.4`, preloaded to fix a ROCm shutdown crash |

---

### `scripts/install_rocm_sdk.sh`

Shared by both images. Downloads and installs the **TheRock nightly ROCm SDK** from AMD's S3 bucket.

**How it works:**
1. Lists all tarballs matching `therock-dist-linux-{GFX}-{ROCM_MAJOR_VER}` in the S3 bucket
2. Selects the latest one using version-sort (`sort -V | tail -n1`)
3. Downloads with `aria2c` (16 parallel connections for speed)
4. Extracts to `/opt/rocm`
5. Writes `/etc/profile.d/rocm-sdk.sh` with all required ROCm environment variables

**Environment variables written to `/etc/profile.d/rocm-sdk.sh`:**

| Variable | Value | Purpose |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` | Root of ROCm installation |
| `HIP_PLATFORM` | `amd` | Tells HIP to use the AMD backend, not NVIDIA |
| `HIP_PATH` | `/opt/rocm` | HIP library root |
| `HIP_CLANG_PATH` | `/opt/rocm/llvm/bin` | Path to ROCm's clang compiler |
| `HIP_DEVICE_LIB_PATH` | (auto-detected bitcode dir) | GPU device library bitcode, needed for kernel compilation |
| `PATH` | prepends `/opt/rocm/bin` and `/opt/rocm/llvm/bin` | Makes `rocm-smi`, `hipcc`, etc. available |
| `LD_LIBRARY_PATH` | prepends ROCm lib dirs | Shared library search path |
| `ROCBLAS_USE_HIPBLASLT` | `1` | Use HIPBlasLt backend for matrix ops (better performance) |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Enables experimental Triton kernels for ROCm (needed for Strix Halo) |
| `VLLM_TARGET_DEVICE` | `rocm` | Carried over from the vLLM toolbox this script was shared from; harmless for Whisper |
| `HIP_FORCE_DEV_KERNARG` | `1` | ROCm performance tuning flag |
| `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES` | `1` | Prevents Ray from interfering with GPU visibility; carried over, harmless |
| `LD_PRELOAD` | `libtcmalloc_minimal.so.4` + `librocm_smi64.so.1.0` | tcmalloc prevents a double-free crash on shutdown; rocm_smi preload improves stability |

> **Note:** This script is copied from the vLLM toolbox unchanged. Some variables (`VLLM_TARGET_DEVICE`, `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES`) are vLLM-specific but are harmless when set in a Whisper environment. If you want a cleaner image you can remove them from this file.

---

### `scripts/01-rocm-env.sh`

Loaded at every interactive shell login via `/etc/profile.d/`. Sets one additional env var on top of what `install_rocm_sdk.sh` already wrote:

| Variable | Value | Purpose |
|---|---|---|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Enables experimental AOTriton kernels — required for Strix Halo (gfx1151) to achieve GPU acceleration in PyTorch |

This is intentionally minimal. `install_rocm_sdk.sh` handles the bulk of the ROCm environment. This file exists as the place to add any Whisper-specific runtime tuning without touching the shared ROCm installer.

---

### `scripts/99-toolbox-banner.sh`

Loaded at interactive shell login. Displays the welcome message when you enter the toolbox. The `99-` prefix ensures it runs last among profile scripts.

**What it shows:** machine name (from DMI), GPU name (from `rocm-smi` or `rocminfo`), ROCm version (from `torch.version.hip`), and quick-start commands.

**To edit the banner text**, find the `cat <<'ASCII' ... ASCII` block and the `printf` lines below it.

**To add new quick-start commands to the banner**, add more `printf` lines in the "Included:" section near the bottom of the file.

---

### `scripts/zz-venv-last.sh`

Loaded at interactive shell login. The `zz-` prefix ensures it runs after all other profile scripts, including any user dotfiles.

**Purpose:** Guarantees `/opt/venv/bin` is always first in `PATH`. Without this, user dotfiles (`.bashrc`, `.cargo/env`, etc.) can prepend their own paths and shadow the container's Python venv, causing the wrong `python` or `pip` to be used.

**Do not edit this file** unless you are changing the venv location from `/opt/venv`.

---

### `scripts/start_whisper.py`

The FastAPI transcription server. Installed as `/opt/start-whisper` and symlinked to `/usr/local/bin/start-whisper`.

**Command-line arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `large-v3` | WhisperX model to load at startup. Downloaded from HuggingFace on first run |
| `--host` | `0.0.0.0` | Network interface to bind. `0.0.0.0` means accessible from outside the container |
| `--port` | `8000` | TCP port for the HTTP server. Change this if 8000 is already in use |
| `--device` | auto | Force `cuda` (ROCm GPU) or `cpu`. By default, uses `cuda` if `torch.cuda.is_available()` returns true |
| `--language` | auto | Set a default language for all requests (e.g. `en`, `fr`, `de`). When not set, WhisperX auto-detects per request |
| `--jobs-dir` | `/tmp/whisper-jobs` | Directory where async job audio uploads and result JSON files are stored |
| `--batch-size` | 16 (GPU) / 4 (CPU) | WhisperX transcription batch size |

**Speaker diarization** (who is speaking) is enabled automatically when the `HF_TOKEN` environment variable is set to a valid HuggingFace access token. The token must have access to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` — accept the model licences at huggingface.co before use.

**Endpoints:**

- `POST /v1/audio/transcriptions` — synchronous transcription (OpenAI-compatible, unchanged)
- `POST /v1/audio/transcriptions/async` — async transcription for large files; returns a job ID immediately
- `GET /jobs/{job_id}` — poll the status of an async job
- `GET /jobs/{job_id}/result` — download the completed JSON transcript
- `GET /health` — returns current model name and device

**How async transcription works:**
1. Audio file is streamed directly to disk in chunks (no full-file RAM buffering)
2. A job ID is returned immediately — the client can close the connection
3. WhisperX pipeline runs in a background thread: transcribe → word-align → speaker diarize
4. When done, the result JSON is written to `{jobs-dir}/{job_id}/result.json`
5. Client downloads the result via `GET /jobs/{job_id}/result`

**To change the default model**, edit the `--model` default in `parse_args()`.

---

### `refresh_toolbox.sh`

Helper script for **Fedora Toolbx users**. Pulls the latest image from the registry, removes the old toolbox, and creates a fresh one with the correct GPU device flags.

**To use a locally built image instead of pulling from a registry**, replace the `podman pull` line with a no-op or remove it, and change `IMAGE` to your local image tag.

**Key variables at the top of the file:**

| Variable | Value | Description |
|---|---|---|
| `TOOLBOX_NAME` | `whisper` | Name of the Toolbx container |
| `IMAGE` | `docker.io/kyuz0/whisper-therock-gfx1151:latest` | Image to pull and use |

---

## Design Decisions

### Why two backends — `openai-whisper` for sync and `whisperx` for async?

**Sync endpoint** uses `openai-whisper` (PyTorch-based). PyTorch inherits full ROCm GPU acceleration from the TheRock nightly builds, so this path is guaranteed to run on the gfx1151 GPU. It returns a standard `{"text": "..."}` response compatible with the OpenAI API.

**Async endpoint** uses `whisperx` (CTranslate2 / faster-whisper backend). WhisperX adds a word alignment pass (wav2vec2) and speaker diarization (pyannote) on top of transcription, producing the rich JSON format with word-level timestamps and per-word speaker labels. CTranslate2's ROCm support is experimental — GPU acceleration may work via ROCm's CUDA compatibility layer, but is not guaranteed. For 30–40 hour background jobs this tradeoff is acceptable: even on CPU the job completes eventually, and speaker diarization is the primary reason to use this path.

### Why PyTorch nightly instead of a stable release?

gfx1151 (Strix Halo) is not in any stable PyTorch release. The TheRock nightly builds from `rocm.nightlies.amd.com/v2-staging/gfx1151/` are the only PyTorch wheels that have GPU kernel support compiled for this architecture. There is no stable alternative.

### Why TheRock ROCm instead of the official AMD ROCm packages?

Official AMD ROCm packages do not include gfx1151. TheRock is AMD's nightly build pipeline for unreleased GPU targets. It is the same approach used by the sibling vLLM toolbox in this project family.

### Why ROCm Clang instead of system GCC for whisper.cpp?

whisper.cpp's HIP backend requires the ROCm Clang compiler (`/opt/rocm/llvm/bin/clang++`) because it compiles `.hip` device code that GCC cannot handle. This is the same reason the vLLM toolbox forces `CC`/`CXX` to ROCm Clang.

### Why two separate images?

The Python image (`Dockerfile`) is heavier (~several GB) due to PyTorch but is more flexible — it accepts any audio format ffmpeg can decode, returns structured JSON, and is directly compatible with OpenAI API clients. The whisper.cpp image (`Dockerfile.whispercpp`) is much lighter, has a built-in web UI, and can be useful for quick CLI transcription or when you don't want the Python stack. They share the same ROCm SDK installer so the GPU layer is identical.

### Why `fp16=True` on GPU?

In `start_whisper.py`, `fp16` is set to `True` whenever the device is not CPU. Half-precision inference is significantly faster on the Strix Halo iGPU and produces negligibly different results for speech recognition. On CPU, fp16 is disabled because most CPUs do not support it natively and it would cause errors.

### Why is the venv path fixed in `zz-venv-last.sh`?

When the toolbox is used as a Fedora Toolbx, the user's host dotfiles (`.bashrc`, `.bash_profile`, etc.) are sourced inside the container because the home directory is shared. Tools like Rust's `cargo`, `pyenv`, and `nvm` prepend to PATH in those dotfiles, which can shadow `/opt/venv/bin/python`. The `zz-` prefix ensures this fix runs after all user dotfiles, reliably putting the container's Python first.

---

## How to Build

### Python image

```bash
cd /home/cake/toolboxes/amd-strix-halo-wisper

podman build -t whisper-gfx1151:latest .
```

Build args (optional overrides):

```bash
# Target a different GPU
podman build --build-arg GFX=gfx1100 -t whisper-gfx1100:latest .

# Pin a specific ROCm major version
podman build --build-arg ROCM_MAJOR_VER=8 -t whisper-gfx1151-rocm8:latest .
```

### whisper.cpp image

```bash
podman build -f Dockerfile.whispercpp -t whisper-cpp-gfx1151:latest .
```

---

## How to Use

### Python Image — Toolbx (Recommended)

Toolbx shares your home directory, so Whisper model downloads persist between container rebuilds.

```bash
# Create the toolbox (GPU access flags required)
toolbox create whisper \
  --image whisper-gfx1151:latest \
  -- --device /dev/dri --device /dev/kfd \
     --group-add video --group-add render \
     --security-opt seccomp=unconfined

# Enter it
toolbox enter whisper

# Start the server (inside the toolbox)
start-whisper --model turbo
```

Models are downloaded to `~/.cache/whisper/` on first use.

### Python Image — Docker/Podman

```bash
podman run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v ~/.cache/whisper:/root/.cache/whisper \
  -p 8000:8000 \
  whisper-gfx1151:latest \
  start-whisper --model turbo --port 8000
```

Mounting `~/.cache/whisper` avoids re-downloading models on every `podman run`.

#### All start-whisper flags

```bash
HF_TOKEN=hf_xxx start-whisper --model large-v3 --port 8000 --language en
start-whisper --model large-v3 --jobs-dir /mnt/data/whisper-jobs
start-whisper --model small    --device cpu   # force CPU
```

| Flag | Default | Options |
|---|---|---|
| `--model` | `large-v3` | `tiny` `base` `small` `medium` `large-v2` `large-v3` `turbo` |
| `--port` | `8000` | any free TCP port |
| `--host` | `0.0.0.0` | `0.0.0.0` (all interfaces) or `127.0.0.1` (localhost only) |
| `--device` | auto | `cuda` (ROCm GPU) or `cpu` |
| `--language` | auto-detect | ISO 639-1 code, e.g. `en` `fr` `de` `ja` |
| `--jobs-dir` | `/tmp/whisper-jobs` | path to store uploaded audio and result JSON files |
| `--batch-size` | 16 GPU / 4 CPU | WhisperX batch size; reduce if you hit GPU OOM |

### whisper.cpp Image

Download a GGML model file first:

```bash
mkdir -p ~/whisper-models
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin \
  -o ~/whisper-models/ggml-large-v3.bin
```

Run the server (web UI on port 8080):

```bash
podman run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v ~/whisper-models:/models \
  -p 8080:8080 \
  whisper-cpp-gfx1151:latest \
  --model /models/ggml-large-v3.bin \
  --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` in a browser to use the built-in upload UI.

Command-line transcription (no server):

```bash
podman run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -v ~/whisper-models:/models \
  -v /path/to/audio:/audio \
  --entrypoint whisper-cli \
  whisper-cpp-gfx1151:latest \
  --model /models/ggml-large-v3.bin \
  --file /audio/recording.wav
```

---

## API Reference

### `POST /v1/audio/transcriptions` — synchronous (OpenAI-compatible)

Blocks until transcription is complete. Suitable for short files. Drop-in replacement for the OpenAI Whisper API.

```bash
# Basic — returns {"text": "..."}
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3"

# Plain text response
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "language=en" \
  -F "response_format=text"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file. Accepts mp3, wav, m4a, flac, ogg, webm, and any format ffmpeg can decode |
| `model` | string | server default | Model name (informational; server uses its loaded model) |
| `language` | string | auto | ISO 639-1 language code. Auto-detect if omitted |
| `response_format` | string | `json` | `json` → `{"text": "..."}` · `text` → plain string · `verbose_json` → full WhisperX output with word timestamps and speaker labels |
| `temperature` | float | `0.0` | Sampling temperature |
| `prompt` | string | — | Optional text prompt to guide transcription |

**Response (`json`):** `{"text": "Hello, this is the transcribed text."}`

---

### `POST /v1/audio/transcriptions/async` — async job queue (for large files)

Streams the upload to disk and returns a job ID immediately. Use this for files where the transcription time would exceed a reasonable HTTP timeout (typically anything over a few minutes).

```bash
# 1. Upload and get job ID
JOB=$(curl -s -X POST http://localhost:8000/v1/audio/transcriptions/async \
  -F "file=@audiobook.mp3" \
  -F "language=en" \
  | jq -r .job_id)

echo "Job ID: $JOB"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file (any format ffmpeg can decode) |
| `model` | string | server default | Model name (informational) |
| `language` | string | auto | ISO 639-1 language code |
| `speaker_name` | string | — | Human-readable label for the primary speaker (stored in job metadata) |

**Response `202`:** `{"job_id": "550e8400-e29b-41d4-a716-446655440000"}`

---

### `GET /jobs/{job_id}` — poll job status

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

`status` values: `queued` → `processing` → `done` or `failed`

---

### `GET /jobs/{job_id}/result` — download completed transcript

```bash
# Download when done
curl -O http://localhost:8000/jobs/$JOB/result
# Saves as <original-filename>.json
```

| Condition | Status | Body |
|---|---|---|
| Job ID not found | `404` | `{"error": "job not found"}` |
| Job still running | `202` | `{"status": "processing"}` |
| Job failed | `422` | `{"error": "...error message..."}` |
| Job done | `200` | JSON file attachment |

The downloaded JSON matches the WhisperX output format — `segments` array with word-level timestamps and speaker labels:

```json
{
  "segments": [
    {
      "start": 0.031, "end": 1.733, "text": " This is Audible.",
      "words": [
        {"word": "This", "start": 0.031, "end": 0.892, "score": 0.93, "speaker": "SPEAKER_00"}
      ],
      "speaker": "SPEAKER_00"
    }
  ]
}
```

Speaker labels require `HF_TOKEN` to be set at server startup. Without it, all `speaker` fields are `"UNKNOWN"`.

#### Full upload → poll → download workflow

```bash
# Upload
JOB=$(curl -s -X POST http://localhost:8000/v1/audio/transcriptions/async \
  -F "file=@audiobook.mp3" | jq -r .job_id)

# Poll until done
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB | jq -r .status)
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ] && break
  sleep 30
done

# Download result
curl -O http://localhost:8000/jobs/$JOB/result
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "large-v3", "device": "cuda"}
```

---

## Changing Models

### Python image

Models are downloaded automatically from HuggingFace on first use and cached in `~/.cache/whisper/`.

Model sizes and approximate VRAM usage on Strix Halo:

| Model | Parameters | VRAM (fp16) | Notes |
|---|---|---|---|
| `tiny` | 39M | ~150 MB | Fastest, least accurate |
| `base` | 74M | ~290 MB | Good for English |
| `small` | 244M | ~960 MB | Good balance |
| `medium` | 769M | ~3 GB | Strong multilingual |
| `large-v2` | 1.5B | ~6 GB | High accuracy |
| `large-v3` | 1.5B | ~6 GB | Best accuracy, improved over v2 |
| `turbo` | 809M | ~3 GB | large-v2 encoder + small decoder — near large quality at small speed |

On Strix Halo with its unified memory, all models fit comfortably in GPU memory.

### whisper.cpp image

Download the corresponding GGML file from `https://huggingface.co/ggerganov/whisper.cpp`. File naming convention: `ggml-{model-name}.bin` (e.g. `ggml-large-v3.bin`, `ggml-turbo.bin`).

---

## Host Configuration

These kernel parameters are required to expose the full iGPU unified memory. Add to `GRUB_CMDLINE_LINUX` in `/etc/default/grub` and run `sudo grub2-mkconfig -o /boot/grub2/grub.cfg`:

```
iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

| Parameter | Purpose |
|---|---|
| `iommu=pt` | IOMMU pass-through mode — reduces overhead for iGPU unified memory access |
| `amdgpu.gttsize=126976` | Caps GPU unified memory to 124 GiB (126976 MiB ÷ 1024) |
| `ttm.pages_limit=32505856` | Caps pinned memory to 124 GiB (32505856 × 4 KiB = 126976 MiB) |

Verify GPU is visible after boot:
```bash
rocm-smi
```

When you run podman build, no files are placed on your host filesystem. The entire build happens inside a container image stored in Podman's local image
  store, typically at:

  ~/.local/share/containers/storage/

  You interact with it by name/tag, not by path. The files inside the image live at these paths within the image:

  ┌──────────────────────────────┬────────────────────────────────────────────────────────┐
  │        Path in image         │                       What it is                       │
  ├──────────────────────────────┼────────────────────────────────────────────────────────┤
  │ /opt/venv/                   │ Python virtualenv with PyTorch, whisper, FastAPI, etc. │
  ├──────────────────────────────┼────────────────────────────────────────────────────────┤
  │ /opt/rocm/                   │ TheRock ROCm SDK                                       │
  ├──────────────────────────────┼────────────────────────────────────────────────────────┤
  │ /opt/start-whisper           │ The start_whisper.py server script                     │
  ├──────────────────────────────┼────────────────────────────────────────────────────────┤
  │ /usr/local/bin/start-whisper │ Symlink to the above                                   │
  ├──────────────────────────────┼────────────────────────────────────────────────────────┤
  │ /etc/profile.d/              │ Login scripts (ROCm env, banner, venv fix)             │
  └──────────────────────────────┴────────────────────────────────────────────────────────┘

  You only see those files when you run the container:

  # Enter a shell and explore
  podman run -it --rm whisper-gfx1151:latest /bin/bash

  # Or via toolbox
  toolbox enter whisper

  The one exception is model weights — those are downloaded at runtime (not build time) and land on your host at:

  ~/.cache/whisper/

  because Toolbx mounts your home directory, and the Docker/Podman examples in the README explicitly mount -v ~/.cache/whisper:/root/.cache/whisper for the
  same reason. That way model files survive container rebuilds.
