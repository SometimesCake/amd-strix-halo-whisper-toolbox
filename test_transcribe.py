#!/usr/bin/env python3
"""
Send an mp3 to the Whisper transcription or diarization service and print the result.

Both services expose identical endpoints — the only difference is that the diarization
service has HF_TOKEN set, so speaker labels (SPEAKER_00, SPEAKER_01, …) are populated.
The transcription service returns "UNKNOWN" for all speaker fields.

  --async works on both services. Use it for large files where the transcription would
  take longer than a reasonable HTTP timeout. The file is uploaded, a job ID is returned
  immediately, and this script polls until the job completes.

  Without --async the sync endpoint is used: the HTTP connection stays open until the
  transcription finishes and the result is returned directly. Fine for short clips.

Ports (set by start-server.sh):
  transcription  →  http://localhost:8001
  diarization    →  http://localhost:8000

Usage:
  python test_transcribe.py recording.mp3
  python test_transcribe.py recording.mp3 --service diarization
  python test_transcribe.py recording.mp3 --async
  python test_transcribe.py recording.mp3 --service diarization --async

Requires: pip install requests
"""

import argparse
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("requests is not installed — run: pip install requests")

SERVICES = {
    "transcription": "http://localhost:8001",
    "diarization":   "http://localhost:8000",
}

# How long to wait between status polls when using --async
POLL_INTERVAL = 10  # seconds


def check_health(base_url: str, service: str) -> None:
    """Hit /health before sending audio so failures are reported clearly."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"Service:  {service} ({base_url})")
        print(f"Model:    {info.get('model')}")
        print(f"Device:   {info.get('device')}")
        print()
    except requests.exceptions.ConnectionError:
        sys.exit(f"Error: cannot reach {service} service at {base_url} — is it running?")


def transcribe_sync(base_url: str, mp3_path: str) -> dict:
    """
    POST to /v1/audio/transcriptions (OpenAI-compatible sync endpoint).
    Blocks until transcription completes. Returns verbose_json so segments
    and word timestamps are included alongside the plain text.
    Timeout is generous (10 min) to handle slow GPU warm-up on first request.
    """
    print("Sending (sync) ...")
    with open(mp3_path, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (mp3_path, f, "audio/mpeg")},
            data={"response_format": "verbose_json"},
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


def transcribe_async(base_url: str, mp3_path: str) -> dict:
    """
    POST to /v1/audio/transcriptions/async — upload returns immediately with a job ID.
    Polls /jobs/{job_id} every POLL_INTERVAL seconds until status is 'done' or 'failed'.
    Downloads and returns the result JSON from /jobs/{job_id}/result.

    The async path uses WhisperX (word-level timestamps + speaker diarization) rather
    than openai-whisper, so the result schema is richer than the sync endpoint.
    """
    print("Uploading (async) ...")
    with open(mp3_path, "rb") as f:
        r = requests.post(
            f"{base_url}/v1/audio/transcriptions/async",
            files={"file": (mp3_path, f, "audio/mpeg")},
            timeout=120,
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"Job ID:   {job_id}")

    while True:
        time.sleep(POLL_INTERVAL)
        r = requests.get(f"{base_url}/jobs/{job_id}", timeout=60)
        r.raise_for_status()
        status = r.json()["status"]
        print(f"  [{time.strftime('%H:%M:%S')}] {status}")
        if status == "done":
            break
        if status == "failed":
            sys.exit(f"Job failed: {r.json().get('error')}")

    r = requests.get(f"{base_url}/jobs/{job_id}/result", timeout=30)
    r.raise_for_status()
    return r.json()


def print_result(result: dict, by_speaker: bool = False) -> None:
    """
    Pretty-print the transcript.

    by_speaker=False (default): one line per segment with timestamps.
    by_speaker=True: consecutive segments from the same speaker are merged
                     into a single block showing the speaker's full run of text.

    When there are no segments (plain {"text": "..."} response), fall back to printing
    the raw text.
    """
    segments = result.get("segments")
    if not segments:
        print("─" * 60)
        print(result.get("text", ""))
        return

    # Only show speaker headers if at least one segment has a real label —
    # openai-whisper results have no speaker field at all, so we'd otherwise
    # print [UNKNOWN] throughout a plain transcription result.
    has_speakers = any(seg.get("speaker", "UNKNOWN") != "UNKNOWN" for seg in segments)

    print("─" * 60)

    if by_speaker and has_speakers:
        # Merge consecutive segments from the same speaker into blocks
        blocks = []
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text    = seg.get("text", "").strip()
            if blocks and blocks[-1]["speaker"] == speaker:
                blocks[-1]["text"] += " " + text
                blocks[-1]["end"]   = seg.get("end", 0)
            else:
                blocks.append({
                    "speaker": speaker,
                    "start":   seg.get("start", 0),
                    "end":     seg.get("end", 0),
                    "text":    text,
                })
        for block in blocks:
            print(f"\n[{block['speaker']}]  {block['start']:.1f}s – {block['end']:.1f}s")
            print(f"  {block['text']}")
    else:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            if has_speakers and speaker != current_speaker:
                current_speaker = speaker
                print(f"\n[{speaker}]")
            start = seg.get("start", 0)
            end   = seg.get("end", 0)
            text  = seg.get("text", "").strip()
            print(f"  {start:6.1f}s – {end:6.1f}s  {text}")

    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Test the Whisper transcription or diarization service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Both services support both endpoints (sync and async).\n"
            "Use --service diarization to get speaker labels in the output.\n"
            "Use --async for large files to avoid HTTP timeout issues."
        ),
    )
    p.add_argument("mp3", help="Path to the mp3 file to transcribe")
    p.add_argument(
        "--service",
        choices=["transcription", "diarization"],
        default="transcription",
        help="Service to target: 'transcription' (port 8001, no speaker labels) or "
             "'diarization' (port 8000, speaker labels). Default: transcription",
    )
    p.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use the async job queue endpoint. Uploads immediately, then polls for "
             "completion. Recommended for files over a few minutes long.",
    )
    p.add_argument(
        "--by-speaker",
        dest="by_speaker",
        action="store_true",
        help="Group output by speaker: merge consecutive segments from the same speaker "
             "into a single block instead of printing one line per segment.",
    )
    args = p.parse_args()

    base_url = SERVICES[args.service]
    check_health(base_url, args.service)

    t0 = time.monotonic()
    result = transcribe_async(base_url, args.mp3) if args.use_async else transcribe_sync(base_url, args.mp3)
    elapsed = time.monotonic() - t0

    print_result(result, by_speaker=args.by_speaker)
    print(f"Elapsed:  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
