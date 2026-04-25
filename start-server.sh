#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="localhost/whisper-therock-gfx1151:latest"

# mode -> (container name, host port, startup script)
declare -A CONTAINER=( [diarization]="whisper-diarization"   [transcription]="whisper-transcription"  )
declare -A PORT=(       [diarization]="8000"                  [transcription]="8001"                   )
declare -A SCRIPT=(     [diarization]="start_speaker_diarization_server.sh" [transcription]="start_transcription_server.sh" )

MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
  esac
done

if [[ -n "$MODE" ]] && [[ -z "${CONTAINER[$MODE]+_}" ]]; then
  echo "Error: --mode must be 'diarization' or 'transcription'" >&2
  exit 1
fi

# Determine which modes to start
MODES=( diarization transcription )
[[ -n "$MODE" ]] && MODES=( "$MODE" )

start_one() {
  local mode="$1"
  local name="${CONTAINER[$mode]}"
  local port="${PORT[$mode]}"
  local host_script="$SCRIPT_DIR/${SCRIPT[$mode]}"

  if podman container exists "$name" 2>/dev/null; then
    echo "Removing existing container: $name"
    podman rm -f "$name"
  fi

  echo "Starting $name ($mode, port $port) ..."
  podman run -d \
    --name "$name" \
    --device /dev/dri \
    --device /dev/kfd \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    -p "$port:8000" \
    -v "$host_script:/opt/start_server.sh:ro" \
    "$IMAGE" \
    bash /opt/start_server.sh
}

for m in "${MODES[@]}"; do
  start_one "$m"
done

echo ""
echo "Containers started — they will keep running after you close this terminal."
echo ""
for m in "${MODES[@]}"; do
  name="${CONTAINER[$m]}"
  port="${PORT[$m]}"
  echo "  [$m]"
  echo "    Logs:   podman logs -f $name"
  echo "    Health: curl http://localhost:$port/health"
  echo "    Stop:   podman stop $name"
done
