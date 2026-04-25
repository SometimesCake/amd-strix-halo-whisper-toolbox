#!/usr/bin/env bash
# Banner for the Whisper toolbox

case $- in *i*) ;; *) return 0 ;; esac

oem_info() {
  local v="" m="" d lv lm
  for d in /sys/class/dmi/id /sys/devices/virtual/dmi/id; do
    [[ -r "$d/sys_vendor" ]] && v=$(<"$d/sys_vendor")
    [[ -r "$d/product_name" ]] && m=$(<"$d/product_name")
    [[ -n "$v" || -n "$m" ]] && break
  done
  if [[ -z "$v" && -z "$m" && -r /proc/device-tree/model ]]; then
    tr -d '\0' </proc/device-tree/model; return
  fi
  lv=$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')
  lm=$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')
  if [[ -n "$m" && "$lm" == "$lv "* ]]; then printf '%s\n' "$m"
  else printf '%s %s\n' "${v:-Unknown}" "${m:-Unknown}"; fi
}

gpu_name() {
  local name=""
  if command -v rocm-smi >/dev/null 2>&1; then
    name=$(rocm-smi --showproductname --csv 2>/dev/null | tail -n1 | cut -d, -f2)
    [[ -z "$name" ]] && name=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -E 'Product Name|Card series' | sed 's/.*: //')
  fi
  if [[ -z "$name" ]] && command -v rocminfo >/dev/null 2>&1; then
    name=$(rocminfo 2>/dev/null | awk -F': ' '/^[[:space:]]*Name:/{print $2; exit}')
  fi
  name=$(printf '%s' "$name" | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//' -e 's/[[:space:]]\{2,\}/ /g')
  printf '%s\n' "${name:-Unknown AMD GPU}"
}

rocm_version() {
  python - <<'PY' 2>/dev/null || true
try:
    import torch
    v = getattr(getattr(torch, "version", None), "hip", "") or ""
    if v: print(v)
    else: raise Exception()
except Exception:
    try:
        import importlib.metadata as im
        try: print(im.version("_rocm_sdk_core"))
        except Exception: print(im.version("rocm"))
    except Exception: print("")
PY
}

MACHINE="$(oem_info)"
GPU="$(gpu_name)"
ROCM_VER="$(rocm_version)"

echo
cat <<'ASCII'
 __        ___     _                     _____           _ _
 \ \      / / |__ (_)___ _ __   ___ _ _|_   _|__   ___ | | |__   _____  __
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|| |/ _ \ / _ \| | '_ \ / _ \ \/ /
   \ V  V / | | | | \__ \ |_) |  __/ |   | | (_) | (_) | | |_) | (_) >  <
    \_/\_/  |_| |_|_|___/ .__/ \___|_|   |_|\___/ \___/|_|_.__/ \___/_/\_\
                         |_|
ASCII
echo
printf 'AMD STRIX HALO — Whisper Toolbox (gfx1151, ROCm via TheRock)\n'
[[ -n "$ROCM_VER" ]] && printf 'ROCm nightly: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
echo
printf 'Included:\n'
printf '  - %-20s → %s\n' "start-whisper" "FastAPI OpenAI-compatible transcription + diarization server"
printf '  - %-20s → %s\n' "Sync endpoint" "POST http://localhost:8000/v1/audio/transcriptions"
printf '  - %-20s → %s\n' "Async endpoint" "POST http://localhost:8000/v1/audio/transcriptions/async"
echo
printf 'Quick start (inside toolbox):\n'
printf '  start-whisper --model large-v3\n'
printf '  HF_TOKEN=hf_xxx start-whisper --model large-v3   # enable speaker diarization\n'
echo
printf 'Background servers (run from host, outside toolbox):\n'
printf '  ./start-server.sh                     # both services: diarization :8000, transcription :8001\n'
printf '  ./start-server.sh --mode diarization  # diarization only  (port 8000, speaker labels)\n'
printf '  ./start-server.sh --mode transcription # transcription only (port 8001, no HF_TOKEN needed)\n\n'

unset PROMPT_COMMAND
PS1='\u@\h:\w\$ '
