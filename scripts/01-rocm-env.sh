#!/usr/bin/env bash
# ROCm environment for Strix Halo / RDNA3.5 — Whisper toolbox
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Suppress all MIOpen log output (includes missing tuning DB warnings)
export MIOPEN_LOG_LEVEL=0

# Suppress ROCm runtime warning: "xnack 'Off' was requested for a processor that does not support it"
export HSA_XNACK=0
