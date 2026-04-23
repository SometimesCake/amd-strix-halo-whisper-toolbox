#!/bin/bash
set -euo pipefail

# Configuration with defaults matching Dockerfile ARGs
ROCM_MAJOR_VER="${ROCM_MAJOR_VER:-7}"
GFX="${GFX:-gfx1151}"

echo "=== Installing ROCm SDK ($GFX / $ROCM_MAJOR_VER) ==="

# 2. Install "TheRock" ROCm SDK (Tarball Method)
# We work in /tmp as per Dockerfile WORKDIR
cd /tmp

BASE="https://therock-nightly-tarball.s3.amazonaws.com"
PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"

# Fetch the Key
KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
  | tr '<' '\n' \
  | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\..*\.tar\.gz" \
  | sort -V | tail -n1)"

if [ -z "$KEY" ]; then
    echo "Error: Could not find tarball key for $PREFIX"
    exit 1
fi

echo "Downloading Latest Tarball: ${KEY}"
aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz

mkdir -p /opt/rocm
tar xzf therock.tar.gz -C /opt/rocm --strip-components=1
rm therock.tar.gz

# 3. Configure Global ROCm Environment
# We add LD_PRELOAD for tcmalloc here to fix the shutdown crash
export ROCM_PATH=/opt/rocm
BITCODE_PATH=$(find /opt/rocm -type d -name bitcode -print -quit)

echo "Generating /etc/profile.d/rocm-sdk.sh..."
printf '%s\n' \
  "export ROCM_PATH=/opt/rocm" \
  "export HIP_PLATFORM=amd" \
  "export HIP_PATH=/opt/rocm" \
  "export HIP_CLANG_PATH=/opt/rocm/llvm/bin" \
  "export HIP_DEVICE_LIB_PATH=$BITCODE_PATH" \
  "export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:\$PATH" \
  "export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:\$LD_LIBRARY_PATH" \
  "export ROCBLAS_USE_HIPBLASLT=1" \
  "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1" \
  "export VLLM_TARGET_DEVICE=rocm" \
  "export HIP_FORCE_DEV_KERNARG=1" \
  "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1" \
  "export LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4:/opt/rocm/lib/librocm_smi64.so.1.0" \
  > /etc/profile.d/rocm-sdk.sh

chmod 0644 /etc/profile.d/rocm-sdk.sh
echo "=== ROCm SDK Installation Complete ==="
