FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
COPY scripts/install_deps.sh /tmp/install_deps.sh
RUN sh /tmp/install_deps.sh

# 2. Install "TheRock" ROCm SDK (Tarball Method)
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
  export ROCM_MAJOR_VER=$ROCM_MAJOR_VER && \
  export GFX=$GFX && \
  /tmp/install_rocm_sdk.sh

# 3. Python Venv Setup
RUN /usr/bin/python3.12 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 4. Install Whisper + API server dependencies
RUN python -m pip install \
  openai-whisper \
  transformers \
  accelerate \
  fastapi \
  "uvicorn[standard]" \
  python-multipart \
  soundfile \
  numpy \
  aiofiles

# 4b. WhisperX (faster-whisper backend + word alignment + speaker diarization)
RUN python -m pip install \
  whisperx \
  "pyannote.audio>=3.1" \
  "librosa>=0.10"

# 5. Install PyTorch (TheRock Nightly) — installed LAST so pyannote/whisperx deps
#    cannot overwrite it with a PyPI CUDA build during their dependency resolution.
RUN python -m pip install \
  --upgrade \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ \
  --pre torch torchaudio torchvision && \
  # Fix caching/JSON serialization bug in recent PyTorch nightlies
  sed -i 's/json.dumps(config_dict, sort_keys=True)/json.dumps(config_dict, sort_keys=True, default=str)/g' /opt/venv/lib64/python3.12/site-packages/torch/_dynamo/utils.py || true

# 6. Cleanup
WORKDIR /opt
RUN chmod -R a+rwX /opt && \
  find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
  rm -rf /root/.cache/pip || true && \
  dnf clean all && rm -rf /var/cache/dnf/*

COPY scripts/01-rocm-env.sh /etc/profile.d/01-rocm-env.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
COPY scripts/start_whisper.py /opt/start-whisper
RUN chmod +x /opt/start-whisper && \
  ln -s /opt/start-whisper /usr/local/bin/start-whisper && \
  chmod 0644 /etc/profile.d/*.sh

RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh
RUN chmod -R a+rwX /opt

CMD ["/bin/bash"]
