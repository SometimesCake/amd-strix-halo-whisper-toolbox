#!/bin/bash
set -e

dnf -y install --setopt=install_weak_deps=False --nodocs \
  python3.12 python3.12-devel git libatomic bash ca-certificates curl \
  gcc gcc-c++ binutils make \
  cmake ninja-build aria2c tar xz vim nano \
  zlib-devel openssl-devel \
  ffmpeg-free ffmpeg-free-devel \
  gperftools-libs \
  && dnf clean all && rm -rf /var/cache/dnf/*
