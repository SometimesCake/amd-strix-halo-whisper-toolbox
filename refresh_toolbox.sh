#!/usr/bin/env bash

set -e

TOOLBOX_NAME="whisper"
IMAGE="docker.io/kyuz0/whisper-therock-gfx1151:latest"

OPTIONS="--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"

for cmd in podman toolbox; do
  command -v "$cmd" > /dev/null || { echo "Error: '$cmd' is not installed." >&2; exit 1; }
done

echo "Refreshing $TOOLBOX_NAME (image: $IMAGE)"

if toolbox list 2>/dev/null | grep -q "$TOOLBOX_NAME"; then
  echo "Removing existing toolbox: $TOOLBOX_NAME"
  toolbox rm -f "$TOOLBOX_NAME"
fi

echo "Pulling latest image: $IMAGE"
podman pull "$IMAGE"

new_id="$(podman image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
new_digest="$(podman image inspect --format '{{.Digest}}' "$IMAGE" 2>/dev/null || true)"

echo "Recreating toolbox: $TOOLBOX_NAME"
toolbox create "$TOOLBOX_NAME" --image "$IMAGE" -- $OPTIONS

repo="${IMAGE%:*}"

while read -r id ref dig; do
  if [[ "$id" != "$new_id" ]]; then
      podman image rm -f "$id" >/dev/null 2>&1 || true
  fi
done < <(podman images --digests --format '{{.ID}} {{.Repository}}:{{.Tag}} {{.Digest}}' \
         | awk -v ref="$IMAGE" -v ndig="$new_digest" '$2==ref && $3!=ndig')

while read -r id; do
  podman image rm -f "$id" >/dev/null 2>&1 || true
done < <(podman images --format '{{.ID}} {{.Repository}}:{{.Tag}}' \
         | awk -v r="$repo" '$2==r":<none>" {print $1}')

echo "$TOOLBOX_NAME refreshed"
