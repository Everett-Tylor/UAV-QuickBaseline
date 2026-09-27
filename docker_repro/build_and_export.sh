#!/bin/sh
set -eu
cd "$(dirname "$0")"
docker build -t uav-round2:74.22 .
docker run --rm --gpus all uav-round2:74.22 check
docker save --output "${1:-uav-round2-image.tar}" uav-round2:74.22
