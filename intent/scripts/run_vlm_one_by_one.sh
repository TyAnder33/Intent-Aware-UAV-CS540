#!/bin/bash

set -u

cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p intent/outputs/vlm

COUNT=0
LIMIT=50

for json_file in role3_perception/outputs/json/*.json; do
    name=$(basename "$json_file")

    echo "======================================"
    echo "Processing $name"
    echo "======================================"

    python intent/scripts/demo_vlm_single_image.py "$name"

    COUNT=$((COUNT + 1))

    if [ "$COUNT" -ge "$LIMIT" ]; then
        break
    fi

    sleep 3
done

echo "Done. Successful outputs:"
ls intent/outputs/vlm | wc -l