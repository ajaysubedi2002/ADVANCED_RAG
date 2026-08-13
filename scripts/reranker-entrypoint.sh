#!/bin/sh
set -e

MODEL_NAME="${RERANKER_MODEL:-BAAI/bge-reranker-base}"
MODEL_DIR="${RERANKER_MODEL_DIR:-/models/bge-reranker-base}"

echo "======================================"
echo "BGE Reranker Initialization"
echo "======================================"

echo "Model: $MODEL_NAME"
echo "Directory: $MODEL_DIR"

if [ -f "$MODEL_DIR/config.json" ]; then

    echo "Reranker already exists."
    echo "Skipping download."

else

    echo "Reranker not found."
    echo "Downloading $MODEL_NAME..."

    python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MODEL_NAME",
    local_dir="$MODEL_DIR"
)

print("Reranker download completed.")
PY

fi

echo "Reranker is ready."

exec "$@"