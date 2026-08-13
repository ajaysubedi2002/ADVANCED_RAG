#!/bin/sh
set -e

export OLLAMA_HOST="0.0.0.0:11434"

echo "Starting Ollama server..."

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to be ready..."

until ollama list >/dev/null 2>&1; do
    sleep 1
done

echo "Ollama is ready."

for MODEL in "${CHAT_MODEL:-}" "${EMBED_MODEL:-}"; do

    if [ -z "$MODEL" ]; then
        continue
    fi

    echo "Ensuring model exists: $MODEL"

    if ! ollama list | grep -q "^${MODEL}"; then
        echo "Pulling $MODEL..."
        ollama pull "$MODEL"
    else
        echo "$MODEL already exists."
    fi

done

echo "All Ollama models are ready."

wait "$OLLAMA_PID"