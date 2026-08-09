#!/bin/sh
set -e

export OLLAMA_HOST="0.0.0.0:11434"

echo "Starting Ollama server..."
ollama serve &
# Save the process ID of the server
OLLAMA_PID=$!

echo "Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Ollama is ready."

if [ -n "${CHAT_MODEL:-}" ]; then
  echo "Ensuring chat model exists: $CHAT_MODEL"
  ollama pull "$CHAT_MODEL" || echo "Failed to pull $CHAT_MODEL"
fi

echo "Done. Ollama running."

# Clean exit handler: forwards Docker stop signals to the Ollama process
trap 'kill -TERM $OLLAMA_PID; wait $OLLAMA_PID' TERM INT

# Wait specifically on the Ollama process, allowing signals to break through
wait $OLLAMA_PID