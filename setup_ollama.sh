#!/bin/bash

# Start Ollama container if not already running
container_id=$(docker ps -q -f name=ollama)
if [ -z "$container_id" ]; then
  echo "Starting Ollama container..."
  docker-compose up -d
else
  echo "Ollama container is already running."
fi

# Wait for the Ollama server to be available
echo "Waiting for the Ollama server to start..."
until curl -s http://localhost:11434/ > /dev/null; do
  echo -n "."
  sleep 1
done
echo "Ollama server is running."

# Pull the model
echo "Pulling model: llama3"
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3
echo "Model llama3 pulled successfully."

echo "Setup completed successfully."
