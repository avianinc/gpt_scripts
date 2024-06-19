#!/bin/bash

# Load configuration
CONFIG_FILE='config.json'
MODEL=$(jq -r '.model' $CONFIG_FILE)
SERVER_URL=$(jq -r '.server_url' $CONFIG_FILE)
DOWNLOAD_MODEL=$(jq -r '.download_model' $CONFIG_FILE)

# Start Ollama container if not already running
container_id=$(docker ps -q -f name=ollama)
if [ -z "$container_id" ]; then
  echo "Starting Ollama container..."
  docker-compose up -d --build
else
  docker-compose restart -d --build
  echo "Ollama container is already running... Restarting"
fi

# Wait for the Ollama server to be available
echo "Waiting for the Ollama server to start..."
until curl -s $SERVER_URL > /dev/null; do
  echo -n "."
  sleep 1
done
echo "Ollama server is running."

# Pull the model
echo "Pulling model: $DOWNLOAD_MODEL"
docker exec -it $(docker ps -q -f name=ollama) ollama pull $DOWNLOAD_MODEL
# docker exec -it $(docker ps -q -f name=ollama) ollama run $DOWNLOAD_MODEL
echo "Model $DOWNLOAD_MODEL pulled successfully."

echo "Setup completed successfully."
