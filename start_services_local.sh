#!/bin/bash
set -e

echo "Starting services..."

source ./.venv/scripts/activate

# Start extract_data service 
echo "Starting extract_data service on port 8000..."
py extract_data.py &

# Start recommendation service
echo "Starting recommendation service on port 8003..."
py recommendation_api.py &

# Start the chatbot service
echo "Starting chatbot service on port 8002..."
py chatbot.py &

wait -n

exit $?