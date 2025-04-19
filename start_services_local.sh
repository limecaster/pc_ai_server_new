#!/bin/bash
set -e

echo "Starting services..."

# Start extract_data service 
echo "Starting extract_data service on port 8000..."
python extract_data.py &

# Start recommendation service
echo "Starting recommendation service on port 3003..."
python recommendation_api.py &

# Start the chatbot service
echo "Starting chatbot service on port 8002..."
python chatbot.py &

wait -n

exit $?