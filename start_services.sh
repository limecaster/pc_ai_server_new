#!/bin/bash
set -e

echo "Starting services..."

# Start extract_data service 
echo "Starting extract_data service on port 8000..."
python /app/extract_data.py &

# Start recommendation service
echo "Starting recommendation service on port 8003..."
python /app/recommendation_api.py &

# Start the chatbot service
echo "Starting chatbot service on port 8002..."
python /app/chatbot.py &

# Create a file to serve as a health check endpoint
echo "Creating health check file..."
echo "OK" > /app/health.txt

# Keep the container running
echo "All services started. Container is running..."
tail -f /dev/null
