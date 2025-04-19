FROM python:3.11-slim

# Create separate directories for application code and data
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for better cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p /app/models
RUN mkdir -p /app/documents
RUN mkdir -p /app/chroma_faq_db

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy production environment file to .env
RUN cp .env.production .env

# Expose ports for all services
EXPOSE 8003 8000 8002

# Command to run all services directly with shell
CMD ["/bin/bash", "-c", "echo 'Starting services...' && python /app/extract_data.py & python /app/recommendation_api.py & python /app/chatbot.py & tail -f /dev/null"] 