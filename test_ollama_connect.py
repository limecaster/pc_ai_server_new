import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

print(f"Testing connectivity to Ollama at: {OLLAMA_URL}")
try:
    resp = requests.get(OLLAMA_URL)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text}")
except Exception as e:
    print(f"Connection error: {e}")
