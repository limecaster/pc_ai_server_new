import os
from langchain_ollama import ChatOllama

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")

print(f"Testing ChatOllama connection to {OLLAMA_URL} with model '{LLM_MODEL}'...")

try:
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, tools=True)
    messages = [
        {"role": "system", "content": "You are a test system."},
        {"role": "user", "content": "ping"}
    ]
    print("Invoking LLM...")
    response = llm.invoke(messages)
    print("LLM responded:")
    print(response)
except Exception as e:
    print("[ERROR] Could not connect to Ollama LLM API:")
    print(str(e))
