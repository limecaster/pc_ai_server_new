from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from langchain_ollama import ChatOllama
from neo4j_retriever import Neo4jRetriever
from intent_classifier import IntentClassifier
from fastapi.responses import StreamingResponse
from load_documents import retrieve_faq_answer
import os
import json
from dotenv import load_dotenv
import pathlib

app = FastAPI()

# Load the appropriate environment file based on NODE_ENV
node_env = os.environ.get('NODE_ENV', 'development')
env_file = f".env.{node_env}" if node_env != 'development' else ".env.local"
env_path = pathlib.Path(__file__).parent / env_file

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

#PC_BACKEND_URL = os.getenv('PC_BACKEND_URL', 'http://localhost:3001')
PC_BACKEND_URL = "http://localhost:3001"

# Initialize Neo4j and LLM
neo4j_uri = os.getenv('NEO4J_URI', 'bolt://pc_neo4j:7687')
neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD', '12345678')

neo4j = Neo4jRetriever(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
llm = ChatOllama(model=LLM_MODEL)


SYSTEM_PROMPT = """You are a customer support chatbot for a PC parts eCommerce website named "Cửa hàng B" serving Vietnamese users, you don't need to answer in English translation for reference.
Key Guidelines: Respond concisely, clearly, and in Vietnamese to ensure a smooth experience for Vietnamese users. 
Do not fabricate information or provide uncertain answers. If the data is unavailable, inform the user. 
Maintain a friendly, professional, and helpful tone.

"""

class ChatRequest(BaseModel):
    message: str
    history: List[str] = []

@app.post("/client-chat")
async def chatbot_endpoint(request: ChatRequest):
    try:
        print(request)
        user_message = request.message
        history = request.history

        # Classify the user message
        intent_classifier = IntentClassifier()
        intent = intent_classifier.classify(user_message)
        print(f"Intent: {intent}")
        # If the intent is about building a PC, call the NestJS at localhost:3000/build/auto-build to get the response
        if intent == "greeting":
            response = {
                "type": "text",
                "data": """Xin chào! Tôi là trợ lý ảo của Cửa hàng B.
                Tôi được training để trả lời câu hỏi về việc đề xuất cấu hình PC, và trả lời một số câu hỏi chung.
                Tôi có thể hỗ trợ gì cho bạn?"""
            }
        elif intent == "auto_build":
            try:
                nestjs_response = requests.post(f"{PC_BACKEND_URL}/build/single-auto-build", json={"userInput": user_message}).json()
                response = {
                    "type": "pc_config",
                    "data": nestjs_response
                }
            except requests.exceptions.ConnectionError:
                response = "The auto-build service is currently unavailable. Please try again later."
        
        # If the intent is about compatibility, call the NestJS compatibility endpoint
        elif intent == "compatibility":
            try:
                # Extract product names from user message
                extraction_prompt = f"Trích xuất tên hai sản phẩm từ câu: {user_message}, trả về JSON với các key 'product1' và 'product2'."
                extraction_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": extraction_prompt}
                ]
                extraction_response = llm.invoke(extraction_messages)
                entities = json.loads(extraction_response)
                product1 = entities.get("product1")
                product2 = entities.get("product2")
                # Call NestJS compatibility endpoint
                nestjs_response = requests.get(
                    f"{PC_BACKEND_URL}/build/compatibility",
                    params={"product1": product1, "product2": product2}
                ).json()
                response = {
                    "type": "compatibility",
                    "data": nestjs_response
                }
            except Exception:
                response = {
                    "type": "error",
                    "data": "Xin lỗi bạn, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
                }
        
        # If the intent is about FAQs, call the LLM
        elif intent == "faq":
            try:
                # Retrieve context from FAQ database
                faq_context = retrieve_faq_answer(user_message)
                # Generate answer using LLM with context
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                # include conversation history
                for msg in history:
                    messages.append({"role": "user", "content": msg})
                # add FAQ context and current message
                messages.append({"role": "system", "content": f"Dữ liệu tham khảo: {faq_context}"})
                messages.append({"role": "user", "content": user_message})
                answer = llm.invoke(messages)
                response = {
                    "type": "faq",
                    "data": answer.content
                }
            except Exception:
                response = {
                    "type": "error",
                    "data": "Xin lỗi bạn, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
                }
        
        # If the intent is unknown, return a default response
        else:
            response = {
                "type": "error",
                "data": "Xin lỗi, tôi không được huấn luyện để trả lời câu hỏi này."
            }    
        return {"response": response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")

def stream_faq_response(llm, user_message: str):
    # Retrieve context from FAQ database
    faq_context = retrieve_faq_answer(user_message)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Dữ liệu tham khảo: {faq_context}"},
        {"role": "user", "content": user_message}
    ]
    for chunk in llm.invoke_stream(messages):
        yield chunk + " "

@app.get("/faq-stream")
def faq_stream(user_message: str):
    return StreamingResponse(stream_faq_response(llm, user_message), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)