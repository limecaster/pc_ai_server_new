from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from langchain_ollama import ChatOllama
from intent_classifier import IntentClassifier
from fastapi.responses import StreamingResponse
from load_documents import retrieve_faq_answer
import os
import json
from dotenv import load_dotenv
import pathlib
import datetime

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

LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
llm = ChatOllama(model=LLM_MODEL, tools=True)

SYSTEM_PROMPT = """You are a customer support chatbot for a PC parts eCommerce website serving Vietnamese users, you don't need to answer in English translation for reference.
Our website named "Cửa hàng B". Do not fabricate information or provide uncertain answers. If the data is unavailable, inform the user. 
Maintain a friendly, professional, and helpful tone.

Your responsibility is to answer the user's question. You must respond in Vietnamese and focus on three main topics: 
Auto-build PC requests: If the user requests a PC configuration based on their needs, 
forward this request to the auto-build service. 
Frequently Asked Questions (FAQ): Give you "Dữ liệu tham khảo": faq_context, if faq_context is not None, you will use it to answer the user's question
otherwise, inform the user that you don't know. Key Guidelines: Respond concisely, clearly, and in Vietnamese to ensure a smooth experience 
for Vietnamese users.
For compatibility questions, you will answer based on your knowledge of the products.
"""

# Chatbot event tracking
def track_chatbot_event(event_type, message, status):
    try:
        event_data = {
            "eventType": event_type,
            "entityType": "chatbot",
            "eventData": {
                "message": message,
                "status": status,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        resp = requests.post(PC_BACKEND_URL + "/events/track", json=event_data, timeout=3)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to track chatbot event: {e}")

class Product(BaseModel):
    name: str
    type: str

class Schema(BaseModel):
    product1: Product
    product2: Product

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
        status = None
        response = None
        if intent == "greeting":
            response = {
                "type": "text",
                "data": """Xin chào! Tôi là trợ lý ảo của Cửa hàng B.
                Tôi được training để trả lời câu hỏi về việc đề xuất cấu hình PC, khả năng tương thích giữa 2 linh kiện, và một số câu hỏi chung.
                Tôi có thể hỗ trợ gì cho bạn?"""
            }
            status = "success"
            track_chatbot_event("greeting", user_message, status)
        elif intent == "auto_build":
            try:
                nestjs_response = requests.post(f"{PC_BACKEND_URL}/build/single-auto-build", json={"userInput": user_message}).json()
                response = {
                    "type": "pc_config",
                    "data": nestjs_response
                }
                status = "success"
            except requests.exceptions.ConnectionError:
                response = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
                status = "error"
            finally:
                track_chatbot_event("auto_build", user_message, status)
        elif intent == "compatibility":
            try:
                extraction_prompt = f"""
Extract exactly two products and their types from the following message: \"{user_message}\"
Type must be one of: CPU, Motherboard, RAM, GraphicsCard, InternalHardDrive, CPUCooler, Case, PowerSupply.
For SSD/HDD, use type \"InternalHardDrive\".
You MUST return ONLY a valid JSON object, with no extra text, no markdown, no explanation. 
NO bullet points, NO bold, NO natural language. 
If you cannot extract both products, return: {{}}
Format:
{{
  \"product1\": {{\"name\": \"...\", \"type\": \"...\"}},
  \"product2\": {{\"name\": \"...\", \"type\": \"...\"}}
}}
"""
                
                messages = [
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": user_message}
                ]
                status = "success"
                try:
                    response = llm.invoke(messages)
                    import re
                    import json
                    content = response.content if hasattr(response, 'content') else str(response)
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        try:
                            entities = json.loads(match.group(0))
                        except Exception as e:
                            print("JSON decode error:", e)
                            entities = {}
                    else:
                        entities = {}

                    # Fallback: Try to recover from markdown output
                    if not entities or not entities.get("product1") or not entities.get("product2"):
                        # Try to extract lines like "**Product Name:** Ryzen 7"
                        name_types = re.findall(r'Product Name[:：]\s*([\w\s\-]+)[\n\r]+\s*\*\*Type[:：]\s*([\w]+)', content, re.IGNORECASE)
                        if len(name_types) >= 2:
                            entities = {
                                "product1": {"name": name_types[0][0].strip(), "type": name_types[0][1].strip()},
                                "product2": {"name": name_types[1][0].strip(), "type": name_types[1][1].strip()},
                            }
                    if not entities or not entities.get("product1") or not entities.get("product2"):
                        response = {
                            "type": "error",
                            "data": "Xin lỗi, hệ thống không thể trích xuất thông tin sản phẩm từ câu hỏi của bạn. Vui lòng thử lại."
                        }
                        status = "error"
                        return response
                finally:
                    track_chatbot_event("product_extraction", user_message, status)
                resp = requests.get(
                    f"{PC_BACKEND_URL}/build/compatibility",
                    params={
                        "product1": json.dumps(entities["product1"]), 
                        "product2": json.dumps(entities["product2"])
                    }
                )
                
                # Check if request was successful
                if resp.status_code != 200:
                    response = {
                        "type": "text", 
                        "data": "Xin lỗi, hệ thống không tìm thấy thông tin về tương thích. Vui lòng thử lại sau."
                    }
                    status = "error"
                else:
                    # Parse response - this is a direct boolean value
                    compatibility_result = resp.json()
                    if compatibility_result == True:
                        response = {
                            "type": "text", 
                            "data": "Có thể kết hợp các sản phẩm này."
                        }
                        status = "success"
                    else:
                        response = {
                            "type": "text", 
                            "data": """Không thể kết hợp các sản phẩm này.
                            Để chắc chắn bạn hãy gửi lại thông tin với định dạng:
                            < Tên đầy đủ sản phẩm 1 + Loại sản phẩm 1> <Tên đầy đủ sản phẩm 2 + Loại sản phẩm 2>.
                            Hoặc bạn có thể sử dụng tính năng "xây dựng cấu hình" của chúng tôi."""
                        }
            except Exception as e:
                print("Exception in compatibility check", e)
                response = {
                    "type": "error",
                    "data": "Xin lỗi bạn, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
                }
                status = "error"
            finally:
                track_chatbot_event("compatibility", user_message, status)
        
        # If the intent is about FAQs, call the LLM
        elif intent == "faq":
            try:
                status = "success"
                faq_context = retrieve_faq_answer(user_message)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in history:
                    messages.append({"role": "user", "content": msg})
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
                status = "error"
            finally:
                track_chatbot_event("faq", user_message, status)
        
        # If the intent is unknown, return a default response
        else:
            response = {
                "type": "error",
                "data": "Xin lỗi, tôi không được huấn luyện để trả lời câu hỏi này."
            }   
            status = "error"
            track_chatbot_event("unknown", user_message, status)
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