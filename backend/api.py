from fastapi import FastAPI
from pydantic import BaseModel

from core.chatbot_engine import ChatbotEngine
from core.vector_store import VectorStore
from database.db import get_recent_history, init_db, save_message


app = FastAPI(title="Educational NLP Chatbot API")

init_db()
vector_store = VectorStore()
chatbot = ChatbotEngine(vector_store)


class ChatRequest(BaseModel):
    user_id: str
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    history = get_recent_history(request.user_id, limit=5)
    answer, sources = chatbot.answer(request.question, history)
    save_message(request.user_id, request.question, answer)
    return {
        "answer": answer,
        "sources": sources,
    }
