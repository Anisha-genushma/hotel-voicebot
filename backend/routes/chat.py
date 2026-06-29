"""
Chat route – accepts query + conversation history for follow-up support.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from backend.services.rag_service import RAGService
from backend.services.tts_service import text_for_browser_tts

router = APIRouter()


class HistoryEntry(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[HistoryEntry] = []


class ChatResponse(BaseModel):
    answer: str
    tts_text: str
    sources: list
    retrieved: bool


@router.post("/query", response_model=ChatResponse)
async def chat_query(body: ChatRequest, request: Request):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    vs = request.app.state.vector_service
    rag = RAGService(vs)
    history = [{"role": h.role, "content": h.content} for h in body.history]
    result = rag.answer(body.query.strip(), history=history)

    return ChatResponse(
        answer=result["answer"],
        tts_text=text_for_browser_tts(result["answer"]),
        sources=result["sources"],
        retrieved=result["retrieved"],
    )