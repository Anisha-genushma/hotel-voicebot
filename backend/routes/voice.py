"""
backend/routes/voice.py

WebSocket live voice route + REST transcribe-and-answer endpoint.

WebSocket protocol (client → server):
  { type: "ping" }                           – keepalive (reply: pong)
  { type: "vad_speech_start" }               – user started speaking
  { type: "vad_speech_end", data, mime, history } – audio blob ready
  { type: "text", query, history }           – typed message in live mode
  { type: "barge_in" }                       – user interrupted Chola

WebSocket protocol (server → client):
  { type: "pong" }
  { type: "listening" }                      – ready to receive speech
  { type: "transcript", text }               – STT result
  { type: "stream_start" }                   – Chola starting to answer
  { type: "token", text }                    – one sentence chunk
  { type: "stream_end", sources, full }      – answer complete
  { type: "barge_in_ack" }                   – barge-in accepted
  { type: "error", message }

Session lifetime:
  - 25-second client keepalive ping prevents nginx/load-balancer timeout
  - Server-side IDLE_TIMEOUT_S (12 min) closes dead sessions gracefully
  - Conversation history is sent from the client on each turn
"""

import asyncio
import base64
import json
import logging
import re

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from backend.services.gemini_client import get_gemini_client
from backend.services.rag_service import RAGService
from backend.services.tts_service import text_for_browser_tts

logger = logging.getLogger(__name__)
router = APIRouter()

# Server closes idle WebSocket after this many seconds.
# Must exceed the client's 25 s ping interval. 12 minutes is generous.
IDLE_TIMEOUT_S = 720

SPLIT_PAT = re.compile(r'(?<=[.!?])\s+')


# ── REST: transcribe audio + return answer (non-session mode) ───────────────

@router.post("/transcribe-and-answer")
async def transcribe_and_answer(
    request: Request,
    audio: UploadFile = File(...),
    history: str = Form(default="[]"),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    try:
        history_list = json.loads(history)
    except json.JSONDecodeError:
        history_list = []

    client = get_gemini_client()

    try:
        transcript = client.transcribe_audio(audio_bytes, mime_type=audio.content_type or "audio/webm")
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(500, "Speech transcription failed")

    if not transcript:
        return {"transcript": "", "answer": "", "sources": []}

    vs = request.app.state.vector_service
    rag = RAGService(vs)
    result = rag.answer(transcript, history=history_list)

    return {
        "transcript": transcript,
        "answer": result["answer"],
        "tts_text": text_for_browser_tts(result["answer"]),
        "sources": result["sources"],
    }


# ── WebSocket: live voice session ───────────────────────────────────────────

@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    logger.info("Live WS session opened: %s", websocket.client)

    # Build RAG service from app state (same pattern as chat.py)
    vs  = websocket.app.state.vector_service
    rag = RAGService(vs)

    # Arm the client immediately
    await _send(websocket, {"type": "listening"})

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=IDLE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.info("WS idle timeout — closing session")
                await websocket.close(1001, "Session idle timeout")
                return

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "ping":
                await _send(websocket, {"type": "pong"})

            elif msg_type == "barge_in":
                await _send(websocket, {"type": "barge_in_ack"})

            elif msg_type == "vad_speech_start":
                pass  # informational; client is already recording

            elif msg_type == "vad_speech_end":
                await _handle_audio(websocket, msg, rag)

            elif msg_type == "text":
                await _handle_text(websocket, msg, rag)

    except WebSocketDisconnect:
        logger.info("Live WS session closed by client")
    except Exception as exc:
        logger.error("Live WS unexpected error: %s", exc, exc_info=True)
        try:
            await _send(websocket, {"type": "error", "message": "Internal server error"})
        except Exception:
            pass


# ── Internal helpers ─────────────────────────────────────────────────────────

async def _send(ws: WebSocket, obj: dict):
    try:
        await ws.send_text(json.dumps(obj))
    except Exception:
        pass


async def _handle_audio(ws: WebSocket, msg: dict, rag: RAGService):
    b64_data = msg.get("data", "")
    mime     = msg.get("mime", "audio/webm")
    history  = msg.get("history", [])

    if not b64_data:
        await _send(ws, {"type": "error", "message": "Empty audio payload"})
        await _send(ws, {"type": "listening"})
        return

    try:
        audio_bytes = base64.b64decode(b64_data)
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid audio encoding"})
        await _send(ws, {"type": "listening"})
        return

    client = get_gemini_client()
    try:
        transcript = await asyncio.to_thread(client.transcribe_audio, audio_bytes, mime)
    except Exception as exc:
        logger.error("Transcription error: %s", exc)
        await _send(ws, {"type": "error", "message": "Could not transcribe audio"})
        await _send(ws, {"type": "listening"})
        return

    if not transcript or not transcript.strip():
        await _send(ws, {"type": "listening"})
        return

    await _send(ws, {"type": "transcript", "text": transcript})
    await _stream_answer(ws, transcript, history, rag)


async def _handle_text(ws: WebSocket, msg: dict, rag: RAGService):
    query   = (msg.get("query") or "").strip()
    history = msg.get("history", [])

    if not query:
        await _send(ws, {"type": "listening"})
        return

    await _stream_answer(ws, query, history, rag)


async def _stream_answer(ws: WebSocket, query: str, history: list, rag: RAGService):
    await _send(ws, {"type": "stream_start"})

    try:
        result = await asyncio.to_thread(rag.answer_stream, query, history)
    except Exception as exc:
        logger.error("RAG stream setup error: %s", exc)
        await _send(ws, {"type": "error", "message": "Failed to retrieve answer"})
        await _send(ws, {"type": "listening"})
        return

    stream_gen = result["stream"]
    sources    = result.get("sources", [])
    full_text  = ""
    sentence_buf = ""

    try:
        async for chunk in _async_iter(stream_gen):
            full_text    += chunk
            sentence_buf += chunk

            parts = SPLIT_PAT.split(sentence_buf)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    s = sentence.strip()
                    if s:
                        await _send(ws, {"type": "token", "text": s})
                sentence_buf = parts[-1]

        if sentence_buf.strip():
            await _send(ws, {"type": "token", "text": sentence_buf.strip()})

    except Exception as exc:
        logger.error("Streaming generation error: %s", exc)
        await _send(ws, {"type": "error", "message": "Answer generation interrupted"})

    await _send(ws, {"type": "stream_end", "full": full_text, "sources": sources})
    await _send(ws, {"type": "listening"})


async def _async_iter(sync_gen):
    """Wrap a synchronous generator for async iteration without blocking the event loop."""
    loop = asyncio.get_event_loop()
    it   = iter(sync_gen)
    sentinel = object()
    while True:
        chunk = await loop.run_in_executor(None, next, it, sentinel)
        if chunk is sentinel:
            break
        yield chunk