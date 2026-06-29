"""
RAG Service – retrieval → prompt → Gemini response.
Now accepts conversation history for follow-up support.
NEW: answer_stream() – yields text chunks for live WebSocket streaming.
"""

import logging
from typing import List, Dict, Any, Generator

from backend.services.gemini_client import get_gemini_client
from backend.services.vector_service import VectorService
from backend.prompts.hotel_prompts import SYSTEM_PROMPT, build_rag_prompt, NO_CONTEXT_RESPONSE

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, vector_service: VectorService):
        self._vs = vector_service

    def answer(self, query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        chunks = self._vs.search(query)

        if not chunks:
            logger.info("No relevant context for: %r", query[:80])
            return {"answer": NO_CONTEXT_RESPONSE, "sources": [], "retrieved": False}

        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"[Context {i}] {chunk['category']} – {chunk['title']}\n{chunk['content']}"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = build_rag_prompt(query=query, context=context_text, history=history or [])
        client = get_gemini_client()
        answer_text = client.generate_text(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        return {
            "answer": answer_text,
            "sources": [
                {"id": c["id"], "title": c["title"], "score": round(c["_score"], 3)}
                for c in chunks
            ],
            "retrieved": True,
        }

    def answer_stream(
        self, query: str, history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          - 'sources': list of source dicts (retrieved before streaming starts)
          - 'retrieved': bool
          - 'stream': Generator[str] – yields raw text chunks from Gemini as they arrive
          - 'no_context': bool – True if no KB match found (stream will yield fallback msg)

        Caller is responsible for consuming 'stream' and assembling full text.
        """
        chunks = self._vs.search(query)

        if not chunks:
            logger.info("No relevant context (stream) for: %r", query[:80])

            def _fallback_gen():
                yield NO_CONTEXT_RESPONSE

            return {
                "stream": _fallback_gen(),
                "sources": [],
                "retrieved": False,
                "no_context": True,
            }

        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"[Context {i}] {chunk['category']} – {chunk['title']}\n{chunk['content']}"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = build_rag_prompt(query=query, context=context_text, history=history or [])
        client = get_gemini_client()

        # generate_text_stream is a sync Generator – caller must wrap in asyncio.to_thread
        stream_gen = client.generate_text_stream(
            prompt=prompt, system_instruction=SYSTEM_PROMPT
        )

        return {
            "stream": stream_gen,
            "sources": [
                {"id": c["id"], "title": c["title"], "score": round(c["_score"], 3)}
                for c in chunks
            ],
            "retrieved": True,
            "no_context": False,
        }