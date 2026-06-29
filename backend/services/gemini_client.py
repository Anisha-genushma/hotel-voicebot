"""
Gemini client with:
  - Multi-key failover (API_KEY_1 → API_KEY_2 → API_KEY_3)
  - Model fallback (gemini-2.5-flash-lite → gemini-2.5-flash → gemini-2.0-flash)
  - Embedding via text-embedding-004
  - NEW: generate_text_stream() – yields text chunks for live streaming
"""

import logging
import time
from typing import Generator, List, Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Thread-safe Gemini wrapper with key rotation and model fallback."""

    def __init__(self):
        self._keys = settings.gemini_api_keys
        if not self._keys:
            raise RuntimeError(
                "No Gemini API keys configured. Set GEMINI_API_KEY_1 (and optionally _2, _3)."
            )
        self._current_key_idx = 0
        self._configure(self._keys[0])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _configure(self, api_key: str):
        genai.configure(api_key=api_key)
        self._current_key = api_key

    def _rotate_key(self) -> bool:
        """Try the next key. Returns False if all keys exhausted."""
        next_idx = self._current_key_idx + 1
        if next_idx >= len(self._keys):
            return False
        self._current_key_idx = next_idx
        self._configure(self._keys[next_idx])
        logger.warning("Rotated to Gemini API key index %d", next_idx)
        return True

    def _model_sequence(self) -> List[str]:
        return [settings.primary_model] + settings.fallback_models

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_text(self, prompt: str, system_instruction: str = "") -> str:
        """Generate text with automatic key rotation and model fallback."""
        retryable_exceptions = (ResourceExhausted, ServiceUnavailable)

        key_idx_start = self._current_key_idx

        for model_name in self._model_sequence():
            # Reset key index to try all keys per model
            self._current_key_idx = key_idx_start
            self._configure(self._keys[self._current_key_idx])

            while True:
                try:
                    logger.debug("generate_text: model=%s key_idx=%d", model_name, self._current_key_idx)
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_instruction or None,
                    )
                    response = model.generate_content(prompt)
                    return response.text.strip()

                except retryable_exceptions as exc:
                    logger.warning("Key %d / model %s hit quota: %s", self._current_key_idx, model_name, exc)
                    if not self._rotate_key():
                        logger.warning("All keys exhausted for model %s, trying next model.", model_name)
                        break
                    time.sleep(1)

                except Exception as exc:
                    logger.error("Unexpected error with model %s: %s", model_name, exc)
                    break  # Try next model

        raise RuntimeError("All Gemini API keys and model fallbacks exhausted.")

    def generate_text_stream(
        self, prompt: str, system_instruction: str = ""
    ) -> Generator[str, None, None]:
        """
        Stream text generation – yields token chunks as they arrive from Gemini.

        Usage:
            for chunk in client.generate_text_stream(prompt, system):
                send_to_websocket(chunk)

        Falls back gracefully: if streaming fails on a model/key, yields the
        complete non-streamed text from the next available option so callers
        always get output.
        """
        retryable_exceptions = (ResourceExhausted, ServiceUnavailable)
        key_idx_start = self._current_key_idx

        for model_name in self._model_sequence():
            self._current_key_idx = key_idx_start
            self._configure(self._keys[self._current_key_idx])

            while True:
                try:
                    logger.debug(
                        "generate_text_stream: model=%s key_idx=%d", model_name, self._current_key_idx
                    )
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_instruction or None,
                    )
                    # stream=True returns an iterable of GenerateContentResponse chunks
                    for chunk in model.generate_content(prompt, stream=True):
                        try:
                            text = chunk.text
                            if text:
                                yield text
                        except Exception:
                            # Some chunks may not carry text (safety metadata etc.)
                            continue
                    return  # Streaming completed successfully

                except retryable_exceptions as exc:
                    logger.warning(
                        "Stream key %d / model %s quota: %s", self._current_key_idx, model_name, exc
                    )
                    if not self._rotate_key():
                        logger.warning("All keys exhausted for streaming model %s.", model_name)
                        break
                    time.sleep(1)

                except Exception as exc:
                    logger.error("Stream error with model %s: %s", model_name, exc)
                    break  # Try next model

        # Last-resort: non-streamed fallback so caller always gets something
        logger.warning("Falling back to non-streamed generate_text for streaming request.")
        try:
            full_text = self.generate_text(prompt, system_instruction)
            yield full_text
        except Exception as exc:
            raise RuntimeError(f"All streaming and fallback attempts failed: {exc}") from exc

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        key_idx_start = self._current_key_idx
        self._current_key_idx = key_idx_start
        self._configure(self._keys[self._current_key_idx])

        while True:
            try:
                result = genai.embed_content(
                    model=f"models/{settings.embedding_model}",
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]

            except (ResourceExhausted, ServiceUnavailable) as exc:
                logger.warning("Embedding quota hit on key %d: %s", self._current_key_idx, exc)
                if not self._rotate_key():
                    raise RuntimeError("All Gemini API keys exhausted during embedding.") from exc
                time.sleep(1)

    def embed_query(self, text: str) -> List[float]:
        """Generate a query embedding (task_type differs from document)."""
        self._configure(self._keys[self._current_key_idx])
        while True:
            try:
                result = genai.embed_content(
                    model=f"models/{settings.embedding_model}",
                    content=text,
                    task_type="retrieval_query",
                )
                return result["embedding"]
            except (ResourceExhausted, ServiceUnavailable) as exc:
                logger.warning("Query embed quota hit: %s", exc)
                if not self._rotate_key():
                    raise RuntimeError("All keys exhausted during query embedding.") from exc
                time.sleep(1)

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """
        Speech-to-text using Gemini multimodal.
        Sends raw audio bytes to the model and returns the transcript.
        """
        import base64

        audio_b64 = base64.b64encode(audio_bytes).decode()

        prompt_parts = [
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": audio_b64,
                }
            },
            "Transcribe the spoken words in this audio accurately. Return only the transcript text with no extra commentary.",
        ]

        for model_name in self._model_sequence():
            try:
                model = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content(prompt_parts)
                return response.text.strip()
            except Exception as exc:
                logger.warning("Transcription failed with %s: %s", model_name, exc)
                if not self._rotate_key():
                    continue

        raise RuntimeError("Audio transcription failed across all models and keys.")


# Singleton
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client