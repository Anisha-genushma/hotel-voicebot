"""
TTS Service – generates Twilio TwiML <Say> responses.
Also provides plain-text passthrough for the browser Web Speech API.

NEW: sentence_chunks() – splits text into speakable units for live streaming TTS.
     The WebSocket voice route yields these chunks one-by-one so the browser
     can call SpeechSynthesisUtterance on each sentence as it arrives, giving
     a ChatGPT-style "speaking while thinking" experience.
"""

import html
import re
from typing import Generator
from config import settings


def _sanitise(text: str) -> str:
    """Remove markdown formatting and XML-escape for TwiML."""
    # Strip markdown bold/italic/code
    text = re.sub(r"[*_`#]+", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    # XML-escape for TwiML embedding
    return html.escape(text)


def build_twiml(answer_text: str) -> str:
    """
    Wrap answer text in Twilio TwiML using Polly.Amy voice.
    Used when the voice agent is invoked via Twilio phone integration.
    """
    safe_text = _sanitise(answer_text)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        f'  <Say voice="{settings.twiml_voice}">{safe_text}</Say>\n'
        "</Response>"
    )
    return twiml


def text_for_browser_tts(answer_text: str) -> str:
    """Clean text for the browser's Web Speech API (no XML escaping needed)."""
    text = re.sub(r"[*_`#]+", "", answer_text)
    return " ".join(text.split())


def sentence_chunks(text: str, min_chars: int = 40) -> Generator[str, None, None]:
    """
    Split answer text into natural sentence-sized chunks optimised for
    browser TTS streaming.

    Yields chunks immediately as they become long enough so the WebSocket
    route can send them to the client for instant SpeechSynthesisUtterance
    playback — no waiting for the full response.

    Rules:
    - Split on sentence-ending punctuation (.  !  ?)
    - Never yield a chunk shorter than `min_chars` unless it's the last piece
      (avoids choppy single-word utterances for very short sentences)
    - Strip markdown before yielding
    - Collapse excessive whitespace

    Example:
        "Welcome to ITC Grand Chola. How may I help? We have great dining."
        → ["Welcome to ITC Grand Chola.", "How may I help?", "We have great dining."]
    """
    # Clean markdown first
    clean = re.sub(r"[*_`#>]+", "", text)
    clean = " ".join(clean.split())

    # Split on sentence boundaries, keeping the delimiter
    parts = re.split(r'(?<=[.!?])\s+', clean.strip())

    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        buffer = (buffer + " " + part).strip() if buffer else part
        # Yield when buffer is long enough (natural sentence break)
        if len(buffer) >= min_chars:
            yield buffer
            buffer = ""

    # Flush remaining
    if buffer.strip():
        yield buffer.strip()