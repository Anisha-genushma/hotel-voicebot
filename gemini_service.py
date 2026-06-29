import json
import os
from google import genai
from google.genai import types

from models import AnalyzeRequest, AnalyzeResponse

GEMINI_MODEL = "gemini-3.1-flash-lite"

_client = None


def get_client() -> genai.Client:
    """Lazily create the Gemini client so import doesn't fail without a key set yet."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy backend/.env.example to backend/.env "
                "and add your key from https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


TRANSCRIBE_PROMPT = """Transcribe this recorded phone call between a hotel staff member and a caller, from the very beginning of the audio to the very end. Do not skip, summarize, or shorten any part of the call.

Formatting rules:
- Label each line with the speaker as "Agent:" or "Caller:" (use your best judgment to distinguish them; if unclear, label as "Speaker 1:" / "Speaker 2:").
- Write out the full spoken content of every line, in order, exactly as said (translated to English if the caller speaks another language).
- Include filler words, false starts, and hesitations only if they're clearly audible — otherwise produce clean, readable sentences.
- Do not include timestamps, sound effect descriptions, or commentary — only the spoken dialogue.

Output ONLY the transcript text, with no preamble or closing remarks."""


def transcribe_audio(file_path: str, mime_type: str) -> str:
    """Uploads an audio file to Gemini and returns a full start-to-end transcript."""
    client = get_client()

    uploaded_file = client.files.upload(file=file_path, config=types.UploadFileConfig(mime_type=mime_type))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[TRANSCRIBE_PROMPT, uploaded_file],
            config=types.GenerateContentConfig(temperature=0.0),
        )
    finally:
        # Clean up the uploaded file on Gemini's side; best-effort, ignore failures.
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

    transcript = (response.text or "").strip()
    if not transcript:
        raise RuntimeError("Gemini returned an empty transcript for this audio file.")
    return transcript


def get_audio_duration_seconds(file_path: str) -> float:
    """Reads the real duration of the audio file directly, rather than trusting model estimates."""
    from mutagen import File as MutagenFile

    audio = MutagenFile(file_path)
    if audio is None or audio.info is None or not getattr(audio.info, "length", None):
        raise RuntimeError("Could not read duration from this audio file. Is it a valid audio format?")
    return float(audio.info.length)


# JSON schema Gemini must conform to. Using response_schema (not just a prompt instruction)
# makes the model return parseable JSON reliably.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_sentiment": {
            "type": "string",
            "enum": ["very negative", "negative", "neutral", "positive", "very positive"],
        },
        "sentiment_score": {"type": "number"},
        "topic_breakdown": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "estimated_seconds": {"type": "number"},
                    "percentage_of_call": {"type": "number"},
                    "summary": {"type": "string"},
                },
                "required": ["topic", "estimated_seconds", "percentage_of_call", "summary"],
            },
        },
        "booking_probability": {"type": "number"},
        "key_signals": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "overall_sentiment",
        "sentiment_score",
        "topic_breakdown",
        "booking_probability",
        "key_signals",
        "reasoning",
    ],
}


def build_prompt(transcript: str, duration_seconds: float, topics: list[str]) -> str:
    topics_list = ", ".join(topics)
    return f"""You are analyzing a transcript of a phone call between a hotel staff member and a caller (a guest or prospective guest).

CALL DURATION: {duration_seconds:.0f} seconds total.

TOPIC CATEGORIES TO CLASSIFY: {topics_list}

TRANSCRIPT:
---
{transcript}
---

TASK:
1. Read the transcript and estimate, in proportion to the total call duration above, roughly how much time ({duration_seconds:.0f} seconds total) was spent discussing each topic category. Base this on the relative amount of dialogue/turns devoted to each topic, not just keyword counts. Categories with no discussion should get 0 seconds. The estimated_seconds across all topics should sum to approximately the total call duration.
2. Assess the caller's overall sentiment across the call (their tone, satisfaction, enthusiasm or frustration).
3. Estimate the probability (0.0 to 1.0) that this caller will go on to make a booking, based on the conversation content, sentiment, urgency signals, objections raised, and how the call concluded.
4. List 3-6 short key signals you used to make the booking probability judgment (e.g. "asked about cancellation policy", "requested a callback", "compared prices with competitor").
5. Give a brief 2-4 sentence reasoning for your booking probability estimate.

Respond ONLY with JSON matching the required schema. Do not include any text outside the JSON."""


def analyze_transcript(request: AnalyzeRequest) -> AnalyzeResponse:
    client = get_client()
    prompt = build_prompt(request.transcript, request.duration_seconds, request.topics)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    data = json.loads(response.text)

    return AnalyzeResponse(
        call_duration_seconds=request.duration_seconds,
        overall_sentiment=data["overall_sentiment"],
        sentiment_score=data["sentiment_score"],
        topic_breakdown=data["topic_breakdown"],
        booking_probability=data["booking_probability"],
        key_signals=data["key_signals"],
        reasoning=data["reasoning"],
    )