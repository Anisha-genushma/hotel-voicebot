"""
Hotel AI Suite — unified entry point
  Project 1 → /receptionist/   AI Call Receptionist (Gemini agent)
  Project 2 → /faq/            ITC Grand Chola Concierge (RAG + FAISS voice agent)
  Project 3 → /analyzer/       Call Sentiment Analyzer (Gemini transcription + analysis)
"""

import os
import tempfile
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # must run before any service imports read env vars

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import Response, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List

# ── Project 1 ─────────────────────────────────────────────────────────────────
from app.agent import ReceptionistAgent

# ── Project 2 (FAQ replacement — ITC Grand Chola Voice Agent) ─────────────────
from backend.routes import voice as faq_voice, chat as faq_chat, health as faq_health
from backend.services.vector_service import VectorService

# ── Project 3 ─────────────────────────────────────────────────────────────────
from models import AnalyzeRequest, AnalyzeResponse, TranscribeResponse
from gemini_service import analyze_transcript, transcribe_audio, get_audio_duration_seconds

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index for Project 2 once at startup."""
    logger.info("Starting Hotel AI Suite…")
    vs = VectorService()
    vs.load_or_build_index()          # fast load if index exists; builds on first run
    app.state.vector_service = vs
    logger.info("Vector service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Hotel AI Suite", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static mounts ─────────────────────────────────────────────────────────────
# /faq-static        → Project 2 CSS + JS  (frontend/static/)
# /analyzer-static   → Project 3 frontend assets
app.mount("/static",          StaticFiles(directory="frontend/static"),   name="static")
app.mount("/analyzer-static", StaticFiles(directory="analyzer_frontend"), name="analyzer-static")

# Project 2 Jinja2 templates
faq_templates = Jinja2Templates(directory="frontend/templates")


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Hotel AI Suite</title>
  <style>
    :root {
      --bg:#0b0f1a;--surface:#131929;--border:#1e2d45;
      --gold:#c9a84c;--gold-dim:#8a6e2f;
      --teal:#3ecfcf;--teal-dim:#1a8080;
      --lav:#a78bfa;--lav-dim:#5b3fc9;
      --text:#e8eaf0;--muted:#6b7a99;--r:14px;
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
      min-height:100vh;display:flex;flex-direction:column;align-items:center;
      justify-content:center;padding:2rem 1.5rem}
    .header{text-align:center;margin-bottom:3.5rem}
    .eyebrow{letter-spacing:.18em;text-transform:uppercase;font-size:.72rem;
      color:var(--gold);margin-bottom:.75rem}
    h1{font-size:clamp(2rem,5vw,3.4rem);font-weight:700;line-height:1.1;letter-spacing:-.02em}
    h1 span{background:linear-gradient(90deg,var(--gold),var(--teal));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .sub{margin-top:1rem;color:var(--muted);font-size:1rem;max-width:420px;
      margin-inline:auto;line-height:1.6}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
      gap:1.5rem;width:100%;max-width:960px}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
      padding:2rem 1.75rem 1.75rem;display:flex;flex-direction:column;gap:1rem;
      transition:transform .2s,border-color .2s,box-shadow .2s;
      text-decoration:none;color:inherit;position:relative;overflow:hidden}
    .card:hover{transform:translateY(-4px)}
    .c1{--a:var(--gold);--ad:var(--gold-dim)}
    .c2{--a:var(--teal);--ad:var(--teal-dim)}
    .c3{--a:var(--lav);--ad:var(--lav-dim)}
    .card:hover{border-color:var(--a);box-shadow:0 0 28px -6px var(--ad)}
    .icon{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;
      justify-content:center;font-size:1.5rem;flex-shrink:0;
      background:color-mix(in srgb,var(--a) 15%,transparent);
      border:1px solid color-mix(in srgb,var(--a) 30%,transparent)}
    .body{flex:1}
    .body h2{font-size:1.1rem;font-weight:650;margin-bottom:.4rem}
    .body p{font-size:.875rem;color:var(--muted);line-height:1.55}
    .cta{display:inline-flex;align-items:center;gap:.45rem;font-size:.82rem;
      font-weight:600;color:var(--a);letter-spacing:.03em;margin-top:.25rem}
    .cta svg{transition:transform .2s}
    .card:hover .cta svg{transform:translateX(4px)}
    footer{margin-top:3.5rem;color:var(--muted);font-size:.78rem;text-align:center}
    @media(max-width:500px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <header class="header">
    <p class="eyebrow">Powered by Gemini AI</p>
    <h1>Hotel <span>AI Suite</span></h1>
    <p class="sub">Three intelligent tools — one unified platform for modern hospitality operations.</p>
  </header>
  <main class="grid">
    <a href="/receptionist/" class="card c1">
      <div class="icon">📞</div>
      <div class="body">
        <h2>AI Call Receptionist</h2>
        <p>Simulate or deploy an AI voice agent that greets callers, collects details, and schedules callbacks — with Twilio integration.</p>
      </div>
      <span class="cta">Open simulator <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
    </a>
    <a href="/faq/" class="card c2">
      <div class="icon">💬</div>
      <div class="body">
        <h2>ITC Grand Chola Concierge</h2>
        <p>RAG-powered AI voice concierge for ITC Grand Chola — answers guest questions from a FAISS knowledge base with live streaming responses.</p>
      </div>
      <span class="cta">Ask Chola <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
    </a>
    <a href="/analyzer/" class="card c3">
      <div class="icon">🎙️</div>
      <div class="body">
        <h2>Call Sentiment Analyzer</h2>
        <p>Upload a hotel call recording to transcribe it and extract sentiment, booking intent, and actionable insights — powered by Gemini.</p>
      </div>
      <span class="cta">Analyze a call <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
    </a>
  </main>
  <footer>Hotel AI Suite &nbsp;·&nbsp; All systems operational</footer>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse(content=LANDING_HTML)


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT 1 — AI CALL RECEPTIONIST  (prefix: /receptionist)
# ═══════════════════════════════════════════════════════════════════════════════

call_context: Dict[str, str] = {
    "name": "", "phone": "", "purpose": "", "priority": "", "callback_time": ""
}
conversation_history: List[Dict[str, Any]] = []
call_summaries: List[Dict[str, Any]] = []
system_state: Dict[str, Any] = {
    "human_available": False,
    "caller_phone": "+14155552671",
    "current_step": "idle",
}
agent = ReceptionistAgent()


class RespondRequest(BaseModel):
    user_input: str


class ToggleHumanRequest(BaseModel):
    available: bool


def _build_summary(ctx: dict, caller_sid: str) -> dict:
    return {
        "name": ctx.get("name"),
        "phone": ctx.get("phone") or caller_sid,
        "purpose": ctx.get("purpose"),
        "priority": ctx.get("priority") or "MEDIUM",
        "callback_time": ctx.get("callback_time"),
        "summary": (
            f"Caller {ctx.get('name')} requested a callback regarding "
            f"'{ctx.get('purpose')}' with priority {ctx.get('priority')}. "
            f"Preferred callback time: {ctx.get('callback_time')}. "
            f"Contact: {ctx.get('phone')}."
        ),
    }


@app.get("/receptionist/", response_class=HTMLResponse)
def receptionist_root():
    with open("receptionist.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/receptionist/api/call/start")
def start_call():
    global call_context, conversation_history
    if system_state["human_available"]:
        system_state["current_step"] = "human_agent"
        return {"status": "transferred", "response_text": "Connecting you to an available human agent...",
                "context": {}, "step": "human_agent"}
    call_context.clear()
    call_context.update({"name": "", "phone": "", "purpose": "", "priority": "", "callback_time": ""})
    conversation_history.clear()
    greeting = (
        "Hello. Thank you for calling. All our representatives are currently assisting other customers. "
        "I can collect your details and arrange a callback. May I know your name please?"
    )
    conversation_history.append({"role": "user", "parts": ["[Incoming Call]"]})
    conversation_history.append({"role": "model", "parts": [greeting]})
    system_state["current_step"] = "collecting_details"
    return {"status": "active", "response_text": greeting, "context": call_context,
            "step": system_state["current_step"]}


@app.post("/receptionist/api/call/respond")
def respond_call(req: RespondRequest):
    global call_context, conversation_history, call_summaries
    conversation_history.append({"role": "user", "parts": [req.user_input]})
    result = agent.process_turn(
        conversation_history=conversation_history,
        user_input=req.user_input,
        current_context=call_context,
        caller_phone=system_state["caller_phone"],
    )
    for key in call_context:
        if key in result.get("context", {}):
            call_context[key] = result["context"][key]
    response_text = result.get("response_text", "")
    conversation_history.append({"role": "model", "parts": [response_text]})
    if result.get("call_ended", False):
        call_summaries.append(_build_summary(call_context, system_state["caller_phone"]))
        call_context.clear()
        conversation_history.clear()
        system_state["current_step"] = "idle"
        return {"status": "ended", "response_text": response_text, "context": {}, "step": "idle"}
    system_state["current_step"] = "collecting_details"
    return {"status": "active", "response_text": response_text, "context": call_context,
            "step": system_state["current_step"]}


@app.post("/receptionist/api/call/end")
def end_call():
    call_context.clear()
    conversation_history.clear()
    system_state["current_step"] = "idle"
    return {"status": "ended", "response_text": "Call ended.", "context": {}}


@app.get("/receptionist/api/call/state")
def get_call_state():
    return {"call_context": call_context, "current_step": system_state["current_step"],
            "human_available": system_state["human_available"],
            "conversation_length": len(conversation_history)}


@app.get("/receptionist/api/call/summaries")
def get_summaries():
    return call_summaries


@app.post("/receptionist/api/call/toggle-human")
def toggle_human(req: ToggleHumanRequest):
    system_state["human_available"] = req.available
    return {"human_available": system_state["human_available"]}


@app.post("/receptionist/twilio/voice")
async def twilio_voice():
    global call_context, conversation_history
    if system_state["human_available"]:
        twiml = ('<?xml version="1.0" encoding="UTF-8"?><Response>'
                 '<Say voice="Polly.Amy">Connecting you to an available representative. Please hold.</Say>'
                 '<Dial>+15555555555</Dial></Response>')
        return Response(content=twiml, media_type="application/xml")
    call_context.clear()
    call_context.update({"name": "", "phone": "", "purpose": "", "priority": "", "callback_time": ""})
    conversation_history.clear()
    greeting = (
        "Hello. Thank you for calling. All our representatives are currently assisting other customers. "
        "I can collect your details and arrange a callback. May I know your name please?"
    )
    conversation_history.append({"role": "user",  "parts": ["[Incoming Call]"]})
    conversation_history.append({"role": "model", "parts": [greeting]})
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        f'<Say voice="Polly.Amy">{greeting}</Say>'
        '<Gather input="speech" action="/receptionist/twilio/gather" timeout="4" speechTimeout="auto" language="en-IN">'
        '<Say voice="Polly.Amy">Please speak after the beep.</Say>'
        '</Gather><Say voice="Polly.Amy">We did not receive any input. Goodbye.</Say><Hangup/></Response>'
    )
    return Response(content=twiml, media_type="application/xml")


@app.post("/receptionist/twilio/gather")
async def twilio_gather(request: Request):
    global call_context, conversation_history, call_summaries
    form_data  = await request.form()
    user_input = form_data.get("SpeechResult", "")
    caller_sid = form_data.get("From", system_state["caller_phone"])
    if not user_input:
        twiml = ('<?xml version="1.0" encoding="UTF-8"?><Response>'
                 '<Say voice="Polly.Amy">I did not catch that. Could you please say that again?</Say>'
                 '<Gather input="speech" action="/receptionist/twilio/gather" timeout="4" speechTimeout="auto" language="en-IN"/>'
                 '</Response>')
        return Response(content=twiml, media_type="application/xml")
    conversation_history.append({"role": "user", "parts": [user_input]})
    result = agent.process_turn(
        conversation_history=conversation_history,
        user_input=user_input,
        current_context=call_context,
        caller_phone=caller_sid,
    )
    for key in call_context:
        if key in result.get("context", {}):
            call_context[key] = result["context"][key]
    response_text = result.get("response_text", "")
    conversation_history.append({"role": "model", "parts": [response_text]})
    if result.get("call_ended", False):
        call_summaries.append(_build_summary(call_context, caller_sid))
        call_context.clear()
        conversation_history.clear()
        twiml = (f'<?xml version="1.0" encoding="UTF-8"?><Response>'
                 f'<Say voice="Polly.Amy">{response_text}</Say><Hangup/></Response>')
        return Response(content=twiml, media_type="application/xml")
    twiml = (f'<?xml version="1.0" encoding="UTF-8"?><Response>'
             f'<Say voice="Polly.Amy">{response_text}</Say>'
             f'<Gather input="speech" action="/receptionist/twilio/gather" timeout="4" speechTimeout="auto" language="en-IN"/>'
             f'</Response>')
    return Response(content=twiml, media_type="application/xml")


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT 2 — ITC GRAND CHOLA CONCIERGE  (prefix: /faq)
# Replaces the old keyword-based FAQ Assistant with a RAG + FAISS voice agent.
# Static assets served from /static (mounted above).
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/faq/", response_class=HTMLResponse)
async def faq_home(request: Request):
    return faq_templates.TemplateResponse("index.html", {"request": request})

app.include_router(faq_health.router, prefix="/faq/api",       tags=["faq-health"])
app.include_router(faq_chat.router,   prefix="/faq/api/chat",  tags=["faq-chat"])
app.include_router(faq_voice.router,  prefix="/faq/api/voice", tags=["faq-voice"])


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT 3 — CALL SENTIMENT ANALYZER  (prefix: /analyzer)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/mp4",
    "audio/m4a", "audio/x-m4a", "audio/aac", "audio/ogg", "audio/flac",
    "audio/webm", "audio/aiff", "audio/x-aiff",
}


@app.get("/analyzer/")
def analyzer_root():
    return FileResponse("analyzer_frontend/index.html")


@app.post("/analyzer/api/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    suffix  = os.path.splitext(file.filename or "")[1] or ".audio"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Audio file too large (200 MB limit).")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty.")
        try:
            duration = get_audio_duration_seconds(tmp_path)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        mime_type = file.content_type or "audio/mpeg"
        try:
            transcript = transcribe_audio(tmp_path, mime_type)
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        except Exception as e:
            raise HTTPException(502, f"Gemini transcription failed: {e}")
        return TranscribeResponse(transcript=transcript, duration_seconds=duration)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/analyzer/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.transcript.strip():
        raise HTTPException(400, "Transcript is empty.")
    try:
        return analyze_transcript(request)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(502, f"Analysis failed: {e}")


@app.get("/analyzer/api/health")
def analyzer_health():
    return {"status": "ok"}