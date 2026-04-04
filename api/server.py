"""FinFlux Pro Server V4.2: 8-Stage Modular Pipeline (Production Expert Module)."""
import os
import shutil
import uuid
import time
import json
import datetime
import base64
import hashlib
import hmac
from typing import Any, Dict
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from dotenv import load_dotenv

# First, ensure .env is available
load_dotenv()

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from finflux.modules.insight_engine.llm_adapters import ExpertSynthesisEngine
from finflux.modules.insight_engine.financial_models import ProductionExpertModule
from api.security import FinFluxSecurity
from api.storage import (
    save_conversation,
    get_all_conversations,
    search_memories,
    create_user,
    authenticate_user,
    get_user_by_username,
    SessionLocal,
    ConversationLog,
    FinancialReminder,
)

app = FastAPI(title="FinFlux Pro", version="4.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Secure Directories
STORAGE_DIR = ROOT_DIR / "data" / "encrypted_audio"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Init Expert Modules Once (Singleton Pattern)
SECURITY = FinFluxSecurity()
EXPERT = ProductionExpertModule()
SYNTHESIS = ExpertSynthesisEngine()

JWT_SECRET = os.environ.get("JWT_SECRET", "finflux-jwt-dev-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))
bearer_scheme = HTTPBearer()


class AuthPayload(BaseModel):
    username: str
    password: str


class ChatPayload(BaseModel):
    text: str
    thread_id: str | None = None


def _make_token(username: str) -> str:
    exp = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": username, "exp": int(exp.timestamp())}
    return _jwt_encode(payload)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def _jwt_encode(payload: Dict[str, Any]) -> str:
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"


def _jwt_decode(token: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")
    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    got_sig = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_sig, got_sig):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    exp = int(payload.get("exp", 0))
    if exp and datetime.datetime.utcnow().timestamp() > exp:
        raise ValueError("Token expired")
    return payload


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    token = credentials.credentials
    try:
        payload = _jwt_decode(token)
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return username
    except ValueError as exc:
        message = str(exc)
        if "expired" in message.lower():
            raise HTTPException(status_code=401, detail="Token expired")
        raise HTTPException(status_code=401, detail="Invalid token")


def _build_analysis_result(
    clean_text: str,
    user_id: str,
    asr_text: str,
    t_asr: float,
    t_norm: float,
    t_start: float,
    thread_id: str,
    input_mode: str,
) -> Dict[str, Any]:
    t_nlp_start = time.time()
    nlp_res = EXPERT.process(clean_text)
    t_nlp = time.time() - t_nlp_start
    memory_context = ""
    try:
        mem_res = search_memories(user_id=user_id, query_text=clean_text, n_results=3)
        result_rows = mem_res.get('results', []) if isinstance(mem_res, dict) else []
        past_summaries = [
            str(row.get('executive_summary', '')).strip()
            for row in result_rows
            if isinstance(row, dict) and str(row.get('executive_summary', '')).strip()
        ]
        if past_summaries:
            memory_context = "\nLONG-TERM MEMORY (PAST CONTEXT):\n" + "\n".join([f"- {s}" for s in past_summaries])
    except Exception:
        memory_context = ""

    t_syn_start = time.time()
    analysis = SYNTHESIS.analyze(
        transcript=clean_text,
        entities=nlp_res.get("entities", []),
        fin_sentiment=nlp_res.get("financial_sentiment", "Neutral"),
        memory_context=memory_context,
    )
    t_syn = time.time() - t_syn_start

    injection_found = SECURITY.detect_injection(clean_text)
    exec_summary = analysis.get("executive_summary", "")
    safe_summary = SECURITY.mask_pii(exec_summary)

    return {
        "conversation_id": f"call_{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "chat_thread_id": thread_id,
        "input_mode": input_mode,
        "raw_user_input": asr_text,
        "timestamp": str(datetime.datetime.utcnow()),
        "language": nlp_res.get("detected_language", "unknown"),
        "language_confidence": nlp_res.get("language_confidence", 0.0),
        "financial_topic": nlp_res.get("topic", "N/A"),
        "topic_top3": nlp_res.get("topic_top3", []),
        "strategic_intent": analysis.get("strategic_intent", ""),
        "risk_level": analysis.get("risk_level", "LOW"),
        "financial_sentiment": nlp_res.get("financial_sentiment", "Neutral"),
        "sentiment_breakdown": nlp_res.get("sentiment_breakdown", {}),
        "advice_request": nlp_res.get("is_advice_request", False) or SECURITY.is_asking_for_advice(clean_text),
        "injection_attempt": injection_found,
        "entities": nlp_res.get("entities", []),
        "executive_summary": safe_summary,
        "summary": safe_summary,
        "future_gearing": analysis.get("future_gearing", ""),
        "risk_assessment": analysis.get("risk_assessment", ""),
        "key_insights": [SECURITY.mask_pii(p) for p in analysis.get("key_insights", [])],
        "key_points": [SECURITY.mask_pii(p) for p in analysis.get("key_insights", [])],
        "transcript": clean_text,
        "expert_reasoning_points": analysis.get("expert_reasoning_points", ""),
        "expert_reasoning": analysis.get("expert_reasoning_points", ""),
        "model_attribution": {
            **nlp_res.get("model_attribution", {}),
            "qwen": {
                "reasoning_available": bool(analysis.get("expert_reasoning_points", "").strip()),
                "section": "Wall of Logic",
            },
        },
        "confidence_score": nlp_res.get("confidence_score", 0.0),
        "timing": {
            "asr_s": round(t_asr, 2),
            "normalization_s": round(t_norm, 2),
            "expert_nlp_s": round(t_nlp, 2),
            "synthesis_s": round(t_syn, 2),
            "total_s": round(time.time() - t_start, 2),
        },
        "raw_asr_text": asr_text,
    }

@app.post("/api/auth/signup")
def signup(payload: AuthPayload):
    username = payload.username.strip().lower()
    if len(username) < 3 or len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Username or password too short")
    user = create_user(username, payload.password)
    if not user:
        raise HTTPException(status_code=409, detail="Username already exists")
    token = _make_token(username)
    return {"access_token": token, "token_type": "bearer", "username": username}


@app.post("/api/auth/login")
def login(payload: AuthPayload):
    username = payload.username.strip().lower()
    user = authenticate_user(username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _make_token(username)
    return {"access_token": token, "token_type": "bearer", "username": username}


@app.get("/api/auth/me")
def me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    thread_id: str = Form(""),
    current_user: str = Depends(get_current_user),
):
    """The 8-Stage Modular Pipeline V4.2."""
    if not file.filename: raise HTTPException(status_code=400, detail="No audio file")
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    t_start = time.time()

    # ── Stage 0: AES Encrypted Audio Persistence ──
    raw_audio = await file.read()
    encrypted_audio = SECURITY.encrypt_audio(raw_audio)
    with open(STORAGE_DIR / f"{call_id}.enc", "wb") as f: f.write(encrypted_audio)

    # Temporary decrypted file for ASR
    temp_wav = f"tmp_{call_id}.wav"
    with open(temp_wav, "wb") as f: f.write(raw_audio)

    try:
        # ── Stage 1: Local-Cloud Hybrid ASR (Groq-Whisper with Local Fallback) ──
        try:
            from finflux.modules.insight_engine.llm_adapters import GroqWhisperAdapter
            whisper = GroqWhisperAdapter()
            asr = whisper.transcribe(temp_wav)
        except Exception as e:
            print(f"[API] Groq ASR Failed: {e}. Switching to Local Fallback...")
            local_text = EXPERT.transcribe_local(temp_wav)
            asr = {"text": local_text, "language": "hindi" if local_text else "unknown"}
            
        raw_text = asr.get("text", "")
        t_asr = time.time() - t_start

        # ── Stage 2: Normalization (Llama 8B) ──
        t2_start = time.time()
        clean_text = SYNTHESIS.normalize_transcript(raw_text)
        t_norm = time.time() - t2_start

        result = _build_analysis_result(
            clean_text=clean_text,
            user_id=current_user,
            asr_text=raw_text,
            t_asr=t_asr,
            t_norm=t_norm,
            t_start=t_start,
            thread_id=thread_id.strip() or f"thr_{uuid.uuid4().hex[:8]}",
            input_mode="audio",
        )

        # Save to Secure DB
        save_conversation(result)
        return result

    except Exception as e:
        print(f"[Server] Pipeline crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

@app.get("/api/warm")
def warm_models():
    """Manually pre-load modular NLP models (VRAM-heavy)."""
    try:
        t_start = time.time()
        EXPERT.warm()
        return {"status": "modular_stack_online", "loading_s": round(time.time() - t_start, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warm-up failed: {str(e)}")

@app.post("/api/chat")
def chat(payload: ChatPayload, current_user: str = Depends(get_current_user)):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t_start = time.time()
    t_asr = 0.0
    t2_start = time.time()
    clean_text = SYNTHESIS.normalize_transcript(text)
    t_norm = time.time() - t2_start

    result = _build_analysis_result(
        clean_text=clean_text,
        user_id=current_user,
        asr_text=text,
        t_asr=t_asr,
        t_norm=t_norm,
        t_start=t_start,
        thread_id=(payload.thread_id or "").strip() or f"thr_{uuid.uuid4().hex[:8]}",
        input_mode="text",
    )
    save_conversation(result)
    return result


@app.get("/api/threads")
def list_threads(current_user: str = Depends(get_current_user)):
    history = get_all_conversations(user_id=current_user).get("results", [])
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in history:
        thread_id = str(row.get("chat_thread_id") or row.get("conversation_id"))
        if thread_id not in grouped:
            grouped[thread_id] = {
                "thread_id": thread_id,
                "last_timestamp": row.get("timestamp", ""),
                "preview": row.get("executive_summary", "")[:110],
                "topic": row.get("financial_topic", "General"),
                "risk_level": row.get("risk_level", "LOW"),
                "count": 0,
            }
        grouped[thread_id]["count"] += 1
        row_ts = str(row.get("timestamp", ""))
        if row_ts > str(grouped[thread_id]["last_timestamp"]):
            grouped[thread_id]["last_timestamp"] = row_ts
            grouped[thread_id]["preview"] = str(row.get("executive_summary", ""))[:110]
            grouped[thread_id]["topic"] = row.get("financial_topic", "General")
            grouped[thread_id]["risk_level"] = row.get("risk_level", "LOW")

    threads = sorted(grouped.values(), key=lambda item: str(item["last_timestamp"]), reverse=True)
    return {"count": len(threads), "results": threads}


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str, current_user: str = Depends(get_current_user)):
    rows = get_all_conversations(user_id=current_user).get("results", [])
    selected = [r for r in rows if str(r.get("chat_thread_id") or r.get("conversation_id")) == thread_id]
    selected.sort(key=lambda item: str(item.get("timestamp", "")))

    messages = []
    for item in selected:
        messages.append({
            "id": f"{item.get('conversation_id')}-user",
            "role": "user",
            "text": item.get("raw_user_input") or item.get("transcript") or "",
            "conversation_id": item.get("conversation_id"),
        })
        messages.append({
            "id": f"{item.get('conversation_id')}-assistant",
            "role": "assistant",
            "text": item.get("executive_summary") or "",
            "conversation_id": item.get("conversation_id"),
            "attached_result": item,
        })
    return {"thread_id": thread_id, "count": len(messages), "results": messages}


@app.get("/api/results")
def list_results(current_user: str = Depends(get_current_user)):
    return get_all_conversations(user_id=current_user)

@app.put("/api/update/{conversation_id}")
async def update_conversation(conversation_id: str, payload: dict, current_user: str = Depends(get_current_user)):
    """Update stored conversation intelligence (e.g. edited transcript)."""
    db = SessionLocal()
    try:
        log = db.query(ConversationLog).filter(
            ConversationLog.id == conversation_id,
            ConversationLog.user_id == current_user,
        ).first()
        if not log: raise HTTPException(status_code=404, detail="Not found")
        
        if "transcript" in payload: log.transcript = payload["transcript"]
        if "executive_summary" in payload: log.executive_summary = payload["executive_summary"]
        if "summary" in payload: log.executive_summary = payload["summary"] # Alias
        
        db.commit()
        return {"status": "updated"}
    finally:
        db.close()

@app.get("/api/report/{conversation_id}")
def generate_report(conversation_id: str, format: str = "pdf", current_user: str = Depends(get_current_user)):
    """Generate a high-fidelity financial analysis report isolated by user."""
    history = get_all_conversations(user_id=current_user)
    data = next((c for c in history["results"] if c["conversation_id"] == conversation_id), None)
    if not data: raise HTTPException(status_code=404, detail="Analysis not found")

    if format == "csv":
        import pandas as pd
        from fastapi.responses import Response
        df = pd.DataFrame([data])
        return Response(content=df.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=finflux_{conversation_id}.csv"})
    
    # PDF Generation (McKinsey Strategic IQ Wall Format)
    import io
    import os
    from fastapi.responses import StreamingResponse
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Register Unicode-capable font if available (Windows Arial)
    font_name = "Helvetica"
    font_path = "C:/Windows/Fonts/arial.ttf"
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('Arial', font_path))
            font_name = "Arial"
        except: pass

    # Custom Styles (using font_name for Unicode support)
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#3b82f6"), spaceAfter=12, fontName=f"{font_name}-Bold" if font_name == "Helvetica" else font_name)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#1e40af"), spaceBefore=18, spaceAfter=8, fontName=f"{font_name}-Bold" if font_name == "Helvetica" else font_name, borderLeftIndent=12, leftIndent=12)
    body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'], fontSize=11, leading=16, spaceAfter=12, fontName=font_name)
    meta_style = ParagraphStyle('MetaStyle', parent=styles['BodyText'], fontSize=9, textColor=colors.grey, fontName=font_name)
    transcript_style = ParagraphStyle('TranscriptStyle', parent=styles['BodyText'], fontSize=9, leading=14, textColor=colors.darkslategrey, leftIndent=12, fontName=font_name)

    story = []

    # 1. Main Header
    story.append(Paragraph("FinFlux Intelligence Report", title_style))
    story.append(Paragraph(f"<b>CONVERSATION ID:</b> {conversation_id}  |  <b>TIMESTAMP:</b> {data.get('timestamp', 'N/A')}", meta_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3b82f6"), spaceAfter=10))

    # 2. Confidence & Sentiment Quick-Wall
    conf = data.get("confidence_score", 0) * 100
    sentiment = data.get("financial_sentiment", "Neutral")
    topic = data.get("financial_topic", "General Analysis")
    
    quick_data = [
        [Paragraph(f"<b>IQ TOPIC:</b> {topic}", body_style), Paragraph(f"<b>SENTIMENT:</b> {sentiment}", body_style)],
        [Paragraph(f"<b>RISK LEVEL:</b> {data.get('risk_level', 'LOW')}", body_style), Paragraph(f"<b>IQ CONFIDENCE:</b> {conf}%", body_style)]
    ]
    
    quick_table = Table(quick_data, colWidths=[3*inch, 3*inch])
    quick_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(quick_table)
    story.append(Spacer(1, 0.3*inch))

    # 3. Executive Summary
    story.append(Paragraph("EXECUTIVE STRATEGIC SUMMARY", header_style))
    story.append(Paragraph(data.get("executive_summary", "No summary captured."), body_style))

    # 4. Strategic Intent & Risk Logic
    story.append(Paragraph("STRATEGIC INTENT & FUTURE GEARING", header_style))
    intent_data = [
        [Paragraph("<b>Intent:</b> " + data.get("strategic_intent", "N/A"), body_style)],
        [Paragraph("<b>Future Gearing:</b> " + data.get("future_gearing", "N/A"), body_style)],
        [Paragraph("<b>Risk Detail:</b> " + data.get("risk_assessment", "N/A"), body_style)]
    ]
    story.append(Table(intent_data, colWidths=[6.5*inch]))
    story.append(Spacer(1, 0.3*inch))

    # 5. Entity Wall (Table)
    entities = data.get("entities", [])
    if entities:
        story.append(Paragraph("PRECISION ENTITY WALL", header_style))
        ent_rows = [["TYPE", "VALUE", "CONTEXT"]]
        for e in entities:
            ent_rows.append([e['type'], e['value'], e.get('context', 'Manual Extraction')])
        
        ent_table = Table(ent_rows, colWidths=[1.5*inch, 2*inch, 3*inch])
        ent_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (1, 1), (-1, -1), 10),
        ]))
        story.append(ent_table)
        story.append(Spacer(1, 0.4*inch))

    # 6. Reasoning Wall (Qwen CFA IQ)
    reasoning = data.get("expert_reasoning_points", "")
    if reasoning:
        story.append(Paragraph("EXPERT ANALYST REASONING (CFA V4.2)", header_style))
        for line in reasoning.split('\n'):
            if line.strip():
                clean_line = line.replace('• ', '').replace('**', '')
                story.append(Paragraph(f"• {clean_line}", body_style))
    
    # 7. Transcript (New Page)
    story.append(PageBreak())
    story.append(Paragraph("VERIFIED TRANSCRIPT RECORD", header_style))
    transcript = data.get("transcript", "")
    for part in transcript.split('\n\n'):
        if part.strip():
            story.append(Paragraph(part, transcript_style))
            story.append(Spacer(1, 0.1*inch))

    doc.build(story)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=finflux_strategic_iq_{conversation_id}.pdf"})

@app.get("/api/health")
def health():
    key_ok = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "finflux-v4.2-online", 
        "security": "AES-256 enabled",
        "api_key_loaded": key_ok,
        "models": ["LangDetect", "FinBERT", "GLiNER", "DeBERTa", "Qwen", "Llama-3.1"]
    }

@app.put("/api/conversations/{conversation_id}/transcript")
async def edit_transcript(conversation_id: str, payload: dict, current_user: str = Depends(get_current_user)):
    """Task 5: User transcript editing with re-analysis trigger."""
    db = SessionLocal()
    try:
        log = db.query(ConversationLog).filter(
            ConversationLog.id == conversation_id,
            ConversationLog.user_id == current_user,
        ).first()
        if not log:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        new_transcript = payload.get("transcript", "").strip()
        if not new_transcript:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")
        
        log.transcript = new_transcript
        
        # Re-run synthesis on edited transcript so insights reflect the correction
        if payload.get("reanalyze", True):
            # 1. Re-run local NLP for updated entities and sentiment
            nlp_res = EXPERT.process(new_transcript)
            # 2. Re-run LLM synthesis for strategic report
            analysis = SYNTHESIS.analyze(
                new_transcript,
                nlp_res.get("entities", []),
                nlp_res.get("financial_sentiment", "Neutral")
            )
            
            # Update DB with high-fidelity insights
            log.executive_summary = analysis.get("executive_summary", log.executive_summary)
            log.strategic_intent = analysis.get("strategic_intent", log.strategic_intent)
            log.future_gearing = analysis.get("future_gearing", log.future_gearing)
            log.risk_assessment = analysis.get("risk_assessment", log.risk_assessment)
            log.expert_reasoning = analysis.get("expert_reasoning_points", log.expert_reasoning)
            log.financial_topic = nlp_res.get("topic", log.financial_topic)
            log.financial_sentiment = nlp_res.get("financial_sentiment", log.financial_sentiment)
            import json
            setattr(log, "entities", json.dumps(nlp_res.get("entities", [])))

        db.commit()
        db.refresh(log)
        return {
            "status": "updated",
            "reanalyzed": payload.get("reanalyze", True),
            "conversation": {
                "conversation_id": log.id,
                "chat_thread_id": getattr(log, "chat_thread_id", None) or log.id,
                "timestamp": log.timestamp,
                "financial_topic": log.financial_topic,
                "risk_level": log.risk_level,
                "financial_sentiment": log.financial_sentiment,
                "confidence_score": log.confidence_score,
                "executive_summary": log.executive_summary,
                "transcript": log.transcript,
                "strategic_intent": log.strategic_intent,
                "future_gearing": log.future_gearing,
                "risk_assessment": log.risk_assessment,
                "expert_reasoning_points": log.expert_reasoning,
                "timing": json.loads(log.timing_data or "{}"),
                "entities": json.loads(log.entities or "[]"),
            },
        }
    finally:
        db.close()

@app.get("/api/conversations/context")
def get_contextual_insights(current_user: str = Depends(get_current_user)):
    """Task 6 bonus: Cross-conversation contextual insights for longitudinal financial tracking."""
    history = get_all_conversations(user_id=current_user)
    results = history.get("results", [])
    if not results:
        return {"contextual_insights": [], "patterns": []}
    
    # Analyze patterns across history
    topics = {}
    risk_levels = []
    total = len(results)
    
    for r in results:
        t = r.get("financial_topic", "general")
        topics[t] = topics.get(t, 0) + 1
        risk_levels.append(r.get("risk_level", "LOW"))
    
    top_topic = max(topics.items(), key=lambda item: item[1])[0] if topics else "general"
    high_risk_count = sum(1 for r in risk_levels if r in ["HIGH", "CRITICAL"])
    
    patterns = []
    if high_risk_count > total * 0.4:
        patterns.append("Repeated high-risk financial discussions detected — recommend formal risk profiling")
    if topics.get("loan", 0) > 2:
        patterns.append(f"Multiple loan discussions ({topics['loan']}) observed — debt servicing optimization could be relevant")
    if topics.get("investment", 0) > 2:
        patterns.append(f"Consistent focus on capital growth ({topics['investment']} sessions) — portfolio diversification analysis suggested")
    
    return {
        "total_conversations": total,
        "dominant_topic": top_topic,
        "topic_distribution": topics,
        "high_risk_frequency": f"{round((high_risk_count/total)*100, 1)}%",
        "contextual_patterns": patterns,
        "metadata": {"analysis_engine": "FinFlux V4.2+ Strategic Pulse"}
    }

@app.post("/api/reminders")
async def create_reminder(payload: dict, current_user: str = Depends(get_current_user)):
    """Bonus PS: Personalised financial reminders from conversation."""
    import uuid, datetime
    db = SessionLocal()
    try:
        reminder = FinancialReminder(
            id=f"rem_{uuid.uuid4().hex[:8]}",
            user_id=current_user,
            conversation_id=payload.get("conversation_id", ""),
            reminder_text=payload.get("text", ""),
            due_date=payload.get("due_date", ""),
            topic=payload.get("topic", "general"),
            created_at=str(datetime.datetime.utcnow()),
            is_done=False
        )
        db.add(reminder)
        db.commit()
        return {"status": "created", "id": reminder.id}
    finally:
        db.close()

@app.get("/api/reminders")
def get_reminders(current_user: str = Depends(get_current_user)):
    """Bonus PS: Retrieve all active financial reminders for the client profile."""
    db = SessionLocal()
    try:
        rows = db.query(FinancialReminder).filter(
            FinancialReminder.user_id == current_user,
            FinancialReminder.is_done == False
        ).order_by(FinancialReminder.due_date).all()
        return {"reminders": [
            {"id": r.id, "text": r.reminder_text, 
             "due_date": r.due_date, "topic": r.topic}
            for r in rows
        ]}
    finally:
        db.close()

@app.post("/api/search/semantic")
async def semantic_search(payload: dict = Body(...), current_user: str = Depends(get_current_user)):
    """AI Memory Search: Retrieve conversations by meaning, intent, and context."""
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    filters = payload.get("filters", {})
    results = search_memories(user_id=current_user, query_text=query, filters=filters, n_results=10)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
