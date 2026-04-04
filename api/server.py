"""FinFlux Pro Server V4.2: 8-Stage Modular Pipeline (Production Expert Module)."""
import os
import shutil
import uuid
import time
import json
import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from api.storage import save_conversation, get_all_conversations

app = FastAPI(title="FinFlux Pro", version="4.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Secure Directories
STORAGE_DIR = ROOT_DIR / "data" / "encrypted_audio"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Init Expert Modules Once (Singleton Pattern)
SECURITY = FinFluxSecurity()
EXPERT = ProductionExpertModule()
SYNTHESIS = ExpertSynthesisEngine()

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
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
        # ── Stage 1: Multilingual ASR (Groq Whisper Turbo) ──
        from finflux.modules.insight_engine.llm_adapters import GroqWhisperAdapter
        whisper = GroqWhisperAdapter()
        asr = whisper.transcribe(temp_wav)
        raw_text = asr.get("text", "")
        t_asr = time.time() - t_start

        # ── Stage 2: Normalization (Llama 8B) ──
        t2_start = time.time()
        clean_text = SYNTHESIS.normalize_transcript(raw_text)
        t_norm = time.time() - t2_start

        # ── Stage 3-6: Advanced Local NLP Stack ──
        # (XLM-Roberta Lang Detect, DeBERTa Topic, FinBERT Sentiment, GLiNER NER)
        t3_start = time.time()
        nlp_res = EXPERT.process(clean_text)
        t_nlp = time.time() - t3_start

        # ── Stage 7-8: Augmented Expert Reasoning & Synthesis (Qwen + Llama 70B) ──
        t4_start = time.time()
        analysis = SYNTHESIS.analyze(clean_text, nlp_res.get("entities", []), nlp_res.get("financial_sentiment", "Neutral"))
        t_syn = time.time() - t4_start

        # ── PII & Security Final Check ──
        injection_found = SECURITY.detect_injection(clean_text)
        
        # McKinsey V4.2+ Higher-Resolution Mapping
        exec_summary = analysis.get("executive_summary", "")
        safe_summary = SECURITY.mask_pii(exec_summary)
        
        result = {
            "conversation_id": call_id,
            "timestamp": str(datetime.datetime.utcnow()),
            "language": nlp_res.get("detected_language", asr.get("language", "unknown")),
            "financial_topic": nlp_res.get("topic", "N/A"),
            "strategic_intent": analysis.get("strategic_intent", ""),
            "risk_level": analysis.get("risk_level", "LOW"),
            "financial_sentiment": nlp_res.get("financial_sentiment", "Neutral"),
            "advice_request": nlp_res.get("is_advice_request", False) or SECURITY.is_asking_for_advice(clean_text),
            "injection_attempt": injection_found,
            "entities": nlp_res.get("entities", []),
            "executive_summary": safe_summary,
            "summary": safe_summary, # Legacy field support
            "future_gearing": analysis.get("future_gearing", ""),
            "risk_assessment": analysis.get("risk_assessment", ""),
            "key_insights": [SECURITY.mask_pii(p) for p in analysis.get("key_insights", [])],
            "key_points": [SECURITY.mask_pii(p) for p in analysis.get("key_insights", [])], # Alias
            "transcript": clean_text,
            "expert_reasoning_points": analysis.get("expert_reasoning_points", ""),
            "expert_reasoning": analysis.get("expert_reasoning_points", ""), # Storage Alias
            "confidence_score": nlp_res.get("confidence_score", 0.0),
            "timing": {
                "asr_s": round(t_asr, 2), "normalization_s": round(t_norm, 2),
                "expert_nlp_s": round(t_nlp, 2), "synthesis_s": round(t_syn, 2),
                "total_s": round(time.time() - t_start, 2)
            }
        }

        # Save to DB
        save_conversation(result)
        return result

    except Exception as e:
        print(f"[Server] Pipeline crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

@app.get("/api/results")
def list_results():
    return get_all_conversations()

@app.put("/api/update/{conversation_id}")
async def update_conversation(conversation_id: str, payload: dict):
    """Update stored conversation intelligence (e.g. edited transcript)."""
    from api.storage import SessionLocal, ConversationLog
    db = SessionLocal()
    try:
        log = db.query(ConversationLog).filter(ConversationLog.id == conversation_id).first()
        if not log: raise HTTPException(status_code=404, detail="Not found")
        
        if "transcript" in payload: log.transcript = payload["transcript"]
        if "executive_summary" in payload: log.executive_summary = payload["executive_summary"]
        if "summary" in payload: log.executive_summary = payload["summary"] # Alias
        
        db.commit()
        return {"status": "updated"}
    finally:
        db.close()

@app.get("/api/report/{conversation_id}")
def generate_report(conversation_id: str, format: str = "pdf"):
    """Generate a high-fidelity financial analysis report."""
    history = get_all_conversations()
    data = next((c for c in history["results"] if c["conversation_id"] == conversation_id), None)
    if not data: raise HTTPException(status_code=404, detail="Analysis not found")

    if format == "csv":
        import pandas as pd
        from fastapi.responses import Response
        df = pd.DataFrame([data])
        return Response(content=df.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=finflux_{conversation_id}.csv"})
    
    # PDF Generation
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    import io
    from fastapi.responses import StreamingResponse

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=LETTER)
    w, h = LETTER

    # Header
    p.setFont("Helvetica-Bold", 24)
    p.drawString(50, h - 50, "FinFlux Intelligence Report")
    p.setFont("Helvetica", 10)
    p.drawString(50, h - 70, f"ID: {conversation_id} | Timestamp: {data['timestamp']}")
    p.line(50, h - 80, w - 50, h - 80)

    # Executive Summary
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, h - 110, "EXECUTIVE SUMMARY")
    p.setFont("Helvetica", 11)
    summary_text = data.get("executive_summary", data.get("summary", ""))
    p.drawString(50, h - 130, summary_text[:120] + "...")
    
    # McKinsey High-Fidelity Sections
    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(colors.blue)
    p.drawString(50, h - 160, "STRATEGIC INTENT")
    p.setFont("Helvetica", 10)
    p.setFillColor(colors.black)
    p.drawString(50, h - 175, data.get("strategic_intent", "N/A")[:100])

    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(colors.purple)
    p.drawString(50, h - 200, "FUTURE GEARING")
    p.setFont("Helvetica", 10)
    p.setFillColor(colors.black)
    p.drawString(50, h - 215, data.get("future_gearing", "N/A")[:100])

    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(colors.red)
    p.drawString(50, h - 240, "RISK ASSESSMENT")
    p.setFont("Helvetica", 10)
    p.setFillColor(colors.black)
    p.drawString(50, h - 255, data.get("risk_assessment", "N/A")[:100])
    
    # Insights
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, h - 290, f"Topic: {data['financial_topic']}")
    p.drawString(50, h - 310, f"Sentiment: {data['financial_sentiment']}")
    p.drawString(50, h - 330, f"Risk Level: {data['risk_level']}")
    
    # Reasoning
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, h - 360, "EXPERT ANALYST REASONING")
    p.setFont("Helvetica", 9)
    p.drawString(50, h - 380, data.get("expert_reasoning", "")[:500])

    p.showPage()
    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=finflux_{conversation_id}.pdf"})

@app.get("/api/health")
def health():
    key_ok = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "finflux-v4.2-online", 
        "security": "AES-256 enabled",
        "api_key_loaded": key_ok,
        "models": ["LangDetect", "FinBERT", "GLiNER", "DeBERTa", "Qwen", "Llama-3.1"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
