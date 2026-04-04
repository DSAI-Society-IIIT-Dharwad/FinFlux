"""FinFlux Pro Server V4.2: 8-Stage Modular Pipeline (Production Expert Module)."""
import os
import importlib
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
sys.path.insert(0, str(ROOT_DIR / "api"))

from finflux.modules.insight_engine.llm_adapters import ExpertSynthesisEngine
from finflux.modules.insight_engine.financial_models import ProductionExpertModule
from api.security import FinFluxSecurity
from api.storage import save_conversation, get_all_conversations

rag_mod = importlib.import_module("rag.router")
rag_router = rag_mod.rag_router
index_result_for_rag = rag_mod.index_result_for_rag
reindex_updated_conversation = rag_mod.reindex_updated_conversation
rag_is_available = rag_mod.rag_is_available

app = FastAPI(title="FinFlux Pro", version="4.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(rag_router)

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
        index_result_for_rag(result)
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
        reindex_updated_conversation(log)

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
        "rag_available": rag_is_available(),
        "models": ["LangDetect", "FinBERT", "GLiNER", "DeBERTa", "Qwen", "Llama-3.1"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
