"""FinFlux SQL Persistence V4.2+: High-Resolution McKinsey-Style History."""
from sqlalchemy import create_engine, Column, String, Float, Boolean, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import json
from pathlib import Path

# Paths
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "analytics.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ConversationLog(Base):
    """V4.2 Log: Fully mapped Strategic Financial Discovery."""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    timestamp = Column(String)
    language = Column(String)
    financial_topic = Column(String)
    risk_level = Column(String)
    financial_sentiment = Column(String)
    
    # McKinsey Gating Fields
    executive_summary = Column(Text)
    future_gearing = Column(Text)
    strategic_intent = Column(Text)
    risk_assessment = Column(Text)
    expert_reasoning = Column(Text)
    transcript = Column(Text) # High-resolution persistence
    
    advice_request = Column(Boolean)
    injection_attempt = Column(Boolean)
    confidence_score = Column(Float)
    
    # Serialized JSON blobs
    entities = Column(Text) # JSON List
    key_points = Column(Text) # JSON List
    timing_data = Column(Text) # JSON Dict

Base.metadata.create_all(bind=engine)

def save_conversation(data):
    """Save multi-model strategic analysis to SQL."""
    db = SessionLocal()
    try:
        log = ConversationLog(
            id=data["conversation_id"],
            timestamp=data["timestamp"],
            language=data.get("language", "unknown"),
            financial_topic=data["financial_topic"],
            risk_level=data["risk_level"],
            financial_sentiment=data.get("financial_sentiment", "Neutral"),
            
            executive_summary=data.get("executive_summary", ""),
            future_gearing=data.get("future_gearing", ""),
            strategic_intent=data.get("strategic_intent", ""),
            risk_assessment=data.get("risk_assessment", ""),
            expert_reasoning=data.get("expert_reasoning", ""),
            transcript=data.get("transcript", ""),
            
            advice_request=data["advice_request"],
            injection_attempt=data["injection_attempt"],
            confidence_score=data["confidence_score"],
            
            entities=json.dumps(data.get("entities", [])),
            key_points=json.dumps(data.get("key_points", [])),
            timing_data=json.dumps(data.get("timing", {}))
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

def get_all_conversations():
    """Retrieve full McKinsey-style strategic history."""
    db = SessionLocal()
    try:
        rows = db.query(ConversationLog).order_by(ConversationLog.timestamp.desc()).all()
        res = []
        for r in rows:
            res.append({
                "conversation_id": r.id,
                "timestamp": r.timestamp,
                "language": r.language,
                "financial_topic": r.financial_topic,
                "risk_level": r.risk_level,
                "financial_sentiment": r.financial_sentiment,
                "executive_summary": r.executive_summary,
                "summary": r.executive_summary, # Frontend Legacy Support
                "future_gearing": r.future_gearing,
                "strategic_intent": r.strategic_intent,
                "risk_assessment": r.risk_assessment,
                "expert_reasoning": r.expert_reasoning,
                "expert_reasoning_points": r.expert_reasoning, # Frontend Consistency
                "transcript": r.transcript,
                "advice_request": r.advice_request,
                "injection_attempt": r.injection_attempt,
                "confidence_score": r.confidence_score,
                "entities": json.loads(r.entities or "[]"),
                "key_points": json.loads(r.key_points or "[]"),
                "timing": json.loads(r.timing_data or "{}")
            })
        return {"count": len(res), "results": res}
    finally:
        db.close()
