from sqlalchemy import create_engine, Column, String, Float, Boolean, Text, DateTime, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import json
import os
from pathlib import Path
import bcrypt

# AI Memory System (ChromaDB + Sentence Transformers)
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "analytics.db"
CHROMA_PATH = DATA_DIR / "chroma_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if HAS_CHROMA:
    # Initialize Embedding Model (MiniLM is fast and efficient)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name="financial_memories")

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ConversationLog(Base):
    """V4.2 Log: Fully mapped Strategic Financial Discovery."""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True) # Added for Multi-Tenancy
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
    chat_thread_id = Column(String, index=True)
    input_mode = Column(String) # text|audio
    raw_user_input = Column(Text)

class FinancialReminder(Base):
    """Bonus: Personalised financial reminders from conversation history."""
    __tablename__ = "reminders"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    conversation_id = Column(String)
    reminder_text = Column(Text)
    due_date = Column(String)
    topic = Column(String)
    created_at = Column(String)
    is_done = Column(Boolean, default=False)

class UserAccount(Base):
    """Application user account for JWT authentication."""
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(String)

# Ensure all tables are created
Base.metadata.create_all(bind=engine)


def _ensure_sqlite_column(table_name: str, column_name: str, column_def: str):
    with engine.connect() as conn:
        cols = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        existing = {str(row[1]) for row in cols}
        if column_name not in existing:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"))
            conn.commit()


def _run_schema_migrations():
    # Keep backwards compatibility for existing analytics.db created before thread-aware chat.
    _ensure_sqlite_column("conversations", "chat_thread_id", "TEXT")
    _ensure_sqlite_column("conversations", "input_mode", "TEXT")
    _ensure_sqlite_column("conversations", "raw_user_input", "TEXT")


_run_schema_migrations()

def get_user_by_username(username: str):
    db = SessionLocal()
    try:
        return db.query(UserAccount).filter(UserAccount.username == username).first()
    finally:
        db.close()

def create_user(username: str, password: str):
    db = SessionLocal()
    try:
        exists = db.query(UserAccount).filter(UserAccount.username == username).first()
        if exists:
            return None

        pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user = UserAccount(
            id=f"usr_{os.urandom(6).hex()}",
            username=username,
            password_hash=pwd_hash,
            created_at=str(datetime.datetime.utcnow())
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

def authenticate_user(username: str, password: str):
    db = SessionLocal()
    try:
        user = db.query(UserAccount).filter(UserAccount.username == username).first()
        if not user:
            return None
        if not bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
            return None
        return user
    finally:
        db.close()

def save_conversation(data):
    """Save multi-model strategic analysis to SQL and isolated ChromaDB memory."""
    db = SessionLocal()
    user_id = data.get("user_id", "default_guest") # Multi-tenant gating
    try:
        log = ConversationLog(
            id=data["conversation_id"],
            user_id=user_id,
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
            timing_data=json.dumps(data.get("timing", {})),
            chat_thread_id=data.get("chat_thread_id", data.get("conversation_id")),
            input_mode=data.get("input_mode", "text"),
            raw_user_input=data.get("raw_user_input", data.get("transcript", "")),
        )
        db.add(log)
        db.commit()

        # Update Semantic Memory (Isolated per user_id)
        if HAS_CHROMA:
            try:
                combined_text = f"{data.get('executive_summary', '')} {data.get('transcript', '')} "
                combined_text += " ".join([f"{e['type']}:{e['value']}" for e in data.get("entities", [])])
                
                collection.add(
                    documents=[combined_text.strip()],
                    metadatas=[{
                        "user_id": user_id, # Strict Ownership
                        "risk_level": data["risk_level"],
                        "timestamp": data["timestamp"],
                        "topic": data["financial_topic"],
                        "language": data.get("language", "unknown"),
                        "sentiment": data.get("financial_sentiment", "Neutral")
                    }],
                    ids=[f"{user_id}_{data['conversation_id']}"] # Unique Global ID
                )
            except Exception as e:
                print(f"[Storage] Multi-tenant ChromaDB indexing error: {e}")
    finally:
        db.close()

def search_memories(user_id: str, query_text: str, filters: dict = None, n_results: int = 5):
    """Secure, user-aware semantic retrieval with mandatory user_id gating."""
    if not HAS_CHROMA:
        return {"error": "Semantic search disabled"}
    
    try:
        # Mandatory Tenant Filter
        where_filter = {"user_id": user_id}
        if filters:
            if filters.get("risk"): where_filter["risk_level"] = filters["risk"]
            if filters.get("topic"): where_filter["topic"] = filters["topic"]
            if filters.get("language"): where_filter["language"] = filters["language"]

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        
        db = SessionLocal()
        final_results = []
        for i, full_id in enumerate(results["ids"][0]):
            inner_id = full_id.split('_', 1)[1] if '_' in full_id else full_id
            log = db.query(ConversationLog).filter(
                ConversationLog.id == inner_id,
                ConversationLog.user_id == user_id # Extra SQL-level guardrail
            ).first()
            if log:
                final_results.append({
                    "conversation_id": log.id,
                    "timestamp": log.timestamp,
                    "topic": log.financial_topic,
                    "risk": log.risk_level,
                    "language": log.language,
                    "executive_summary": log.executive_summary,
                    "similarity_score": round(1 - results["distances"][0][i], 2),
                    "entities": json.loads(log.entities or "[]")
                })
        db.close()
        return {"results": final_results}
    except Exception as e:
        print(f"[Storage] Secure semantic search error: {e}")
        return {"results": [], "error": str(e)}

def get_all_conversations(user_id: str = "guest_001"):
    """Retrieve full McKinsey-style strategic history for a specific tenant."""
    db = SessionLocal()
    try:
        rows = db.query(ConversationLog).filter(
            ConversationLog.user_id == user_id
        ).order_by(ConversationLog.timestamp.desc()).all()
        res = []
        for r in rows:
            res.append({
                "conversation_id": r.id,
                "user_id": r.user_id,
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
                "timing": json.loads(r.timing_data or "{}"),
                "chat_thread_id": r.chat_thread_id or r.id,
                "input_mode": r.input_mode or "text",
                "raw_user_input": r.raw_user_input or r.transcript,
            })
        return {"count": len(res), "results": res}
    finally:
        db.close()
