from sqlalchemy import create_engine, Column, String, Float, Boolean, Text, DateTime, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import json
import os
from pathlib import Path
import bcrypt
import requests
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "analytics.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Embedding Model for Supabase vector RPC query vectors
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_VECTOR_RPC = os.environ.get("SUPABASE_VECTOR_RPC", "search_user_embeddings_bridge_service")
SUPABASE_CONV_THREADS_TABLE = os.environ.get("SUPABASE_CONV_THREADS_TABLE", "ai_conversation_threads")
SUPABASE_CONV_MESSAGES_TABLE = os.environ.get("SUPABASE_CONV_MESSAGES_TABLE", "ai_conversation_messages")
SUPABASE_EMBEDDINGS_TABLE = os.environ.get("SUPABASE_EMBEDDINGS_TABLE", "ai_message_embeddings")


def _supabase_vector_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def _supabase_conversation_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_CONV_THREADS_TABLE and SUPABASE_CONV_MESSAGES_TABLE)


def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _supabase_request(method: str, path: str, payload: Optional[Any] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    if not SUPABASE_URL:
        raise RuntimeError("Supabase URL not configured")
    url = f"{SUPABASE_URL}{path}"
    res = requests.request(method=method.upper(), url=url, headers=_supabase_headers(), json=payload, params=params, timeout=20)
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase request failed ({res.status_code}): {res.text}")
    if not res.text:
        return None
    try:
        return res.json()
    except Exception:
        return None


def _supabase_vector_search(user_id: str, query_text: str, n_results: int = 5, thread_id: str = "") -> List[Dict[str, Any]]:
    if not _supabase_vector_enabled():
        return []

    query_vector = embed_model.encode(query_text).tolist()
    payload: Dict[str, Any] = {
        "query_embedding": query_vector,
        "p_user_id": str(user_id),
        "match_count": max(1, int(n_results)),
        "filter_thread_id": thread_id or None,
    }
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    rpc_candidates = [
        SUPABASE_VECTOR_RPC,
        "search_user_embeddings_bridge_service",
        "search_user_message_embeddings_service",
        "search_user_message_embeddings",
    ]

    rows: List[Dict[str, Any]] = []
    last_error: Optional[Exception] = None
    for rpc_name in rpc_candidates:
        url = f"{SUPABASE_URL}/rest/v1/rpc/{rpc_name}"
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=20)
            res.raise_for_status()
            parsed = res.json() if res.text else []
            rows = parsed if isinstance(parsed, list) else []
            break
        except Exception as exc:
            last_error = exc
            continue

    if not rows and last_error:
        raise last_error

    mapped: List[Dict[str, Any]] = []
    for row in rows:
        payload_obj = row.get("payload", {}) if isinstance(row, dict) else {}
        mapped.append({
            "conversation_id": row.get("message_id"),
            "timestamp": row.get("created_at"),
            "topic": payload_obj.get("financial_topic", "N/A"),
            "risk": payload_obj.get("risk_level", "LOW"),
            "executive_summary": payload_obj.get("executive_summary", ""),
            "similarity_score": round(float(row.get("similarity", 0.0)), 4),
            "entities": [],
        })
    return mapped

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

def _supabase_insert_conversation(data: Dict[str, Any]) -> None:
    user_id = str(data.get("user_id", "default_guest"))
    thread_id = str(data.get("chat_thread_id") or data.get("conversation_id"))
    conversation_id = str(data.get("conversation_id", ""))
    ts = str(data.get("timestamp") or datetime.datetime.utcnow().isoformat())

    # Upsert thread with explicit user filter guard fields.
    _supabase_request(
        "POST",
        f"/rest/v1/{SUPABASE_CONV_THREADS_TABLE}",
        payload={
            "id": thread_id,
            "user_id": user_id,
            "title": str(data.get("financial_topic", "Financial Discussion"))[:180],
            "last_message_at": ts,
        },
        params={"on_conflict": "id", "columns": "id,user_id,title,last_message_at"},
    )

    sequence_rows = _supabase_request(
        "GET",
        f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}",
        params={
            "select": "sequence_no",
            "thread_id": f"eq.{thread_id}",
            "user_id": f"eq.{user_id}",
            "order": "sequence_no.desc",
            "limit": 1,
        },
    ) or []
    next_seq = int(sequence_rows[0]["sequence_no"]) + 1 if sequence_rows else 1

    user_msg = {
        "thread_id": thread_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "role": "user",
        "input_mode": data.get("input_mode", "text"),
        "sequence_no": next_seq,
        "raw_user_input": data.get("raw_user_input", ""),
        "transcript": data.get("transcript", ""),
        "created_at": ts,
    }
    assistant_msg = {
        "thread_id": thread_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "role": "assistant",
        "input_mode": data.get("input_mode", "text"),
        "sequence_no": next_seq + 1,
        "raw_user_input": data.get("raw_user_input", ""),
        "transcript": data.get("transcript", ""),
        "executive_summary": data.get("executive_summary", ""),
        "strategic_intent": data.get("strategic_intent", ""),
        "future_gearing": data.get("future_gearing", ""),
        "risk_level": data.get("risk_level", "LOW"),
        "risk_assessment": data.get("risk_assessment", ""),
        "financial_topic": data.get("financial_topic", "N/A"),
        "financial_sentiment": data.get("financial_sentiment", "Neutral"),
        "confidence_score": data.get("confidence_score", 0.0),
        "model_attribution": data.get("model_attribution", {}),
        "expert_reasoning_points": data.get("expert_reasoning", ""),
        "entities": data.get("entities", []),
        "timing": data.get("timing", {}),
        "created_at": ts,
    }

    # PostgREST bulk insert requires identical keys across objects, so insert separately.
    _supabase_request("POST", f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}", payload=user_msg)
    assistant_rows = _supabase_request("POST", f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}", payload=assistant_msg) or []

    assistant_message_id = ""
    if isinstance(assistant_rows, list) and assistant_rows:
        assistant_message_id = str(assistant_rows[0].get("id") or "")

    # Write embedding row for semantic retrieval. Skip hard-fail if embedding write fails.
    if assistant_message_id:
        try:
            embedding_vec = embed_model.encode(
                f"{data.get('executive_summary', '')}\n{data.get('transcript', '')}"
            ).tolist()
            _supabase_request(
                "POST",
                f"/rest/v1/{SUPABASE_EMBEDDINGS_TABLE}",
                payload={
                    "message_id": assistant_message_id,
                    "user_id": user_id,
                    "embedding": embedding_vec,
                    "embedding_model": "all-MiniLM-L6-v2",
                },
                params={"on_conflict": "message_id"},
            )
        except Exception as exc:
            print(f"[Storage] Supabase embedding insert skipped: {exc}")


def save_conversation(data):
    """Save analysis to Supabase bridge tables; fallback to local SQLite if bridge is unavailable."""
    if _supabase_conversation_enabled():
        try:
            _supabase_insert_conversation(data)
            return
        except Exception as exc:
            print(f"[Storage] Supabase conversation insert failed, falling back to SQLite: {exc}")

    db = SessionLocal()
    user_id = data.get("user_id", "default_guest")
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
    finally:
        db.close()

def search_memories(user_id: str, query_text: str, filters: dict = None, n_results: int = 5):
    """Supabase vector RPC semantic retrieval with mandatory user_id gating."""
    thread_id = ""
    if filters and isinstance(filters, dict):
        thread_id = str(filters.get("thread_id", ""))

    try:
        if not _supabase_vector_enabled():
            return {"results": [], "error": "Supabase vector search not configured"}

        supa_results = _supabase_vector_search(user_id=user_id, query_text=query_text, n_results=n_results, thread_id=thread_id)
        return {"results": supa_results, "source": "supabase"}
    except Exception as e:
        print(f"[Storage] Supabase semantic search error: {e}")
        return {"results": [], "error": str(e)}

def _map_bridge_message_to_conversation(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "conversation_id": row.get("conversation_id") or row.get("id"),
        "user_id": row.get("user_id", ""),
        "timestamp": row.get("created_at", ""),
        "language": row.get("language", "unknown"),
        "financial_topic": row.get("financial_topic", "N/A"),
        "risk_level": row.get("risk_level", "LOW"),
        "financial_sentiment": row.get("financial_sentiment", "Neutral"),
        "executive_summary": row.get("executive_summary", ""),
        "summary": row.get("executive_summary", ""),
        "future_gearing": row.get("future_gearing", ""),
        "strategic_intent": row.get("strategic_intent", ""),
        "risk_assessment": row.get("risk_assessment", ""),
        "expert_reasoning": row.get("expert_reasoning_points", ""),
        "expert_reasoning_points": row.get("expert_reasoning_points", ""),
        "transcript": row.get("transcript", ""),
        "confidence_score": row.get("confidence_score", 0.0),
        "entities": row.get("entities") or [],
        "key_points": [],
        "timing": row.get("timing") or {},
        "chat_thread_id": row.get("thread_id") or row.get("conversation_id") or row.get("id"),
        "input_mode": row.get("input_mode") or "text",
        "raw_user_input": row.get("raw_user_input") or row.get("transcript") or "",
    }


def get_conversation_by_id(user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    if _supabase_conversation_enabled():
        try:
            rows = _supabase_request(
                "GET",
                f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}",
                params={
                    "select": "*",
                    "user_id": f"eq.{user_id}",
                    "conversation_id": f"eq.{conversation_id}",
                    "role": "eq.assistant",
                    "limit": 1,
                },
            ) or []
            if rows:
                return _map_bridge_message_to_conversation(rows[0])
        except Exception as exc:
            print(f"[Storage] Supabase get conversation failed: {exc}")

    db = SessionLocal()
    try:
        r = db.query(ConversationLog).filter(
            ConversationLog.id == conversation_id,
            ConversationLog.user_id == user_id,
        ).first()
        if not r:
            return None
        return {
            "conversation_id": r.id,
            "user_id": r.user_id,
            "timestamp": r.timestamp,
            "language": r.language,
            "financial_topic": r.financial_topic,
            "risk_level": r.risk_level,
            "financial_sentiment": r.financial_sentiment,
            "executive_summary": r.executive_summary,
            "summary": r.executive_summary,
            "future_gearing": r.future_gearing,
            "strategic_intent": r.strategic_intent,
            "risk_assessment": r.risk_assessment,
            "expert_reasoning": r.expert_reasoning,
            "expert_reasoning_points": r.expert_reasoning,
            "transcript": r.transcript,
            "confidence_score": r.confidence_score,
            "entities": json.loads(r.entities or "[]"),
            "key_points": json.loads(r.key_points or "[]"),
            "timing": json.loads(r.timing_data or "{}"),
            "chat_thread_id": r.chat_thread_id or r.id,
            "input_mode": r.input_mode or "text",
            "raw_user_input": r.raw_user_input or r.transcript,
        }
    finally:
        db.close()


def update_conversation(user_id: str, conversation_id: str, updates: Dict[str, Any]) -> bool:
    if _supabase_conversation_enabled():
        try:
            _supabase_request(
                "PATCH",
                f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}",
                payload=updates,
                params={
                    "user_id": f"eq.{user_id}",
                    "conversation_id": f"eq.{conversation_id}",
                    "role": "eq.assistant",
                },
            )
            return True
        except Exception as exc:
            print(f"[Storage] Supabase update conversation failed: {exc}")

    db = SessionLocal()
    try:
        log = db.query(ConversationLog).filter(
            ConversationLog.id == conversation_id,
            ConversationLog.user_id == user_id,
        ).first()
        if not log:
            return False

        field_map = {
            "transcript": "transcript",
            "executive_summary": "executive_summary",
            "financial_topic": "financial_topic",
            "financial_sentiment": "financial_sentiment",
            "strategic_intent": "strategic_intent",
            "future_gearing": "future_gearing",
            "risk_assessment": "risk_assessment",
            "expert_reasoning_points": "expert_reasoning",
            "entities": "entities",
            "timing": "timing_data",
        }
        for k, v in updates.items():
            if k not in field_map:
                continue
            attr = field_map[k]
            if k in {"entities", "timing"}:
                setattr(log, attr, json.dumps(v))
            else:
                setattr(log, attr, v)
        db.commit()
        return True
    finally:
        db.close()


def clear_user_history(user_id: str) -> Dict[str, int]:
    deleted = {"messages": 0, "threads": 0, "local_conversations": 0, "local_reminders": 0}

    if _supabase_conversation_enabled():
        try:
            # Read first for accurate count before deletion.
            msgs = _supabase_request(
                "GET",
                f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}",
                params={"select": "id", "user_id": f"eq.{user_id}"},
            ) or []
            ths = _supabase_request(
                "GET",
                f"/rest/v1/{SUPABASE_CONV_THREADS_TABLE}",
                params={"select": "id", "user_id": f"eq.{user_id}"},
            ) or []
            deleted["messages"] = len(msgs)
            deleted["threads"] = len(ths)

            _supabase_request("DELETE", f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}", params={"user_id": f"eq.{user_id}"})
            _supabase_request("DELETE", f"/rest/v1/{SUPABASE_CONV_THREADS_TABLE}", params={"user_id": f"eq.{user_id}"})
        except Exception as exc:
            print(f"[Storage] Supabase clear history failed: {exc}")

    db = SessionLocal()
    try:
        deleted["local_conversations"] = db.query(ConversationLog).filter(ConversationLog.user_id == user_id).delete()
        deleted["local_reminders"] = db.query(FinancialReminder).filter(FinancialReminder.user_id == user_id).delete()
        db.commit()
    finally:
        db.close()
    return deleted


def get_all_conversations(user_id: str = "guest_001"):
    """Retrieve strategic history for a specific user from Supabase bridge; fallback to SQLite."""
    if _supabase_conversation_enabled():
        try:
            rows = _supabase_request(
                "GET",
                f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}",
                params={
                    "select": "*",
                    "user_id": f"eq.{user_id}",
                    "role": "eq.assistant",
                    "order": "created_at.desc",
                },
            ) or []
            res = [_map_bridge_message_to_conversation(r) for r in rows]
            return {"count": len(res), "results": res}
        except Exception as exc:
            print(f"[Storage] Supabase list conversations failed, fallback to SQLite: {exc}")

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
                "summary": r.executive_summary,
                "future_gearing": r.future_gearing,
                "strategic_intent": r.strategic_intent,
                "risk_assessment": r.risk_assessment,
                "expert_reasoning": r.expert_reasoning,
                "expert_reasoning_points": r.expert_reasoning,
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
