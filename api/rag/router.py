from __future__ import annotations

import importlib
import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

storage_mod = importlib.import_module("api.storage")
service_mod = importlib.import_module("rag.service")
settings_mod = importlib.import_module("rag.settings")

get_all_conversations = storage_mod.get_all_conversations
FinancialRAGService = service_mod.FinancialRAGService
RAG_COLLECTION_NAME = settings_mod.RAG_COLLECTION_NAME
RAG_EMBEDDING_MODEL = settings_mod.RAG_EMBEDDING_MODEL

rag_router = APIRouter(prefix="/api/rag", tags=["rag"])
rag_service = FinancialRAGService()


def rag_is_available() -> bool:
    return rag_service.available


def index_result_for_rag(result: Dict[str, Any]) -> None:
    if not rag_service.available:
        return
    try:
        rag_service.index_conversation(result)
    except Exception as exc:
        print(f"[RAG] index failed for {result.get('conversation_id', 'unknown')}: {exc}")


def reindex_updated_conversation(log: Any) -> None:
    if not rag_service.available:
        return

    refreshed = {
        "conversation_id": getattr(log, "id", ""),
        "timestamp": getattr(log, "timestamp", ""),
        "language": getattr(log, "language", ""),
        "financial_topic": getattr(log, "financial_topic", ""),
        "risk_level": getattr(log, "risk_level", "LOW"),
        "financial_sentiment": getattr(log, "financial_sentiment", "neutral"),
        "executive_summary": getattr(log, "executive_summary", ""),
        "summary": getattr(log, "executive_summary", ""),
        "transcript": getattr(log, "transcript", ""),
        "entities": json.loads(getattr(log, "entities", "[]") or "[]"),
    }
    try:
        rag_service.index_conversation(refreshed)
    except Exception as exc:
        print(f"[RAG] reindex failed for {refreshed.get('conversation_id', 'unknown')}: {exc}")


@rag_router.get("/status")
def status() -> Dict[str, Any]:
    return {
        "available": rag_service.available,
        "collection": RAG_COLLECTION_NAME,
        "embedding_model": RAG_EMBEDDING_MODEL,
    }


@rag_router.post("/reindex-all")
def reindex_all() -> Dict[str, Any]:
    if not rag_service.available:
        raise HTTPException(status_code=503, detail="RAG is not available. Install packages from api/rag/requirements.txt")

    history = get_all_conversations().get("results", [])
    outcome = rag_service.index_many(history)
    return {"status": "ok", **outcome}


@rag_router.post("/query")
def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not rag_service.available:
        raise HTTPException(status_code=503, detail="RAG is not available. Install packages from api/rag/requirements.txt")

    user_query = str(payload.get("query", "")).strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = int(payload.get("top_k", 3) or 3)
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else None
    return rag_service.query(user_query=user_query, top_k=top_k, filters=filters)


@rag_router.post("/insight")
def insight(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not rag_service.available:
        raise HTTPException(status_code=503, detail="RAG is not available. Install packages from api/rag/requirements.txt")

    user_query = str(payload.get("query", "")).strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query is required")

    top_k = int(payload.get("top_k", 3) or 3)
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else None
    fin_sentiment = str(payload.get("financial_sentiment", "Neutral"))

    retrieved = rag_service.query(user_query=user_query, top_k=top_k, filters=filters)
    context_block = rag_service.build_context_block(retrieved)
    generated = rag_service.generate_contextual_insight(
        user_query=user_query,
        retrieved_context=context_block,
        fin_sentiment=fin_sentiment,
    )

    return {
        "query": user_query,
        "retrieval": retrieved,
        "rag_context": context_block,
        "insight": generated,
    }
