from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, cast

import requests

from finflux import config

from .settings import (
    RAG_CHROMA_DIR,
    RAG_COLLECTION_NAME,
    RAG_EMBEDDING_MODEL,
    RAG_GROQ_INSIGHT_MODEL,
    RAG_GROQ_QUERY_MODEL,
    RAG_TOP_K,
)


@dataclass
class QueryPlan:
    semantic_query: str
    filters: Dict[str, Any]
    mode: str
    top_k: int


class FinancialRAGService:
    """All RAG responsibilities are isolated here: index, retrieve, and generate."""

    def __init__(self) -> None:
        self.available = False
        self._collection: Any | None = None
        self._api_key = config.GROQ_API_KEY
        self._llm_url = "https://api.groq.com/openai/v1/chat/completions"

        try:
            chromadb = importlib.import_module("chromadb")
            embedding_module = importlib.import_module("chromadb.utils.embedding_functions")
            SentenceTransformerEmbeddingFunction = getattr(
                embedding_module,
                "SentenceTransformerEmbeddingFunction",
            )
        except ImportError:
            return

        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=RAG_EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=RAG_CHROMA_DIR)
        self._collection = client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.available = True

    def _to_epoch(self, ts: str | None) -> int:
        if not ts:
            return int(datetime.now(tz=timezone.utc).timestamp())
        try:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
        except Exception:
            return int(datetime.now(tz=timezone.utc).timestamp())

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    def _entity_value(self, entities: List[Dict[str, Any]], labels: set[str]) -> str:
        for entity in entities:
            etype = str(entity.get("type", "")).upper()
            if etype in labels:
                return str(entity.get("value", "")).strip()
        return ""

    def _extract_metadata(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        entities = conversation.get("entities", []) or []
        return {
            "conversation_id": str(conversation.get("conversation_id", "")),
            "risk_level": str(conversation.get("risk_level", "LOW")).upper(),
            "topic": str(conversation.get("financial_topic", "general")).lower(),
            "language": str(conversation.get("language", "unknown")).lower(),
            "sentiment": str(conversation.get("financial_sentiment", "neutral")).lower(),
            "institution": self._entity_value(entities, {"BANK", "ORGANIZATION", "INSTITUTION"}),
            "product": self._entity_value(
                entities,
                {"LOAN", "EMI", "INSURANCE", "MUTUAL_FUND", "STOCK", "GOLD", "PROPERTY", "INVESTMENT"},
            ),
            "amount": self._entity_value(entities, {"AMOUNT"}),
            "timestamp": str(conversation.get("timestamp", "")),
            "timestamp_epoch": self._to_epoch(conversation.get("timestamp")),
        }

    def _semantic_payload(self, conversation: Dict[str, Any]) -> str:
        summary = conversation.get("executive_summary") or conversation.get("summary") or ""
        transcript = conversation.get("transcript") or ""
        return (
            "Executive Summary:\n"
            f"{summary.strip()}\n\n"
            "Transcript:\n"
            f"{transcript.strip()}"
        ).strip()

    def index_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return {"indexed": False, "reason": "RAG dependencies missing"}

        conv_id = str(conversation.get("conversation_id", "")).strip()
        if not conv_id:
            return {"indexed": False, "reason": "conversation_id is required"}

        collection = cast(Any, self._collection)
        collection.upsert(
            ids=[conv_id],
            documents=[self._semantic_payload(conversation)],
            metadatas=[self._extract_metadata(conversation)],
        )
        return {"indexed": True, "conversation_id": conv_id}

    def index_many(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        indexed = 0
        for conversation in conversations:
            if self.index_conversation(conversation).get("indexed"):
                indexed += 1
        return {"indexed": indexed, "total": len(conversations)}

    def _llm_query_plan(self, query: str, top_k: int) -> QueryPlan:
        if not self._api_key:
            return QueryPlan(semantic_query=query, filters={}, mode="semantic", top_k=top_k)

        system_prompt = (
            "Convert user financial query into retrieval plan JSON. Return JSON only with keys: "
            "semantic_query (string), filters (object), mode (semantic|metadata_only|hybrid), top_k (int). "
            "Metadata fields: risk_level, language, topic, institution, sentiment, timestamp_epoch. "
            "Use operators: $eq, $gt, $gte, $lt, $lte, $in."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"top_k: {top_k}\n"
            "metadata_only=hard filter only. semantic=meaning search only. hybrid=both."
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": RAG_GROQ_QUERY_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(self._llm_url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            parsed = self._extract_json(response.json()["choices"][0]["message"]["content"])
            return QueryPlan(
                semantic_query=str(parsed.get("semantic_query", query)).strip() or query,
                filters=parsed.get("filters", {}) if isinstance(parsed.get("filters", {}), dict) else {},
                mode=str(parsed.get("mode", "semantic")).lower(),
                top_k=int(parsed.get("top_k", top_k) or top_k),
            )
        except Exception:
            return QueryPlan(semantic_query=query, filters={}, mode="semantic", top_k=top_k)

    def _metadata_rows(self, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        collection = cast(Any, self._collection)
        data = collection.get(where=filters if filters else None, limit=top_k, include=["documents", "metadatas"])
        rows: List[Dict[str, Any]] = []
        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []

        for idx, conv_id in enumerate(ids):
            rows.append(
                {
                    "conversation_id": conv_id,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "document": docs[idx] if idx < len(docs) else "",
                    "score": 1.0,
                }
            )
        return rows

    def _vector_rows(self, semantic_query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        collection = cast(Any, self._collection)
        # Wrap multiple filters in $and for ChromaDB compatibility
        if filters and len(filters) > 1:
            chroma_filter = {"$and": [{k: v} for k, v in filters.items()]}
        elif filters and len(filters) == 1:
            chroma_filter = filters
        else:
            chroma_filter = None

        data = collection.query(
            query_texts=[semantic_query],
            n_results=top_k,
            where=chroma_filter if chroma_filter else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (data.get("ids") or [[]])[0]
        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]

        rows: List[Dict[str, Any]] = []
        for idx, conv_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else 0.0
            rows.append(
                {
                    "conversation_id": conv_id,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "document": docs[idx] if idx < len(docs) else "",
                    "score": max(0.0, round(1.0 - float(distance), 4)),
                }
            )
        return rows

    def query(self, user_query: str, top_k: int | None = None, filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "query": user_query, "results": [], "reason": "RAG dependencies missing"}

        k = int(top_k or RAG_TOP_K)
        # Always use semantic search without LLM-generated filters to avoid metadata mismatch
        plan = QueryPlan(semantic_query=user_query, filters=filters or {}, mode="semantic", top_k=k)

        if plan.mode == "metadata_only":
            rows = self._metadata_rows(plan.filters, plan.top_k)
        elif plan.mode == "hybrid":
            rows = self._vector_rows(plan.semantic_query, plan.filters, plan.top_k)
        else:
            rows = self._vector_rows(plan.semantic_query, {}, plan.top_k)

        return {
            "available": True,
            "plan": {
                "semantic_query": plan.semantic_query,
                "filters": plan.filters,
                "mode": plan.mode,
                "top_k": plan.top_k,
            },
            "results": rows,
        }

    def build_context_block(self, retrieved: Dict[str, Any]) -> str:
        rows = retrieved.get("results", []) or []
        if not rows:
            return "No historical sessions retrieved."

        lines: List[str] = []
        for row in rows:
            metadata = row.get("metadata", {})
            doc = row.get("document", "")
            summary = ""
            if "Executive Summary:" in doc:
                summary = doc.split("Executive Summary:", 1)[1].split("Transcript:", 1)[0].strip()
            lines.append(
                f"[Date: {metadata.get('timestamp', 'N/A')}] "
                f"[Summary: {summary or 'N/A'}] "
                f"[Risk: {metadata.get('risk_level', 'UNKNOWN')}] "
                f"[Topic: {metadata.get('topic', 'unknown')}]"
            )
        return "\n".join(lines)

    def generate_contextual_insight(self, user_query: str, retrieved_context: str, fin_sentiment: str = "Neutral") -> Dict[str, Any]:
        if not self._api_key:
            return {
                "contextual_insight": "GROQ_API_KEY missing; generation unavailable.",
                "pattern_summary": [],
                "risk_trajectory": "unknown",
                "recommended_next_focus": [],
                "confidence": "low",
            }

        system_prompt = """You are FinFlux ARMOR financial insight engine.
Use retrieved history as factual memory.
RULES:
1) No financial advice.
2) Use neutral, strategic language.
    3) Ground every claim in retrieved sessions.
    4) Match the response language to the user's query language.
    5) Return strict JSON.
6) Be specific and substantial, not generic.
7) Mention concrete evidence from retrieved sessions (for example date, risk, topic).

JSON SCHEMA:
{
    "contextual_insight": "120-200 words with evidence-backed synthesis from history.",
    "pattern_summary": ["4-8 bullets of recurring behaviors/risks with specifics."],
    "risk_trajectory": "2-4 sentence trend summary across retrieved sessions.",
    "recommended_next_focus": ["3-6 concrete checkpoints/questions for next interaction."],
  "confidence": "low|medium|high"
}
"""
        user_prompt = (
            f"User query: {user_query}\n"
            f"Sentiment: {fin_sentiment}\n"
            f"Retrieved historical sessions:\n{retrieved_context}"
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": RAG_GROQ_INSIGHT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(self._llm_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return self._extract_json(response.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            return {
                "contextual_insight": f"Failed to generate contextual insight: {exc}",
                "pattern_summary": [],
                "risk_trajectory": "unknown",
                "recommended_next_focus": [],
                "confidence": "low",
            }
