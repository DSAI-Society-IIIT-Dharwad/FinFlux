"""FinFlux Pro Server V4.2: 8-Stage Modular Pipeline (Production Expert Module)."""
import os
import shutil
import uuid
import time
import json
import datetime
import requests
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List
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
from finflux import config as finflux_config
from api.security import FinFluxSecurity
from api.storage import (
    save_conversation,
    save_quality_metrics,
    get_quality_summary,
    get_all_conversations,
    search_memories,
    get_conversation_by_id,
    update_conversation as storage_update_conversation,
    clear_user_history,
    delete_thread_history,
    SessionLocal,
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

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
bearer_scheme = HTTPBearer()


class AuthPayload(BaseModel):
    username: str
    password: str


class ChatPayload(BaseModel):
    text: str
    thread_id: str | None = None


class RealtimeDetectPayload(BaseModel):
    text: str


def _require_supabase_auth_config() -> None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Supabase Auth is not configured")


def _normalize_auth_email(username: str) -> str:
    normalized = username.strip().lower()
    if "@" not in normalized or "." not in normalized.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    return normalized


def _supabase_auth_headers(token: str | None = None) -> Dict[str, str]:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _supabase_get_user(access_token: str) -> Dict[str, Any]:
    _require_supabase_auth_config()
    res = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers=_supabase_auth_headers(token=access_token),
        timeout=15,
    )
    if res.status_code >= 400:
        raise HTTPException(status_code=401, detail="Invalid or expired auth token")
    data = res.json() if res.text else {}
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(status_code=401, detail="Invalid auth user payload")
    return data


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    user = _supabase_get_user(credentials.credentials)
    return str(user.get("id"))


RESET_PATTERNS = [
    r"\b(start\s+fresh|fresh\s+start|new\s+chat|new\s+conversation|reset|ignore\s+previous|from\s+scratch)\b",
    r"\b(naya\s+chat|nayi\s+baat|purana\s+ignore|reset\s+karo|shuru\s+se)\b",
]
GREETING_PATTERN = r"^(hi|hello|hey|namaste|namaskar|good\s+(morning|afternoon|evening)|hii+)\b"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "to", "of", "in", "on", "for", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "as",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them", "their", "with", "about", "into",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "not",
    "no", "yes", "also", "just", "very", "so", "than", "too", "up", "down", "over", "under", "again", "all",
    "मैं", "मेरे", "मेरी", "है", "हैं", "था", "थे", "को", "से", "पर", "और", "या", "की", "का", "के", "में",
}

FINANCIAL_LEXICON = [
    "emi", "equated monthly installment", "instalment", "loan", "home loan", "personal loan", "car loan", "education loan",
    "gold loan", "business loan", "overdraft", "loan amount", "principal", "interest", "interest rate", "floating rate",
    "fixed rate", "prepayment", "foreclosure", "tenure", "moratorium", "outstanding", "credit", "credit card", "minimum due",
    "credit limit", "statement", "billing cycle", "late fee", "cibil", "credit score", "debt", "debt consolidation",
    "liability", "repayment", "default", "penalty", "bounce", "salary", "income", "monthly income", "cashflow", "expense",
    "expenses", "budget", "savings", "monthly savings", "surplus", "emergency fund", "buffer", "contingency", "liquidity",
    "reserve", "corpus", "invest", "investment", "investing", "sip", "systematic investment plan", "lumpsum", "mutual fund",
    "amc", "fund house", "folio", "nav", "equity", "debt fund", "hybrid fund", "index fund", "etf", "smallcap", "midcap",
    "largecap", "diversification", "returns", "yield", "xirr", "cagr", "alpha", "beta", "volatility", "risk", "risk profile",
    "asset allocation", "rebalancing", "fixed deposit", "fd", "recurring deposit", "rd", "bond", "debenture", "treasury", "gsec",
    "ppf", "epf", "vpf", "nps", "elss", "ulip", "retirement", "pension", "annuity", "superannuation", "insurance",
    "term insurance", "health insurance", "mediclaim", "premium", "sum assured", "coverage", "deductible", "claim", "tax",
    "income tax", "tax saving", "section 80c", "section 80d", "section 87a", "hra", "tds", "capital gains", "gst", "itr",
    "property", "real estate", "down payment", "registry", "stamp duty", "maintenance", "rent", "rental yield", "gold",
    "sovereign gold bond", "sgb", "digital gold", "silver", "commodity", "crypto", "bitcoin", "ethereum", "bank", "bank account",
    "savings account", "current account", "ifsc", "neft", "rtgs", "imps", "upi", "demat", "broker", "trading", "networth",
    "wealth", "financial planning", "goal planning", "child education", "marriage fund", "house purchase", "vehicle purchase",
    "inflation", "cost of living", "rupee", "rs", "inr", "lakh", "crore", "thousand", "k", "million", "billion", "किस्त",
    "ब्याज", "बचत", "निवेश", "ऋण", "कर्ज", "आय", "खर्च", "बीमा", "प्रीमियम", "टैक्स", "म्यूचुअल फंड", "सिप", "एफडी",
    "पीपीएफ", "एनपीएस", "क्रेडिट कार्ड", "होम लोन", "पर्सनल लोन", "डाउन पेमेंट", "जोखिम", "रिटर्न", "धन", "सम्पत्ति",
    "संपत्ति", "बैंक", "इमरजेंसी फंड", "फाइनेंस", "वित्तीय", "रुपया",
]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[\w\u0900-\u097F]+", str(text or "").lower())


def _asr_confidence_from_verbose(asr_meta: Dict[str, Any] | None) -> float:
    meta = asr_meta or {}
    try:
        raw = meta.get("asr_confidence", 0.85)
        val = float(raw)
        if 0.0 <= val <= 1.0:
            return val
    except Exception:
        pass

    segments = meta.get("segments", []) if isinstance(meta, dict) else []
    probs: List[float] = []
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            words = seg.get("words", [])
            if not isinstance(words, list):
                continue
            for w in words:
                if not isinstance(w, dict):
                    continue
                try:
                    p = float(w.get("probability"))
                    if 0.0 <= p <= 1.0:
                        probs.append(p)
                except Exception:
                    continue
    if probs:
        return float(sum(probs) / len(probs))
    return 0.85


def _ner_coverage_pct(transcript: str) -> float:
    text = str(transcript or "").lower()
    matched = 0
    for term in FINANCIAL_LEXICON:
        t = str(term).strip().lower()
        if not t:
            continue
        if " " in t or any(ord(ch) > 127 for ch in t):
            if t in text:
                matched += 1
        else:
            if re.search(rf"\b{re.escape(t)}\b", text):
                matched += 1
    if not FINANCIAL_LEXICON:
        return 0.0
    return round((matched / len(FINANCIAL_LEXICON)) * 100.0, 2)


def _rouge1_recall(summary: str, transcript: str) -> float:
    s_words = {w for w in _tokenize_words(summary) if w not in STOPWORDS}
    t_words = {w for w in _tokenize_words(transcript) if w not in STOPWORDS}
    if not t_words:
        return 0.0
    inter = s_words.intersection(t_words)
    return round(len(inter) / len(t_words), 4)


def _contains_fuzzy(haystack: str, needle: str, threshold: float = 0.80) -> bool:
    h = str(haystack or "").lower().strip()
    n = str(needle or "").lower().strip()
    if not h or not n:
        return False
    if n in h:
        return True

    if SequenceMatcher(None, n, h).ratio() >= threshold:
        return True

    # Character-window fuzzy substring check.
    n_len = len(n)
    if n_len == 0:
        return False
    step = max(1, n_len // 4)
    for i in range(0, max(1, len(h) - n_len + 1), step):
        chunk = h[i:i + n_len]
        if SequenceMatcher(None, n, chunk).ratio() >= threshold:
            return True
    return False


def _entity_alignment_pct(entities: List[Dict[str, Any]], summary: str, key_insights: List[str]) -> float:
    if not entities:
        return 0.0
    target = " ".join([str(summary or "")] + [str(k) for k in (key_insights or [])])
    aligned = 0
    total = 0
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        val = str(ent.get("value") or ent.get("text") or ent.get("entity") or "").strip()
        if not val:
            continue
        total += 1
        if _contains_fuzzy(target, val, threshold=0.80):
            aligned += 1
    if total == 0:
        return 0.0
    return round((aligned / total) * 100.0, 2)


def _build_model_versions(input_mode: str) -> Dict[str, Any]:
    whisper_model = finflux_config.GROQ_STT_MODEL if input_mode == "audio" else "none_text_input"
    return {
        "whisper": whisper_model,
        "llm_synthesis": finflux_config.GROQ_LLM_MODEL,
        "llm_reasoning": finflux_config.GROQ_LLM_REASON_MODEL,
        "ner_model": finflux_config.HF_NER_FINANCIAL,
        "sentiment_model": finflux_config.HF_FINBERT,
        "language_model": finflux_config.HF_LANG_DETECT,
    }


def _compute_quality_metrics(
    *,
    transcript: str,
    executive_summary: str,
    key_insights: List[str],
    entities: List[Dict[str, Any]],
    language_confidence: float,
    financial_relevance_score: float,
    asr_meta: Dict[str, Any] | None,
    input_mode: str,
) -> Dict[str, Any]:
    asr_conf = _asr_confidence_from_verbose(asr_meta)
    ner_cov = _ner_coverage_pct(transcript)
    rouge = _rouge1_recall(executive_summary, transcript)
    ent_align = _entity_alignment_pct(entities, executive_summary, key_insights)
    lang_conf = float(language_confidence or 0.0)
    fin_rel = float(financial_relevance_score or 0.0)

    overall = (
        (asr_conf * 0.30)
        + ((ner_cov / 100.0) * 0.20)
        + (rouge * 0.20)
        + ((ent_align / 100.0) * 0.15)
        + (lang_conf * 0.10)
        + (fin_rel * 0.05)
    )
    if overall > 0.82:
        tier = "EXCELLENT"
    elif overall > 0.65:
        tier = "GOOD"
    elif overall > 0.45:
        tier = "ACCEPTABLE"
    else:
        tier = "LOW"

    return {
        "asr_confidence": round(asr_conf, 4),
        "ner_coverage_pct": round(ner_cov, 2),
        "rouge1_recall": round(rouge, 4),
        "entity_alignment_pct": round(ent_align, 2),
        "language_confidence": round(lang_conf, 4),
        "financial_relevance_score": round(fin_rel, 4),
        "overall_quality_score": round(overall, 4),
        "quality_tier": tier,
        "model_versions": _build_model_versions(input_mode),
    }


def _is_fresh_start_intent(clean_text: str, thread_id: str, history_rows: List[Dict[str, Any]], current_topic: str) -> bool:
    text = clean_text.strip().lower()
    same_thread = [
        row for row in history_rows
        if str(row.get("chat_thread_id") or row.get("conversation_id")) == thread_id
    ]
    is_new_thread = len(same_thread) == 0

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in RESET_PATTERNS):
        return True

    words = re.findall(r"\w+", text)
    is_opening_greeting = bool(re.search(GREETING_PATTERN, text)) and len(words) <= 10
    if is_new_thread and is_opening_greeting:
        return True

    if is_new_thread and current_topic and current_topic != "N/A":
        latest_any = history_rows[0] if history_rows else {}
        prev_topic = str(latest_any.get("financial_topic") or "").strip()
        if prev_topic and prev_topic != "N/A" and prev_topic.lower() != str(current_topic).lower():
            return True

    return False


def _apply_optional_memory_filters(rows: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Narrow semantic retrieval results with optional metadata filters; never block all results."""
    if not rows or not filters:
        return rows

    def _parse_iso(ts: str) -> datetime.datetime | None:
        text = str(ts or "").strip()
        if not text:
            return None
        try:
            return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    narrowed: List[Dict[str, Any]] = []
    for row in rows:
        row_topic = str(row.get("financial_topic", "")).strip().lower()
        row_risk = str(row.get("risk_level") or row.get("risk") or "").strip().upper()
        row_sentiment = str(row.get("financial_sentiment", "")).strip().capitalize()
        row_dt = _parse_iso(str(row.get("timestamp", "")))

        ok = True
        f_topic = str(filters.get("financial_topic", "")).strip().lower()
        if f_topic and f_topic not in row_topic:
            ok = False

        f_risk = str(filters.get("risk_level", "")).strip().upper()
        if f_risk and f_risk != row_risk:
            ok = False

        f_sentiment = str(filters.get("financial_sentiment", "")).strip().capitalize()
        if f_sentiment and f_sentiment != row_sentiment:
            ok = False

        start_s = str(filters.get("created_at_start", "")).strip()
        if start_s and row_dt:
            try:
                start_dt = datetime.datetime.fromisoformat(start_s)
                if row_dt.date() < start_dt.date():
                    ok = False
            except Exception:
                pass

        end_s = str(filters.get("created_at_end", "")).strip()
        if end_s and row_dt:
            try:
                end_dt = datetime.datetime.fromisoformat(end_s)
                if row_dt.date() > end_dt.date():
                    ok = False
            except Exception:
                pass

        if ok:
            narrowed.append(row)

    return narrowed if narrowed else rows


def _parse_ts(ts: str) -> datetime.datetime | None:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed
    except Exception:
        return None


def _history_product(row: Dict[str, Any]) -> str:
    model_attr = row.get("model_attribution", {})
    if isinstance(model_attr, dict):
        deberta = model_attr.get("deberta", {})
        if isinstance(deberta, dict):
            p = str(deberta.get("level2_top_product", "")).strip().lower()
            if p:
                return p
    return str(row.get("financial_topic", "")).strip().lower()


def _generate_deterministic_reminders(history_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reminders: List[Dict[str, Any]] = []
    if not history_rows:
        return reminders

    # Rule 1: same product in 3+ conversations.
    product_counts: Dict[str, int] = {}
    for row in history_rows:
        p = _history_product(row)
        if not p or p == "n/a":
            continue
        product_counts[p] = product_counts.get(p, 0) + 1
    for p, count in product_counts.items():
        if count >= 3:
            reminders.append(
                {
                    "reminder_type": "PRODUCT_PATTERN",
                    "reminder_text": f"{p.replace('_', ' ')} has appeared in {count} conversations; schedule a focused follow-up review.",
                    "urgency": "MEDIUM",
                }
            )

    # Sort history by time asc for sequence checks.
    ordered = sorted(history_rows, key=lambda r: str(r.get("timestamp", "")))

    # Rule 2: DECISION commitment older than 7 days without FULFILLED marker.
    now = datetime.datetime.now(datetime.timezone.utc)
    fulfilled_marker = any(
        "fulfilled" in str(r.get("executive_summary", "")).lower()
        or "fulfilled" in str(r.get("transcript", "")).lower()
        or "fulfilled" in str(r.get("raw_user_input", "")).lower()
        for r in ordered
    )
    if not fulfilled_marker:
        for r in ordered:
            ts = _parse_ts(str(r.get("timestamp", "")))
            if not ts:
                continue
            text_blob = " ".join(
                [
                    " ".join(r.get("key_points", [])) if isinstance(r.get("key_points"), list) else "",
                    str(r.get("executive_summary", "")),
                    str(r.get("risk_assessment", "")),
                    str(r.get("strategic_intent", "")),
                ]
            ).lower()
            if "decision" in text_blob and (now - ts).days > 7:
                reminders.append(
                    {
                        "reminder_type": "COMMITMENT_PENDING",
                        "reminder_text": "A prior decision-level commitment is older than 7 days and has no fulfilled marker; prompt execution check.",
                        "urgency": "HIGH",
                    }
                )
                break

    # Rule 3: HIGH risk for 2 consecutive conversations.
    consecutive_high = 0
    for r in ordered:
        risk = str(r.get("risk_level", "")).strip().upper()
        if risk == "HIGH":
            consecutive_high += 1
            if consecutive_high >= 2:
                reminders.append(
                    {
                        "reminder_type": "RISK_ALERT",
                        "reminder_text": "Risk level has remained HIGH across consecutive conversations; immediate rebalancing review is recommended.",
                        "urgency": "HIGH",
                    }
                )
                break
        else:
            consecutive_high = 0

    # Keep output concise and deterministic.
    return reminders[:5]


def _build_analysis_result(
    clean_text: str,
    user_id: str,
    asr_text: str,
    t_asr: float,
    t_norm: float,
    t_start: float,
    thread_id: str,
    input_mode: str,
    asr_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    t_nlp_start = time.time()
    nlp_res = EXPERT.process(clean_text)
    t_nlp = time.time() - t_nlp_start
    thread_history_context = ""
    long_term_memory_context = ""
    retrieval_spec = SYNTHESIS.decompose_retrieval_query(
        user_message=clean_text,
        detected_topic=str(nlp_res.get("topic") or ""),
    )
    semantic_query = str(retrieval_spec.get("semantic_query") or clean_text)
    retrieval_filters = retrieval_spec.get("filters", {}) if isinstance(retrieval_spec, dict) else {}
    history_rows: List[Dict[str, Any]] = []
    try:
        history_rows = get_all_conversations(user_id=user_id).get("results", [])
        same_thread = [
            row for row in history_rows
            if str(row.get("chat_thread_id") or row.get("conversation_id")) == thread_id
        ]
        same_thread = same_thread[:4]
        is_fresh_start = _is_fresh_start_intent(
            clean_text=clean_text,
            thread_id=thread_id,
            history_rows=history_rows,
            current_topic=str(nlp_res.get("topic") or "N/A"),
        )
        thread_lines = []
        for row in same_thread:
            user_line = str(row.get("raw_user_input") or row.get("transcript") or "").strip()
            assistant_line = str(row.get("executive_summary") or "").strip()
            if user_line or assistant_line:
                thread_lines.append(f"- User: {user_line}\n  Assistant: {assistant_line}")
        if thread_lines:
            thread_history_context = "\n".join(thread_lines)

        # Fresh-start threads should not pull global historical memory.
        if not is_fresh_start:
            memory_rows: List[Dict[str, Any]] = []
            thread_mem_res = search_memories(
                user_id=user_id,
                query_text=semantic_query,
                filters={
                    "thread_id": thread_id,
                    "financial_topic": retrieval_filters.get("financial_topic", ""),
                    "risk_level": retrieval_filters.get("risk_level", ""),
                },
                n_results=8,
                min_similarity=0.72,
            )
            thread_rows = thread_mem_res.get("results", []) if isinstance(thread_mem_res, dict) else []
            thread_rows = _apply_optional_memory_filters(thread_rows, retrieval_filters)

            for row in thread_rows:
                if not isinstance(row, dict):
                    continue
                summary = str(row.get("executive_summary", "")).strip()
                if summary:
                    memory_rows.append(row)

            # Escalate to global only when thread context is sparse.
            if len(same_thread) < 3:
                mem_res = search_memories(
                    user_id=user_id,
                    query_text=semantic_query,
                    filters={
                        "financial_topic": retrieval_filters.get("financial_topic", ""),
                        "risk_level": retrieval_filters.get("risk_level", ""),
                    },
                    n_results=8,
                    min_similarity=0.72,
                )
                result_rows = mem_res.get("results", []) if isinstance(mem_res, dict) else []
                result_rows = _apply_optional_memory_filters(result_rows, retrieval_filters)
                for row in result_rows:
                    if not isinstance(row, dict):
                        continue
                    row_thread_id = str(row.get("thread_id") or "")
                    if row_thread_id and row_thread_id == thread_id:
                        continue
                    summary = str(row.get("executive_summary", "")).strip()
                    if summary and all(str(existing.get("executive_summary", "")).strip() != summary for existing in memory_rows):
                        memory_rows.append(row)

            if memory_rows:
                long_term_memory_context = SYNTHESIS.build_memory_context(
                    transcript=clean_text,
                    memory_records=memory_rows,
                    max_keep=3,
                )
    except Exception:
        thread_history_context = ""
        long_term_memory_context = ""

    t_syn_start = time.time()
    analysis = SYNTHESIS.analyze(
        transcript=clean_text,
        entities=nlp_res.get("entities", []),
        fin_sentiment=nlp_res.get("financial_sentiment", "Neutral"),
        thread_history_context=thread_history_context,
        long_term_memory_context=long_term_memory_context,
    )
    future_insights = SYNTHESIS.generate_future_insights(
        transcript=clean_text,
        current_analysis={
            **analysis,
            "financial_topic": nlp_res.get("topic", "N/A"),
        },
        memory_context=long_term_memory_context,
    )
    reminders = _generate_deterministic_reminders(history_rows)
    t_syn = time.time() - t_syn_start

    injection_found = SECURITY.detect_injection(clean_text)
    exec_summary = analysis.get("executive_summary", "")
    safe_summary = SECURITY.mask_pii(exec_summary)
    masked_key_insights = [SECURITY.mask_pii(p) for p in analysis.get("key_insights", [])]

    quality_metrics = _compute_quality_metrics(
        transcript=clean_text,
        executive_summary=safe_summary,
        key_insights=masked_key_insights,
        entities=nlp_res.get("entities", []),
        language_confidence=float(nlp_res.get("language_confidence", 0.0) or 0.0),
        financial_relevance_score=float(nlp_res.get("confidence_score", 0.0) or 0.0),
        asr_meta=asr_meta,
        input_mode=input_mode,
    )

    return {
        "response_mode": "analysis",
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
        "future_insights": future_insights,
        "reminders": reminders,
        "quality_metrics": quality_metrics,
        "risk_assessment": analysis.get("risk_assessment", ""),
        "key_insights": masked_key_insights,
        "key_points": masked_key_insights,
        "transcript": clean_text,
        "expert_reasoning_points": analysis.get("expert_reasoning_points", ""),
        "expert_reasoning": analysis.get("expert_reasoning_points", ""),
        "model_attribution": {
            **nlp_res.get("model_attribution", {}),
            "response_mode": "analysis",
            "retrieval_spec": {
                "semantic_query": semantic_query,
                "filters": retrieval_filters,
            },
            "future_insights": future_insights,
            "reminders": reminders,
            "quality_metrics": quality_metrics,
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


def _dominant_language_from_breakdown(breakdown: Dict[str, Any]) -> str:
    h = float(breakdown.get("hindi", 0.0))
    e = float(breakdown.get("english", 0.0))
    hg = float(breakdown.get("hinglish", 0.0))
    if h > 60.0:
        return "hindi"
    if e > 60.0:
        return "english"
    if hg >= max(h, e):
        return "hinglish"
    return "hinglish"


def _build_direct_response_result(
    *,
    user_id: str,
    thread_id: str,
    input_text: str,
    assistant_text: str,
    route: str,
    route_label: str,
    route_score: float,
) -> Dict[str, Any]:
    response_mode = "general_conversation" if route == "general" else "financial_inquiry"
    safe_text = assistant_text.strip() or input_text.strip()
    return {
        "response_mode": response_mode,
        "conversation_id": f"call_{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "chat_thread_id": thread_id,
        "input_mode": "text",
        "raw_user_input": input_text,
        "timestamp": str(datetime.datetime.utcnow()),
        "language": "unknown",
        "language_confidence": 0.0,
        "financial_topic": "general",
        "topic_top3": [],
        "strategic_intent": "",
        "risk_level": "LOW",
        "financial_sentiment": "neutral",
        "sentiment_breakdown": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        "advice_request": False,
        "injection_attempt": False,
        "entities": [],
        "executive_summary": safe_text,
        "summary": safe_text,
        "assistant_text": safe_text,
        "future_gearing": "",
        "future_insights": [],
        "reminders": [],
        "quality_metrics": {},
        "risk_assessment": "",
        "key_insights": [],
        "key_points": [],
        "transcript": input_text,
        "expert_reasoning_points": "",
        "expert_reasoning": "",
        "model_attribution": {
            "response_mode": response_mode,
            "route_label": route_label,
            "route_score": route_score,
        },
        "confidence_score": route_score,
        "timing": {"total_s": 0.0},
        "raw_asr_text": input_text,
    }

@app.post("/api/auth/signup")
def signup(payload: AuthPayload):
    username = payload.username.strip().lower()
    if len(username) < 3 or len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Username or password too short")
    _require_supabase_auth_config()
    email = _normalize_auth_email(username)
    res = requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        headers=_supabase_auth_headers(),
        json={"email": email, "password": payload.password},
        timeout=20,
    )
    data = res.json() if res.text else {}
    if res.status_code >= 400:
        message = data.get("msg") if isinstance(data, dict) else None
        raise HTTPException(status_code=400, detail=message or "Supabase signup failed")

    session = data.get("session") if isinstance(data, dict) else None
    if not session or not session.get("access_token"):
        return {
            "access_token": None,
            "token_type": "bearer",
            "username": email,
            "user_id": data.get("user", {}).get("id"),
            "requires_email_confirmation": True,
            "message": "Verification email sent. Confirm your email, then sign in.",
        }
    return {
        "access_token": session.get("access_token"),
        "token_type": "bearer",
        "username": email,
        "user_id": data.get("user", {}).get("id"),
        "requires_email_confirmation": False,
    }


@app.post("/api/auth/login")
def login(payload: AuthPayload):
    _require_supabase_auth_config()
    email = _normalize_auth_email(payload.username)
    res = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers=_supabase_auth_headers(),
        json={"email": email, "password": payload.password},
        timeout=20,
    )
    data = res.json() if res.text else {}
    if res.status_code >= 400:
        message = data.get("msg") if isinstance(data, dict) else None
        raise HTTPException(status_code=401, detail=message or "Invalid credentials")
    return {
        "access_token": data.get("access_token"),
        "token_type": "bearer",
        "username": email,
        "user_id": data.get("user", {}).get("id"),
    }


@app.get("/api/auth/me")
def me(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    user = _supabase_get_user(credentials.credentials)
    return {
        "user_id": user.get("id"),
        "username": user.get("email") or user.get("phone") or "unknown",
    }


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    thread_id: str = Form(""),
    asr_language: str = Form(""),
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
            lang_hint = asr_language.strip().lower()
            if lang_hint not in {"hi", "en", "hindi", "english"}:
                lang_hint = ""
            asr = whisper.transcribe(temp_wav, language=lang_hint or None)
            if not str(asr.get("text", "")).strip():
                raise RuntimeError(str(asr.get("error", "empty_transcript")))
        except Exception as e:
            print(f"[API] Groq ASR Failed: {e}. Switching to Local Fallback...")
            local_text = EXPERT.transcribe_local(temp_wav)
            asr = {"text": local_text, "language": "hindi" if local_text else "unknown", "asr_confidence": 0.85}
            
        raw_text = asr.get("text", "")
        t_asr = time.time() - t_start

        # ── Stage 2: Normalization (Llama 8B) ──
        t2_start = time.time()
        pre_nlp = EXPERT.process(raw_text)
        language_breakdown = pre_nlp.get("language_breakdown", {}) if isinstance(pre_nlp, dict) else {}
        dominant_language = _dominant_language_from_breakdown(language_breakdown if isinstance(language_breakdown, dict) else {})
        clean_text = SYNTHESIS.normalize_transcript(
            raw_text,
            dominant_language=dominant_language,
            language_breakdown=language_breakdown if isinstance(language_breakdown, dict) else None,
        )
        t_norm = time.time() - t2_start

        thread_value = thread_id.strip() or f"thr_{uuid.uuid4().hex[:8]}"
        try:
            route_info = EXPERT.route_intent(clean_text)
            route = route_info.get("route", "financial_inquiry")
            route_label = route_info.get("label", "financial information request")
            route_score = float(route_info.get("score", 0.0))
        except Exception as exc:
            print(f"[Server] Audio route classification failed: {exc}")
            route = "analysis"
            route_label = "personal financial situation discussion"
            route_score = 0.0

        if route == "analysis":
            result = _build_analysis_result(
                clean_text=clean_text,
                user_id=current_user,
                asr_text=raw_text,
                t_asr=t_asr,
                t_norm=t_norm,
                t_start=t_start,
                thread_id=thread_value,
                input_mode="audio",
                asr_meta=asr,
            )
            result["language"] = pre_nlp.get("detected_language", result.get("language", "unknown"))
            result["language_confidence"] = pre_nlp.get("language_confidence", result.get("language_confidence", 0.0))
            result["language_breakdown"] = language_breakdown if isinstance(language_breakdown, dict) else {"hindi": 0.0, "english": 0.0, "hinglish": 0.0}
            result["script_preserved"] = True
            result.setdefault("model_attribution", {})
            result["model_attribution"]["script_preserved"] = True
            result["model_attribution"]["language_breakdown"] = result["language_breakdown"]
        elif route == "general":
            assistant_text = EXPERT.answer_casual(clean_text)
            result = _build_direct_response_result(
                user_id=current_user,
                thread_id=thread_value,
                input_text=clean_text,
                assistant_text=assistant_text,
                route=route,
                route_label=route_label,
                route_score=route_score,
            )
            result["input_mode"] = "audio"
            result["raw_asr_text"] = raw_text
            result["timing"]["asr_s"] = round(t_asr, 2)
            result["timing"]["normalization_s"] = round(t_norm, 2)
            result["language_breakdown"] = language_breakdown if isinstance(language_breakdown, dict) else {"hindi": 0.0, "english": 0.0, "hinglish": 0.0}
            result["script_preserved"] = True
        else:
            assistant_text = EXPERT.answer_financial_inquiry(clean_text)
            result = _build_direct_response_result(
                user_id=current_user,
                thread_id=thread_value,
                input_text=clean_text,
                assistant_text=assistant_text,
                route=route,
                route_label=route_label,
                route_score=route_score,
            )
            result["input_mode"] = "audio"
            result["raw_asr_text"] = raw_text
            result["timing"]["asr_s"] = round(t_asr, 2)
            result["timing"]["normalization_s"] = round(t_norm, 2)
            result["language_breakdown"] = language_breakdown if isinstance(language_breakdown, dict) else {"hindi": 0.0, "english": 0.0, "hinglish": 0.0}
            result["script_preserved"] = True

        # Save to Secure DB
        save_conversation(result)
        if result.get("response_mode") == "analysis":
            try:
                save_quality_metrics(
                    user_id=current_user,
                    conversation_id=str(result.get("conversation_id", "")),
                    metrics=result.get("quality_metrics", {}),
                )
            except Exception as exc:
                print(f"[Server] Quality metrics save skipped: {exc}")
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
    thread_id = (payload.thread_id or "").strip() or f"thr_{uuid.uuid4().hex[:8]}"

    try:
        route_info = EXPERT.route_intent(text)
        route = route_info.get("route", "financial_inquiry")
        route_label = route_info.get("label", "financial information request")
        route_score = float(route_info.get("score", 0.0))
    except Exception as exc:
        print(f"[Server] Route classification failed: {exc}")
        route = "financial_inquiry"
        route_label = "financial information request"
        route_score = 0.0

    if route == "analysis":
        t_asr = 0.0
        t2_start = time.time()
        pre_nlp = EXPERT.process(text)
        language_breakdown = pre_nlp.get("language_breakdown", {}) if isinstance(pre_nlp, dict) else {}
        dominant_language = _dominant_language_from_breakdown(language_breakdown if isinstance(language_breakdown, dict) else {})
        clean_text = SYNTHESIS.normalize_transcript(
            text,
            dominant_language=dominant_language,
            language_breakdown=language_breakdown if isinstance(language_breakdown, dict) else None,
        )
        t_norm = time.time() - t2_start
        result = _build_analysis_result(
            clean_text=clean_text,
            user_id=current_user,
            asr_text=text,
            t_asr=t_asr,
            t_norm=t_norm,
            t_start=t_start,
            thread_id=thread_id,
            input_mode="text",
            asr_meta={"asr_confidence": 0.85},
        )
        result["language"] = pre_nlp.get("detected_language", result.get("language", "unknown"))
        result["language_confidence"] = pre_nlp.get("language_confidence", result.get("language_confidence", 0.0))
        result["language_breakdown"] = language_breakdown if isinstance(language_breakdown, dict) else {"hindi": 0.0, "english": 0.0, "hinglish": 0.0}
        result["script_preserved"] = True
        result.setdefault("model_attribution", {})
        result["model_attribution"]["script_preserved"] = True
        result["model_attribution"]["language_breakdown"] = result["language_breakdown"]
    elif route == "general":
        assistant_text = EXPERT.answer_casual(text)
        result = _build_direct_response_result(
            user_id=current_user,
            thread_id=thread_id,
            input_text=text,
            assistant_text=assistant_text,
            route=route,
            route_label=route_label,
            route_score=route_score,
        )
        pre_nlp = EXPERT.process(text)
        result["language_breakdown"] = pre_nlp.get("language_breakdown", {"hindi": 0.0, "english": 0.0, "hinglish": 0.0})
        result["script_preserved"] = True
    else:
        assistant_text = EXPERT.answer_financial_inquiry(text)
        result = _build_direct_response_result(
            user_id=current_user,
            thread_id=thread_id,
            input_text=text,
            assistant_text=assistant_text,
            route=route,
            route_label=route_label,
            route_score=route_score,
        )
        pre_nlp = EXPERT.process(text)
        result["language_breakdown"] = pre_nlp.get("language_breakdown", {"hindi": 0.0, "english": 0.0, "hinglish": 0.0})
        result["script_preserved"] = True

    save_conversation(result)
    if result.get("response_mode") == "analysis":
        try:
            save_quality_metrics(
                user_id=current_user,
                conversation_id=str(result.get("conversation_id", "")),
                metrics=result.get("quality_metrics", {}),
            )
        except Exception as exc:
            print(f"[Server] Quality metrics save skipped: {exc}")
    return result


@app.post("/api/realtime/financial-detect")
def realtime_financial_detect(payload: RealtimeDetectPayload):
    """Lightweight keyword detector for rolling 5-second recording chunks."""
    return EXPERT.detect_financial_keywords(payload.text)


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


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str, current_user: str = Depends(get_current_user)):
    try:
        if not thread_id or not current_user:
            raise HTTPException(status_code=400, detail="thread_id and user_id are required")
        
        deleted = delete_thread_history(user_id=current_user, thread_id=thread_id)
        return {
            "status": "deleted",
            "thread_id": thread_id,
            "user_id": current_user,
            "deleted": deleted,
            "success": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {str(e)}")


@app.get("/api/results")
def list_results(current_user: str = Depends(get_current_user)):
    return get_all_conversations(user_id=current_user)


@app.get("/api/quality/summary")
def quality_summary(current_user: str = Depends(get_current_user)):
    return get_quality_summary(user_id=current_user)

@app.put("/api/update/{conversation_id}")
async def update_conversation_endpoint(conversation_id: str, payload: dict, current_user: str = Depends(get_current_user)):
    """Update stored conversation intelligence (e.g. edited transcript)."""
    update_fields: Dict[str, Any] = {}
    if "transcript" in payload:
        update_fields["transcript"] = payload["transcript"]
    if "executive_summary" in payload:
        update_fields["executive_summary"] = payload["executive_summary"]
    if "summary" in payload:
        update_fields["executive_summary"] = payload["summary"]

    if not update_fields:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    ok = storage_update_conversation(user_id=current_user, conversation_id=conversation_id, updates=update_fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": "updated"}

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
    existing = get_conversation_by_id(user_id=current_user, conversation_id=conversation_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_transcript = payload.get("transcript", "").strip()
    if not new_transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    update_fields: Dict[str, Any] = {"transcript": new_transcript}

    # Re-run synthesis on edited transcript so insights reflect the correction
    if payload.get("reanalyze", True):
        nlp_res = EXPERT.process(new_transcript)
        analysis = SYNTHESIS.analyze(
            new_transcript,
            nlp_res.get("entities", []),
            nlp_res.get("financial_sentiment", "Neutral")
        )
        update_fields.update({
            "executive_summary": analysis.get("executive_summary", existing.get("executive_summary", "")),
            "strategic_intent": analysis.get("strategic_intent", existing.get("strategic_intent", "")),
            "future_gearing": analysis.get("future_gearing", existing.get("future_gearing", "")),
            "risk_assessment": analysis.get("risk_assessment", existing.get("risk_assessment", "")),
            "expert_reasoning_points": analysis.get("expert_reasoning_points", existing.get("expert_reasoning_points", "")),
            "financial_topic": nlp_res.get("topic", existing.get("financial_topic", "N/A")),
            "financial_sentiment": nlp_res.get("financial_sentiment", existing.get("financial_sentiment", "Neutral")),
            "entities": nlp_res.get("entities", []),
        })

    ok = storage_update_conversation(user_id=current_user, conversation_id=conversation_id, updates=update_fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")

    updated = get_conversation_by_id(user_id=current_user, conversation_id=conversation_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "status": "updated",
        "reanalyzed": payload.get("reanalyze", True),
        "conversation": {
            "conversation_id": updated.get("conversation_id"),
            "chat_thread_id": updated.get("chat_thread_id") or updated.get("conversation_id"),
            "timestamp": updated.get("timestamp"),
            "financial_topic": updated.get("financial_topic"),
            "risk_level": updated.get("risk_level"),
            "financial_sentiment": updated.get("financial_sentiment"),
            "confidence_score": updated.get("confidence_score"),
            "executive_summary": updated.get("executive_summary"),
            "transcript": updated.get("transcript"),
            "strategic_intent": updated.get("strategic_intent"),
            "future_gearing": updated.get("future_gearing"),
            "risk_assessment": updated.get("risk_assessment"),
            "expert_reasoning_points": updated.get("expert_reasoning_points"),
            "timing": updated.get("timing", {}),
            "entities": updated.get("entities", []),
        },
    }


@app.post("/api/history/purge")
def purge_my_history(payload: dict = Body(default={}), current_user: str = Depends(get_current_user)):
    """Securely purge current user's history only.

    Requires explicit confirmation flag to avoid accidental destructive actions.
    """
    if not payload.get("confirm"):
        raise HTTPException(status_code=400, detail="Set {\"confirm\": true} to purge history")
    deleted = clear_user_history(user_id=current_user)
    return {"status": "purged", "user_id": current_user, "deleted": deleted}

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
