# FinFlux: Multilingual Financial Conversation Intelligence

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb)
![Supabase](https://img.shields.io/badge/Storage-Supabase%20%2B%20pgvector-3ecf8e)
![Security](https://img.shields.io/badge/Security-AES--256%20%2B%20PII%20Masking-red)

Captures informal Indian financial conversations in Hindi, Hinglish, and English, transcribes them securely, and converts unstructured speech into structured financial intelligence using a 10-model AI pipeline.

## 1) Problem and Solution

In India, high-impact financial decisions often happen in informal spoken conversations in Hindi, Hinglish, or mixed-language contexts. These discussions include commitments, debt pressure, risk concerns, and investment intent, but they are rarely recorded in a structured, auditable format. As a result, key details are lost, follow-ups are inconsistent, and strategic decision quality depends on memory rather than data.

FinFlux solves this with secure audio capture, multilingual transcription, and a 10-model NLP intelligence pipeline that converts speech into structured outputs: topic, sentiment, entities, risk signals, strategic intent, reminders, future insights, and quality metrics. The platform is built as a live FastAPI + React system backed by Supabase bridge tables and pgvector retrieval, producing executive-grade outputs while preserving security and auditability.

## 2) Architecture Diagram

Diagram style follows software architecture and flowchart conventions.

- Architecture reference: [AWS Architecture Diagramming](https://aws.amazon.com/what-is/architecture-diagramming/)
- Flowchart reference: [Flowchart](https://en.wikipedia.org/wiki/Flowchart)

```text
+-------------+
| Microphone  |
+------+------+ 
       |
       v
+-----------------------+
| AES-256 Encryption    |
| (Fernet + PBKDF2)     |
+-----------+-----------+
            |
            v
+-----------------------+
| Groq Whisper Turbo    |
| whisper-large-v3-turbo|
+-----------+-----------+
            |
            v
+-----------------------+
| Llama-3.1-8B Instant  |
| Transcript Normalize  |
+-----------+-----------+
            |
            v
+-----------------------+
| XLM-Roberta           |
| Language Detection    |
+-----------+-----------+
            |
            v
+-----------------------+
| DeBERTa-v3            |
| Topic Classification  |
+-----------+-----------+
            |
            v
+-----------------------+
| FinBERT               |
| Sentiment             |
+-----------+-----------+
            |
            v
+-----------------------+
| GLiNER-Medium         |
| Financial NER         |
+-----------+-----------+
            |
            v
+-----------------------+
| Qwen-32B              |
| Expert Reasoning      |
+-----------+-----------+
            |
            v
+-----------------------+
| Llama-3.3-70B         |
| Strategic Synthesis   |
+-----------+-----------+
            |
            v
+-----------------------+
| Supabase Bridge +     |
| pgvector RAG          |
| (history + retrieval) |
+-----------------------+
```

## 3) Model Attribution Table

| Stage | Model | Provider | Role |
| --- | --- | --- | --- |
| 1 | `whisper-large-v3-turbo` | Groq | Multilingual ASR |
| 2 | `llama-3.1-8b-instant` | Groq | Transcript normalization |
| 3 | `papluca/xlm-roberta-base-language-detection` | Hugging Face | Language detection |
| 4 | `cross-encoder/nli-deberta-v3-small` | Hugging Face | Topic classification / route intent |
| 5 | `ProsusAI/finbert` | Hugging Face | Financial sentiment |
| 6 | `urchade/gliner_medium-v2.1` | Hugging Face | Financial NER |
| 7 | `qwen/qwen3-32b` | Groq | Expert reasoning wall |
| 8 | `llama-3.3-70b-versatile` | Groq | Strategic synthesis |
| 9 | `all-MiniLM-L6-v2` | Local | Embedding generation |
| 10 | `Supabase pgvector` | Supabase | Semantic retrieval (RAG) |

## 4) Prerequisites

- Python 3.11+
- Node.js 20+
- CUDA-capable GPU optional but recommended
- Groq API key
- Supabase project with `pgvector` extension enabled

## 5) Setup in 10 Commands

```bash
# 1
git clone https://github.com/<your-org>/finflux.git

# 2
cd finflux

# 3
cp .env.example .env

# 4
python -m pip install -r requirements.txt --break-system-packages

# 5
python scripts/download_models.py

# 6
# Run SQL migrations 001 through 012 in Supabase SQL Editor in order.
# (Open Supabase -> SQL Editor -> execute files from sql/001...sql/012)

# 7
cd frontend && npm install && cd ..

# 8
python -m uvicorn api.server:app --port 8000

# 9
cd frontend && npm run dev

# 10
# Open in browser
# http://localhost:5173
```

### Full Validation + RAG + Runtime Checklist

```bash
# Backend health
curl -s http://127.0.0.1:8000/api/health

# End-to-end test flow (auth + chat + analyze + history + purge)
python scripts/_e2e_live_check.py

# Quality dashboard aggregate endpoint (requires auth in real flow)
# GET /api/quality/summary
```

## 6) Environment Variables

| Variable | Required | Example (masked) | What breaks if missing |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Yes | `gsk_live_xxxxxxxxxxxxxxxxx` | Groq ASR + LLM pipeline degrades/fails; local fallback only where available |
| `SUPABASE_URL` | Yes | `https://abcxyz.supabase.co` | Supabase bridge storage, retrieval, quality metrics disabled |
| `SUPABASE_SERVICE_KEY` | Optional alias | `eyJhbGciOi...` | No direct break if `SUPABASE_SERVICE_ROLE_KEY` is set |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes (runtime key used by code) | `eyJhbGciOi...` | Supabase writes/reads fail for threads/messages/embeddings/quality |
| `FINFLUX_SECRET` | Yes | `a-strong-rotated-random-secret` | Audio encryption/decryption integrity compromised |
| `USE_CUDA` | Optional | `true` | No break; slower inference on CPU |
| `DEMO_MODE` | Optional | `false` | No break; when true, synthetic responses are used for some LLM paths |

## 7) API Reference

| Method | Endpoint | Description | Key Response Fields |
| --- | --- | --- | --- |
| `POST` | `/api/analyze` | Main audio intelligence pipeline | `conversation_id`, `financial_topic`, `risk_level`, `entities`, `executive_summary`, `quality_metrics` |
| `GET` | `/api/results` | User conversation history | `count`, `results[]` |
| `PUT` | `/api/update/{id}` | Update stored transcript/summary fields | `status` |
| `GET` | `/api/report/{id}` | Export PDF or CSV | file stream |
| `GET` | `/api/health` | Model/service status | `status`, `api_key_loaded`, `models[]` |
| `GET` | `/api/quality/summary` | Aggregate validation dashboard stats | `average_overall_quality_score`, `quality_tier_distribution`, `quality_trend_last_10` |
| `GET` | `/api/benchmark` | Transcription benchmark proof endpoint | Current branch behavior: `404` with `{"detail":"Not Found"}` |

### Response Shape: `POST /api/analyze`

```json
{
  "response_mode": "analysis",
  "conversation_id": "call_abc12345",
  "user_id": "<uuid>",
  "chat_thread_id": "thr_efgh6789",
  "input_mode": "audio",
  "raw_user_input": "...",
  "timestamp": "2026-04-05T00:00:00Z",
  "language": "hi",
  "language_confidence": 0.93,
  "language_breakdown": {
    "hindi": 40.0,
    "english": 25.0,
    "hinglish": 35.0
  },
  "script_preserved": true,
  "financial_topic": "DEBT_MANAGEMENT",
  "topic_top3": [
    {
      "topic": "DEBT_MANAGEMENT",
      "score": 0.83
    },
    {
      "topic": "INVESTMENT_PLANNING",
      "score": 0.74
    },
    {
      "topic": "BUDGETING",
      "score": 0.69
    }
  ],
  "strategic_intent": "Debt Management",
  "risk_level": "MEDIUM",
  "financial_sentiment": "Neutral",
  "sentiment_breakdown": {
    "positive": 0.21,
    "neutral": 0.56,
    "negative": 0.23
  },
  "advice_request": false,
  "injection_attempt": false,
  "entities": [
    {
      "type": "LOAN",
      "value": "home loan",
      "confidence": 0.93,
      "source": "gliner"
    }
  ],
  "executive_summary": "...",
  "summary": "...",
  "future_gearing": "...",
  "future_insights": [
    {
      "insight_type": "COMMITMENT_FOLLOWUP",
      "insight_text": "...",
      "urgency": "HIGH",
      "days_relevant": 14
    }
  ],
  "reminders": [
    {
      "reminder_type": "RISK_ALERT",
      "reminder_text": "...",
      "urgency": "HIGH"
    }
  ],
  "quality_metrics": {
    "asr_confidence": 0.91,
    "ner_coverage_pct": 73.5,
    "rouge1_recall": 0.64,
    "entity_alignment_pct": 79.0,
    "language_confidence": 0.93,
    "financial_relevance_score": 0.83,
    "overall_quality_score": 0.79,
    "quality_tier": "GOOD",
    "model_versions": {
      "whisper": "whisper-large-v3-turbo",
      "llm_synthesis": "llama-3.3-70b-versatile",
      "llm_reasoning": "qwen/qwen3-32b",
      "ner_model": "urchade/gliner_medium-v2.1",
      "sentiment_model": "ProsusAI/finbert",
      "language_model": "papluca/xlm-roberta-base-language-detection"
    }
  },
  "risk_assessment": "...",
  "key_insights": ["..."],
  "key_points": ["..."],
  "transcript": "...",
  "expert_reasoning_points": "...",
  "expert_reasoning": "...",
  "model_attribution": {},
  "confidence_score": 0.83,
  "timing": {
    "asr_s": 2.6,
    "normalization_s": 1.2,
    "expert_nlp_s": 1.8,
    "synthesis_s": 3.4,
    "total_s": 9.7
  },
  "raw_asr_text": "..."
}
```

### Response Shape: `GET /api/results`

```json
{
  "count": 2,
  "results": [
    {
      "conversation_id": "call_abc12345",
      "chat_thread_id": "thr_efgh6789",
      "timestamp": "2026-04-05T00:00:00Z",
      "financial_topic": "DEBT_MANAGEMENT",
      "risk_level": "MEDIUM",
      "executive_summary": "...",
      "quality_metrics": {}
    }
  ]
}
```

### Response Shape: `PUT /api/update/{id}`

```json
{
  "status": "updated"
}
```

### Response Shape: `GET /api/report/{id}`

```text
Content-Type: application/pdf  (or text/csv)
Body: file stream
```

### Response Shape: `GET /api/health`

```json
{
  "status": "finflux-v4.2-online",
  "security": "AES-256 enabled",
  "api_key_loaded": true,
  "models": [
    "LangDetect",
    "FinBERT",
    "GLiNER",
    "DeBERTa",
    "Qwen",
    "Llama-3.1"
  ]
}
```

### Response Shape: `GET /api/quality/summary`

```json
{
  "average_overall_quality_score": 0.7812,
  "quality_tier_distribution": {
    "EXCELLENT": 7,
    "GOOD": 19,
    "ACCEPTABLE": 4,
    "LOW": 1
  },
  "average_asr_confidence_by_language": {
    "hi": 0.91,
    "en": 0.89,
    "hinglish": 0.9
  },
  "quality_trend_last_10": [
    {
      "conversation_id": "call_a1",
      "overall_quality_score": 0.82,
      "quality_tier": "GOOD",
      "created_at": "2026-04-05T00:00:00Z"
    }
  ],
  "count": 31
}
```

### Response Shape: `GET /api/benchmark`

```json
{
  "detail": "Not Found"
}
```

## 8) Bonus Features Implemented

- [x] Real-time financial topic detection during recording
- [x] Emotion and sentiment analysis via FinBERT with positive/negative/neutral breakdown
- [x] Personalized financial reminders from pattern matching across conversation history
- [x] Risk score estimation with four-tier CRITICAL/HIGH/MEDIUM/LOW classification
- [x] Visualization dashboard with risk trends and entity analytics
- [x] Hybrid self-querying RAG with pgvector semantic search and metadata filtering
- [x] AES-256 audio encryption with PBKDF2 key derivation
- [x] PII masking for Indian context covering PAN/Aadhaar/phone/email
- [x] Prompt injection detection with keyword pattern matching
- [x] Language percentage breakdown showing Hindi/English/Hinglish distribution per conversation

## 9) Sample Output (Complete)

### Input (Hinglish)

```text
Yaar mera home loan ka EMI bahut zyada ho gaya hai, soch raha hoon prepayment kar doon, aur SIP bhi five thousand se badha ke eight thousand karna chahta hoon.
```

### Output (Realistic Structured JSON)

```json
{
  "response_mode": "analysis",
  "conversation_id": "call_7f31a9c2",
  "user_id": "2dd9f6a2-2ed4-4c5f-8e8f-23f7b851cf2a",
  "chat_thread_id": "thr_b40f9ad1",
  "input_mode": "text",
  "raw_user_input": "Yaar mera home loan ka EMI bahut zyada ho gaya hai, soch raha hoon prepayment kar doon, aur SIP bhi five thousand se badha ke eight thousand karna chahta hoon.",
  "timestamp": "2026-04-05T08:37:10Z",
  "language": "hi",
  "language_confidence": 0.92,
  "language_breakdown": {
    "hindi": 40.1,
    "hinglish": 34.8,
    "english": 25.1
  },
  "script_preserved": true,
  "financial_topic": "DEBT_MANAGEMENT",
  "topic_top3": [
    {
      "topic": "DEBT_MANAGEMENT",
      "score": 0.86
    },
    {
      "topic": "INVESTMENT_PLANNING",
      "score": 0.81
    },
    {
      "topic": "BUDGETING",
      "score": 0.63
    }
  ],
  "strategic_intent": "Consolidation",
  "risk_level": "MEDIUM",
  "financial_sentiment": "Neutral",
  "sentiment_breakdown": {
    "positive": 0.19,
    "neutral": 0.54,
    "negative": 0.27
  },
  "advice_request": false,
  "injection_attempt": false,
  "entities": [
    {
      "type": "LOAN",
      "value": "home loan",
      "confidence": 0.95,
      "source": "gliner"
    },
    {
      "type": "EMI",
      "value": "EMI",
      "confidence": 0.92,
      "source": "gliner"
    },
    {
      "type": "ACTION",
      "value": "prepayment",
      "confidence": 0.9,
      "source": "rule_based"
    },
    {
      "type": "SIP AMOUNT",
      "value": "five thousand",
      "confidence": 0.91,
      "source": "rule_based"
    },
    {
      "type": "SIP AMOUNT",
      "value": "eight thousand",
      "confidence": 0.91,
      "source": "rule_based"
    }
  ],
  "executive_summary": "The user is signaling pressure from a rising home-loan EMI and is actively considering a prepayment move to reduce debt stress. In the same turn, they indicate an intent to increase SIP from approximately INR 5,000 to INR 8,000, which implies confidence in future monthly surplus but also creates dual cash-flow obligations. The key near-term decision is sequencing: prepayment timing should be aligned so liquidity buffer is not compromised while stepping up investment. The conversation reflects a practical consolidation posture where debt burden reduction and disciplined long-term investing are being balanced in one plan.",
  "summary": "The user is signaling pressure from a rising home-loan EMI and is actively considering a prepayment move to reduce debt stress. In the same turn, they indicate an intent to increase SIP from approximately INR 5,000 to INR 8,000, which implies confidence in future monthly surplus but also creates dual cash-flow obligations. The key near-term decision is sequencing: prepayment timing should be aligned so liquidity buffer is not compromised while stepping up investment. The conversation reflects a practical consolidation posture where debt burden reduction and disciplined long-term investing are being balanced in one plan.",
  "future_gearing": "If surplus remains stable over the next 60-90 days, the user can phase prepayment first and then lock the higher SIP amount. A staggered execution pattern should lower stress while preserving compounding intent.",
  "future_insights": [
    {
      "insight_type": "COMMITMENT_FOLLOWUP",
      "insight_text": "Reconfirm whether the home-loan prepayment is executed this month before finalizing the SIP step-up.",
      "urgency": "HIGH",
      "days_relevant": 14
    },
    {
      "insight_type": "RISK_REBALANCING",
      "insight_text": "Rising fixed EMI load and higher SIP together warrant a liquidity buffer check before additional commitments.",
      "urgency": "MEDIUM",
      "days_relevant": 21
    },
    {
      "insight_type": "SAVINGS_OPPORTUNITY",
      "insight_text": "A small discretionary spend reduction can fund part of the SIP increase without delaying debt reduction.",
      "urgency": "LOW",
      "days_relevant": 30
    }
  ],
  "reminders": [
    {
      "reminder_type": "COMMITMENT_PENDING",
      "reminder_text": "A prior decision-level commitment is older than 7 days and has no fulfilled marker; prompt execution check.",
      "urgency": "HIGH"
    }
  ],
  "quality_metrics": {
    "asr_confidence": 0.9,
    "ner_coverage_pct": 71.0,
    "rouge1_recall": 0.63,
    "entity_alignment_pct": 80.0,
    "language_confidence": 0.92,
    "financial_relevance_score": 0.86,
    "overall_quality_score": 0.794,
    "quality_tier": "GOOD",
    "model_versions": {
      "whisper": "none_text_input",
      "llm_synthesis": "llama-3.3-70b-versatile",
      "llm_reasoning": "qwen/qwen3-32b",
      "ner_model": "urchade/gliner_medium-v2.1",
      "sentiment_model": "ProsusAI/finbert",
      "language_model": "papluca/xlm-roberta-base-language-detection"
    }
  },
  "risk_assessment": "Risk is MEDIUM because debt-service pressure is explicitly increasing while investment contribution is also being stepped up. The simultaneous obligation increase can stress cash-flow if income variability or unplanned expenses occur. Maintaining emergency liquidity before executing both actions is the recommended control.",
  "key_insights": [
    "The user explicitly reports higher home-loan EMI burden, indicating debt pressure is current and not hypothetical.",
    "The statement about prepayment shows active debt optimization intent rather than passive concern.",
    "SIP increase from 5,000 to 8,000 indicates growth intent but raises fixed monthly commitment.",
    "The best execution path is staged: protect liquidity first, then scale SIP after prepayment visibility."
  ],
  "key_points": [
    "The user explicitly reports higher home-loan EMI burden, indicating debt pressure is current and not hypothetical.",
    "The statement about prepayment shows active debt optimization intent rather than passive concern.",
    "SIP increase from 5,000 to 8,000 indicates growth intent but raises fixed monthly commitment.",
    "The best execution path is staged: protect liquidity first, then scale SIP after prepayment visibility."
  ],
  "transcript": "Yaar mera home loan ka EMI bahut zyada ho gaya hai, soch raha hoon prepayment kar doon, aur SIP bhi five thousand se badha ke eight thousand karna chahta hoon.",
  "expert_reasoning_points": "- STRATEGIC WALL [Debt-First Stability]: EMI stress is immediate, so debt burden control has priority before aggressive contribution expansion.\n- TECHNICAL WALL [Cashflow Sequencing]: Prepayment and SIP increase should be phased to avoid over-commitment in one billing cycle.\n- COMPLIANCE WALL [Decision Control]: Mark prepayment execution as a trackable milestone and validate 30-day liquidity adequacy before SIP step-up lock.",
  "confidence_score": 0.86,
  "timing": {
    "asr_s": 0.0,
    "normalization_s": 1.1,
    "expert_nlp_s": 1.6,
    "synthesis_s": 3.3,
    "total_s": 6.7
  }
}
```

## 10) Validation Metrics Explained

| Metric | What it measures | How it is calculated | Practical expected range |
| --- | --- | --- | --- |
| `asr_confidence` | Speech transcription certainty | Average word probability from Whisper `verbose_json`; fallback `0.85` when unavailable | `0.87-0.94` for clear Indian-accented speech |
| `ner_coverage_pct` | Financial term capture density | Matched financial lexicon terms / total lexicon size * 100 | `65-80%` in financial conversations |
| `rouge1_recall` | Summary faithfulness | Token overlap recall of summary against transcript (stopwords removed) | `0.55-0.72` |
| `entity_alignment_pct` | Entity grounding in summary | Percent of extracted entities appearing in summary/key insights via fuzzy matching | `70-85%` |
| `language_confidence` | Language classifier certainty | Direct pass-through from XLM-Roberta score | `0.88-0.96` |
| `financial_relevance_score` | Financial intent certainty | DeBERTa top-topic confidence | `0.72-0.91` |

Overall quality formula:

```text
overall_quality_score =
  asr_confidence * 0.30 +
  (ner_coverage_pct/100) * 0.20 +
  rouge1_recall * 0.20 +
  (entity_alignment_pct/100) * 0.15 +
  language_confidence * 0.10 +
  financial_relevance_score * 0.05
```

Tier mapping:

- `> 0.82` => `EXCELLENT`
- `> 0.65` => `GOOD`
- `> 0.45` => `ACCEPTABLE`
- `<= 0.45` => `LOW`

## 11) Security Architecture

### 11.1 AES-256 Audio Encryption

Stored audio is encrypted using Fernet with PBKDF2HMAC key derivation (`SHA-256`, 100,000 iterations). This protects at-rest audio artifacts even if file storage is exposed.

### 11.2 PII Masking Before LLM Exposure

Regex masking covers Indian PAN, 12-digit Aadhaar, Indian mobile numbers, email addresses, and generic bank-account patterns. Sensitive values are masked before synthesis prompts.

### 11.3 Prompt Injection Detection

Keyword pattern matching detects override-style phrases and risky prompt manipulations. Injection markers are included in response metadata and logs.

### 11.4 Non-Advisory Guardrail

The system is designed for structured intelligence extraction, not direct financial advice. Prompt constraints enforce analytical framing and avoid prescriptive recommendation behavior.

## 12) Project Structure

```text
finflux/
├── api/                      # FastAPI backend orchestration + endpoints
│   ├── server.py             # Main API pipeline, auth, routes, quality summary
│   ├── security.py           # AES encryption, PII masking, injection checks
│   └── storage.py            # Supabase bridge + SQLite fallback + retrieval
├── src/finflux/              # Core NLP modules and configuration
│   ├── config.py             # Model identifiers and runtime config
│   └── modules/
│       └── insight_engine/   # ASR adapters, synthesis, financial model stack
├── frontend/                 # React + Vite client
│   ├── src/pages/            # Dashboard, Record, History views
│   └── package.json          # Frontend dependencies and scripts
├── scripts/                  # Utilities, dataset, e2e checks
│   ├── _e2e_live_check.py    # End-to-end API validation script
│   ├── download_models.py    # Model warm-download utility
│   └── train_asr.py          # LoRA fine-tuning pipeline entrypoint
├── sql/                      # Supabase migrations (001-012)
├── configs/                  # YAML pipeline/training/dataset configs
├── data/                     # Runtime data, encrypted audio, fixtures
├── requirements.txt          # Python dependencies
└── README.md                 # Submission-ready technical documentation
```

## 13) Known Limitations and Honest Scope

FinFlux performs strongly on Hindi/Hinglish/English conversations for transcription, entity extraction, strategic synthesis, secure storage, and semantic retrieval over conversation memory. The system is production-structured with endpoint-level observability and quality scoring suitable for evaluation dashboards.

LoRA fine-tuning scripts and pipeline assets are present, but full training was not executed during the hackathon window due to compute/time constraints. For languages beyond Hindi (for example Tamil/Telugu/Kannada), Whisper transcription works, but current financial NER coverage is lower. Real-time recording detection uses keyword matching rather than full DeBERTa inference to keep latency low.

## 14) Non-Breaking Code Quality Notes

To stay fully non-breaking in this live system, quality improvements are documentation-first only.

- Add docstrings to existing functions without changing function signatures.
- Add type hint annotations as comments only where useful.
- Improve inline comments for complex logic paths.
- Avoid renames of variables, response fields, imports, routes, and modules.

This README and `.env.example` are safe additions and do not alter runtime interfaces.
