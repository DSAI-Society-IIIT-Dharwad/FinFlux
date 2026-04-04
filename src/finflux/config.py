"""FinFlux Global Configuration: V4.2 Modular Pro."""
import os
from dotenv import load_dotenv

# Automatic .env loading (Resolves 401 Unauthorized)
load_dotenv()

# ── GROQ API ──────────────────────────────────────
GROQ_API_KEY          = os.environ.get("GROQ_API_KEY")
GROQ_STT_MODEL        = "whisper-large-v3-turbo"
GROQ_STT_HQ_MODEL     = "whisper-large-v3"
GROQ_LLM_MODEL        = "llama-3.3-70b-versatile"
GROQ_LLM_FAST_MODEL   = "llama-3.1-8b-instant"
GROQ_LLM_REASON_MODEL = "qwen/qwen3-32b" # Specialist Reasoner

# ── HUGGING FACE ──────────────────────────────────
HF_FINBERT            = "ProsusAI/finbert" # Financial Sentiment
HF_ZERO_SHOT          = "cross-encoder/nli-deberta-v3-small" # Generic Topic
HF_NER_FINANCIAL      = "urchade/gliner_medium-v2.1" # Zero-Shot NER
HF_NER_GENERAL        = None  # skip, wrong model
HF_LANG_DETECT        = "papluca/xlm-roberta-base-language-detection" # Precise Lang Gating
HF_INDIC_NER          = None  # skip, not downloaded
HF_INDIC_STT          = None  # skip, not downloaded

# ── CONFIG ────────────────────────────────────────
USE_CUDA              = os.environ.get("USE_CUDA", "true").lower() == "true"
DEBUG_MODE            = os.environ.get("DEBUG_MODE", "false").lower() == "true"
