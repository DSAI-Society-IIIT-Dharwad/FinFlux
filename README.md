# FinFlux: Multilingual Financial Decision Intelligence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production--Grade-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-Vite%20%2B%20Tailwind-61dafb?style=for-the-badge&logo=react)](https://reactjs.org/)
[![Groq](https://img.shields.io/badge/Intelligence-Groq%20Llama%203.3--70B-orange?style=for-the-badge)](https://groq.com/)

> **Project "Armor" Submission:** A high-fidelity system that captures informal Indian financial conversations (Hindi, English, Hinglish) and converts unstructured speech into executive-grade structured intelligence.

---

## 🌟 The Vision: Why FinFlux?

In the Indian context, critical financial decisions aren't made in bank branches—they happen over dinner, in family group chats, and during casual office calls. These conversations are **multilingual**, **informal**, and **fleeting**. When the "EMI manage ho jayega" or "SIP badha dete hain" moments pass without documentation, financial accountability dies.

**FinFlux** is an audio-native AI platform designed to capture these moments securely. It doesn't just transcribe; it **thinks** like a McKinsey analyst—extracting risk, intent, and commitments while maintaining a persistent "Financial Memory" for the user.

---

## 🚀 Key Features (V4.2+ Finalized)

### 📊 Elite 3-Page PDF Strategic Reports
No more messy CSVs. FinFlux now generates a professional, multi-page compliance-ready report containing:
- **Executive Cover**: High-level intent and summary.
- **Risk & Metric Audit**: Visual NLP distributions and entity grounding.
- **Expert Reasoning Walls**: Deep-dive strategy synthesized by Qwen-32B & Llama-70B.

### 🌓 Premium "Independent Dual-Scroll" UI
A state-of-the-art terminal experience:
- **Left Pane**: Dynamic recording and chat interface.
- **Right Pane**: Auto-snapping intelligence insights that stay locked to your viewport.
- **Micro-Animations**: Decrypted-text branding effects and glassmorphic neon design.

### 🌍 Multilingual NLP (Hinglish-First)
- **Zero-Shot Mastery**: Using `mDeBERTa-v3` label augmentation to achieve high-confidence classification even in pure Hindi script.
- **Precision NER**: Special rule-based catchers for Indian financial terms like **GST, GSTIN, ITR, TDS, and Lakhs**.
- **PII Guardrail**: Automatic masking of Aadhaar, PAN, and phone numbers before data ever touches an LLM.

---

## 🏗️ Technical Architecture

FinFlux employs a sophisticated **10-Model Orchestration Pipeline**:

1.  **ASR**: `whisper-large-v3-turbo` (Groq) for sub-second multilingual transcription.
2.  **Gating**: `XLM-Roberta` identifies if the speaker is using Hindi, English, or Hinglish.
3.  **Synthesis**: Dual-layer reasoning using **Llama 3.3-70B** and **Qwen-32B** to avoid hallucination.
4.  **NER**: `GLiNER` + RegEx rule-engine for capturing entities (EMI, SIP, Loans).
5.  **Sentiment**: `FinBERT` specifically tuned for financial market emotion.
6.  **Storage**: Supabase PGVector for semantic retrieval and "Financial Memory" RAG.

---

## 🛠️ Humanized Setup Guide

### 1. The Essentials
Ensure you have **Python 3.11+**, **Node.js 20+**, and a **Groq API Key**.

### 2. Backend Ignition
```bash
# Clone and enter the vault
git clone https://github.com/your-repo/finflux.git
cd finflux

# Create your identity
cp .env.example .env  # Fill in your GROQ_API_KEY and SUPABASE_URL

# Install the brains
pip install -r requirements.txt
python scripts/download_models.py

# Launch the server
python -m uvicorn api.server:app --port 8000
```

### 3. Frontend Fusion
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173 to see the neon glow
```

---

## 🔒 Security & Privacy

We treat financial data with extreme caution:
- **Audio Vault**: All audio is encrypted with **AES-256 (Fernet)** using PBKDF2 key derivation before hitting the disk.
- **Sanitization**: Every transcript is scrubbed for PII (Personally Identifiable Information) before synthesis.
- **Guardrails**: System prompts strictly enforce **Non-Advisory status**—we provide intelligence, not financial advice.

---

## 🏆 Hackathon Goal Alignment

| Goal | FinFlux implementation Status |
| --- | --- |
| Multilingual Speech | **Achieved** (Whisper-v3 + XLM-R Gating) |
| Structured Summaries | **Achieved** (Elite 3-Page PDF + McKinsey Synthesis) |
| Conversation History | **Achieved** (Thread-aware RAG + Semantic Search) |
| User Privacy | **Achieved** (AES-256 + PII Masking) |
| Review & Edit | **Achieved** (Inline Transcript Editor with real audio-sync) |

---

## 👨‍💻 Developed By
*Built for the AI/FinTech/Speech Processing Track.*

"Transforming informal conversations into actionable financial intelligence." 🚀
