# FinFlux: Strategic Financial Intelligence Platform

## Problem Statement

The financial advisory and wealth management sector faces significant challenges in capturing and analyzing unstructured client communications. Manual transcription and summary generation lead to:

1. **Data Loss**: Critical financial details, subtle strategic intents, and specific commitments are often lost or incorrectly recorded during manual entry.
2. **Inconsistency**: Subjective interpretation of risk and sentiment varies drastically between analysts, creating inconsistent reporting.
3. **Multilingual Friction**: Standard automated speech recognition (ASR) systems struggle with code-switching (Hinglish) and regional nuances in financial terminology.
4. **Audit Latency**: High-fidelity records and PDF reports for compliance take hours or days to generate, delaying strategic response time.

FinFlux addresses these challenges by implementing an automated, high-fidelity 12-model pipeline that transforms raw audio into consistent, logic-backed strategic intelligence.

## Current Application Structure

The repository is organized into four distinct specialized layers:

### 1. API Services (api/)

The backend orchestration layer built with FastAPI. It handles:

- **Audio Processing**: Secure audio ingestion and landing zones.
- **Reporting Engine**: Dynamic PDF generation using ReportLab.
- **Data Persistence**: SQLAlchemy-based mapping of high-fidelity strategic fields.
- **Security**: PII masking during transcript normalization and AES-256 encryption for stored audio.

### 2. Strategic Frontend (frontend/)

A React/Vite application that serves as the executive command center.

- **DashboardView**: Real-time summary of historical intelligence and risk levels.
- **RecordView**: Interactive capture terminal with live editable transcripts and expert reasoning panels.
- **HistoryView**: Longitudinal tracking of recorded intelligence with PDF export capabilities.

### 3. Source Modules (src/finflux/)

The core NLP inference engine providing specialized local model execution:

- **insight_engine/**: High-level adapters for Groq-based cloud inference and McKinsey-style synthesis.
- **modules/**: Local specialized models for financial sentiment (FinBERT) and named entity recognition (GLiNER).

### 4. Roadmap Utilities (scripts/)

A technical suite for future research and optimization:

- **training/**: Entry points for Whisper LoRA fine-tuning and dataset manifest generation.
- **generation/**: Synthetic financial dialogue engines for Hindi/Hinglish RAG (Retrieval-Augmented Generation) training.

## Models and Pipeline Stages

FinFlux utilizes a sophisticated 12-model stack to achieve executive-grade strategic synthesis:

| Stage | Model / Component | Technical Implementation |
| :--- | :--- | :--- |
| Stage 1 | Whisper Turbo (Groq) | Near-real-time multilingual transcription (Hinglish/English). |
| Stage 2 | Llama-3-8B Fast | High-speed transcript normalization and PII identification. |
| Stage 3 | XLM-Roberta | Precise language identification for downstream model selection. |
| Stage 4 | DeBERTa-v3-small | Logical commitment and financial obligation detection. |
| Stage 5 | DeBERTa-v3-base | 14-dimensional financial topic classification and entity grounding. |
| Stage 6 | FinBERT | Financial-specific sentiment analysis focused on asset-class sentiment. |
| Stage 7 | GLiNER-Medium | Zero-shot named entity recognition for specialized financial types. |
| Stage 8 | Qwen-32B | Expert technical reasoning (Reasoning Chain) for Logic verification. |
| Stage 9 | Llama-3.1-70B | Executive strategic synthesis (Intent, Gearing, and Risk). |
| Stage 10 | ReportLab | Automated conversion of intelligence into professional PDF assets. |

## System Architecture and Technical Implementation

### Supabase Secure Bridge (RLS Enabled)

Use a controlled backend bridge: keep RLS enabled, keep service-role key only in backend, and always filter by current user in backend logic.

Set these environment variables for backend:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (server-side only, never in frontend)
- `SUPABASE_VECTOR_RPC=search_user_embeddings_bridge_service`
- `SUPABASE_CONV_THREADS_TABLE=ai_conversation_threads`
- `SUPABASE_CONV_MESSAGES_TABLE=ai_conversation_messages`

Run SQL migrations in order, including:

- `sql/010_backend_bridge_tables.sql`
- `sql/011_rpc_semantic_search_bridge_service.sql`

This gives AI retrieval through a safe backend endpoint with explicit user filtering while RLS remains enabled.

### Backend Stack

- **Web Framework**: FastAPI (Uvicorn) with asynchronous request handling.
- **Database**: SQLite with SQLAlchemy ORM for high-resolution strategic history tracking.
- **NLP Execution**: Integrated usage of local Transformers (PyTorch) and high-speed Groq inference via REST.
- **Storage Strategy**: Local storage for processed datasets and encrypted audio shards.

### Frontend Stack

- **Framework**: React 18+ with Vite for ultra-fast HMR (Hot Module Replacement).
- **Styling**: Custom CSS architecture focusing on glassmorphism, depth, and professional dark-mode aesthetics.
- **Dynamic Interactions**: Lucide-based iconography and real-time state management for transcript editing and persistence.

### Logic Flow

1. **Ingestion**: Audio is captured via React components and streamed to the API.
2. **ASR Layer**: Groq Whisper handles transcription, followed by Stage 2 normalization.
3. **Extraction**: The 4-model local stack (LangDetect, DeBERTa, FinBERT, GLiNER) extracts technical metadata and sentiment.
4. **Synthesis**: Qwen-32B generates internal logic points (Expert Technical Wall) which Llama-3.1-70B used to produce the final Strategic JSON.
5. **Persistence**: The resulting structured intelligence is mapped to the relational database and made available for the React Dashboard.
6. **Reporting**: Users can export the entire intelligence chain into a high-fidelity PDF report with a single click.
