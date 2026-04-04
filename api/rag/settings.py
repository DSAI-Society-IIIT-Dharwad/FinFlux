import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAG_CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", str(ROOT_DIR / "data" / "chroma"))
RAG_COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "finflux_sessions")
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
RAG_GROQ_QUERY_MODEL = os.environ.get("RAG_GROQ_QUERY_MODEL", "llama-3.1-8b-instant")
RAG_GROQ_INSIGHT_MODEL = os.environ.get("RAG_GROQ_INSIGHT_MODEL", "llama-3.3-70b-versatile")
