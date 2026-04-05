"""Semantic memory and Supabase configuration tests."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_supabase_semantic_config_present():
    """Semantic search is Supabase-only; ensure required env vars are configured."""
    required = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_VECTOR_RPC"]
    missing = [name for name in required if not os.environ.get(name)]
    assert not missing, f"Missing required Supabase semantic search env vars: {', '.join(missing)}"


def test_search_memories_returns_dict():
    """search_memories must return a dict with results key."""
    from api.storage import search_memories

    result = search_memories(
        user_id="test_semantic_user",
        query_text="investment portfolio SIP mutual fund",
        n_results=5,
    )
    assert isinstance(result, dict)
    assert "results" in result
    assert isinstance(result["results"], list)


def test_search_memories_has_degraded_flag():
    """search_memories must include a degraded flag for RAG health awareness."""
    from api.storage import search_memories

    result = search_memories(
        user_id="test_degraded_flag",
        query_text="loan EMI",
        n_results=3,
    )
    assert "degraded" in result, "search_memories must return 'degraded' flag"
    assert isinstance(result["degraded"], bool)


def test_embedding_dimensions():
    """Embedding model must produce 384-dimensional vectors for all-MiniLM-L6-v2."""
    from api.storage import embed_model

    vec = embed_model.encode("financial investment query").tolist()
    assert len(vec) == 384


def test_build_embedding_source():
    """_build_embedding_source should combine summary, intent, topic, and entities."""
    from api.storage import _build_embedding_source

    data = {
        "executive_summary": "Client plans SIP investment",
        "strategic_intent": "Growth & Wealth Building",
        "entities": [
            {"value": "SIP", "confidence": 0.95},
            {"value": "5 lakh", "confidence": 0.90},
        ],
    }
    source = _build_embedding_source(data, "INVESTMENT_PLANNING")
    assert "SIP" in source
    assert "INVESTMENT_PLANNING" in source
    assert "Growth" in source


if __name__ == "__main__":
    test_supabase_semantic_config_present()
    test_search_memories_returns_dict()
    test_search_memories_has_degraded_flag()
    test_embedding_dimensions()
    test_build_embedding_source()
    print("All semantic memory tests passed.")
