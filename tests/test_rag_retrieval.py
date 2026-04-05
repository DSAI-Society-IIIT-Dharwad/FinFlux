"""Tests for RAG retrieval, user isolation, and semantic search degradation handling."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_search_memories_returns_proper_structure():
    """search_memories must always return dict with 'results' key and 'degraded' flag."""
    from api.storage import search_memories

    result = search_memories(
        user_id="nonexistent_user_test",
        query_text="investment planning",
        filters={},
        n_results=5,
    )
    assert isinstance(result, dict)
    assert "results" in result
    assert isinstance(result["results"], list)
    # Must have 'degraded' flag to distinguish "no data" from "search failed"
    assert "degraded" in result


def test_search_memories_user_isolation():
    """Searches for different users must not leak data across users."""
    from api.storage import search_memories

    r1 = search_memories(user_id="isolation_user_A", query_text="loan EMI", n_results=5)
    r2 = search_memories(user_id="isolation_user_B", query_text="loan EMI", n_results=5)

    ids_1 = {r.get("conversation_id") for r in r1.get("results", []) if isinstance(r, dict)}
    ids_2 = {r.get("conversation_id") for r in r2.get("results", []) if isinstance(r, dict)}

    # No ID overlap between isolated users
    assert ids_1.isdisjoint(ids_2), f"User isolation violated: {ids_1 & ids_2}"


def test_embedding_model_loads():
    """The sentence transformer embedding model must load and produce 384-dim vectors."""
    from api.storage import embed_model

    vec = embed_model.encode("test financial query").tolist()
    assert isinstance(vec, list)
    assert len(vec) == 384  # all-MiniLM-L6-v2 produces 384-dim vectors


def test_supabase_vector_enabled_check():
    """_supabase_vector_enabled should reflect environment configuration."""
    from api.storage import _supabase_vector_enabled

    result = _supabase_vector_enabled()
    assert isinstance(result, bool)
    # If env vars are set, it should be True
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        assert result is True


def test_decompose_retrieval_query_structure():
    """Self-query decomposition must return semantic_query and filters dict."""
    from api.server import SYNTHESIS

    result = SYNTHESIS.decompose_retrieval_query(
        user_message="What was my home loan EMI discussion?",
        detected_topic="DEBT_MANAGEMENT",
    )
    assert isinstance(result, dict)
    assert "semantic_query" in result
    assert "filters" in result
    assert isinstance(result["semantic_query"], str)
    assert isinstance(result["filters"], dict)
    assert len(result["semantic_query"]) > 0
