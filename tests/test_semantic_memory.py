import os

def test_supabase_semantic_config_present():
    """Semantic search is Supabase-only; ensure required env vars are configured."""
    required = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_VECTOR_RPC"]
    missing = [name for name in required if not os.environ.get(name)]
    assert not missing, f"Missing required Supabase semantic search env vars: {', '.join(missing)}"

if __name__ == "__main__":
    test_supabase_semantic_config_present()
