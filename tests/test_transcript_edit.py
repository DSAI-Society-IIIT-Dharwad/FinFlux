"""Tests for transcript edit and reanalyze flow."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_update_conversation_field_mapping():
    """storage.update_conversation field_map must include all editable fields."""
    from api.storage import update_conversation

    # Verify the function exists and accepts expected signature
    import inspect
    sig = inspect.signature(update_conversation)
    params = list(sig.parameters.keys())
    assert "user_id" in params
    assert "conversation_id" in params
    assert "updates" in params


def test_conversation_log_has_all_required_columns():
    """ConversationLog model must have all columns needed for transcript edit flow."""
    from api.storage import ConversationLog

    required_columns = [
        "id", "user_id", "timestamp", "language", "financial_topic",
        "risk_level", "financial_sentiment", "executive_summary",
        "future_gearing", "strategic_intent", "risk_assessment",
        "expert_reasoning", "transcript", "advice_request",
        "injection_attempt", "confidence_score", "entities",
        "key_points", "timing_data", "chat_thread_id",
        "input_mode", "raw_user_input", "future_insights", "reminders",
    ]
    model_columns = {col.name for col in ConversationLog.__table__.columns}
    for col in required_columns:
        assert col in model_columns, f"Missing column: {col}"


def test_get_conversation_by_id_structure():
    """get_conversation_by_id should return None for nonexistent IDs."""
    from api.storage import get_conversation_by_id

    result = get_conversation_by_id(user_id="test_user_xxx", conversation_id="nonexistent_id")
    # Should return None, not raise an exception
    assert result is None


def test_quality_metrics_computation():
    """Quality metrics should produce valid scores and tiers."""
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from server import _compute_quality_metrics

    metrics = _compute_quality_metrics(
        transcript="I want to invest 5 lakh in SIP mutual fund for 10 years",
        executive_summary="Client seeks SIP investment of 5 lakh for long-term wealth creation",
        key_insights=["SIP investment plan", "10-year horizon"],
        entities=[{"type": "SIP AMOUNT", "value": "5 lakh"}],
        language_confidence=0.95,
        financial_relevance_score=0.88,
        asr_meta={"asr_confidence": 0.92},
        input_mode="text",
    )

    assert "overall_quality_score" in metrics
    assert "quality_tier" in metrics
    assert "asr_confidence" in metrics
    assert "ner_coverage_pct" in metrics
    assert "rouge1_recall" in metrics
    assert "entity_alignment_pct" in metrics
    assert "model_versions" in metrics
    assert metrics["quality_tier"] in {"EXCELLENT", "GOOD", "ACCEPTABLE", "LOW"}
    assert 0.0 <= metrics["overall_quality_score"] <= 1.0


def test_ner_coverage_with_financial_text():
    """NER coverage should be > 0 for text containing financial terms."""
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from server import _ner_coverage_pct

    coverage = _ner_coverage_pct("I want to invest in SIP mutual fund with EMI payment for home loan")
    assert coverage > 0.0, "Financial text should have non-zero NER coverage"


def test_rouge1_recall_matching():
    """ROUGE-1 recall should be > 0 when summary overlaps with transcript."""
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from server import _rouge1_recall

    recall = _rouge1_recall(
        summary="Client wants SIP investment for long-term growth",
        transcript="I want SIP investment for long-term growth and returns"
    )
    assert recall > 0.0, "Overlapping summary and transcript should have positive ROUGE-1 recall"
