"""Tests for multi-turn conversation handling: fresh-start detection, thread continuity, summary consistency."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "api"))

from server import _is_fresh_start_intent, _apply_optional_memory_filters, _generate_deterministic_reminders


# ── Fresh-Start Intent Detection ────────────────────

def test_fresh_start_explicit_reset():
    """Explicit reset phrases should trigger fresh start."""
    assert _is_fresh_start_intent("start fresh", "thr_001", [], "N/A") is True
    assert _is_fresh_start_intent("new chat please", "thr_001", [], "N/A") is True
    assert _is_fresh_start_intent("shuru se baat karo", "thr_001", [], "N/A") is True


def test_fresh_start_greeting_on_new_thread():
    """Greeting on a new (empty) thread should trigger fresh start."""
    assert _is_fresh_start_intent("Hello!", "thr_new", [], "N/A") is True
    assert _is_fresh_start_intent("Namaste", "thr_new", [], "N/A") is True


def test_no_fresh_start_on_financial_message():
    """Financial discussion should NOT trigger fresh start."""
    history = [{"chat_thread_id": "thr_001", "financial_topic": "INVESTMENT_PLANNING"}]
    assert _is_fresh_start_intent(
        "I want to invest 5 lakh in SIP", "thr_001", history, "INVESTMENT_PLANNING"
    ) is False


def test_no_fresh_start_on_existing_thread_greeting():
    """Greeting on an existing thread with history should NOT trigger fresh start."""
    history = [{"chat_thread_id": "thr_001", "financial_topic": "DEBT_MANAGEMENT"}]
    assert _is_fresh_start_intent("Hi there", "thr_001", history, "DEBT_MANAGEMENT") is False


# ── Memory Filter Application ───────────────────────

def test_memory_filter_by_topic():
    """Filters should narrow results by financial_topic."""
    rows = [
        {"financial_topic": "insurance", "executive_summary": "health plan"},
        {"financial_topic": "loan", "executive_summary": "home EMI"},
        {"financial_topic": "insurance", "executive_summary": "term plan"},
    ]
    filtered = _apply_optional_memory_filters(rows, {"financial_topic": "insurance"})
    assert len(filtered) == 2
    assert all(r["financial_topic"] == "insurance" for r in filtered)


def test_memory_filter_empty_returns_all():
    """No filters should return all rows."""
    rows = [{"financial_topic": "loan"}, {"financial_topic": "investment"}]
    filtered = _apply_optional_memory_filters(rows, {})
    assert len(filtered) == 2


def test_memory_filter_no_match_returns_all():
    """When filter matches nothing, should return original rows (fallback safeguard)."""
    rows = [{"financial_topic": "loan"}, {"financial_topic": "investment"}]
    filtered = _apply_optional_memory_filters(rows, {"financial_topic": "crypto"})
    assert len(filtered) == 2  # falls back to all rows if narrowed is empty


# ── Deterministic Reminders ──────────────────────────

def test_reminders_product_pattern():
    """Products appearing 3+ times should generate PRODUCT_PATTERN reminder."""
    history = [
        {"financial_topic": "home_loan", "model_attribution": {}, "timestamp": "2026-01-01T00:00:00"},
        {"financial_topic": "home_loan", "model_attribution": {}, "timestamp": "2026-01-02T00:00:00"},
        {"financial_topic": "home_loan", "model_attribution": {}, "timestamp": "2026-01-03T00:00:00"},
    ]
    reminders = _generate_deterministic_reminders(history)
    product_reminders = [r for r in reminders if r["reminder_type"] == "PRODUCT_PATTERN"]
    assert len(product_reminders) >= 1
    assert "home loan" in product_reminders[0]["reminder_text"].lower()


def test_reminders_empty_history():
    """Empty history should produce no reminders."""
    assert _generate_deterministic_reminders([]) == []


def test_reminders_max_5():
    """Reminders should never exceed 5."""
    # Create a large history that could trigger many reminders
    history = []
    for i in range(20):
        history.append({
            "financial_topic": f"topic_{i % 3}",
            "model_attribution": {},
            "timestamp": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
            "risk_level": "HIGH",
            "key_points": ["decision to proceed"],
            "executive_summary": "decision made to proceed",
        })
    reminders = _generate_deterministic_reminders(history)
    assert len(reminders) <= 5
