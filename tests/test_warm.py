"""Tests for model warm-up and NLP pipeline initialization."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def test_config_loads():
    """FinFlux config must load with all required model identifiers."""
    from finflux import config

    assert config.GROQ_STT_MODEL is not None
    assert config.GROQ_LLM_MODEL is not None
    assert config.HF_FINBERT is not None
    assert config.HF_ZERO_SHOT is not None
    assert config.HF_NER_FINANCIAL is not None
    assert config.HF_LANG_DETECT is not None


def test_expert_module_singleton():
    """ProductionExpertModule must follow singleton pattern."""
    from finflux.modules.insight_engine.financial_models import ProductionExpertModule

    a = ProductionExpertModule()
    b = ProductionExpertModule()
    assert a is b, "ProductionExpertModule must be a singleton"


def test_synthesis_engine_initializes():
    """ExpertSynthesisEngine must initialize without errors."""
    from finflux.modules.insight_engine.llm_adapters import ExpertSynthesisEngine

    engine = ExpertSynthesisEngine()
    assert engine.main_model is not None
    assert engine.reason_model is not None
    assert engine.fast_model is not None
    assert engine.url is not None


def test_financial_lexicon_exists():
    """Server must have a populated financial lexicon for NER coverage."""
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from server import FINANCIAL_LEXICON

    assert isinstance(FINANCIAL_LEXICON, list)
    assert len(FINANCIAL_LEXICON) > 100, "Financial lexicon should have 100+ terms"


def test_keyword_detection_structure():
    """Financial keyword detection must return expected response structure."""
    from finflux.modules.insight_engine.financial_models import ProductionExpertModule

    module = ProductionExpertModule()
    # Must warm before use, but detect_financial_keywords auto-warms
    result = module.detect_financial_keywords("I want to invest in SIP mutual fund")
    assert isinstance(result, dict)
    assert "financial_detected" in result
    assert "matched_keywords" in result
    assert isinstance(result["matched_keywords"], list)
    assert result["financial_detected"] is True


def test_keyword_detection_non_financial():
    """Non-financial text should not trigger financial detection."""
    from finflux.modules.insight_engine.financial_models import ProductionExpertModule

    module = ProductionExpertModule()
    result = module.detect_financial_keywords("What is the weather today in Bangalore?")
    assert result["financial_detected"] is False
    assert len(result["matched_keywords"]) == 0
