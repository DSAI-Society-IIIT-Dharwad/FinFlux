"""V4.2 Full Model Stack: LangDetect, DeBERTa, FinBERT, and GLiNER."""
import os
import torch
import json
from typing import Dict, Any, List
from transformers import pipeline
from finflux import config

try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

class ProductionExpertModule:
    """Consolidated 4-Model High-Performance NLP Stack (V4.2)."""

    _instance = None

    def __new__(cls):
        """Singleton pattern: Load models ONCE to save 3GB VRAM."""
        if cls._instance is None:
            cls._instance = super(ProductionExpertModule, cls).__new__(cls)
            cls._instance._init_stack()
        return cls._instance

    def _init_stack(self):
        self.device = 0 if config.USE_CUDA and torch.cuda.is_available() else -1
        print(f"[ProductionExpertModule] Initializing Modular Stack on device={self.device}...")

        # ── Stage 1: Precise Language Detection (XLM-Roberta) ──
        self.lang_pipe = pipeline("text-classification", model=config.HF_LANG_DETECT, device=self.device)

        # ── Stage 2: Topic & Advice Classification (DeBERTa v3) ──
        self.topic_pipe = pipeline("zero-shot-classification", model=config.HF_ZERO_SHOT, device=self.device)
        self.topics = ["investment", "loan", "EMI", "insurance", "mutual fund", "gold", "stock", "crypto", "property", "general"]
        self.advice_labels = ["asking for financial advice", "general discussion"]

        # ── Stage 3: Financial Sentiment & Strategy (FinBERT) ──
        self.fin_pipe = pipeline("sentiment-analysis", model=config.HF_FINBERT, device=self.device)

        # ── Stage 4: Zero-Shot Entity Extraction (GLiNER) ──
        if HAS_GLINER:
            print(f"[ProductionExpertModule] Loading GLiNER Specialist: {config.HF_NER_FINANCIAL}")
            self.gliner = GLiNER.from_pretrained(config.HF_NER_FINANCIAL).to("cuda" if self.device == 0 else "cpu")
        else:
            self.gliner = None
            self.ner_fallback = pipeline("ner", model=config.HF_NER_GENERAL, aggregation_strategy="simple", device=self.device)

        # Labels for GLiNER semantic extraction
        self.labels = [
            "INVESTMENT", "LOAN", "EMI", "INSURANCE", "MUTUAL FUND", "GOLD", "STOCK", 
            "PROPERTY", "AMOUNT", "INTEREST RATE", "TENURE", "BANK", "FINANCIAL GOAL"
        ]

    def process(self, text: str) -> Dict[str, Any]:
        """Run the full integrated pipeline with truncation for stability."""
        if not text.strip(): return {"error": "Empty input"}

        try:
            # 0. Truncation Gating (Prevent model overflow)
            # Most local models (DeBERTa, FinBERT) have 512 token limit.
            # Truncating to ~1800 chars safely fits.
            safe_text = text[:1800]

            # 1. Language Logic
            lang_res = self.lang_pipe(safe_text)[0]
            detected_lang = lang_res["label"]

            # 2. Topic & Advice Logic
            topic_res = self.topic_pipe(safe_text, candidate_labels=self.topics, multi_label=True)
            advice_res = self.topic_pipe(safe_text, candidate_labels=self.advice_labels, multi_label=False)
            top_topic = topic_res["labels"][0]
            confidence_score = round(topic_res["scores"][0], 2)

            # 3. FinBERT Sentiment/Urgency
            fin_res = self.fin_pipe(safe_text)[0]
            sentiment_label = fin_res["label"]

            # 4. NER Extraction (GLiNER handled separately but safe_text used for consistency)
            ner_items = []
            if self.gliner:
                entities = self.gliner.predict_entities(safe_text, self.labels)
                for ent in entities:
                    label = ent["label"].replace(" ", "_").upper()
                    if label == "MUTUAL_FUND": label = "MUTUAL_FUND"
                    ner_items.append({"type": label, "value": ent["text"], "context": f"Extracted via GLiNER {config.HF_NER_FINANCIAL}"})
            else:
                for item in self.ner_fallback(text):
                    ner_items.append({"type": item["entity_group"], "value": item["word"], "context": "Fallback NER"})

            return {
                "detected_language": detected_lang,
                "topic": top_topic,
                "confidence_score": confidence_score,
                "financial_sentiment": sentiment_label,
                "is_advice_request": advice_res["labels"][0] == "asking for financial advice" and advice_res["scores"][0] > 0.6,
                "entities": ner_items
            }
        except Exception as e:
            print(f"[ProductionExpertModule] Pipeline error: {e}")
            return {"error": str(e)}
