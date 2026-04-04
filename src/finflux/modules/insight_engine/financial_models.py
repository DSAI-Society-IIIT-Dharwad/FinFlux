"""V4.2 Full Model Stack: LangDetect, DeBERTa, FinBERT, and GLiNER."""
import os
import torch
import json
import re
from typing import Dict, Any, List
from transformers import pipeline
from finflux import config

try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

class ProductionExpertModule:
    """Consolidated 7-Model High-Performance NLP Stack (V4.2+)."""

    _instance = None

    def __new__(cls):
        """Singleton pattern: Optimized lazy loading stack."""
        if cls._instance is None:
            cls._instance = super(ProductionExpertModule, cls).__new__(cls)
            cls._instance.loaded = False
        return cls._instance

    def warm(self):
        """Manually trigger model stack loading."""
        if not self.loaded:
            self._init_stack()
            self.loaded = True

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

        # ── Stage 4: Zero-Shot Entity Extraction (GLiNER & General NER) ──
        if HAS_GLINER:
            print(f"[ProductionExpertModule] Loading GLiNER Specialist: {config.HF_NER_FINANCIAL}")
            self.gliner = GLiNER.from_pretrained(config.HF_NER_FINANCIAL).to("cuda" if self.device == 0 else "cpu")
        else:
            self.gliner = None
        
        self.ner_general = pipeline("ner", model=config.HF_NER_GENERAL, aggregation_strategy="simple", device=self.device)

        # ── Stage 5: Indic Specialist (Indic-BERT) ──
        print(f"[ProductionExpertModule] Loading Indic Specialist: {config.HF_INDIC_NER}")
        self.indic_ner = pipeline("ner", model=config.HF_INDIC_NER, device=self.device)

        # ── Stage 6: Local STT Fallback (Whisper Hindi) ──
        print(f"[ProductionExpertModule] Loading Local STT Fallback: {config.HF_INDIC_STT}")
        self.stt_pipe = pipeline("automatic-speech-recognition", model=config.HF_INDIC_STT, device=self.device)

        # Labels for GLiNER semantic extraction
        self.labels = [
            "INVESTMENT", "LOAN", "EMI", "INSURANCE", "MUTUAL FUND", "GOLD", "STOCK", 
            "PROPERTY", "AMOUNT", "INTEREST RATE", "TENURE", "BANK", "FINANCIAL GOAL"
        ]

    def transcribe_local(self, audio_path: str) -> str:
        """Local ASR fallback using Whisper-Hindi-Small."""
        if not hasattr(self, 'stt_pipe'): return ""
        try:
            res = self.stt_pipe(audio_path)
            return res.get("text", "")
        except Exception as e:
            print(f"[ProductionExpertModule] Local STT Error: {e}")
            return ""

    def _gliner_safe(self, text: str) -> list:
        """Split text into chunks of max 300 chars to avoid GLiNER memory/token issues."""
        if not self.gliner: return []
        
        # Split on sentence boundaries (including Hindi full stop)
        sentences = re.split(r'[।.!?]\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) < 300:
                current += s + ". "
            else:
                if current: chunks.append(current.strip())
                current = s + ". "
        if current: chunks.append(current.strip())
        
        all_entities = []
        for chunk in chunks:
            try:
                entities = self.gliner.predict_entities(chunk, self.labels)
                all_entities.extend(entities)
            except Exception:
                continue
        return all_entities

    def process(self, text: str) -> Dict[str, Any]:
        """Run the full integrated pipeline with lazy loading."""
        if not self.loaded: self.warm()
        if not text.strip(): return {"error": "Empty input"}

        try:
            # 0. Truncation Gating (Prevent model overflow)
            # Most local models (DeBERTa, FinBERT) have 512 token limit.
            # Truncating to ~1200 chars safely fits well within 512 tokens.
            safe_text = text[:1200]

            # 1. Language Logic
            lang_res = self.lang_pipe(safe_text, truncation=True)[0]
            detected_lang = lang_res["label"]

            # 2. Topic & Advice Logic
            topic_res = self.topic_pipe(safe_text, candidate_labels=self.topics, multi_label=True, truncation=True)
            advice_res = self.topic_pipe(safe_text, candidate_labels=self.advice_labels, multi_label=False, truncation=True)
            top_topic = topic_res["labels"][0]
            confidence_score = round(topic_res["scores"][0], 2)

            # 3. FinBERT Sentiment/Urgency
            fin_res = self.fin_pipe(safe_text, truncation=True)[0]
            sentiment_label = fin_res["label"]

            # 4. Entity Extraction (Multi-Pass: GLiNER + General + Indic)
            ner_items = []
            if self.gliner:
                entities = self._gliner_safe(safe_text)
                for ent in entities:
                    label = ent["label"].replace(" ", "_").upper()
                    ner_items.append({"type": label, "value": ent["text"], "context": "GLiNER Specialist"})

            # Add general entities for robustness
            gen_entities = self.ner_general(safe_text)
            for ent in gen_entities:
                if ent['score'] > 0.4:
                    ner_items.append({"type": ent['entity_group'], "value": ent['word'], "context": "General Pass"})

            # Indic-BERT specialized pass for Hindi (Davlan HRL Specialist)
            if detected_lang in ["hi", "hindi"]:
                indic_res = self.indic_ner(safe_text[:512]) # bert-base has 512 token limit
                for ent in indic_res:
                    if ent.get('word') and ent.get('entity'):
                        # Map Davlan tags to clean labels
                        raw_tag = ent['entity']
                        label = "INDIC_ENTITY"
                        if "PER" in raw_tag: label = "PERSON"
                        elif "ORG" in raw_tag: label = "ORGANIZATION"
                        elif "LOC" in raw_tag: label = "LOCATION"
                        
                        ner_items.append({
                            "type": label, 
                            "value": ent['word'].replace("##", ""), 
                            "context": f"Indic-HRL Pass ({raw_tag})"
                        })

            return {
                "detected_language": detected_lang,
                "topic": top_topic,
                "confidence_score": confidence_score,
                "financial_sentiment": sentiment_label,
                "is_advice_request": advice_res["labels"][0] == "asking for financial advice" and advice_res["scores"][0] > 0.6,
                "entities": ner_items,
                "stt_engine": "Groq-Whisper (Primary)" # Metadata
            }
        except Exception as e:
            print(f"[ProductionExpertModule] Pipeline error: {e}")
            return {"error": str(e)}
