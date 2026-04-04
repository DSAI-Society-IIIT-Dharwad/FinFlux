"""V4.2 Full Model Stack: LangDetect, DeBERTa, FinBERT, and GLiNER."""
import re
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import pipeline

from finflux import config

try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

class ProductionExpertModule:
    """Consolidated 4-model NLP stack with rule-assisted financial extraction."""

    _instance = None

    def __new__(cls):
        """Singleton pattern: Optimized lazy loading stack."""
        if cls._instance is None:
            cls._instance = super(ProductionExpertModule, cls).__new__(cls)
            cls._instance.loaded = False
            cls._instance.stt_pipe = None
            cls._instance.device = 0 if config.USE_CUDA and torch.cuda.is_available() else -1
        return cls._instance

    def warm(self):
        """Manually trigger model stack loading."""
        if not self.loaded:
            self._init_stack()
            self.loaded = True

    def _init_stack(self):
        self.device = 0 if config.USE_CUDA and torch.cuda.is_available() else -1
        print(f"[ProductionExpertModule] Initializing 4-Model Stack on device={self.device}...")

        # Stage 1: Precise Language Detection (XLM-Roberta)
        self.lang_pipe = pipeline("text-classification", model=config.HF_LANG_DETECT, device=self.device)

        # Stage 2: Topic & Advice Classification (DeBERTa v3)
        self.topic_pipe = pipeline("zero-shot-classification", model=config.HF_ZERO_SHOT, device=self.device)
        self.topics = config.FINANCIAL_TOPICS
        self.advice_labels = ["asking for financial advice", "general discussion"]

        # Stage 3: Financial Sentiment & Strategy (FinBERT)
        self.fin_pipe = pipeline("sentiment-analysis", model=config.HF_FINBERT, device=self.device)

        # Stage 4: Zero-Shot Entity Extraction (GLiNER handled everything)
        if HAS_GLINER:
            print(f"[ProductionExpertModule] Loading GLiNER Specialized: {config.HF_NER_FINANCIAL}")
            self.gliner = GLiNER.from_pretrained(config.HF_NER_FINANCIAL).to("cuda" if self.device == 0 else "cpu")
        else:
            self.gliner = None

        # Disabled models (skip loading to save VRAM/stability)
        self.ner_general = None
        self.indic_ner = None
        self.stt_pipe = None

        self.labels = config.GLINER_LABELS

    def _ensure_local_stt(self) -> bool:
        """Lazy-load local STT so fallback works without bloating initial startup."""
        if getattr(self, "stt_pipe", None) is not None:
            return True
        try:
            model_id = getattr(config, "HF_LOCAL_ASR", "openai/whisper-small")
            stt_device = getattr(self, "device", 0 if config.USE_CUDA and torch.cuda.is_available() else -1)
            self.stt_pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=stt_device,
            )
            return True
        except Exception as e:
            print(f"[ProductionExpertModule] Local STT init error: {e}")
            self.stt_pipe = None
            return False

    @staticmethod
    def _parse_amount(raw: str) -> Optional[float]:
        txt = raw.lower().replace(",", "").strip()
        mult = 1.0
        if "crore" in txt:
            mult = 10_000_000
        elif "lakh" in txt:
            mult = 100_000
        elif txt.endswith("k"):
            mult = 1_000
        txt = re.sub(r"(rs\.?|inr|₹|rupees|crores?|lakhs?|k)", "", txt, flags=re.IGNORECASE).strip()
        num_match = re.search(r"\d+(?:\.\d+)?", txt)
        if not num_match:
            return None
        return round(float(num_match.group(0)) * mult, 2)

    @staticmethod
    def _parse_rate(raw: str) -> Optional[float]:
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return round(float(m.group(0)), 4) if m else None

    @staticmethod
    def _parse_tenure_months(raw: str) -> Optional[int]:
        m = re.search(r"\d+", raw)
        if not m:
            return None
        n = int(m.group(0))
        txt = raw.lower()
        if "year" in txt or "saal" in txt:
            return n * 12
        if "day" in txt:
            return max(1, n // 30)
        return n

    @staticmethod
    def _dedupe_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for item in items:
            key = (str(item.get("type", "")).upper(), str(item.get("value", "")).strip().lower())
            if key in seen or not key[1]:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _extract_rule_entities(self, text: str) -> List[Dict[str, Any]]:
        patterns = {
            "AMOUNT": r"(?:₹|rs\.?|inr)?\s*\d+(?:,\d{2,3})*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|k)?",
            "INTEREST_RATE": r"\b\d+(?:\.\d+)?\s*(?:%|percent|pa)\b",
            "TENURE": r"\b\d+\s*(?:day|days|month|months|year|years|mahine|saal)\b",
        }
        keyword_types = {
            "emi": "EMI",
            "sip": "SIP",
            "loan": "LOAN",
            "mutual fund": "MUTUAL_FUND",
            "insurance": "INSURANCE",
            "credit card": "CREDIT_CARD",
            "income": "INCOME",
            "salary": "INCOME",
            "expense": "EXPENSE",
            "budget": "EXPENSE",
        }

        entities: List[Dict[str, Any]] = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                if not value:
                    continue
                entities.append(
                    {
                        "type": entity_type,
                        "value": value,
                        "context": "Regex financial extractor",
                        "confidence": 0.86,
                    }
                )

        lowered = text.lower()
        for term, entity_type in keyword_types.items():
            start = 0
            while True:
                idx = lowered.find(term, start)
                if idx < 0:
                    break
                entities.append(
                    {
                        "type": entity_type,
                        "value": text[idx: idx + len(term)],
                        "context": "Lexicon financial extractor",
                        "confidence": 0.8,
                    }
                )
                start = idx + len(term)
        return entities

    def _extract_financial_parameters(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        lower = text.lower()
        parameters: Dict[str, Any] = {
            "emi_amount": None,
            "sip_amount": None,
            "loan_amount": None,
            "income_amount": None,
            "interest_rate_pct": None,
            "tenure_months": None,
            "total_amount_mentioned": 0.0,
        }

        amount_values: List[float] = []
        for entity in entities:
            if str(entity.get("type", "")).upper() in {"AMOUNT", "INVESTMENT", "LOAN", "EMI", "SIP"}:
                parsed = self._parse_amount(str(entity.get("value", "")))
                if parsed is not None:
                    amount_values.append(parsed)
        if amount_values:
            parameters["total_amount_mentioned"] = round(sum(amount_values), 2)

        regex_rules = {
            "emi_amount": [r"(?:emi|ईएमआई)[^\d]{0,12}(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)", r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:emi|ईएमआई)"],
            "sip_amount": [r"(?:sip|systematic investment plan)[^\d]{0,12}(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)", r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:sip)"],
            "loan_amount": [r"(?:loan|ऋण|लोन)[^\d]{0,16}(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)"],
            "income_amount": [r"(?:income|salary|monthly income)[^\d]{0,16}(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)"],
        }
        for key, patterns in regex_rules.items():
            for pattern in patterns:
                m = re.search(pattern, lower, flags=re.IGNORECASE)
                if not m:
                    continue
                parsed = self._parse_amount(m.group(1))
                if parsed is not None:
                    parameters[key] = parsed
                    break

        rate_match = re.search(r"(?:interest|rate|roi)[^\d]{0,12}(\d+(?:\.\d+)?\s*(?:%|percent|pa))", lower, flags=re.IGNORECASE)
        if rate_match:
            parameters["interest_rate_pct"] = self._parse_rate(rate_match.group(1))

        tenure_match = re.search(r"(\d+\s*(?:day|days|month|months|year|years|mahine|saal))", lower, flags=re.IGNORECASE)
        if tenure_match:
            parameters["tenure_months"] = self._parse_tenure_months(tenure_match.group(1))

        filled = sum(1 for key in ["emi_amount", "sip_amount", "loan_amount", "income_amount", "interest_rate_pct", "tenure_months"] if parameters.get(key) is not None)
        parameters["parameter_coverage_score"] = round(filled / 6.0, 3)
        return parameters

    def _estimate_risk(self, text: str, parameters: Dict[str, Any], sentiment: str) -> Dict[str, Any]:
        lowered = text.lower()
        score = 0.22
        reasons: List[str] = []

        high_hits = [kw for kw in config.RISK_KEYWORDS_HIGH if kw in lowered]
        medium_hits = [kw for kw in config.RISK_KEYWORDS_MEDIUM if kw in lowered]
        low_hits = [kw for kw in config.RISK_KEYWORDS_LOW if kw in lowered]

        if high_hits:
            score += min(0.4, 0.12 * len(high_hits))
            reasons.append(f"high-risk terms: {', '.join(high_hits[:3])}")
        if medium_hits:
            score += min(0.22, 0.05 * len(medium_hits))
        if low_hits:
            score -= min(0.18, 0.04 * len(low_hits))

        emi = parameters.get("emi_amount")
        income = parameters.get("income_amount")
        if isinstance(emi, (int, float)) and isinstance(income, (int, float)) and income > 0:
            burden = emi / income
            if burden >= 0.6:
                score += 0.28
                reasons.append("EMI burden above 60% of income")
            elif burden >= 0.4:
                score += 0.15
                reasons.append("EMI burden above 40% of income")

        if "neg" in str(sentiment).lower():
            score += 0.08

        score = max(0.0, min(1.0, score))
        if score >= 0.8:
            level = "CRITICAL"
        elif score >= 0.62:
            level = "HIGH"
        elif score >= 0.38:
            level = "MEDIUM"
        else:
            level = "LOW"
        return {"risk_level": level, "risk_score": round(score, 3), "risk_reasons": reasons}

    @staticmethod
    def _build_recommendation_hints(topic: str, parameters: Dict[str, Any], is_advice_request: bool) -> List[str]:
        hints: List[str] = []
        if not is_advice_request:
            return hints
        if topic in {"loan", "emi"}:
            hints.append("Compare current EMI burden against monthly net income before increasing commitments.")
        if topic in {"investment", "sip", "mutual fund", "stock"}:
            hints.append("Review risk appetite and time horizon before changing allocation decisions.")
        if parameters.get("interest_rate_pct") is not None:
            hints.append("Validate APR/ROI and hidden charges before finalizing loan-linked choices.")
        if parameters.get("tenure_months") is not None:
            hints.append("Assess tenure impact on total interest outflow, not only monthly installment.")
        return hints

    @staticmethod
    def _calibrate_confidence(topic_score: float, language_conf: float, sentiment_score: float, entity_count: int, parameter_coverage: float) -> float:
        entity_signal = min(1.0, entity_count / 8.0)
        score = (0.36 * topic_score) + (0.2 * language_conf) + (0.12 * sentiment_score) + (0.2 * entity_signal) + (0.12 * parameter_coverage)
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _estimate_language_mix(text: str, detected_lang: str = "unknown", language_confidence: float = 0.0) -> Dict[str, Any]:
        latin_tokens = re.findall(r"[A-Za-z]+", text.lower())
        devanagari_tokens = re.findall(r"[\u0900-\u097F]+", text)
        other_script_chars = sum(
            1
            for ch in text
            if ch.isalpha() and not (("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("\u0900" <= ch <= "\u097F"))
        )
        hindi_roman_markers = {
            "hai", "haan", "nahi", "nahin", "nhi", "ka", "ki", "ke", "ko", "se", "par", "mein",
            "main", "mera", "meri", "mere", "hum", "aap", "tum", "jo", "kya", "kaise", "kyu", "kyun",
            "acha", "achha", "theek", "thik", "sir", "ji", "wali", "wala", "kar", "karo", "karna", "raha", "rahi",
            "gaya", "gayi", "hoga", "hogi", "tha", "thi", "chahiye", "matlab", "haanji", "bahut", "zyada",
            "kam", "sahi", "mujhe", "samajh", "kyunki", "agar", "lekin", "aur",
        }
        hindi_roman_hits = sum(1 for tok in latin_tokens if tok in hindi_roman_markers)

        hindi_units = float(len(devanagari_tokens)) + float(hindi_roman_hits)
        english_units = float(max(0, len(latin_tokens) - hindi_roman_hits))
        other_units = float(other_script_chars)

        total = hindi_units + english_units + other_units
        if total == 0:
            return {
                "hindi_pct": 0.0,
                "english_pct": 0.0,
                "other_pct": 0.0,
                "dominant_language": "unknown",
            }

        hindi_pct = round((hindi_units / total) * 100.0, 1)
        english_pct = round((english_units / total) * 100.0, 1)
        other_pct = round(max(0.0, 100.0 - hindi_pct - english_pct), 1)

        # Calibrate with language model confidence so Hindi-leaning Hinglish is not forced to 0%.
        lang_tag = detected_lang.lower()
        if lang_tag.startswith("hi") and language_confidence >= 0.7 and hindi_pct < 25.0:
            boost = min(45.0, round((language_confidence * 50.0), 1))
            shift = min(boost - hindi_pct, english_pct)
            if shift > 0:
                hindi_pct = round(hindi_pct + shift, 1)
                english_pct = round(max(0.0, english_pct - shift), 1)
                other_pct = round(max(0.0, 100.0 - hindi_pct - english_pct), 1)

        dominant = "hindi"
        dominant_value = hindi_pct
        if english_pct > dominant_value:
            dominant = "english"
            dominant_value = english_pct
        if other_pct > dominant_value:
            dominant = "other"

        return {
            "hindi_pct": hindi_pct,
            "english_pct": english_pct,
            "other_pct": other_pct,
            "dominant_language": dominant,
        }

    def transcribe_local(self, audio_path: str) -> str:
        """Local ASR fallback using Whisper-Hindi-Small."""
        if not self._ensure_local_stt():
            return ""
        try:
            # Decode with soundfile so local STT works even when ffmpeg is not on PATH.
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                audio = resample_poly(audio, 16000, sr).astype(np.float32)
                sr = 16000
            res = self.stt_pipe({"array": audio, "sampling_rate": sr})
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
            detected_lang = str(lang_res.get("label", "unknown")).lower()
            language_confidence = float(lang_res.get("score", 0.0))

            # 2. Topic & Advice Logic
            topic_res = self.topic_pipe(safe_text, candidate_labels=self.topics, multi_label=True, truncation=True)
            advice_res = self.topic_pipe(safe_text, candidate_labels=self.advice_labels, multi_label=False, truncation=True)
            top_topic = str(topic_res["labels"][0]).lower()
            topic_score = float(topic_res["scores"][0])
            topic_top3 = [
                {"topic": str(topic_res["labels"][i]).lower(), "score": round(float(topic_res["scores"][i]), 4)}
                for i in range(min(3, len(topic_res.get("labels", []))))
            ]

            # 3. FinBERT Sentiment/Urgency
            fin_res = self.fin_pipe(safe_text, truncation=True)[0]
            sentiment_label = str(fin_res.get("label", "neutral")).lower()
            sentiment_score = float(fin_res.get("score", 0.0))
            sentiment_breakdown = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            if "pos" in sentiment_label:
                sentiment_breakdown["positive"] = round(sentiment_score, 4)
            elif "neg" in sentiment_label:
                sentiment_breakdown["negative"] = round(sentiment_score, 4)
            else:
                sentiment_breakdown["neutral"] = round(sentiment_score, 4)

            # 4. GLiNER Specialist Extraction (Strict mode, no fallback noise)
            ner_items = []
            if self.gliner:
                entities = self._gliner_safe(safe_text)
                for ent in entities:
                    label = ent["label"].replace(" ", "_").upper()
                    ner_items.append({
                        "type": label, 
                        "value": ent["text"], 
                        "context": "GLiNER Specialist",
                        "confidence": round(float(ent.get("score", 0.0)), 4),
                    })

            ner_items.extend(self._extract_rule_entities(safe_text))
            ner_items = self._dedupe_entities(ner_items)

            is_advice_request = advice_res["labels"][0] == "asking for financial advice" and float(advice_res["scores"][0]) > 0.6
            financial_parameters = self._extract_financial_parameters(safe_text, ner_items)
            risk_meta = self._estimate_risk(safe_text, financial_parameters, sentiment_label)
            recommendation_hints = self._build_recommendation_hints(top_topic, financial_parameters, is_advice_request)
            language_mix = self._estimate_language_mix(
                safe_text,
                detected_lang=detected_lang,
                language_confidence=language_confidence,
            )
            calibrated_conf = self._calibrate_confidence(
                topic_score=topic_score,
                language_conf=language_confidence,
                sentiment_score=sentiment_score,
                entity_count=len(ner_items),
                parameter_coverage=float(financial_parameters.get("parameter_coverage_score", 0.0)),
            )

            model_attribution = {
                "xlm_roberta": {
                    "detected_language": detected_lang,
                    "confidence": round(language_confidence, 4),
                    "language_mix": language_mix,
                },
                "deberta": {"top_topic": top_topic, "top3_topics": topic_top3, "advice_score": round(float(advice_res["scores"][0]), 4)},
                "finbert": {"label": sentiment_label, "score": round(sentiment_score, 4), "breakdown": sentiment_breakdown},
                "entity_engine": {"gliner_plus_rules_count": len(ner_items)},
                "risk_engine": risk_meta,
            }

            return {
                "detected_language": detected_lang,
                "language_confidence": round(language_confidence, 4),
                "language_mix": language_mix,
                "topic": top_topic,
                "topic_top3": topic_top3,
                "confidence_score": calibrated_conf,
                "financial_sentiment": sentiment_label,
                "sentiment_breakdown": sentiment_breakdown,
                "is_advice_request": is_advice_request,
                "entities": ner_items,
                "financial_parameters": financial_parameters,
                "recommendation_hints": recommendation_hints,
                "risk_level": risk_meta["risk_level"],
                "risk_score": risk_meta["risk_score"],
                "risk_reasons": risk_meta["risk_reasons"],
                "model_attribution": model_attribution,
                "stt_engine": "Groq-Whisper (Primary)",
            }
        except Exception as e:
            print(f"[ProductionExpertModule] Pipeline error: {e}")
            return {"error": str(e)}
