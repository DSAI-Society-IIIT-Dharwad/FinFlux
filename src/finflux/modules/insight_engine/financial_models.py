"""V4.2 Full Model Stack: LangDetect, DeBERTa, FinBERT, and GLiNER."""
import os
import torch
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, cast
from transformers import pipeline
from finflux import config

try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    GLiNER = None
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
        print(f"[ProductionExpertModule] Initializing 4-Model Stack on device={self.device}...")

        # Pylance in some transformers versions narrows task overloads too aggressively.
        # Cast once here so runtime behavior stays unchanged while static analysis remains clean.
        make_pipeline = cast(Any, pipeline)

        # Stage 1: Precise Language Detection (XLM-Roberta)
        self.lang_pipe = make_pipeline("text-classification", model=config.HF_LANG_DETECT, device=self.device)

        # Stage 2: Topic & Advice Classification (DeBERTa v3)
        self.topic_pipe = make_pipeline("zero-shot-classification", model=config.HF_ZERO_SHOT, device=self.device)
        self.topic_level1_labels = [
            "DEBT_MANAGEMENT (Loan, EMI, कर्ज)",
            "INVESTMENT_PLANNING (SIP, Mutual Fund, निवेश)",
            "INSURANCE_PROTECTION (Premium, Policy, क्लैम, बीमा)",
            "TAX_PLANNING (GST, ITR, टैक्स)",
            "BUDGETING (Expenses, Salary, बजट, खर्च)",
            "ASSET_PURCHASE (Property, Gold, Car, संपत्ति)",
            "RETIREMENT (NPS, Pension, रिटायरमेंट)",
            "EMERGENCY_FUND (Savings, बचत)",
            "GENERAL_FINANCIAL (Finance, बैंकिंग)",
        ]
        self.topic_level2_labels = [
            "home_loan",
            "personal_loan",
            "car_loan",
            "education_loan",
            "credit_card",
            "SIP",
            "mutual_fund",
            "direct_equity",
            "FD",
            "PPF",
            "NPS",
            "term_insurance",
            "health_insurance",
            "gold",
            "real_estate",
            "crypto",
        ]
        self.advice_labels = ["asking for financial advice", "general discussion"]

        # Lightweight dictionary for real-time chunk detection (DeBERTa-free).
        self.financial_keywords = [
            "loan", "home loan", "personal loan", "car loan", "education loan", "gold loan", "business loan", "emi", "instalment",
            "interest", "interest rate", "floating rate", "fixed rate", "prepayment", "foreclosure", "tenure", "principal", "outstanding",
            "credit", "credit card", "credit limit", "minimum due", "billing cycle", "statement", "late fee", "cibil", "credit score",
            "debt", "debt management", "debt consolidation", "liability", "repayment", "default", "penalty", "bounce", "overdue",
            "salary", "income", "monthly income", "cashflow", "expenses", "expense", "budget", "savings", "monthly savings",
            "emergency fund", "rainy day fund", "contingency", "buffer", "liquidity", "reserve", "corpus",
            "invest", "investment", "investing", "sip", "lumpsum", "mutual fund", "nav", "amc", "fund house", "folio",
            "equity", "stock", "shares", "index fund", "etf", "smallcap", "midcap", "largecap", "diversification",
            "returns", "yield", "xirr", "cagr", "alpha", "beta", "volatility", "risk", "risk profile", "asset allocation",
            "fixed deposit", "fd", "rd", "recurring deposit", "bond", "debenture", "treasury", "gsec",
            "ppf", "nps", "epf", "vpf", "elss", "ulip", "retirement", "pension", "annuity", "superannuation",
            "insurance", "term insurance", "health insurance", "mediclaim", "premium", "sum assured", "coverage", "deductible", "claim",
            "tax", "income tax", "tax saving", "80c", "80d", "87a", "hra", "tds", "capital gains", "gst", "itr",
            "property", "real estate", "down payment", "registry", "stamp duty", "maintenance", "rent", "rental yield",
            "gold", "digital gold", "sovereign gold bond", "sgb", "silver", "commodity", "crypto", "bitcoin", "ethereum",
            "upi", "bank", "bank account", "savings account", "current account", "ifsc", "neft", "rtgs", "imps",
            "balance", "statement", "nominee", "kyc", "pan", "aadhaar", "demat", "broker", "trading",
            "किस्त", "ब्याज", "बचत", "निवेश", "ऋण", "कर्ज", "आय", "खर्च", "बीमा", "प्रीमियम", "कर", "टैक्स",
            "होम लोन", "पर्सनल लोन", "क्रेडिट कार्ड", "म्यूचुअल फंड", "सिप", "एफडी", "पीपीएफ", "एनपीएस", "रिटायरमेंट",
            "डाउन पेमेंट", "इमरजेंसी फंड", "रेटन", "रिटर्न", "लाभ", "हानि", "जोखिम", "सम्पत्ति", "संपत्ति", "बैंक",
            "क्रिप्टो", "शेयर", "पोर्टफोलियो", "मार्केट", "बाजार", "जीएसटी", "फंड",
            "emi", "sip amount", "loan amount", "interest pa", "percent returns", "tax section", "fund name", "bank name",
            "insurance premium", "fd amount", "investment goal", "expense category", "monthly income", "monthly savings",
            "cash", "wallet", "networth", "wealth", "financial", "finance", "advisor", "advice", "planning", "goal",
            "child education", "marriage fund", "vacation fund", "house purchase", "vehicle purchase", "inflation", "cost of living",
        ]

        # Stage 3: Financial Sentiment & Strategy (FinBERT)
        self.fin_pipe = make_pipeline("sentiment-analysis", model=config.HF_FINBERT, device=self.device)

        # Stage 4: Zero-Shot Entity Extraction (GLiNER handled everything)
        if HAS_GLINER and GLiNER is not None:
            print(f"[ProductionExpertModule] Loading GLiNER Specialized: {config.HF_NER_FINANCIAL}")
            self.gliner = GLiNER.from_pretrained(config.HF_NER_FINANCIAL).to("cuda" if self.device == 0 else "cpu")
        else:
            self.gliner = None

        # Disabled models (skip loading to save VRAM/stability)
        self.ner_general = None
        self.indic_ner = None

        # Stage 5: Local ASR Fallback (Whisper-small for cloud failure resilience)
        try:
            from transformers import pipeline as hf_pipeline
            stt_device = 0 if self.device == 0 else -1
            self.stt_pipe = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=stt_device,
            )
            print("[ProductionExpertModule] Local ASR fallback (whisper-small) loaded.")
        except Exception as stt_err:
            print(f"[ProductionExpertModule] Local ASR fallback unavailable: {stt_err}")
            self.stt_pipe = None

        # Consolidated labels for GLiNER semantic extraction (High-Resolution V4.2+)
        self.labels = [
            "EMI AMOUNT",
            "SIP AMOUNT",
            "LOAN AMOUNT",
            "LOAN TENURE",
            "INTEREST RATE",
            "CIBIL SCORE",
            "FUND NAME",
            "BANK NAME",
            "INSURANCE PREMIUM",
            "FD AMOUNT",
            "TAX SECTION",
            "INVESTMENT GOAL",
            "EXPENSE CATEGORY",
            "MONTHLY INCOME",
            "MONTHLY SAVINGS",
            # Existing high-signal labels retained for broader extraction coverage.
            "RETURNS PERCENTAGE",
            "MUTUAL FUND",
            "RISK LEVEL",
            "FUND CATEGORY",
            "PROPERTY",
            "GOLD",
            "STOCK",
            "CRYPTO",
        ]

    def _rule_based_entities(self, text: str) -> List[Dict[str, Any]]:
        """Rule-based catcher for Indian financial formats and Hindi terms missed by GLiNER."""
        results: List[Dict[str, Any]] = []

        # Indian number formats: 1.5 lakh, 50k, 2 crore, Rs 50,000, ₹1,20,000
        amount_pattern = re.compile(
            r"(?:₹|rs\.?\s*)?\b\d+(?:[.,]\d+)?(?:\s?[,\d]{0,10})\s?(?:k|thousand|lakh|lakhs|lac|crore|crores|cr)?\b",
            flags=re.IGNORECASE,
        )
        for m in amount_pattern.finditer(text):
            val = m.group(0).strip()
            if not val:
                continue
            lower = val.lower()
            if any(tok in lower for tok in ["%", "percent", "pa", "p.a"]):
                continue
            results.append({"type": "LOAN AMOUNT", "value": val, "confidence": 0.90, "source": "rule_based"})

        # Percentage expressions: 8.5% pa, 12 percent returns, 10% per annum
        pct_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\s?(?:%|percent)\s?(?:pa|p\.a\.?|per\s+annum)?\b",
            flags=re.IGNORECASE,
        )
        for m in pct_pattern.finditer(text):
            val = m.group(0).strip()
            if val:
                results.append({"type": "INTEREST RATE", "value": val, "confidence": 0.90, "source": "rule_based"})

        # Financial keywords rule-based extractor
        keywords = {
            "GST": "TAX ENTITY",
            "ITR": "COMPLIANCE",
            "TDS": "TAX ENTITY",
            "Income Tax": "TAX ENTITY",
            "Mutual Fund": "FUND NAME",
            "SIP": "INVESTMENT TYPE",
            "FD": "INVESTMENT TYPE",
            "Crypto": "ASSET CLASS",
            "Bitcoin": "ASSET CLASS",
            "Portfolio": "FINANCIAL PROFILE",
        }
        for kw, etype in keywords.items():
            if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
                results.append({"type": etype, "value": kw, "confidence": 0.85, "source": "rule_based"})

        # Hindi financial terms mapping.
        hindi_terms = {
            "किस्त": "EMI AMOUNT",
            "ब्याज": "INTEREST RATE",
            "बचत": "MONTHLY SAVINGS",
            "निवेश": "INVESTMENT GOAL",
            "जीएसटी": "TAX ENTITY",
            "क्रिप्टो": "ASSET CLASS",
        }
        for term, mapped_type in hindi_terms.items():
            for _ in re.finditer(re.escape(term), text):
                results.append({"type": mapped_type, "value": term, "confidence": 0.90, "source": "rule_based"})

        # De-duplicate by (type, value).
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in results:
            key = f"{item['type']}::{str(item['value']).strip().lower()}"
            dedup[key] = item
        return list(dedup.values())

    def transcribe_local(self, audio_path: str) -> str:
        """Local ASR fallback using Whisper-small."""
        if not hasattr(self, 'stt_pipe') or self.stt_pipe is None:
            raise RuntimeError(
                "Local ASR fallback is not available: stt_pipe was not initialized. "
                "Ensure 'openai/whisper-small' can be loaded or set DEMO_MODE=true."
            )
        try:
            res = self.stt_pipe(audio_path)
            text = res.get("text", "") if isinstance(res, dict) else str(res)
            if not text.strip():
                print("[ProductionExpertModule] Local STT returned empty transcript")
            return text
        except Exception as e:
            print(f"[ProductionExpertModule] Local STT Error: {e}")
            raise RuntimeError(f"Local ASR transcription failed: {e}") from e

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

    def _call_fast_llm(self, system_prompt: str, user_prompt: str) -> str:
        api_key = config.GROQ_API_KEY
        if not api_key:
            return ""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.GROQ_LLM_FAST_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

    def route_intent(self, text: str) -> Dict[str, Any]:
        """Three-way router: analysis, financial inquiry, or general conversation."""
        if not self.loaded:
            self.warm()

        route_labels = [
            "personal financial situation discussion",
            "financial information request",
            "non-financial conversation",
        ]
        safe_text = text[:800]
        result = self.topic_pipe(safe_text, candidate_labels=route_labels, multi_label=False, truncation=True)
        top_label = str(result["labels"][0])
        top_score = float(result["scores"][0])

        score_map = {str(label): float(score) for label, score in zip(result["labels"], result["scores"])}

        if top_score < 0.55:
            route = "financial_inquiry"
        elif top_label == "personal financial situation discussion":
            route = "analysis"
        elif top_label == "non-financial conversation":
            route = "general"
        else:
            route = "financial_inquiry"

        return {
            "route": route,
            "label": top_label,
            "score": top_score,
            "scores": score_map,
        }

    def answer_financial_inquiry(self, question: str) -> str:
        """Answer direct finance questions without strategic analysis."""
        system_prompt = (
            "You are a knowledgeable Indian financial educator. Answer the question directly, concisely, and helpfully. "
            "Provide factual information without giving personalized financial advice. Use simple language. "
            "Do not generate strategic analysis or McKinsey-style summaries. Just answer the question."
        )
        try:
            answer = self._call_fast_llm(system_prompt, question)
            return answer.strip() if answer.strip() else "I can help with that finance question."
        except Exception as exc:
            print(f"[ProductionExpertModule] Inquiry response error: {exc}")
            return "I can help with that finance question."

    def answer_casual(self, message: str) -> str:
        """Answer casual conversation naturally without finance framing."""
        system_prompt = (
            "You are a friendly conversational assistant. Reply naturally, warmly, and briefly. "
            "Do not mention financial analysis, strategy, or advisory framing. If the user greets you, greet back. "
            "Keep the response short and human."
        )
        try:
            answer = self._call_fast_llm(system_prompt, message)
            return answer.strip() if answer.strip() else "Hello. How can I help?"
        except Exception as exc:
            print(f"[ProductionExpertModule] Casual response error: {exc}")
            return "Hello. How can I help?"

    def detect_financial_keywords(self, text: str) -> Dict[str, Any]:
        """Keyword-only realtime detector for small rolling audio chunks."""
        if not self.loaded:
            self.warm()
        probe = str(text or "").lower()
        if not probe.strip():
            return {"financial_detected": False, "matched_keywords": []}

        matched: List[str] = []
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", probe))
        for kw in self.financial_keywords:
            key = kw.lower().strip()
            if not key:
                continue
            if " " in key or any(ord(ch) > 127 for ch in key):
                if key in probe:
                    matched.append(kw)
            else:
                if key in tokens:
                    matched.append(kw)

        # Keep a compact unique list for badge rendering.
        unique = []
        seen = set()
        for item in matched:
            k = item.lower()
            if k in seen:
                continue
            seen.add(k)
            unique.append(item)

        return {
            "financial_detected": len(unique) > 0,
            "matched_keywords": unique[:12],
        }

    def _language_breakdown(self, text: str, detected_lang: str) -> Dict[str, float]:
        """Character-level language split: Devanagari -> Hindi, roman words -> English/Hinglish."""
        financial_english_terms = {
            "loan", "emi", "sip", "mutual", "fund", "insurance", "premium", "debt", "equity", "stock",
            "return", "returns", "risk", "portfolio", "income", "expense", "budget", "credit", "score",
            "cibil", "prepayment", "interest", "tenure", "corpus", "liquidity", "nps", "ppf", "elss",
            "fd", "bond", "gold", "crypto", "property", "tax", "invest", "investment", "finance",
        }
        general_english_terms = {
            "the", "is", "are", "was", "were", "my", "your", "our", "and", "or", "but", "if", "then",
            "what", "how", "when", "where", "why", "can", "could", "should", "would", "please", "help",
            "plan", "today", "tomorrow", "month", "year", "salary", "home", "car", "bank", "money",
        }

        hindi_chars = 0
        english_chars = 0
        hinglish_chars = 0

        for ch in text:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                hindi_chars += 1

        latin_words = re.findall(r"[A-Za-z]+", text)
        for word in latin_words:
            token = word.lower()
            if token in financial_english_terms or token in general_english_terms:
                english_chars += len(word)
            else:
                hinglish_chars += len(word)

        total = hindi_chars + english_chars + hinglish_chars
        if total <= 0:
            label = str(detected_lang).lower()
            if "hi" in label or "hindi" in label:
                return {"hindi": 100.0, "english": 0.0, "hinglish": 0.0}
            if "en" in label or "english" in label:
                return {"hindi": 0.0, "english": 100.0, "hinglish": 0.0}
            return {"hindi": 0.0, "english": 0.0, "hinglish": 100.0}

        h = round((hindi_chars / total) * 100, 2)
        e = round((english_chars / total) * 100, 2)
        hg = round((hinglish_chars / total) * 100, 2)
        delta = round(100.0 - (h + e + hg), 2)

        # Keep sums exactly 100 by absorbing rounding delta into the largest bucket.
        if delta != 0:
            buckets = {"hindi": h, "english": e, "hinglish": hg}
            k = max(buckets, key=buckets.get)
            buckets[k] = round(buckets[k] + delta, 2)
            return buckets

        return {"hindi": h, "english": e, "hinglish": hg}

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
            language_confidence = float(lang_res.get("score", 0.0))
            language_breakdown = self._language_breakdown(safe_text, detected_lang)

            # 2. Topic & Advice Logic
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_l1 = pool.submit(
                    self.topic_pipe,
                    safe_text,
                    candidate_labels=self.topic_level1_labels,
                    multi_label=True,
                    truncation=True,
                )
                f_l2 = pool.submit(
                    self.topic_pipe,
                    safe_text,
                    candidate_labels=self.topic_level2_labels,
                    multi_label=True,
                    truncation=True,
                )
                topic_res_l1 = f_l1.result()
                topic_res_l2 = f_l2.result()

            advice_res = self.topic_pipe(safe_text, candidate_labels=self.advice_labels, multi_label=False, truncation=True)
            top_topic = topic_res_l1["labels"][0]
            top_topic_score = float(topic_res_l1["scores"][0])
            top_product = topic_res_l2["labels"][0]
            top_product_score = float(topic_res_l2["scores"][0])
            confidence_score = round(top_topic_score, 2)
            topic_top3 = [
                {
                    "topic": topic_res_l1["labels"][i],
                    "score": float(topic_res_l1["scores"][i])
                }
                for i in range(min(3, len(topic_res_l1.get("labels", []))))
            ]

            # 3. FinBERT Sentiment/Urgency
            fin_res = self.fin_pipe(safe_text, truncation=True)[0]
            sentiment_label = fin_res["label"]
            sentiment_score = float(fin_res.get("score", 0.0))
            sentiment_map = {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            }
            normalized_sentiment = str(sentiment_label).lower()
            if "pos" in normalized_sentiment:
                sentiment_map["positive"] = sentiment_score
            elif "neg" in normalized_sentiment:
                sentiment_map["negative"] = sentiment_score
            else:
                sentiment_map["neutral"] = sentiment_score

            # 4. GLiNER Specialist Extraction (Strict mode, no fallback noise)
            ner_items = []
            if self.gliner:
                entities = self._gliner_safe(safe_text)
                for ent in entities:
                    score = float(ent.get("score", 0.0))
                    if score <= 0.55:
                        continue
                    label = str(ent.get("label", "")).strip().upper()
                    val = str(ent.get("text", "")).strip()
                    if not label or not val:
                        continue
                    ner_items.append({
                        "type": label,
                        "value": val,
                        "confidence": score,
                        "source": "gliner",
                    })

            # 5. Rule-based catcher for patterns GLiNER commonly misses.
            rule_items = self._rule_based_entities(safe_text)

            # Merge and deduplicate entities by (type, value) while keeping highest confidence.
            merged: Dict[str, Dict[str, Any]] = {}
            for item in ner_items + rule_items:
                key = f"{item.get('type','')}::{str(item.get('value','')).lower()}"
                existing = merged.get(key)
                if not existing or float(item.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                    merged[key] = item
            ner_items = list(merged.values())

            model_attribution = {
                "xlm_roberta": {
                    "detected_language": detected_lang,
                    "confidence": language_confidence,
                    "language_breakdown": language_breakdown,
                },
                "deberta": {
                    "level1_top_category": top_topic,
                    "level1_confidence": top_topic_score,
                    "level2_top_product": top_product,
                    "level2_confidence": top_product_score,
                    "top_topic": top_topic,
                    "top3_topics": topic_top3,
                },
                "finbert": {
                    "label": sentiment_label,
                    "breakdown": sentiment_map,
                },
                "gliner": {
                    "entity_count": len(ner_items),
                },
                "qwen": {
                    "reasoning_available": True,
                },
            }

            return {
                "detected_language": detected_lang,
                "language_confidence": language_confidence,
                "language_breakdown": language_breakdown,
                "topic": top_topic,
                "topic_level1": top_topic,
                "topic_level1_confidence": top_topic_score,
                "topic_level2": top_product,
                "topic_level2_confidence": top_product_score,
                "confidence_score": confidence_score,
                "topic_top3": topic_top3,
                "financial_sentiment": sentiment_label,
                "sentiment_breakdown": sentiment_map,
                "is_advice_request": advice_res["labels"][0] == "asking for financial advice" and advice_res["scores"][0] > 0.6,
                "entities": ner_items,
                "stt_engine": "Groq-Whisper (Primary)", # Metadata
                "script_preserved": True,
                "model_attribution": model_attribution,
            }
        except Exception as e:
            print(f"[ProductionExpertModule] Pipeline error: {e}")
            return {"error": str(e)}
