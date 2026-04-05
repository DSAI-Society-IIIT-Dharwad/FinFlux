from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class FinancialNLPResult:
    detected_language: str
    finance_topic: str
    topic_confidence: float
    topic_scores: List[Dict[str, Any]]
    terms: List[str]
    entities: List[Dict[str, Any]]
    parameters: Dict[str, List[str]]
    is_finance_related: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_language": self.detected_language,
            "topic": self.finance_topic,
            "topic_confidence": self.topic_confidence,
            "topic_top3": self.topic_scores,
            "terms": self.terms,
            "entities": self.entities,
            "parameters": self.parameters,
            "is_finance_related": self.is_finance_related,
        }


class FinancialNLPAnalyzer:
    """Rule-based finance NLP for terms, parameters, and topic hints."""

    _AMOUNT_PATTERN = re.compile(
        r"(?:rs\.?|inr|₹)\s*\d+(?:,\d{2,3})*(?:\.\d+)?|\b\d+(?:,\d{2,3})*(?:\.\d+)?\s*(?:rupees|lakh|lakhs|crore|crores|k)\b",
        flags=re.IGNORECASE,
    )
    _RATE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|pa)\b", flags=re.IGNORECASE)
    _TENURE_PATTERN = re.compile(
        r"\b\d+\s*(?:day|days|week|weeks|month|months|year|years|mahine|saal)\b",
        flags=re.IGNORECASE,
    )
    _DATE_PATTERN = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
        flags=re.IGNORECASE,
    )

    _PRODUCT_TERMS = {
        "sip": "sip",
        "emi": "emi",
        "loan": "loan",
        "home loan": "home_loan",
        "personal loan": "personal_loan",
        "education loan": "education_loan",
        "car loan": "car_loan",
        "mutual fund": "mutual_fund",
        "fixed deposit": "fixed_deposit",
        "fd": "fixed_deposit",
        "insurance": "insurance",
        "stock": "stock",
        "shares": "stock",
        "gold": "gold",
        "crypto": "crypto",
        "property": "property",
        "nps": "nps",
        "elss": "elss",
        "ppf": "ppf",
        "investment": "investment",
        "savings": "savings",
        "bank": "banking",
        "account": "banking",
        "credit card": "credit_card",
        "debt": "debt",
        "interest": "interest",
        "installment": "installment",
        "loan emi": "loan_emi",
        "ईएमआई": "emi",
        "एसआईपी": "sip",
        "लोन": "loan",
        "ऋण": "loan",
        "बचत": "savings",
        "निवेश": "investment",
        "बीमा": "insurance",
    }

    _INSTITUTION_TERMS = {
        "hdfc": "hdfc",
        "sbi": "sbi",
        "state bank of india": "sbi",
        "icici": "icici",
        "axis": "axis",
        "kotak": "kotak",
        "pnb": "pnb",
        "bank of baroda": "bank_of_baroda",
        "bob": "bank_of_baroda",
        "lic": "lic",
        "tata": "tata",
        "bajaj": "bajaj",
        "nippon": "nippon",
        "uti": "uti",
        "mirae": "mirae",
        "quant": "quant",
        "aditya birla": "aditya_birla",
        "fund house": "fund_house",
        "bank": "bank",
        "nbfc": "nbfc",
    }

    _TOPIC_KEYWORDS = {
        "loan": {"loan", "emi", "interest", "tenure", "repay", "debt", "mortgage", "installment"},
        "investment": {"sip", "mutual fund", "stock", "equity", "portfolio", "returns", "nps", "elss", "ppf"},
        "insurance": {"insurance", "premium", "policy", "claim"},
        "savings": {"fd", "fixed deposit", "savings", "deposit", "bank account"},
        "banking": {"bank", "account", "transfer", "transaction", "loan account"},
        "property": {"property", "home", "house", "plot", "real estate"},
        "crypto": {"crypto", "bitcoin", "ethereum", "blockchain"},
        "general": set(),
    }

    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars

    def analyze(self, text: str, detected_language: str = "unknown") -> FinancialNLPResult:
        normalized = self._normalize(text)
        lowered = normalized.lower()

        entities: List[Dict[str, Any]] = []
        parameters: Dict[str, List[str]] = {
            "amounts": [],
            "rates": [],
            "tenures": [],
            "dates": [],
            "institutions": [],
            "products": [],
        }
        terms: List[str] = []

        self._collect_regex_entities(normalized, self._AMOUNT_PATTERN, "AMOUNT", 0.96, entities, parameters["amounts"], terms)
        self._collect_regex_entities(normalized, self._RATE_PATTERN, "RATE", 0.94, entities, parameters["rates"], terms)
        self._collect_regex_entities(normalized, self._TENURE_PATTERN, "TENURE", 0.92, entities, parameters["tenures"], terms)
        self._collect_regex_entities(normalized, self._DATE_PATTERN, "DATE", 0.88, entities, parameters["dates"], terms)

        self._collect_lexicon_entities(lowered, normalized, self._PRODUCT_TERMS, "FINANCIAL_PRODUCT", 0.9, entities, parameters["products"], terms)
        self._collect_lexicon_entities(lowered, normalized, self._INSTITUTION_TERMS, "INSTITUTION", 0.87, entities, parameters["institutions"], terms)

        topic_scores = self._score_topics(lowered)
        finance_topic = topic_scores[0]["topic"] if topic_scores else "general"
        topic_confidence = topic_scores[0]["score"] if topic_scores else 0.0

        is_finance_related = bool(terms or any(parameters.values()) or finance_topic != "general")

        return FinancialNLPResult(
            detected_language=detected_language,
            finance_topic=finance_topic,
            topic_confidence=round(float(topic_confidence), 3),
            topic_scores=topic_scores[:3],
            terms=self._unique_list(terms),
            entities=self._dedupe_entities(entities),
            parameters={key: self._unique_list(values) for key, values in parameters.items()},
            is_finance_related=is_finance_related,
        )

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()[: self.max_chars]

    def _collect_regex_entities(
        self,
        source_text: str,
        pattern: re.Pattern[str],
        entity_type: str,
        confidence: float,
        entities: List[Dict[str, Any]],
        bucket: List[str],
        terms: List[str],
    ) -> None:
        for match in pattern.finditer(source_text):
            value = match.group(0).strip()
            entities.append(
                {
                    "type": entity_type,
                    "value": value,
                    "context": "rule_based_financial_nlp",
                    "confidence": confidence,
                }
            )
            bucket.append(value)
            terms.append(value.lower())

    def _collect_lexicon_entities(
        self,
        lowered_text: str,
        original_text: str,
        lexicon: Dict[str, str],
        entity_type: str,
        confidence: float,
        entities: List[Dict[str, Any]],
        bucket: List[str],
        terms: List[str],
    ) -> None:
        for term, normalized_value in lexicon.items():
            start = 0
            while True:
                index = lowered_text.find(term, start)
                if index < 0:
                    break
                end = index + len(term)
                value_text = original_text[index:end]
                entities.append(
                    {
                        "type": entity_type,
                        "value": value_text,
                        "context": "rule_based_financial_nlp",
                        "confidence": confidence,
                    }
                )
                bucket.append(normalized_value)
                terms.append(normalized_value)
                start = index + 1

    def _score_topics(self, lowered_text: str) -> List[Dict[str, Any]]:
        counts: List[Dict[str, Any]] = []
        for topic, keywords in self._TOPIC_KEYWORDS.items():
            if not keywords:
                counts.append({"topic": topic, "score": 0.0})
                continue
            hits = 0
            for keyword in keywords:
                if keyword in lowered_text:
                    hits += 1
            score = min(1.0, hits / max(1, len(keywords)))
            counts.append({"topic": topic, "score": round(score, 3)})
        counts.sort(key=lambda item: item["score"], reverse=True)
        if counts and counts[0]["score"] == 0:
            return [{"topic": "general", "score": 1.0}]
        return counts

    @staticmethod
    def _unique_list(values: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for value in values:
            cleaned = str(value).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    @staticmethod
    def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        result: List[Dict[str, Any]] = []
        for entity in entities:
            entity_type = str(entity.get("type", "")).upper()
            value = str(entity.get("value", "")).strip()
            if not entity_type or not value:
                continue
            key = (entity_type, value.lower())
            if key in seen:
                continue
            seen.add(key)
            result.append(entity)
        return result
