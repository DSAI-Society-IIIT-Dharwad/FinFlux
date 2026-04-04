from __future__ import annotations
import json
from finflux.modules.llm_wrapper.client import GroqClient
from finflux.modules.llm_wrapper.prompts import (
    CLEAN_TRANSCRIPT_SYSTEM,
    TOPIC_DETECTION_SYSTEM,
    INSIGHT_SUMMARY_SYSTEM
)

class LLMProcessor:
    def __init__(self, model: str = "llama3-8b-8192"):
        self.client = GroqClient(model=model)
        
    def clean_transcript(self, raw_text: str) -> str:
        if not raw_text.strip():
            return raw_text
        cleaned = self.client.generate(prompt=raw_text, system=CLEAN_TRANSCRIPT_SYSTEM)
        return cleaned if cleaned else raw_text
        
    def detect_financial_topic(self, text: str) -> dict:
        if not text.strip():
            return {"is_financial": False, "topic": None, "entities_found": []}
            
        result = self.client.generate(prompt=f"Text: {text}", system=TOPIC_DETECTION_SYSTEM)
        
        try:
            # Attempt to parse json out of the markdown if necessary
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            parsed = json.loads(result.strip())
            return {
                "is_financial": bool(parsed.get("is_financial", False)),
                "topic": parsed.get("topic"),
                "entities_found": parsed.get("entities_found", [])
            }
        except Exception:
            return {"is_financial": False, "topic": None, "entities_found": []}

    def generate_summary(self, text: str) -> str:
        return self.client.generate(prompt=text, system=INSIGHT_SUMMARY_SYSTEM)
