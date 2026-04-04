"""V4.2+ McKinsey-Level Groq Synthesis Adapter: Strategic IQ & Expert Reasoning."""
import os
import requests
import json
from typing import Dict, Any, List
from pathlib import Path
from finflux import config

class GroqWhisperAdapter:
    """Groq Whisper API for multilingual transcription."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """High-resolution transcription via Groq Whisper."""
        if not self.api_key:
            return {"error": "GROQ_API_KEY not set", "text": ""}

        # Check file size before sending — Groq limit is 25MB
        import os
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 24:
            print(f"[GroqWhisper] File too large ({file_size_mb:.1f}MB), switching to local fallback")
            return {"error": "file_too_large", "text": ""}
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": config.GROQ_STT_MODEL,
            "response_format": "verbose_json",
        }
        if language: data["language"] = language
        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    self.url, headers=headers,
                    files={"file": (Path(audio_path).name, f, "audio/wav")},
                    data=data, timeout=60
                )
            response.raise_for_status()
            res = response.json()
            return {"text": res.get("text", "").strip(), "language": res.get("language", "unknown")}
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}", "text": ""}

class ExpertSynthesisEngine:
    """World-Class McKinsey-Style Strategic Synthesis for FinFlux V4.2+."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.main_model = config.GROQ_LLM_MODEL
        self.reason_model = config.GROQ_LLM_REASON_MODEL
        self.fast_model = config.GROQ_LLM_FAST_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _call_groq(self, system_prompt: str, user_prompt: str, model_override: str = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_override or self.main_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"} if "JSON" in system_prompt else None
        }
        res = requests.post(self.url, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def normalize_transcript(self, transcript: str) -> str:
        """Stage 2: Transcript Normalization (Llama-3-8B Fast)."""
        sys_p = (
            "You are a professional Transcript Cleaner. Correct ASR errors, fix spelling, "
            "preserve the exact meaning. Return ONLY cleaned text in the same language as the transcript."
        )
        return self._call_groq(sys_p, transcript, model_override=self.fast_model)

    def analyze(self, transcript: str, entities: List[Dict[str, Any]] = None, fin_sentiment: str = "Neutral") -> Dict[str, Any]:
        """Stage 7: World-Class Strategic Synthesis & Expert Reasoning (Llama-70B + Qwen-32B)."""
        ent_text = json.dumps(entities or [], indent=2)
        
        # ─── Part 1: Expert Technical Reasoning (Qwen) ───
        reason_p = f"Transcript: {transcript}\nEntities: {ent_text}\nSentiment: {fin_sentiment}"
        reason_sys = (
            "You are a senior Financial Analyst (CFA). Provide 3-4 bullet points of high-level technical "
            "logic for this conversation. No advice, just technical reasoning. "
            "Respond in the same language as the transcript."
        )
        expert_reasoning = self._call_groq(reason_sys, reason_p, model_override=self.reason_model)

        # ─── Part 2: McKinsey-Style Final Synthesis (Llama-70B) ───
        final_sys = """You are a senior Financial Consultant (McKinsey/GS level). Generate a structured STRATEGIC JSON.
    RULES: 1. NO FINANCIAL ADVICE. 2. USE NEUTRAL, STRATEGIC TERMINOLOGY. 3. RESPOND IN THE SAME LANGUAGE AS THE TRANSCRIPT.
JSON SCHEMA:
{
  "executive_summary": "A world-class strategic summary (concise, high-impact).",
  "key_insights": ["High-impact bullet points focusing on financial leverage/operations."],
  "risk_assessment": "Technical analysis of risk (Low/Medium/High/Critical) and its logic.",
  "future_gearing": "Contextual insights for future financial decisions/considerations.",
  "strategic_intent": "Deep analysis of the customer's intent (e.g., Growth, Consolidation, Liquidity).",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL"
}"""
        final_user = f"TRANSCRIPT: {transcript}\nVERIFIED ENTITIES: {ent_text}\nEXPERT ANALYST NOTES: {expert_reasoning}"
        analysis_json = json.loads(self._call_groq(final_sys, final_user, model_override=self.main_model))
        
        # Merge technical reasoning into the response
        analysis_json["expert_reasoning_points"] = expert_reasoning
        return analysis_json

