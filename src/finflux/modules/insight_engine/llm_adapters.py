"""V4.2+ McKinsey-Level Groq Synthesis Adapter: Strategic IQ & Expert Reasoning."""
import os
import requests
import json
import re
import mimetypes
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional
from pathlib import Path
from finflux import config

class GroqWhisperAdapter:
    """Groq Whisper API for multilingual transcription."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    @staticmethod
    def _hindi_signal_score(text: str) -> float:
        if not text:
            return 0.0
        hindi_script_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        hindi_markers = {
            "hai", "haan", "nahi", "nahin", "nhi", "ka", "ki", "ke", "ko", "se", "par", "mein",
            "main", "mera", "meri", "mere", "hum", "aap", "tum", "jo", "kyu", "kyun", "kya", "kaise",
            "acha", "achha", "theek", "thik", "sir", "ji", "wali", "wala", "kar", "karo", "raha", "rahi",
            "gaya", "gayi", "hoga", "hogi", "tha", "thi", "tha", "chahiye", "matlab",
        }
        marker_hits = sum(1 for t in tokens if t in hindi_markers)
        return float(hindi_script_chars) + (2.5 * float(marker_hits))

    @staticmethod
    def _build_asr_payload(language: Optional[str] = None, strict_mixed: bool = False, model_override: Optional[str] = None) -> Dict[str, Any]:
        prompt = (
            "Audio may contain Hindi, Hinglish, and English mixed naturally. "
            "Transcribe exactly as spoken. Do not translate Hindi/Hinglish into English. "
            "Preserve original wording and financial terms."
        )
        if strict_mixed:
            prompt += " Prefer preserving Hindi words in Devanagari when clearly spoken in Hindi."

        payload: Dict[str, Any] = {
            "model": model_override or config.GROQ_STT_MODEL,
            "response_format": "verbose_json",
            "temperature": "0",
            "prompt": prompt,
        }
        if language:
            payload["language"] = language
        return payload

    def _transcribe_once(self, audio_path: str, language: Optional[str] = None, strict_mixed: bool = False, model_override: Optional[str] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._build_asr_payload(language=language, strict_mixed=strict_mixed, model_override=model_override)
        content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        with open(audio_path, "rb") as f:
            response = requests.post(
                self.url,
                headers=headers,
                files={"file": (Path(audio_path).name, f, content_type)},
                data=data,
                timeout=60,
            )
        if response.status_code >= 400:
            body = (response.text or "").strip().replace("\n", " ")
            body = body[:400]
            raise requests.HTTPError(
                f"{response.status_code} {response.reason}: {body}",
                response=response,
            )
        res = response.json()
        return {"text": res.get("text", "").strip(), "language": str(res.get("language", "unknown")).lower()}

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """High-resolution transcription via Groq Whisper."""
        if not self.api_key:
            return {"error": "GROQ_API_KEY not set", "text": ""}

        # Check file size before sending — Groq limit is 25MB
        import os
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 24:
            print(f"[GroqWhisper] File too large ({file_size_mb:.1f}MB), switching to local fallback")
            return {"error": "file_too_large", "text": ""}
            
        try:
            primary = self._transcribe_once(audio_path, language=language, strict_mixed=False)

            # Retry with Hindi hint when auto-detection returns English and no language was forced.
            if language is None and primary.get("language", "").startswith("en"):
                try:
                    retry_hi = self._transcribe_once(audio_path, language="hi", strict_mixed=True)
                    primary_score = self._hindi_signal_score(primary.get("text", ""))
                    retry_score = self._hindi_signal_score(retry_hi.get("text", ""))
                    if retry_score > primary_score + 1.0:
                        return retry_hi
                except Exception as retry_error:
                    # Keep primary transcript if retry path fails.
                    print(f"[GroqWhisper] Hindi retry failed, keeping primary transcript: {retry_error}")

            return primary
        except Exception as e:
            err_text = str(e)
            # Retry once with alternate Groq STT model if the primary model is rejected.
            invalid_model = "model" in err_text.lower() or "invalid" in err_text.lower() or "not found" in err_text.lower()
            if invalid_model and config.GROQ_STT_HQ_MODEL and config.GROQ_STT_HQ_MODEL != config.GROQ_STT_MODEL:
                try:
                    retry = self._transcribe_once(
                        audio_path,
                        language=language,
                        strict_mixed=False,
                        model_override=config.GROQ_STT_HQ_MODEL,
                    )
                    return retry
                except Exception as retry_error:
                    return {"error": f"Transcription failed: {err_text}; fallback model failed: {retry_error}", "text": ""}
            return {"error": f"Transcription failed: {err_text}", "text": ""}

class ExpertSynthesisEngine:
    """World-Class McKinsey-Style Strategic Synthesis for FinFlux V4.2+."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.main_model = config.GROQ_LLM_MODEL
        self.reason_model = config.GROQ_LLM_REASON_MODEL
        self.fast_model = config.GROQ_LLM_FAST_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _call_groq(self, system_prompt: str, user_prompt: str, model_override: Optional[str] = None, temperature: float = 0.1) -> str:
        # Check DEMO_MODE
        if os.environ.get("DEMO_MODE", "").lower() == "true":
            return self._get_demo_response(system_prompt)
            
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_override or self.main_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": temperature,
        }
        
        # Remove response_format for non-JSON calls — don't pass None
        if "JSON" in system_prompt or "json" in system_prompt.lower():
            payload["response_format"] = {"type": "json_object"}

        try:
            res = requests.post(self.url, headers=headers, json=payload, timeout=45)
            res.raise_for_status()
            raw = res.json()["choices"][0]["message"]["content"]
            
            # CRITICAL: Strip Qwen3 chain-of-thought thinking tokens
            cleaned_raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            
            return cleaned_raw
        except Exception as e:
            print(f"[GroqExpert] API Failure: {e}")
            return self._get_demo_response(system_prompt) if "JSON" in system_prompt else user_prompt

    def _get_demo_response(self, system_prompt: str) -> str:
        """Convincing McKinsey-style fixture data."""
        if "STRATEGIC JSON" in system_prompt:
            return json.dumps({
                "executive_summary": "The client is seeking to optimize their diversified investment portfolio with a focus on long-term capital appreciation and risk mitigation. Their current posture indicates a strong preference for secure assets while maintaining exposure to emerging opportunities.",
                "key_insights": ["High-yield fixed income remains the primary liquidity driver.", "Strategic reallocation into low-volatility equity indexes is recommended.", "Current EMI obligations are well-covered by operational cash flow."],
                "risk_assessment": "Moderate risk primarily focused on market volatility. Structural risk is low due to robust asset backing.",
                "future_gearing": "Positioning for potential interest rate adjustments is critical in the next 18-24 months.",
                "strategic_intent": "Growth & Liquidity Optimization",
                "risk_level": "LOW|MEDIUM"
            })
        elif "Financial Analyst" in system_prompt:
            return "- Portfolio diversification index suggests a defensive tilt.\n- Interest coverage ratio is within optimal bounds for current debt levels.\n- Strategic alignment with regional market growth targets is evident."
        else:
            return "Transcript cleaned and normalized for professional financial review."

    @staticmethod
    def _has_repetition_artifact(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 6:
            return False
        repeats = 0
        for i in range(1, len(lines)):
            if lines[i] == lines[i - 1]:
                repeats += 1
        return repeats >= 3

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _normalization_drift_too_high(self, original: str, edited: str) -> bool:
        o = self._normalize_for_similarity(original)
        e = self._normalize_for_similarity(edited)
        if not e:
            return True

        ratio = SequenceMatcher(None, o, e).ratio()
        len_ratio = (len(e) / max(1, len(o)))

        o_nums = re.findall(r"\d+(?:\.\d+)?", o)
        e_nums = re.findall(r"\d+(?:\.\d+)?", e)
        numbers_diverged = set(e_nums) - set(o_nums)

        if ratio < 0.72:
            return True
        if len_ratio < 0.6 or len_ratio > 1.55:
            return True
        if len(numbers_diverged) > 1:
            return True
        if self._has_repetition_artifact(edited):
            return True
        return False

    def normalize_transcript(self, transcript: str) -> str:
        """Stage 2: Transcript Normalization (Llama-3-8B Fast)."""
        sys_p = """You are a strict ASR correction engine for Indian financial conversations.

TASK: Perform MINIMAL correction only.

RULES:
1. Keep the original sentence order and overall length close to input.
2. Do NOT add new facts, numbers, recommendations, or repeated lines.
3. Do NOT translate language; preserve Hindi/Hinglish/English as spoken.
4. Correct only obvious ASR mistakes (spelling, token spacing, punctuation).
5. Keep financial terms exactly (EMI, SIP, CIBIL, FD, NPS, ROI, etc.).
6. Keep existing speaker labels only if already present; do NOT invent speaker labels.
7. Return ONLY corrected transcript text.
"""
        try:
            result = self._call_groq(sys_p, transcript, model_override=self.fast_model, temperature=0.0)
            if not result.strip():
                return transcript
            if self._normalization_drift_too_high(transcript, result):
                print("[GroqExpert] Normalization drift too high, using raw transcript")
                return transcript
            return result
        except:
            return transcript # Safe fallback

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Validate and fix LLM JSON outputs."""
        default_analysis = {
            "executive_summary": "Analysis currently unavailable due to processing limitations.",
            "key_insights": ["System is currently utilizing fallback analysis measures."],
            "risk_assessment": "Risk assessment pending technical verification.",
            "future_gearing": "Strategic outlook is currently conservative.",
            "strategic_intent": "General Inquiry",
            "risk_level": "LOW"
        }
        try:
            cleaned = text.strip()
            # Handle markdown code blocks if LLM adds them
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[-1].split("```")[0].strip()
            
            data = json.loads(cleaned)
            # Ensure all fields exist
            for field, val in default_analysis.items():
                if field not in data: data[field] = val
            return data
        except Exception as e:
            print(f"[GroqExpert] JSON Parse Guard Triggered: {e}")
            return default_analysis

    def analyze(self, transcript: str, entities: Optional[List[Dict[str, Any]]] = None, fin_sentiment: str = "Neutral", memory_context: str = "") -> Dict[str, Any]:
        """Stage 7: World-Class Strategic Synthesis & Expert Reasoning (Llama-70B + Qwen-32B)."""
        ent_text = json.dumps(entities or [], indent=2)
        
        # ─── Part 1: Expert Technical Reasoning (Qwen) ───
        reason_p = f"Transcript: {transcript}\nEntities: {ent_text}\nSentiment: {fin_sentiment}\nMemoryContext: {memory_context}"
        reason_sys = """You are a Senior Financial Analyst (CFA Level 3, 15 years experience in Indian markets).

Analyze this financial conversation and provide STRUCTURED TECHNICAL REASONING.

OUTPUT FORMAT — return exactly 4 bullet points, each on a new line starting with "•":

- **[TOPIC LABEL]**: [2-3 sentences of technical analysis. Be specific, cite exact figures mentioned.]
- **[TOPIC LABEL]**: [2-3 sentences. Focus on financial mechanics and implications.]
- **[TOPIC LABEL]**: [2-3 sentences. Address risk dimensions relevant to the conversation.]
- **[TOPIC LABEL]**: [2-3 sentences. Address future positioning or decision framework.]

RULES:
- NO financial advice. Observations and technical analysis only.
- Reference specific numbers, products, or terms from the transcript
- Use professional financial terminology (liquidity, leverage, tenor, yield, etc.)
- Each bullet must be self-contained and substantive
- If Hindi/Hinglish terms are used, acknowledge them with English translation in brackets

TOPIC LABELS to choose from based on content:
Debt Servicing Analysis | Investment Strategy | Risk Profile | Liquidity Assessment | 
Asset Allocation | Tax Efficiency | Insurance Coverage | Credit Utilization | 
Portfolio Concentration | Cash Flow Management | Commitment Tracking | Market Exposure"""
        try:
            expert_reasoning = self._call_groq(reason_sys, reason_p, model_override=self.reason_model)
        except:
            expert_reasoning = "- Analysis engine is currently using local reasoning patterns.\n- Technical extraction continues on cached parameters."

        # ─── Part 2: McKinsey-Style Final Synthesis (Llama-70B) ───
        final_sys = """You are a Senior Financial Intelligence Analyst producing a STRUCTURED STRATEGIC REPORT.

Generate a comprehensive financial intelligence JSON. This is NOT financial advice — it is structured intelligence extraction.

STRICT JSON SCHEMA (return ONLY valid JSON, no text outside):
{
  "executive_summary": "3-4 sentences. Opening sentence states the primary financial activity. Second sentence identifies key entities (amounts, products, timelines). Third sentence notes the risk dimension. Fourth sentence on strategic positioning.",
  
  "key_insights": [
    "Insight 1: Specific observation about a financial commitment or pattern mentioned",
    "Insight 2: Observation about risk exposure or vulnerability",  
    "Insight 3: Observation about financial behavior or decision-making pattern",
    "Insight 4: Observation about future financial trajectory based on conversation"
  ],
  
  "risk_assessment": "2-3 sentences. State risk level clearly (LOW/MEDIUM/HIGH/CRITICAL). Explain the specific factors driving this level. Note any mitigating factors.",
  
  "future_gearing": "2-3 sentences. What financial considerations will be relevant for this person going forward based on what was discussed. Frame as contextual observations, not advice.",
  
  "strategic_intent": "1-2 sentences. Classify the primary intent: [Growth | Consolidation | Debt Management | Risk Mitigation | Wealth Preservation | Liquidity Building | Learning/Research]. Explain why.",
  
  "risk_level": "LOW or MEDIUM or HIGH or CRITICAL"
}

QUALITY STANDARDS:
- executive_summary must be minimum 80 words
- Each key_insight must reference something SPECIFICALLY mentioned in the transcript
- Never use generic filler phrases like "the customer discussed" or "the conversation covered"
- Use precise financial language
- If the conversation is in Hindi/Hinglish, the insights should still be in English but reference the Hindi terms used"""
        final_user = f"TRANSCRIPT: {transcript}\nVERIFIED ENTITIES: {ent_text}\nMEMORY CONTEXT: {memory_context}\nEXPERT ANALYST NOTES: {expert_reasoning}"
        
        raw_response = self._call_groq(final_sys, final_user, model_override=self.main_model)
        analysis_json = self._safe_json_parse(raw_response)
        
        # Merge technical reasoning into the response
        analysis_json["expert_reasoning_points"] = expert_reasoning
        return analysis_json
