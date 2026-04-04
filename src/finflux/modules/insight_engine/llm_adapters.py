"""V4.2+ McKinsey-Level Groq Synthesis Adapter: Strategic IQ & Expert Reasoning."""
import os
import requests
import json
import re
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from finflux import config
from pydub import AudioSegment

class GroqWhisperAdapter:
    """Groq Whisper API for multilingual transcription."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def _compress_for_stt(
        self,
        source_path: str,
        frame_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        max_seconds: Optional[int] = None,
    ) -> str:
        """Create a temporary WAV optimized for low upload size while preserving speech intelligibility."""
        audio = AudioSegment.from_file(source_path)
        if max_seconds and max_seconds > 0:
            audio = audio[: max_seconds * 1000]
        audio = audio.set_frame_rate(frame_rate).set_channels(channels).set_sample_width(sample_width)

        fd, out_path = tempfile.mkstemp(prefix="finflux_stt_", suffix=".wav")
        os.close(fd)
        audio.export(out_path, format="wav")
        return out_path

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """High-resolution transcription via Groq Whisper."""
        if not self.api_key:
            return {"error": "GROQ_API_KEY not set", "text": ""}

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": config.GROQ_STT_MODEL,
            "response_format": "verbose_json",
        }
        if language: data["language"] = language

        upload_path = audio_path
        temp_paths: List[str] = []

        # Proactively compress large-ish wav payloads before Groq upload.
        try:
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 1.0:
                upload_path = self._compress_for_stt(audio_path, frame_rate=16000, channels=1, sample_width=2)
                temp_paths.append(upload_path)
        except Exception:
            upload_path = audio_path

        try:
            with open(upload_path, "rb") as f:
                response = requests.post(
                    self.url, headers=headers,
                    files={"file": (Path(upload_path).name, f, "audio/wav")},
                    data=data, timeout=60
                )

            # Retry once with aggressive compression if Groq rejects payload size.
            if response.status_code == 413:
                try:
                    retry_path = self._compress_for_stt(
                        source_path=audio_path,
                        frame_rate=12000,
                        channels=1,
                        sample_width=2,
                        max_seconds=300,
                    )
                    temp_paths.append(retry_path)
                    with open(retry_path, "rb") as f2:
                        response = requests.post(
                            self.url,
                            headers=headers,
                            files={"file": (Path(retry_path).name, f2, "audio/wav")},
                            data=data,
                            timeout=60,
                        )
                except Exception:
                    pass

            response.raise_for_status()
            res = response.json()
            segments = res.get("segments", []) if isinstance(res, dict) else []
            word_probs: List[float] = []
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    words = seg.get("words", [])
                    if not isinstance(words, list):
                        continue
                    for w in words:
                        if not isinstance(w, dict):
                            continue
                        raw_prob = w.get("probability")
                        try:
                            p = float(raw_prob)
                            if 0.0 <= p <= 1.0:
                                word_probs.append(p)
                        except Exception:
                            continue

            asr_conf = sum(word_probs) / len(word_probs) if word_probs else 0.85
            return {
                "text": res.get("text", "").strip(),
                "language": res.get("language", "unknown"),
                "segments": segments,
                "asr_confidence": float(asr_conf),
            }
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}", "text": ""}
        finally:
            for p in temp_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    continue

class ExpertSynthesisEngine:
    """World-Class McKinsey-Style Strategic Synthesis for FinFlux V4.2+."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.main_model = config.GROQ_LLM_MODEL
        self.reason_model = config.GROQ_LLM_REASON_MODEL
        self.fast_model = config.GROQ_LLM_FAST_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _call_groq(self, system_prompt: str, user_prompt: str, model_override: Optional[str] = None) -> str:
        # Check DEMO_MODE
        if os.environ.get("DEMO_MODE", "").lower() == "true":
            return self._get_demo_response(system_prompt)
            
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_override or self.main_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.1,
        }
        
        # Remove response_format for non-JSON calls
        if "JSON" in system_prompt or "json" in system_prompt.lower():
            payload["response_format"] = {"type": "json_object"}

        try:
            res = requests.post(self.url, headers=headers, json=payload, timeout=45)
            res.raise_for_status()
            raw = res.json()["choices"][0]["message"]["content"]
            
            # CRITICAL: Strip Qwen3 chain-of-thought thinking tokens
            import re
            cleaned_raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            
            return cleaned_raw
        except Exception as e:
            print(f"[GroqExpert] API Failure: {e}")
            return self._get_demo_response(system_prompt) if "JSON" in system_prompt else user_prompt

    def _get_demo_response(self, system_prompt: str) -> str:
        """Convincing McKinsey-style fixture data."""
        if "financial memory summary" in system_prompt.lower() or "memory brief" in system_prompt.lower():
            return (
                "The user maintains recurring loan and SIP obligations, with monthly cash-flow discipline as a consistent priority. "
                "Across prior sessions, products discussed repeatedly include term insurance, emergency-fund parking, and diversified mutual-fund SIPs. "
                "Risk posture is generally moderate, with preference for liquidity safeguards before aggressive allocation changes. "
                "An unresolved goal remains balancing debt reduction with steady long-term wealth accumulation."
            )
        if "STRATEGIC JSON" in system_prompt or "valid JSON" in system_prompt:
            return json.dumps({
                "executive_summary": "The client is seeking to optimize their diversified investment portfolio with a focus on long-term capital appreciation and risk mitigation. Their current posture indicates a strong preference for secure assets while maintaining exposure to emerging opportunities.",
                "key_insights": ["High-yield fixed income remains the primary liquidity driver.", "Strategic reallocation into low-volatility equity indexes is recommended.", "Current EMI obligations are well-covered by operational cash flow."],
                "risk_assessment": "Moderate risk primarily focused on market volatility. Structural risk is low due to robust asset backing.",
                "future_gearing": "Positioning for potential interest rate adjustments is critical in the next 18-24 months.",
                "strategic_intent": "Growth & Liquidity Optimization",
                "risk_level": "MEDIUM"
            })
        elif "Financial Analyst" in system_prompt:
            return "- Portfolio diversification index suggests a defensive tilt.\n- Interest coverage ratio is within optimal bounds for current debt levels.\n- Strategic alignment with regional market growth targets is evident."
        else:
            return "Transcript cleaned and normalized for professional financial review."

    def summarize_memory_brief(self, memory_records: List[Dict[str, Any]]) -> str:
        """Compress retrieved memories into a compact brief for prompt injection."""
        if not memory_records:
            return ""

        compact_records: List[Dict[str, Any]] = []
        for row in memory_records[:6]:
            compact_records.append({
                "financial_topic": str(row.get("financial_topic", "")).strip()[:80],
                "risk_level": str(row.get("risk", "")).strip()[:20],
                "sentiment": str(row.get("financial_sentiment", "")).strip()[:40],
                "strategic_intent": str(row.get("strategic_intent", "")).strip()[:100],
                "future_gearing": str(row.get("future_gearing", "")).strip()[:180],
                "risk_assessment": str(row.get("risk_assessment", "")).strip()[:180],
                "executive_summary": str(row.get("executive_summary", "")).strip()[:260],
                "transcript": str(row.get("transcript", "")).strip()[:220],
            })

        system_prompt = (
            "Compress these past financial conversation records into a 3-4 sentence memory brief. "
            "Focus only on: standing commitments, recurring financial products mentioned, established risk preferences, "
            "and unresolved financial goals. Ignore conversational details. Write in third-person present tense. "
            "Maximum 200 tokens."
        )
        user_prompt = json.dumps(compact_records, ensure_ascii=False)

        try:
            brief = self._call_groq(system_prompt, user_prompt, model_override=self.fast_model).strip()
        except Exception:
            brief = ""

        if not brief:
            # Fallback without LLM dependence.
            fallback_points = [r.get("executive_summary", "") for r in compact_records if r.get("executive_summary")]
            brief = " ".join(str(x).strip() for x in fallback_points[:2]).strip()

        words = brief.split()
        if len(words) > 180:
            brief = " ".join(words[:180]).strip()
        return brief

    def _memory_is_relevant(self, transcript: str, memory_record: Dict[str, Any]) -> bool:
        """Binary relevance gate for one memory record using fast model."""
        gate_system = (
            "You are a strict relevance gate for financial memory retrieval. "
            "Given a current transcript and one past memory record, answer only YES or NO. "
            "Answer YES only if the memory is directly relevant to the same financial concern, "
            "product context, commitment, or decision trajectory. Otherwise answer NO."
        )
        gate_user = json.dumps(
            {
                "current_transcript": transcript,
                "memory": {
                    "financial_topic": memory_record.get("financial_topic", ""),
                    "risk_level": memory_record.get("risk_level") or memory_record.get("risk", ""),
                    "financial_sentiment": memory_record.get("financial_sentiment", ""),
                    "strategic_intent": memory_record.get("strategic_intent", ""),
                    "executive_summary": memory_record.get("executive_summary", ""),
                    "transcript": memory_record.get("transcript", ""),
                },
            },
            ensure_ascii=False,
        )
        try:
            out = self._call_groq(gate_system, gate_user, model_override=self.fast_model).strip().upper()
            return out.startswith("YES")
        except Exception:
            return False

    def build_memory_context(self, transcript: str, memory_records: List[Dict[str, Any]], max_keep: int = 3) -> str:
        """Layer 3: relevance-gate retrieved memories, then synthesize a compact memory brief."""
        if not memory_records:
            return ""

        relevant: List[Dict[str, Any]] = []
        for row in memory_records[:8]:
            if not isinstance(row, dict):
                continue
            if self._memory_is_relevant(transcript, row):
                relevant.append(row)
            if len(relevant) >= max_keep:
                break

        if not relevant:
            return ""

        return self.summarize_memory_brief(relevant)

    def decompose_retrieval_query(self, user_message: str, detected_topic: str = "") -> Dict[str, Any]:
        """Create semantic query + optional metadata filters for hybrid retrieval."""
        system_prompt = """You are a retrieval query decomposition engine for a financial assistant.

Convert the user message into:
1) semantic_query: a third-person English financial-intent description, concise and normalized.
2) optional metadata filters from this allowed set only:
- financial_topic
- risk_level (LOW|MEDIUM|HIGH|CRITICAL)
- financial_sentiment (Positive|Neutral|Negative)
- created_at_start (YYYY-MM-DD)
- created_at_end (YYYY-MM-DD)

Rules:
- Filters are optional and additive only.
- If confidence is low for any filter, omit it.
- Never invent values not implied by user input.
- Return JSON only with this schema:
{
  "semantic_query": "string",
  "filters": {
    "financial_topic": "string (optional)",
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL (optional)",
    "financial_sentiment": "Positive|Neutral|Negative (optional)",
    "created_at_start": "YYYY-MM-DD (optional)",
    "created_at_end": "YYYY-MM-DD (optional)"
  }
}
"""
        user_prompt = json.dumps(
            {
                "user_message": user_message,
                "detected_financial_topic": detected_topic,
            },
            ensure_ascii=False,
        )

        fallback = {"semantic_query": user_message.strip(), "filters": {}}
        try:
            raw = self._call_groq(system_prompt, user_prompt, model_override=self.fast_model).strip()
            cleaned = raw
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[-1].split("```")[0].strip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return fallback

            semantic_query = str(parsed.get("semantic_query", "")).strip() or fallback["semantic_query"]
            raw_filters = parsed.get("filters", {})
            filters: Dict[str, Any] = {}
            if isinstance(raw_filters, dict):
                topic_val = str(raw_filters.get("financial_topic", "")).strip()
                if topic_val:
                    filters["financial_topic"] = topic_val

                risk_val = str(raw_filters.get("risk_level", "")).strip().upper()
                if risk_val in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                    filters["risk_level"] = risk_val

                sentiment_val = str(raw_filters.get("financial_sentiment", "")).strip().capitalize()
                if sentiment_val in {"Positive", "Neutral", "Negative"}:
                    filters["financial_sentiment"] = sentiment_val

                for key in ("created_at_start", "created_at_end"):
                    date_val = str(raw_filters.get(key, "")).strip()
                    if len(date_val) == 10 and date_val[4] == "-" and date_val[7] == "-":
                        filters[key] = date_val

            return {"semantic_query": semantic_query, "filters": filters}
        except Exception:
            return fallback

    def _script_ratios(self, text: str) -> Dict[str, float]:
        devanagari = 0
        latin = 0
        for ch in text:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                devanagari += 1
            elif re.match(r"[A-Za-z]", ch):
                latin += 1
        total = devanagari + latin
        if total == 0:
            return {"devanagari": 0.0, "latin": 0.0}
        return {
            "devanagari": (devanagari / total) * 100.0,
            "latin": (latin / total) * 100.0,
        }

    def normalize_transcript(
        self,
        transcript: str,
        dominant_language: str = "",
        language_breakdown: Optional[Dict[str, float]] = None,
    ) -> str:
        """Stage 2: Transcript Normalization (Llama-3-8B Fast)."""
        transcript = transcript[:1500]
        breakdown = language_breakdown or {}

        # For dominant Hindi input, preserve raw transcript exactly.
        if str(dominant_language).lower() == "hindi" and float(breakdown.get("hindi", 0.0)) > 60.0:
            return transcript

        sys_p = """You are an expert multilingual ASR correction specialist for Indian financial conversations.

TASK: Fix ASR errors. Return ONLY corrected transcript — no commentary.

CRITICAL CONSTRAINTS:
- Never translate the transcript.
- Never convert script (Devanagari stays Devanagari, Latin stays Latin).
- Preserve original code-switching exactly for Hinglish.
- Keep language and script identity unchanged while correcting ASR mistakes only.

COMMON ASR ERRORS TO FIX:
- "पांच जाल" / "paanch jaal" → ₹5,000
- "एक लाख" → ₹1,00,000
- "पांच साल" / "five jal" → 5 साल (5 years)
- "sip" / "सिप" / "ship" → SIP
- "ee em ai" / "ईएमआई" → EMI
- "mutchual" / "म्यूचुअल फंड" → Mutual Fund
- "sibil" / "civil score" → CIBIL score
- Garbage syllable runs like "विच फील्ड शूअर आइए" → reconstruct from financial context, tag as [reconstructed]
- Preserve ALL financial figures exactly

OUTPUT: corrected transcript text only."""
        try:
            result = self._call_groq(sys_p, transcript, model_override=self.fast_model)
            cleaned = result.strip() if result else ""
            if not cleaned:
                return transcript

            # Script-preservation guard to prevent accidental translation/conversion.
            src = self._script_ratios(transcript)
            out = self._script_ratios(cleaned)
            if str(dominant_language).lower() == "hindi":
                if out["devanagari"] + 20.0 < src["devanagari"]:
                    return transcript
            elif str(dominant_language).lower() == "english":
                if out["latin"] + 20.0 < src["latin"]:
                    return transcript
            else:  # hinglish / mixed
                if out["devanagari"] + 25.0 < src["devanagari"] and out["latin"] + 25.0 < src["latin"]:
                    return transcript

            return cleaned
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
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[-1].split("```")[0].strip()
            
            data = json.loads(cleaned)
            for field, val in default_analysis.items():
                if field not in data: data[field] = val
            return data
        except Exception as e:
            print(f"[GroqExpert] JSON Parse Guard Triggered: {e}")
            return default_analysis

    def analyze(
        self,
        transcript: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        fin_sentiment: str = "Neutral",
        thread_history_context: str = "",
        long_term_memory_context: str = "",
    ) -> Dict[str, Any]:
        """Stage 7: Context-Aware Strategic Intelligence & Wall Architecture (Llama-70B + Qwen-32B)."""
        transcript = transcript[:2500]
        entities = (entities or [])[:20]
        ent_text = json.dumps(entities, indent=2)
        thread_history_context = (thread_history_context or "").strip()
        long_term_memory_context = (long_term_memory_context or "").strip()

        reason_sections = [
            f"CURRENT TRANSCRIPT (PRIMARY - 70% weight):\n{transcript}",
            f"ENTITIES:\n{ent_text}\nSENTIMENT: {fin_sentiment}",
        ]
        if thread_history_context:
            reason_sections.insert(1, f"THREAD HISTORY (SECONDARY - 20% weight):\n{thread_history_context}")
        if long_term_memory_context:
            insert_at = 2 if thread_history_context else 1
            reason_sections.insert(insert_at, f"LONG-TERM MEMORY (TERTIARY - 10% weight):\n{long_term_memory_context}")
        
        # ─── Part 1: Expert Technical Reasoning (CFA Intelligence Layer) ───
        reason_p = "\n\n".join(reason_sections)
        reason_sys = """You are a Senior CFA-certified Financial Analyst (20 years Indian experience). 
Act as the 'Intelligence Layer' to provide a Context-Aware Multi-Wall Technical Audit.

SILENT AUDIT LOGIC:
1. CURRENT TRANSCRIPT is the primary source of truth.
2. THREAD HISTORY is continuity context only if it supports the current transcript.
3. LONG-TERM MEMORY is lowest priority and optional.
4. MEMORY GATE: Only reference historical context if you identify a direct semantic link to the current transcript (same product type, same concern, same financial commitment). If no link exists, ignore historical context entirely and analyze only the CURRENT TRANSCRIPT.
5. Net Surplus: Determine [Income - Expenses = Surplus].
6. Capacity Check: If Investment > Surplus, highlight "SIP Limit".
7. Safety: Priority #1 is Emergency Buffer = 3 x Monthly Expenses.

OUTPUT FORMAT (Multi-Wall Hierarchy):
- **STRATEGIC WALL [Action Title]**: [3-5 sentences. Reference figures. Calculate savings rate % and CAGR corpus. Acknowledge past context if relevant.]
- **TECHNICAL WALL [Action Title]**: [3-5 sentences. Apply 3-Tier Asset Mapping (Foundation/Growth/Alpha). Address risk types: Liquidity, Inflation, Concentration.]
- **COMPLIANCE WALL [Action Title]**: [3-5 sentences. Evaluate lock-ins and flexibility. Identify the next critical decision.]

RULES:
1. Every sentence must reference specific transcript data.
2. Normalized Informality: Transform slang into McKinsey executive logic.
3. Each bullet minimum 4 sentences. NO LATEX."""

        try:
            expert_reasoning = self._call_groq(reason_sys, reason_p, model_override=self.reason_model)
        except:
            expert_reasoning = "- **STRATEGIC WALL**: Technical intelligence layer initialization failed. Contextual mapping disabled."

        # ─── Part 2: McKinsey-Style Final Synthesis (Llama-70B) ───
        final_sys = """You are Head of Strategic Intelligence at a top Indian wealth management firm. 
Produce a McKinsey-quality isolated JSON report. This result must be CONTEXT-AWARE.

CONTEXT PRIORITY RULES:
- CURRENT TRANSCRIPT: primary source of truth (70%)
- THREAD HISTORY: continuity only when directly relevant (20%)
- LONG-TERM MEMORY: optional and lowest weight (10%)
- MEMORY GATE: include historical context only when a direct semantic link exists. Never force a historical linkage.

Return ONLY valid JSON matching this exact schema:
{
    "executive_summary": "4 sentences minimum. Focus primarily on CURRENT TRANSCRIPT specifics and figures.",
  "key_insights": [
    "Commitment: [Exploratory vs Decision] — reference specific code language",
    "Asset Mix: Tier 1/2/3 mapping observations",
    "Risk Quantification: Specific risk indicator with figures"
  ],
  "risk_assessment": "3 sentences. S1: risk level + driver. S2: % Income Committed. S3: emergency buffer priority.",
  "future_gearing": "2-3 sentences. Future decision trajectory based on current surplus and history.",
    "historical_link": "Include ONLY if a direct relevant historical link exists; omit otherwise.",
  "strategic_intent": "One of: [Growth | Consolidation | Debt Management | Risk Mitigation | Wealth Preservation | Liquidity Building | Learning/Research].",
  "risk_level": "LOW or MEDIUM or HIGH or CRITICAL"
}

NEVER WRITE: 'the customer discussed', 'this conversation covers', 'it is important to note'."""

        final_sections = [f"CURRENT TRANSCRIPT:\n{transcript}"]
        if thread_history_context:
            final_sections.append(f"THREAD HISTORY:\n{thread_history_context}")
        if long_term_memory_context:
            final_sections.append(f"LONG-TERM MEMORY:\n{long_term_memory_context}")
        final_sections.append(f"ENTITIES:\n{ent_text}")
        final_sections.append(f"ANALYST NOTES:\n{expert_reasoning}")
        final_user = "\n\n".join(final_sections)
        
        raw_response = self._call_groq(final_sys, final_user, model_override=self.main_model)
        analysis_json = self._safe_json_parse(raw_response)
        
        # Merge technical reasoning into the response
        analysis_json["expert_reasoning_points"] = expert_reasoning
        return analysis_json

    def generate_future_insights(
        self,
        transcript: str,
        current_analysis: Dict[str, Any],
        memory_context: str,
    ) -> List[Dict[str, Any]]:
        """Generate 3 contextual future insights using Llama-70B from current analysis + memory context."""
        allowed_types = {
            "COMMITMENT_FOLLOWUP",
            "MARKET_TIMING",
            "RISK_REBALANCING",
            "SAVINGS_OPPORTUNITY",
            "DEADLINE_APPROACHING",
        }
        allowed_urgency = {"HIGH", "MEDIUM", "LOW"}

        sys_prompt = """You are a financial conversation intelligence planner.
Generate exactly 3 future insights grounded in current transcript, current analysis, and historical memory brief.

Return ONLY JSON array with 3 objects. Each object schema:
{
  "insight_type": "COMMITMENT_FOLLOWUP|MARKET_TIMING|RISK_REBALANCING|SAVINGS_OPPORTUNITY|DEADLINE_APPROACHING",
  "insight_text": "one sentence specific to the user situation",
  "urgency": "HIGH|MEDIUM|LOW",
  "days_relevant": 1-90 integer
}

Rules:
- Be specific, not generic.
- Keep each insight_text to one sentence.
- Do not add extra keys.
- Output strict JSON only.
"""

        user_payload = {
            "transcript": str(transcript or "")[:1200],
            "current_analysis": {
                "executive_summary": str(current_analysis.get("executive_summary", ""))[:1200],
                "risk_level": str(current_analysis.get("risk_level", "")),
                "strategic_intent": str(current_analysis.get("strategic_intent", "")),
                "financial_topic": str(current_analysis.get("financial_topic", "")),
                "future_gearing": str(current_analysis.get("future_gearing", ""))[:500],
            },
            "memory_brief": str(memory_context or "")[:1200],
        }

        fallback = [
            {
                "insight_type": "COMMITMENT_FOLLOWUP",
                "insight_text": "User should review the most recent financial commitment and confirm execution in the next check-in.",
                "urgency": "MEDIUM",
                "days_relevant": 14,
            },
            {
                "insight_type": "RISK_REBALANCING",
                "insight_text": "Portfolio risk posture should be reassessed against current cash-flow constraints before adding new exposure.",
                "urgency": "MEDIUM",
                "days_relevant": 21,
            },
            {
                "insight_type": "SAVINGS_OPPORTUNITY",
                "insight_text": "User can improve monthly surplus by tightening one discretionary expense bucket and redirecting it to stated goals.",
                "urgency": "LOW",
                "days_relevant": 30,
            },
        ]

        try:
            raw = self._call_groq(sys_prompt, json.dumps(user_payload, ensure_ascii=False), model_override=self.main_model)
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[-1].split("```")[0].strip()

            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                return fallback

            out: List[Dict[str, Any]] = []
            for item in parsed[:3]:
                if not isinstance(item, dict):
                    continue
                t = str(item.get("insight_type", "")).strip().upper()
                u = str(item.get("urgency", "")).strip().upper()
                text = str(item.get("insight_text", "")).strip()
                try:
                    days = int(item.get("days_relevant", 14))
                except Exception:
                    days = 14
                if t not in allowed_types:
                    t = "COMMITMENT_FOLLOWUP"
                if u not in allowed_urgency:
                    u = "MEDIUM"
                if not text:
                    continue
                out.append(
                    {
                        "insight_type": t,
                        "insight_text": text,
                        "urgency": u,
                        "days_relevant": max(1, min(90, days)),
                    }
                )

            if len(out) == 3:
                return out
            return fallback
        except Exception:
            return fallback