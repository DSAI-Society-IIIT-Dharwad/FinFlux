from __future__ import annotations
import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    """Client for Groq LLM API."""
    def __init__(self, model: str = "llama3-8b-8192", api_key: str | None = None):
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json"
        }
        
    def generate(self, prompt: str, system: str = "") -> str:
        """Call Groq chat completions endpoint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return ""
