"""FinFlux Security Module: AES Encryption, PII Masking, and Injection Detection."""

import re
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class FinFluxSecurity:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.environ.get("FINFLUX_SECRET", "dev-secret-key-12345")
        # Initialize AES Fernet
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'finflux-salt-v1',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
        self.fernet = Fernet(key)

        # PII Regex Patterns (Indian Context)
        self.pii_patterns = {
            "PAN": r"[A-Z]{5}[0-9]{4}[A-Z]{1}",
            "AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b",
            "PHONE": r"\b(?:\+91|0)?[6-9]\d{9}\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ACCOUNT": r"\b\d{9,18}\b" # Generic bank account
        }

        # Injection Patterns
        self.injection_keywords = [
            "ignore previous", "ignore all instructions", "system override",
            "act as an advisor", "bypass rules", "you are now a financial advisor",
            "recommend stocks", "predict the market", "override", "system prompt",
            "forget your instructions", "new persona", "act as", "pretend you are",
            "guarantee returns", "100% returns", "risk free profit",
            "invest in my scheme", "send money to", "wire transfer to"
        ]

    def encrypt_audio(self, data: bytes) -> bytes:
        """Encrypt raw audio data using AES-256."""
        return self.fernet.encrypt(data)

    def decrypt_audio(self, encrypted_data: bytes) -> bytes:
        """Decrypt audio data."""
        return self.fernet.decrypt(encrypted_data)

    def mask_pii(self, text: str) -> str:
        """Mask PII (PAN, Aadhaar, Phone, Email) with labels."""
        masked_text = text
        for label, pattern in self.pii_patterns.items():
            masked_text = re.sub(pattern, f"[MASKED_{label}]", masked_text)
        return masked_text

    def detect_injection(self, text: str) -> bool:
        """Check for prompt injection attempts in the transcript."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.injection_keywords)

    def is_asking_for_advice(self, text: str) -> bool:
        """Heuristic check for advice requests (Stage 1)."""
        advice_triggers = ["should i", "recommend", "best path", "what to buy", "invest in", "is it good", "buy or sell"]
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in advice_triggers)
