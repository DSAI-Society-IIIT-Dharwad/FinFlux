"""Replaceable model loading contracts for ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ASRModelHandle:
    """Opaque model handle wrapper used by inference engines."""

    model_id: str
    model: Any


class ModelLoader(Protocol):
    """Loads and returns an ASR model handle."""

    def load(self) -> ASRModelHandle:
        ...


@dataclass
class StaticModelLoader:
    """Deterministic loader that returns a static in-memory model handle."""

    model_id: str = "mock-asr"
    model_object: Any = None

    def load(self) -> ASRModelHandle:
        return ASRModelHandle(model_id=self.model_id, model=self.model_object)


__all__ = ["ASRModelHandle", "ModelLoader", "StaticModelLoader"]
