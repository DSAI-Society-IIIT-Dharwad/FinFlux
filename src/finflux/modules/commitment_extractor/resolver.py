"""Resolver utilities for commitment actor and due-date extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from finflux.contracts.events import FinancialEntity


@dataclass(frozen=True)
class ActorResolution:
    actor: str
    unresolved_reason: str | None


class ActorResolver:
    """Resolves actor with precision-first heuristics."""

    def resolve(self, text: str, entities: Sequence[FinancialEntity]) -> ActorResolution:
        lowered = text.lower()

        if re.search(r"\b(i|we|main|mai|hum)\b", lowered):
            return ActorResolution(actor="speaker", unresolved_reason=None)

        party_entities = [entity for entity in entities if entity.entity_type == "party"]
        if len(party_entities) == 1:
            return ActorResolution(actor=party_entities[0].normalized_value, unresolved_reason=None)
        if len(party_entities) > 1:
            return ActorResolution(actor="unknown", unresolved_reason="multiple_actor_candidates")

        return ActorResolution(actor="unknown", unresolved_reason="actor_unclear")


class TimeResolver:
    """Resolves due date from date/tenure entities when available."""

    _DATE_PATTERN = re.compile(r"^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$")

    def resolve_due_date_iso(self, entities: Sequence[FinancialEntity]) -> str | None:
        date_entities = [entity for entity in entities if entity.entity_type == "date"]
        if not date_entities:
            return None

        for entity in date_entities:
            parsed = self._parse_date(entity.normalized_value)
            if parsed is not None:
                return parsed

        return None

    def _parse_date(self, value: str) -> str | None:
        candidate = value.strip().lower()
        match = self._DATE_PATTERN.match(candidate)
        if not match:
            return None

        day = int(match.group(1))
        month = int(match.group(2))
        year_raw = match.group(3)

        if year_raw is None:
            year = datetime.utcnow().year
        else:
            year = int(year_raw)
            if year < 100:
                year = 2000 + year

        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return None


__all__ = ["ActorResolution", "ActorResolver", "TimeResolver"]
