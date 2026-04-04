"""Unified manifest schema and multilingual coverage validation helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

UNIFIED_MANIFEST_FIELDS = (
    "audio_path",
    "text",
    "language",
    "source",
    "duration_seconds",
)

LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "mixed": "hinglish",
    "code-mixed": "hinglish",
    "code_mixed": "hinglish",
    "hinglish": "hinglish",
}


@dataclass(frozen=True)
class ManifestCoverage:
    total_rows: int
    by_language: dict[str, int]
    by_source: dict[str, int]


def normalize_language_label(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return "en"
    return LANGUAGE_ALIASES.get(lowered, lowered)


def build_unified_row(
    *,
    audio_path: str,
    text: str,
    language: str,
    source: str,
    duration_seconds: float,
) -> dict[str, str]:
    return {
        "audio_path": audio_path,
        "text": text,
        "language": normalize_language_label(language),
        "source": source,
        "duration_seconds": f"{float(duration_seconds):.3f}",
    }


def write_unified_manifest(rows: Iterable[dict[str, str]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(UNIFIED_MANIFEST_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return path


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_manifest_schema(path: str | Path) -> list[dict[str, str]]:
    rows = load_manifest_rows(path)
    if not rows:
        raise ValueError("Manifest is empty or missing")

    first = rows[0]
    missing = [field for field in UNIFIED_MANIFEST_FIELDS if field not in first]
    if missing:
        raise ValueError(f"Manifest schema mismatch. Missing columns: {', '.join(missing)}")

    bad_rows = [
        index
        for index, row in enumerate(rows, start=2)
        if not row.get("audio_path") or not row.get("text") or not row.get("language")
    ]
    if bad_rows:
        preview = ", ".join(str(index) for index in bad_rows[:10])
        raise ValueError(f"Manifest has incomplete rows at lines: {preview}")

    return rows


def compute_manifest_coverage(rows: list[dict[str, str]]) -> ManifestCoverage:
    by_language: dict[str, int] = {}
    by_source: dict[str, int] = {}

    for row in rows:
        language = normalize_language_label(row.get("language", ""))
        source = row.get("source", "unknown") or "unknown"

        by_language[language] = by_language.get(language, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1

    return ManifestCoverage(
        total_rows=len(rows),
        by_language=by_language,
        by_source=by_source,
    )


def assert_multilingual_coverage(
    coverage: ManifestCoverage,
    *,
    min_total_rows: int,
    min_rows_by_language: dict[str, int],
) -> None:
    errors: list[str] = []
    if coverage.total_rows < min_total_rows:
        errors.append(f"total_rows={coverage.total_rows} is below required {min_total_rows}")

    for language, minimum in min_rows_by_language.items():
        normalized = normalize_language_label(language)
        actual = coverage.by_language.get(normalized, 0)
        if actual < minimum:
            errors.append(f"language[{normalized}]={actual} is below required {minimum}")

    if errors:
        raise ValueError("Coverage check failed: " + "; ".join(errors))
