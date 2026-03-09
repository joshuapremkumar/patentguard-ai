"""
modules/patent_extract.py
==========================
Clean and normalise raw search records into validated PatentRecord objects.

This module applies:
  • Title trimming and deduplication
  • Year extraction from abstract / title text
  • Source classification from URL patterns
  • Abstract truncation to 1 000 chars for downstream embedding efficiency
  • Deduplication by URL

Usage
-----
    from modules.patent_extract import extract_records
    from utils.models import PatentRecord

    clean: list[PatentRecord] = extract_records(raw_patent_records)
"""

from __future__ import annotations

import logging
import re

from utils.models import PatentRecord
from utils.source_mapper import infer_source

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19[89]\d|20[012]\d)\b")
_MAX_ABSTRACT_CHARS: int = 1_000
_GENERIC_SOURCES = ("", "Web", "Patent DB", "Unknown")


def _resolve_source(url: str, current_source: str) -> str:
    """Re-infer source from URL only when the current value is generic."""
    if current_source not in _GENERIC_SOURCES:
        return current_source
    return infer_source(url, fallback=current_source or "Unknown")


def _extract_year(text: str) -> int | None:
    """Return the first plausible 4-digit year found in text."""
    m = _YEAR_RE.search(text)
    return int(m.group()) if m else None


def _clean_title(title: str) -> str:
    """Strip excess whitespace and truncate to 200 chars."""
    return " ".join(title.split())[:200] or "Untitled"


def _clean_abstract(abstract: str) -> str:
    """Normalise whitespace and truncate to MAX_ABSTRACT_CHARS."""
    cleaned = " ".join(abstract.split())
    return cleaned[:_MAX_ABSTRACT_CHARS]


# ── Public API ────────────────────────────────────────────────────────────

def extract_records(records: list[PatentRecord]) -> list[PatentRecord]:
    """
    Clean and deduplicate a list of PatentRecord objects.

    Processing steps applied to every record:
    1. Clean title (whitespace + truncate)
    2. Resolve source from URL when it's generic
    3. Attempt to extract a filing / publication year from the abstract
    4. Clean and truncate the abstract
    5. Deduplicate the list by URL (first occurrence wins)

    Parameters
    ----------
    records : list[PatentRecord]
        Raw records from ``patent_search`` or ``prior_art_search``.

    Returns
    -------
    list[PatentRecord]
        Normalised, deduplicated records ready for embedding.
    """
    if not records:
        return []

    seen_urls: set[str] = set()
    cleaned: list[PatentRecord] = []

    for rec in records:
        # Deduplicate
        url = rec.url.strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        # Skip records with no usable text
        if not rec.title.strip() and not rec.abstract.strip():
            continue

        year = rec.filing_year or _extract_year(rec.abstract) or _extract_year(rec.title)

        cleaned.append(
            PatentRecord(
                title=_clean_title(rec.title),
                abstract=_clean_abstract(rec.abstract),
                url=url,
                source=_resolve_source(url, rec.source),
                filing_year=year,
                authors=rec.authors,
                record_type=rec.record_type,
            )
        )

    logger.info(
        "extract_records: %d → %d records after cleaning.",
        len(records), len(cleaned),
    )
    return cleaned
