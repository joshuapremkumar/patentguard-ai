"""
modules/patent_drafter.py
==========================
Generate a structured patent application draft using a local Ollama LLM.

The LLM is prompted to produce a JSON-structured draft. If JSON parsing
fails, a plain-text fallback extracts sections using header markers.

Draft Sections
--------------
  1. Title of the Invention
  2. Background of the Invention
  3. Summary of the Invention
  4. Detailed Description of the Preferred Embodiment
  5. Claims (1 independent + 4 dependent)
  6. Abstract

Usage
-----
    from modules.patent_drafter import generate_draft
    from utils.models import PatentDraft, ScoredRecord

    draft: PatentDraft = generate_draft(
        invention_text="A wearable NIR glucose sensor…",
        top_matches=report.top_matches[:3],
    )
    print(draft.to_plain_text())
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from utils.llm_parsing import strip_json_fences
from utils.models import PatentDraft
from utils.ollama_client import generate

if TYPE_CHECKING:
    from utils.models import ScoredRecord

logger = logging.getLogger(__name__)

OLLAMA_MODEL: str = "llama3"

_SYSTEM_PROMPT = (
    "You are a senior USPTO patent attorney with 20 years of experience. "
    "You write precise, professional patent applications using standard legal language."
)

_DRAFT_PROMPT = """\
Generate a complete patent application draft for the invention described below.
Return ONLY a valid JSON object — no markdown, no explanation, no extra text.

Invention:
{invention}

Conflicting / blocking patents to distinguish from:
{blockers}

JSON schema (match exactly, all fields required):
{{
  "title": "<concise title, max 15 words>",
  "abstract": "<single paragraph, max 150 words>",
  "background": "<2 paragraphs: field of invention + problem in the art>",
  "summary": "<1-2 paragraphs: invention overview + primary advantages>",
  "description": "<3-4 paragraphs: detailed embodiment description with components>",
  "claims": [
    "<Claim 1 – independent: A [device/method] comprising …, wherein …>",
    "<Claim 2 – dependent: The [device/method] of claim 1, wherein …>",
    "<Claim 3 – dependent: The [device/method] of claim 1, further comprising …>",
    "<Claim 4 – dependent: The [device/method] of claim 2, wherein …>",
    "<Claim 5 – dependent: The [device/method] of claim 1, wherein …>"
  ]
}}
"""


# ── JSON helpers ──────────────────────────────────────────────────────────

def _parse_draft_json(raw: str) -> PatentDraft | None:
    """Attempt to parse the LLM response as JSON into a PatentDraft."""
    try:
        data = json.loads(strip_json_fences(raw))
        return PatentDraft(
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            background=data.get("background", ""),
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            claims=data.get("claims", []),
        )
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_section(text: str, header: str) -> str:
    """Regex-extract a section following a header marker from plain text."""
    pattern = rf"{re.escape(header)}[:\-\s]*(.*?)(?=\n[A-Z][A-Z\s]{{3,}}[:\-]|\Z)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _fallback_parse(text: str, invention: str) -> PatentDraft:
    """
    Best-effort section extraction from unstructured plain-text LLM output.
    Used when JSON parsing fails.
    """
    title_m = re.search(r"title[:\-\s]+(.+)", text, re.IGNORECASE)
    title = title_m.group(1).strip()[:120] if title_m else "Patent Application"

    claims_block = _extract_section(text, "CLAIMS")
    claims_raw = re.findall(r"\d+\.\s+.+", claims_block)
    claims = claims_raw[:5] if claims_raw else [f"1. A device comprising: {invention[:80]}"]

    return PatentDraft(
        title=title,
        abstract=_extract_section(text, "ABSTRACT") or invention[:200],
        background=_extract_section(text, "BACKGROUND") or f"This invention relates to: {invention[:200]}",
        summary=_extract_section(text, "SUMMARY") or "A novel invention as described herein.",
        description=_extract_section(text, "DESCRIPTION") or _extract_section(text, "DETAILED") or invention[:500],
        claims=claims,
    )


# ── Public API ────────────────────────────────────────────────────────────

def generate_draft(
    invention_text: str,
    top_matches: list["ScoredRecord"] | None = None,
    model: str = OLLAMA_MODEL,
) -> PatentDraft:
    """
    Generate a structured patent application draft via local Ollama LLM.

    Parameters
    ----------
    invention_text : str
        The inventor's full description of their idea.
    top_matches : list[ScoredRecord] | None
        Top-similar patents from the novelty report (used to shape claims).
    model : str
        Ollama model tag (default ``llama3``).

    Returns
    -------
    PatentDraft
        Structured patent draft with all sections populated.

    Raises
    ------
    ValueError
        If ``invention_text`` is empty.
    """
    if not invention_text.strip():
        raise ValueError("invention_text must not be empty.")

    # Format blocking patents for the prompt
    if top_matches:
        blocker_lines = [
            f"  - {sr.record.title} ({sr.record.source}, {sr.record.filing_year or 'N/A'}) "
            f"[{sr.similarity * 100:.1f}% similar]"
            for sr in top_matches[:3]
        ]
        blockers = "\n".join(blocker_lines)
    else:
        blockers = "  None identified."

    prompt = _DRAFT_PROMPT.format(
        invention=invention_text[:1500],
        blockers=blockers,
    )

    logger.info("patent_drafter: calling Ollama model='%s'…", model)
    raw_response = generate(prompt, model=model, system=_SYSTEM_PROMPT)

    if raw_response.startswith("[Ollama error]"):
        logger.error("patent_drafter: Ollama unavailable — using fallback draft.")
        return _make_simple_fallback(invention_text)

    # Try JSON first, then plain-text fallback
    draft = _parse_draft_json(raw_response)
    if draft is None:
        logger.warning("patent_drafter: JSON parse failed — using text extraction fallback.")
        draft = _fallback_parse(raw_response, invention_text)

    logger.info("patent_drafter: draft generated — '%s'", draft.title)
    return draft


def _make_simple_fallback(invention: str) -> PatentDraft:
    """Minimal placeholder draft when Ollama is unavailable."""
    return PatentDraft(
        title="Patent Application (Draft)",
        abstract=f"A novel invention: {invention[:150]}",
        background=(
            "This invention relates to the technical field described herein. "
            "There exists a need in the art for improved solutions to the identified problems."
        ),
        summary="The present invention provides a novel solution as described in detail below.",
        description=invention[:600],
        claims=[
            f"1. An invention comprising: {invention[:120]}",
            "2. The invention of claim 1, wherein the device is portable.",
            "3. The invention of claim 1, further comprising a wireless communication interface.",
            "4. The invention of claim 2, wherein the device is battery-powered.",
            "5. The invention of claim 1, wherein the device includes a user interface.",
        ],
    )
