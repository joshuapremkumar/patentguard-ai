"""
modules/invention_improver.py
==============================
Suggest 5 actionable improvements to increase invention novelty
and avoid blocking prior art using a local Ollama LLM.

Improvement Categories
----------------------
  • Technical       – modify core mechanism, materials, or components
  • Claim Scope     – narrow / broaden claims strategically
  • Use Case        – target an unexplored application vertical
  • Combination     – add a non-obvious secondary feature
  • Jurisdictional  – file in geographies with weaker prior art

Usage
-----
    from modules.invention_improver import suggest_improvements
    from utils.models import Improvement

    suggestions: list[Improvement] = suggest_improvements(
        invention_text="A wearable NIR glucose sensor…",
        novelty_score=42.0,
        risk_level="HIGH",
        top_matches=report.top_matches[:3],
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from utils.llm_parsing import strip_json_fences
from utils.models import Improvement
from utils.ollama_client import generate

if TYPE_CHECKING:
    from utils.models import ScoredRecord

logger = logging.getLogger(__name__)

OLLAMA_MODEL: str = "llama3"

_SYSTEM_PROMPT = (
    "You are a patent strategy consultant who specialises in increasing "
    "invention novelty and designing around blocking prior art."
)

_IMPROVE_PROMPT = """\
Analyse the invention and blocking patents below. Generate exactly 5 specific,
actionable improvement suggestions that will:
  1. Differentiate the invention from each blocker
  2. Raise the novelty score above 70/100
  3. Strengthen the patent application

Invention:
{invention}

Current Novelty Score: {score}/100  |  Risk: {risk}

Blocking Patents:
{blockers}

Return ONLY a valid JSON array of exactly 5 objects — no markdown, no extra text:
[
  {{
    "category": "<Technical | Claim Scope | Use Case | Combination | Jurisdictional>",
    "suggestion": "<specific improvement, 1-2 sentences>",
    "rationale": "<why this increases novelty vs the listed art, 1 sentence>",
    "priority": <integer 1-5, 1=highest impact>
  }}
]
"""

# ── Fallback suggestions ──────────────────────────────────────────────────

_FALLBACK: list[dict] = [
    {
        "category": "Technical",
        "suggestion": "Introduce a novel material or sub-component not present in any existing claims.",
        "rationale": "Material differentiation creates a new technical feature that prior art cannot anticipate.",
        "priority": 1,
    },
    {
        "category": "Use Case",
        "suggestion": "Target a specific patient population (e.g. paediatric or geriatric) not covered by existing claims.",
        "rationale": "Application-specific claims reduce direct conflict with broad existing patents.",
        "priority": 2,
    },
    {
        "category": "Claim Scope",
        "suggestion": "Narrow the independent claim to focus exclusively on the most unique technical feature of the invention.",
        "rationale": "Narrower independent claims are more defensible and less likely to overlap with prior art.",
        "priority": 2,
    },
    {
        "category": "Combination",
        "suggestion": "Combine the core invention with a secondary non-obvious feature to create a novel combination claim.",
        "rationale": "Combination novelty is recognised when neither element individually predicts the other.",
        "priority": 3,
    },
    {
        "category": "Jurisdictional",
        "suggestion": "File first in jurisdictions where no equivalent prior art has been registered (e.g. emerging markets).",
        "rationale": "Jurisdictional gaps in prior art coverage can provide stronger IP protection in key markets.",
        "priority": 4,
    },
]


# ── JSON helpers ──────────────────────────────────────────────────────────

def _parse_improvements(raw: str) -> list[Improvement] | None:
    """Attempt to parse LLM output as a JSON array of improvements."""
    try:
        items: list[dict] = json.loads(strip_json_fences(raw, array=True))
        return [
            Improvement(
                category=item.get("category", "General"),
                suggestion=item.get("suggestion", "").strip(),
                rationale=item.get("rationale", "").strip(),
                priority=int(item.get("priority", 3)),
            )
            for item in items[:5]
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_plain_text(text: str) -> list[Improvement]:
    """
    Extract numbered suggestions from unstructured plain-text LLM output.
    Used as a secondary fallback when JSON parsing fails.
    """
    improvements: list[Improvement] = []
    # Match numbered items like "1. " or "1) "
    chunks = re.split(r"\n\s*\d+[\.\)]\s+", text)
    categories = ["Technical", "Use Case", "Claim Scope", "Combination", "Jurisdictional"]
    for i, chunk in enumerate(chunks[1:6]):  # skip preamble
        lines = [l.strip() for l in chunk.strip().splitlines() if l.strip()]
        suggestion = lines[0] if lines else "See full output."
        rationale = lines[1] if len(lines) > 1 else "Refer to analysis above."
        improvements.append(
            Improvement(
                category=categories[i % len(categories)],
                suggestion=suggestion[:300],
                rationale=rationale[:200],
                priority=i + 1,
            )
        )
    return improvements or _fallback_improvements()


def _fallback_improvements() -> list[Improvement]:
    return [Improvement(**item) for item in _FALLBACK]


# ── Public API ────────────────────────────────────────────────────────────

def suggest_improvements(
    invention_text: str,
    novelty_score: float,
    risk_level: str,
    top_matches: list["ScoredRecord"] | None = None,
    model: str = OLLAMA_MODEL,
) -> list[Improvement]:
    """
    Generate 5 actionable novelty-improvement suggestions via Ollama.

    Parameters
    ----------
    invention_text : str
        The inventor's description of their idea.
    novelty_score : float
        Current novelty score (0–100) from ``similarity_engine.analyze()``.
    risk_level : str
        Risk classification: ``LOW`` | ``MEDIUM`` | ``HIGH``.
    top_matches : list[ScoredRecord] | None
        Top-similar patent records driving the risk score.
    model : str
        Ollama model tag (default ``llama3``).

    Returns
    -------
    list[Improvement]
        5 improvement suggestions sorted by priority (1 = highest impact).
    """
    if not invention_text.strip():
        raise ValueError("invention_text must not be empty.")

    # Format blocking patents
    if top_matches:
        blocker_lines = [
            f"  [{sr.similarity * 100:.1f}%] {sr.record.title} "
            f"({sr.record.source}, {sr.record.filing_year or 'N/A'})\n"
            f"    {sr.record.abstract[:150]}"
            for sr in top_matches[:4]
        ]
        blockers = "\n\n".join(blocker_lines)
    else:
        blockers = "  No specific blocking patents identified."

    prompt = _IMPROVE_PROMPT.format(
        invention=invention_text[:1200],
        score=round(novelty_score, 1),
        risk=risk_level,
        blockers=blockers,
    )

    logger.info("invention_improver: calling Ollama model='%s'…", model)
    raw = generate(prompt, model=model, system=_SYSTEM_PROMPT)

    if raw.startswith("[Ollama error]"):
        logger.error("invention_improver: Ollama unavailable — using fallback.")
        return _fallback_improvements()

    # Try JSON → plain-text fallback
    improvements = _parse_improvements(raw)
    if not improvements:
        logger.warning("invention_improver: JSON parse failed — using text extraction.")
        improvements = _parse_plain_text(raw)

    # Sort by priority ascending (1 = highest impact first)
    improvements.sort(key=lambda i: i.priority)

    logger.info(
        "invention_improver: %d suggestions, risk=%s, score=%.1f",
        len(improvements), risk_level, novelty_score,
    )
    return improvements
