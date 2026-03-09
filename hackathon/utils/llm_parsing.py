"""
utils/llm_parsing.py
====================
Shared helpers for parsing JSON out of Ollama LLM responses.
Used by patent_drafter and invention_improver.
"""

from __future__ import annotations


def strip_json_fences(text: str, array: bool = False) -> str:
    """
    Remove markdown code fences from an LLM response.

    Parameters
    ----------
    text : str
        Raw LLM output, possibly wrapped in ```json … ``` fences.
    array : bool
        If True, look for a JSON array opening ``[``.
        If False (default), look for a JSON object opening ``{``.

    Returns
    -------
    str
        Text with fences stripped, or original text if no fences found.
    """
    text = text.strip()
    marker = "[" if array else "{"

    if text.startswith("```"):
        parts = text.split("```")
        for part in parts[1:]:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith(marker):
                return stripped

    return text
