"""
utils/ollama_client.py
======================
Minimal HTTP client for the local Ollama inference server.

Ollama must be running at http://localhost:11434 with the desired model
already pulled:
    ollama pull llama3

Supported endpoints
-------------------
  POST /api/generate  – single-turn text generation (used here)
  GET  /api/tags      – list available models (used for health-check)

Usage
-----
    from utils.ollama_client import OllamaClient, generate

    # Module-level convenience function (uses default model)
    reply = generate("Write a patent claim for a wearable glucose sensor.")

    # Or use the class directly
    client = OllamaClient(model="llama3")
    reply = client.complete("Summarise this patent abstract…")
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_MODEL: str = "llama3"
GENERATE_TIMEOUT: int = 180     # seconds – LLM generation can be slow locally
HEALTH_TIMEOUT: int = 5


class OllamaClient:
    """
    Lightweight wrapper around the Ollama REST API.

    Parameters
    ----------
    model : str
        Ollama model tag to use (default ``llama3``).
    base_url : str
        Ollama server base URL (default ``http://localhost:11434``).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    # ── Health check ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the Ollama server responds to a health ping."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=HEALTH_TIMEOUT)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return a list of locally available model tags."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=HEALTH_TIMEOUT)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning("OllamaClient.list_models failed: %s", exc)
            return []

    # ── Text generation ───────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a prompt to Ollama and return the generated text.

        Parameters
        ----------
        prompt : str
            User prompt / instruction.
        system : str | None
            Optional system message prepended to the conversation.
        temperature : float
            Sampling temperature (lower = more deterministic). Default 0.3.
        max_tokens : int
            Maximum tokens to generate. Default 2048.

        Returns
        -------
        str
            The model's response text, or an error message prefixed with
            ``[Ollama error]`` if the request fails.
        """
        if not prompt.strip():
            return ""

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=GENERATE_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            msg = "Ollama is not running. Start it with: ollama serve"
            logger.error(msg)
            return f"[Ollama error] {msg}"
        except requests.exceptions.Timeout:
            msg = f"Ollama request timed out after {GENERATE_TIMEOUT}s."
            logger.error(msg)
            return f"[Ollama error] {msg}"
        except Exception as exc:
            logger.error("OllamaClient.complete unexpected error: %s", exc)
            return f"[Ollama error] {exc}"


# ── Module-level singleton + convenience function ─────────────────────────

_client: Optional[OllamaClient] = None


def get_client(model: str = DEFAULT_MODEL) -> OllamaClient:
    """Return a cached OllamaClient instance (re-created on model change)."""
    global _client
    if _client is None or _client.model != model:
        _client = OllamaClient(model=model)
    return _client


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """
    Module-level convenience wrapper for single-turn text generation.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str
        Ollama model tag (default ``llama3``).
    system : str | None
        Optional system message.
    temperature : float
        Sampling temperature (default 0.3).

    Returns
    -------
    str
        Generated response text.
    """
    return get_client(model).complete(prompt, system=system, temperature=temperature)
