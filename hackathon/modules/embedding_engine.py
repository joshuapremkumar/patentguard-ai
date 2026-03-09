"""
modules/embedding_engine.py
============================
Generate dense vector embeddings using SentenceTransformers.

Model: all-MiniLM-L6-v2  (384-dim, fast, accurate for semantic similarity)

The module exposes a lazily-loaded singleton so the model is downloaded
and loaded only once per Python process — critical for Streamlit which
re-runs the script on every interaction.

Usage
-----
    from modules.embedding_engine import embed_text, embed_batch, get_engine

    # Single text
    vector: list[float] = embed_text("wearable glucose sensor NIR")

    # Batch (more efficient for many records)
    matrix: np.ndarray = embed_batch(["text 1", "text 2", ...])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME: str = "all-MiniLM-L6-v2"
BATCH_SIZE: int = 32


# ── Singleton model (Streamlit resource cache) ────────────────────────────

@st.cache_resource
def get_engine() -> SentenceTransformer:
    """
    Return the cached SentenceTransformer model.
    Decorated with @st.cache_resource so the model is loaded once per
    app process and persists across Streamlit script reruns.
    """
    logger.info("Loading SentenceTransformer model '%s'…", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded (%d dims).", model.get_sentence_embedding_dimension())
    return model


# ── Public helpers ────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Encode a single text string and return its embedding as a Python list.

    Parameters
    ----------
    text : str
        Input text (patent abstract, invention description, etc.).

    Returns
    -------
    list[float]
        384-dimensional embedding vector.

    Raises
    ------
    ValueError
        If ``text`` is empty or whitespace-only.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed an empty string.")
    model = get_engine()
    vector: np.ndarray = model.encode(text, show_progress_bar=False)
    return vector.tolist()


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts in one batched forward pass.

    Parameters
    ----------
    texts : list[str]
        Input texts. Empty / whitespace-only strings are replaced with a
        placeholder to avoid errors.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 384)`` float32 matrix.

    Raises
    ------
    ValueError
        If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts list must not be empty.")

    # Replace empty strings with a neutral placeholder
    cleaned = [t.strip() if t and t.strip() else "[empty]" for t in texts]

    model = get_engine()
    matrix: np.ndarray = model.encode(
        cleaned,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return matrix.astype(np.float32)


def embed_texts_as_list(texts: list[str]) -> list[list[float]]:
    """
    Convenience wrapper: encode a batch and return as a nested Python list.
    Suitable for JSON serialisation or Pydantic model storage.
    """
    return embed_batch(texts).tolist()
