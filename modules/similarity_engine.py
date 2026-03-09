"""
modules/similarity_engine.py
=============================
Compute cosine similarity between an invention and a patent corpus,
then derive a novelty score and risk classification.

Scoring Rules
-------------
  Novelty Score  =  100 − (max_cosine_similarity × 100)

Risk Classification
-------------------
  similarity > 0.80  → HIGH    risk  (potential blocking patent)
  0.60 ≤ sim ≤ 0.80  → MEDIUM  risk  (related art detected)
  sim < 0.60         → LOW     risk  (likely novel)

Usage
-----
    from modules.similarity_engine import analyze
    from utils.models import NoveltyReport

    report: NoveltyReport = analyze(
        invention_text="A wearable NIR glucose sensor…",
        corpus=extracted_patent_records,
    )
    print(report.novelty_score, report.risk_level)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from modules.embedding_engine import embed_batch, embed_text
from utils.models import NoveltyReport, PatentRecord, ScoredRecord

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD: float   = 0.80
MEDIUM_RISK_THRESHOLD: float = 0.60
TOP_K: int = 10


# ── Internal maths ────────────────────────────────────────────────────────

def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and a row matrix.

    Parameters
    ----------
    query_vec : (D,) float32
    matrix    : (N, D) float32

    Returns
    -------
    (N,) float32  – cosine similarities clipped to [0, 1]
    """
    norm_q = np.linalg.norm(query_vec)
    if norm_q == 0:
        return np.zeros(len(matrix), dtype=np.float32)

    norms_m = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(norms_m == 0, 1e-9, norms_m)
    scores = (matrix @ query_vec) / (safe_norms * norm_q)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def _classify_risk(max_sim: float) -> str:
    if max_sim > HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif max_sim >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ── Public API ────────────────────────────────────────────────────────────

def analyze(
    invention_text: str,
    corpus: list[PatentRecord],
    top_k: int = TOP_K,
    invention_embedding: Optional[list[float]] = None,
) -> NoveltyReport:
    """
    Embed the invention and all corpus records, compute cosine similarity,
    calculate the novelty score, and classify patent risk.

    Parameters
    ----------
    invention_text : str
        The inventor's description of their idea.
    corpus : list[PatentRecord]
        Patent and prior-art records to compare against.
    top_k : int
        Number of top-similar records to include in the report (default 10).
    invention_embedding : list[float] | None
        Pre-computed invention embedding (avoids re-encoding if cached).

    Returns
    -------
    NoveltyReport
        Full analysis with novelty score, risk level, and ranked matches.

    Raises
    ------
    ValueError
        If ``invention_text`` is empty.
    """
    if not invention_text.strip():
        raise ValueError("invention_text must not be empty.")

    # ── 1. Handle empty corpus ────────────────────────────────────────────
    if not corpus:
        logger.warning("analyze: empty corpus — returning 100%% novelty.")
        inv_emb = invention_embedding or embed_text(invention_text)
        return NoveltyReport(
            novelty_score=100.0,
            risk_level="LOW",
            max_similarity=0.0,
            invention_embedding=inv_emb,
        )

    # ── 2. Embed invention ────────────────────────────────────────────────
    inv_vec = np.array(
        invention_embedding if invention_embedding else embed_text(invention_text),
        dtype=np.float32,
    )

    # ── 3. Embed corpus ───────────────────────────────────────────────────
    texts = [r.text_for_embedding for r in corpus]
    corpus_matrix: np.ndarray = embed_batch(texts)        # (N, D)

    # ── 4. Cosine similarity ──────────────────────────────────────────────
    scores: np.ndarray = _cosine_scores(inv_vec, corpus_matrix)

    # ── 5. Build ranked ScoredRecord list ─────────────────────────────────
    scored = [
        ScoredRecord(record=rec, similarity=float(sim))
        for rec, sim in zip(corpus, scores)
    ]
    scored.sort(key=lambda s: s.similarity, reverse=True)
    for rank, sr in enumerate(scored, start=1):
        sr.rank = rank

    top_matches = scored[:top_k]
    max_similarity = scored[0].similarity if scored else 0.0

    # ── 6. Novelty score & risk ───────────────────────────────────────────
    novelty_score = round(max(0.0, min(100.0, 100.0 - max_similarity * 100.0)), 2)
    risk_level = _classify_risk(max_similarity)

    logger.info(
        "analyze: novelty=%.1f, risk=%s, max_sim=%.4f, corpus=%d",
        novelty_score, risk_level, max_similarity, len(corpus),
    )

    return NoveltyReport(
        novelty_score=novelty_score,
        risk_level=risk_level,
        max_similarity=max_similarity,
        top_matches=top_matches,
        invention_embedding=inv_vec.tolist(),
        corpus_embeddings=corpus_matrix.tolist(),
    )
