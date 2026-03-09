"""
modules/landscape_map.py
========================
Generate an interactive 2-D patent landscape visualisation using
PCA dimensionality reduction + K-Means clustering + Plotly.

The scatter plot shows:
  • All retrieved patents and prior art as coloured cluster dots
  • The inventor's idea as a distinct red star marker
  • Hover tooltips with title, source, year, and similarity %

Usage
-----
    from modules.landscape_map import build_landscape
    import plotly.graph_objects as go

    fig: go.Figure = build_landscape(
        corpus=extracted_patent_records,
        report=novelty_report,
        invention_text="My wearable glucose sensor",
    )
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.models import NoveltyReport, PatentRecord

logger = logging.getLogger(__name__)

CLUSTER_PALETTE: list[str] = [
    "#4E79A7", "#F28E2B", "#E15759",
    "#76B7B2", "#59A14F", "#EDC948",
]
INVENTION_COLOR: str = "#FF2B2B"
MIN_POINTS_FOR_PCA: int = 3
DEFAULT_N_CLUSTERS: int = 5


def build_landscape(
    corpus: list[PatentRecord],
    report: NoveltyReport,
    invention_text: str = "Your Invention",
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> go.Figure:
    """
    Build and return an interactive Plotly scatter chart of the patent landscape.

    Parameters
    ----------
    corpus : list[PatentRecord]
        Patent and prior-art records (must align with ``report.corpus_embeddings``).
    report : NoveltyReport
        Output from ``similarity_engine.analyze()`` containing pre-computed
        embeddings and similarity scores.
    invention_text : str
        Label for the inventor's idea point (default ``"Your Invention"``).
    n_clusters : int
        Number of K-Means clusters for colour grouping (default 5).

    Returns
    -------
    go.Figure
        Interactive Plotly scatter figure ready for ``st.plotly_chart()``.
    """
    # ── Guard: need at least MIN_POINTS ──────────────────────────────────
    if (
        not corpus
        or not report.corpus_embeddings
        or len(corpus) < MIN_POINTS_FOR_PCA
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data points to build the landscape map.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="Patent Landscape Map")
        return fig

    # ── 1. Build embedding matrix ─────────────────────────────────────────
    corpus_matrix = np.array(report.corpus_embeddings, dtype=np.float32)
    inv_vec = np.array(report.invention_embedding, dtype=np.float32)

    has_invention = inv_vec.size > 0 and inv_vec.shape[0] == corpus_matrix.shape[1]

    if has_invention:
        full_matrix = np.vstack([inv_vec[np.newaxis, :], corpus_matrix])
        offset = 1
    else:
        full_matrix = corpus_matrix
        offset = 0

    # ── 2. Standardise ────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaled = scaler.fit_transform(full_matrix)

    # ── 3. PCA → 2D ───────────────────────────────────────────────────────
    n_components = min(2, scaled.shape[0] - 1, scaled.shape[1])
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(scaled)

    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    explained = pca.explained_variance_ratio_
    corpus_coords = coords[offset:]

    # ── 4. K-Means clustering ─────────────────────────────────────────────
    k = min(n_clusters, len(corpus))
    if k >= 2:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels: list[int] = kmeans.fit_predict(corpus_coords).tolist()
    else:
        labels = [0] * len(corpus)

    # ── 5. Extract similarity scores ──────────────────────────────────────
    sim_map: dict[str, float] = {
        sr.record.url: sr.similarity for sr in report.top_matches
    }

    # ── 6. Build Plotly traces per cluster ────────────────────────────────
    fig = go.Figure()

    cluster_groups: dict[int, list[tuple]] = {}
    for i, (rec, label) in enumerate(zip(corpus, labels)):
        cluster_groups.setdefault(label, []).append(
            (rec, corpus_coords[i, 0], corpus_coords[i, 1], sim_map.get(rec.url, 0.0))
        )

    for cluster_id, items in cluster_groups.items():
        color = CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=[item[1] for item in items],
                y=[item[2] for item in items],
                mode="markers",
                marker=dict(size=9, color=color, opacity=0.8, line=dict(width=1, color="white")),
                name=f"Cluster {cluster_id + 1}",
                text=[
                    (
                        f"<b>{item[0].title[:60]}</b><br>"
                        f"Source: {item[0].source}<br>"
                        f"Year: {item[0].filing_year or '—'}<br>"
                        f"Similarity: {item[3] * 100:.1f}%"
                    )
                    for item in items
                ],
                hoverinfo="text",
                customdata=[item[0].url for item in items],
            )
        )

    # ── 7. Invention star marker ──────────────────────────────────────────
    if has_invention:
        fig.add_trace(
            go.Scatter(
                x=[coords[0, 0]],
                y=[coords[0, 1]],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    size=18,
                    color=INVENTION_COLOR,
                    line=dict(width=1.5, color="white"),
                ),
                text=["★ Your Invention"],
                textposition="top center",
                textfont=dict(size=11, color=INVENTION_COLOR),
                name="Your Invention",
                hovertext=(
                    f"<b>Your Invention</b><br>"
                    f"Novelty Score: {report.novelty_score:.1f}/100<br>"
                    f"Risk: {report.risk_level}"
                ),
                hoverinfo="text",
            )
        )

    # ── 8. Layout ─────────────────────────────────────────────────────────
    ev1 = f"{explained[0] * 100:.1f}%" if len(explained) > 0 else "PC1"
    ev2 = f"{explained[1] * 100:.1f}%" if len(explained) > 1 else "PC2"

    fig.update_layout(
        title=dict(
            text="📍 Patent Landscape Map",
            font=dict(size=18),
        ),
        xaxis_title=f"Principal Component 1 ({ev1} variance)",
        yaxis_title=f"Principal Component 2 ({ev2} variance)",
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
        hovermode="closest",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="#2D2D2D", zerolinecolor="#444"),
        yaxis=dict(gridcolor="#2D2D2D", zerolinecolor="#444"),
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    logger.info(
        "build_landscape: %d points, %d clusters, PCA variance %.1f%% + %.1f%%",
        len(corpus) + (1 if has_invention else 0),
        k,
        explained[0] * 100 if len(explained) > 0 else 0,
        explained[1] * 100 if len(explained) > 1 else 0,
    )
    return fig
