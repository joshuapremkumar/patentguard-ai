#!/usr/bin/env bash
set -e
echo "🛡️ Setting up PatentGuard AI..."
mkdir -p patentguard-ai/modules patentguard-ai/utils

cat > patentguard-ai/.env.example << 'EOF'
# PatentGuard AI – Environment Variables
# Copy this file to .env and fill in your values:
#   cp .env.example .env

# ── Required ───────────────────────────────────────────────────────────────
# Get your free API key at https://tavily.com
TAVILY_API_KEY=your_tavily_api_key_here

# ── Optional overrides ─────────────────────────────────────────────────────
# Ollama server URL (default: http://localhost:11434)
# OLLAMA_BASE_URL=http://localhost:11434

# Default Ollama model (default: llama3)
# OLLAMA_MODEL=llama3
EOF

cat > patentguard-ai/requirements.txt << 'EOF'
# PatentGuard AI – Hackathon Prototype
# Install: pip install -r requirements.txt

# ── Streamlit UI ───────────────────────────────────────────────────────────
streamlit>=1.35.0

# ── Search ─────────────────────────────────────────────────────────────────
tavily-python>=0.3.0
requests>=2.32.0

# ── Embeddings ─────────────────────────────────────────────────────────────
sentence-transformers>=3.0.0

# ── ML / Dimensionality Reduction ─────────────────────────────────────────
scikit-learn>=1.5.0
numpy>=1.26.0

# ── Visualisation ──────────────────────────────────────────────────────────
plotly>=5.22.0

# ── Data models ────────────────────────────────────────────────────────────
pydantic>=2.7.0

# ── Config ─────────────────────────────────────────────────────────────────
python-dotenv>=1.0.0
EOF

cat > patentguard-ai/app.py << 'EOF'
"""
app.py – PatentGuard AI Hackathon Prototype
============================================
Streamlit single-page application.

Run:
    streamlit run app.py

Requirements:
    • TAVILY_API_KEY set in .env
    • Ollama running locally:  ollama serve && ollama pull llama3
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv

# ── Internal modules ──────────────────────────────────────────────────────
from modules.patent_search import search_patents
from modules.prior_art_search import search_prior_art
from modules.patent_extract import extract_records
from modules.similarity_engine import analyze
from modules.landscape_map import build_landscape
from modules.patent_drafter import generate_draft
from modules.invention_improver import suggest_improvements
from utils.models import PatentDraft, NoveltyReport
from utils.ollama_client import get_client as get_ollama

# ── Module-level constants ────────────────────────────────────────────────
_CATEGORY_ICONS: dict[str, str] = {
    "Technical":      "⚙️",
    "Claim Scope":    "📐",
    "Use Case":       "🎯",
    "Combination":    "🔗",
    "Jurisdictional": "🌍",
}

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PatentGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-card {
            background: #1E1E2E;
            border-radius: 12px;
            padding: 1.2rem 1.6rem;
            border: 1px solid #2D2D3F;
        }
        .risk-high   { color: #FF4B4B; font-weight: 700; font-size: 1.1rem; }
        .risk-medium { color: #FFD700; font-weight: 700; font-size: 1.1rem; }
        .risk-low    { color: #00C851; font-weight: 700; font-size: 1.1rem; }
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            margin: 1.5rem 0 0.5rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #667eea;
        }
        .improvement-card {
            background: #1A1A2E;
            border-left: 4px solid #667eea;
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 2rem;
            font-size: 1rem;
        }
        .stButton > button:hover { opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return configuration dict."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/64/shield.png", width=56)
        st.markdown("### ⚙️ Settings")

        max_patents = st.slider(
            "Max patent results", min_value=3, max_value=20, value=8, step=1,
            help="Number of patents fetched per Tavily query",
        )
        max_prior_art = st.slider(
            "Max prior art results", min_value=3, max_value=15, value=6, step=1,
            help="Number of prior-art records fetched",
        )
        ollama_model = st.selectbox(
            "Ollama model",
            options=["llama3", "llama3.1", "mistral", "gemma2", "phi3"],
            index=0,
            help="Local Ollama model for draft & improvements",
        )

        st.divider()

        # Connectivity status
        st.markdown("### 🔌 Service Status")
        tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
        ollama_ok = get_ollama(ollama_model).is_available()
        st.write(
            f"{'✅' if tavily_ok else '❌'} Tavily API key "
            f"{'configured' if tavily_ok else 'NOT configured'}"
        )
        st.write(
            f"{'✅' if ollama_ok else '❌'} Ollama "
            f"{'online' if ollama_ok else 'offline (run: ollama serve)'}"
        )

        st.divider()
        st.caption("PatentGuard AI — Hackathon Prototype\nBuilt with Streamlit + Ollama + Tavily")

    return {
        "max_patents": max_patents,
        "max_prior_art": max_prior_art,
        "ollama_model": ollama_model,
    }


# ── Pipeline execution ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(
    invention: str,
    max_patents: int,
    max_prior_art: int,
    ollama_model: str,
) -> dict:
    """
    Execute the full PatentGuard AI analysis pipeline.
    Results are cached by Streamlit so re-runs don't re-fetch.
    """
    results: dict = {}

    progress = st.progress(0, text="🔍 Searching patents & prior art in parallel…")

    # 1 & 2. Patent + prior-art searches run concurrently (independent I/O)
    with ThreadPoolExecutor(max_workers=2) as pool:
        patent_future   = pool.submit(search_patents,   invention, max_patents)
        prior_art_future = pool.submit(search_prior_art, invention, max_prior_art)
        raw_patents  = patent_future.result()
        raw_prior_art = prior_art_future.result()

    progress.progress(30, text="🧹 Extracting & cleaning records…")

    # 3. Extract and clean
    patents = extract_records(raw_patents)
    prior_art = extract_records(raw_prior_art)
    corpus = patents + prior_art
    progress.progress(45, text="🧠 Computing semantic similarity…")

    progress.progress(60, text="📐 Computing semantic similarity…")

    # 5. Similarity & novelty score
    report: NoveltyReport = analyze(invention, corpus)
    progress.progress(72, text="🗺️ Building patent landscape map…")

    # 6. Landscape map
    fig = build_landscape(corpus, report, invention_text=invention[:60])
    progress.progress(82, text="💡 Generating improvement suggestions…")

    # 7. Improvement suggestions
    improvements = suggest_improvements(
        invention_text=invention,
        novelty_score=report.novelty_score,
        risk_level=report.risk_level,
        top_matches=report.top_matches[:4],
        model=ollama_model,
    )
    progress.progress(93, text="📝 Drafting patent application…")

    # 8. Patent draft
    draft: PatentDraft = generate_draft(
        invention_text=invention,
        top_matches=report.top_matches[:3],
        model=ollama_model,
    )
    progress.progress(100, text="✅ Analysis complete!")
    progress.empty()

    results["patents"] = patents
    results["prior_art"] = prior_art
    results["corpus"] = corpus
    results["report"] = report
    results["landscape_fig"] = fig
    results["improvements"] = improvements
    results["draft"] = draft

    return results


# ── Section renderers ─────────────────────────────────────────────────────

def render_intelligence_report(results: dict) -> None:
    """Render the Patent Intelligence Report section."""
    report: NoveltyReport = results["report"]
    patents = results["patents"]
    prior_art = results["prior_art"]

    st.markdown('<p class="section-header">📊 Patent Intelligence Report</p>', unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Novelty Score", f"{report.novelty_score:.1f} / 100")
    with col2:
        st.metric("⚠️ Risk Level", f"{report.risk_color} {report.risk_level}")
    with col3:
        st.metric("🔍 Patents Found", len(patents))
    with col4:
        st.metric("📄 Prior Art Found", len(prior_art))

    # ── Novelty interpretation ─────────────────────────────────────────
    st.markdown(
        f"**Analysis:** {report.score_label} &nbsp;|&nbsp; "
        f"Max similarity to existing patents: **{report.max_similarity * 100:.1f}%**"
    )

    # ── Top similar patents ────────────────────────────────────────────
    if report.top_matches:
        st.markdown("#### 🏆 Most Similar Patents & Prior Art")
        rows = [sr.to_row() for sr in report.top_matches[:8]]
        st.dataframe(
            rows,
            use_container_width=True,
            column_config={
                "URL": st.column_config.LinkColumn("Link"),
                "Similarity": st.column_config.TextColumn("Similarity"),
            },
            hide_index=True,
        )

    # ── Expanders for full lists ───────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander(f"📜 All Patents ({len(patents)})", expanded=False):
            if patents:
                st.dataframe(
                    [r.to_row() for r in patents],
                    use_container_width=True,
                    hide_index=True,
                    column_config={"URL": st.column_config.LinkColumn("Link")},
                )
            else:
                st.info("No patents found.")
    with col_b:
        with st.expander(f"🔬 All Prior Art ({len(prior_art)})", expanded=False):
            if prior_art:
                st.dataframe(
                    [r.to_row() for r in prior_art],
                    use_container_width=True,
                    hide_index=True,
                    column_config={"URL": st.column_config.LinkColumn("Link")},
                )
            else:
                st.info("No prior art found.")


def render_landscape(results: dict) -> None:
    """Render the Patent Landscape Map section."""
    st.markdown('<p class="section-header">🗺️ Patent Landscape Map</p>', unsafe_allow_html=True)
    st.caption(
        "2-D visualisation using PCA dimensionality reduction + K-Means clustering. "
        "Your invention is shown as a ★ red star. Hover over any point for details."
    )
    st.plotly_chart(results["landscape_fig"], use_container_width=True)


def render_draft(draft: PatentDraft) -> None:
    """Render the AI Patent Draft section."""
    st.markdown('<p class="section-header">📝 AI Patent Application Draft</p>', unsafe_allow_html=True)
    st.caption("Generated by local Ollama LLM. Review with a qualified patent attorney before filing.")

    if draft.title:
        st.markdown(f"#### {draft.title}")

    tab_abstract, tab_background, tab_summary, tab_desc, tab_claims = st.tabs(
        ["Abstract", "Background", "Summary", "Description", "Claims"]
    )

    with tab_abstract:
        st.write(draft.abstract or "_No abstract generated._")

    with tab_background:
        st.write(draft.background or "_No background generated._")

    with tab_summary:
        st.write(draft.summary or "_No summary generated._")

    with tab_desc:
        st.write(draft.description or "_No description generated._")

    with tab_claims:
        if draft.claims:
            for i, claim in enumerate(draft.claims, start=1):
                st.markdown(f"**{i}.** {claim}")
        else:
            st.info("No claims generated.")

    # Download button
    plain_text = draft.to_plain_text()
    st.download_button(
        label="⬇️ Download Patent Draft (.txt)",
        data=plain_text,
        file_name=f"patent_draft_{draft.title[:30].replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=False,
    )


def render_improvements(results: dict) -> None:
    """Render the Invention Improvement Suggestions section."""
    st.markdown('<p class="section-header">💡 Invention Improvement Suggestions</p>', unsafe_allow_html=True)
    st.caption("AI-powered suggestions to increase novelty score and avoid blocking prior art.")

    improvements = results["improvements"]
    if not improvements:
        st.info("No improvement suggestions generated.")
        return

    for i, imp in enumerate(improvements, start=1):
        icon = _CATEGORY_ICONS.get(imp.category, "💡")
        st.markdown(
            f"""
            <div class="improvement-card">
                <b>{i}. {icon} {imp.category}</b><br>
                <span style="font-size:0.97rem">{imp.suggestion}</span><br>
                <span style="color:#9999BB; font-size:0.85rem">💬 {imp.rationale}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    config = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="main-header">🛡️ PatentGuard AI</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Discover if your invention idea is already patented — globally.**  \n"
        "Powered by Tavily · SentenceTransformers · Ollama · Plotly"
    )
    st.divider()

    # ── Input area ────────────────────────────────────────────────────────
    st.markdown("### 💡 Describe Your Invention")
    invention = st.text_area(
        label="Invention Description",
        placeholder=(
            "Example: A wearable patch that uses near-infrared spectroscopy to "
            "continuously monitor blood glucose levels without breaking the skin, "
            "transmitting real-time data to a smartphone app for diabetic management."
        ),
        height=140,
        label_visibility="collapsed",
    )

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Invention", use_container_width=True)
    with col_note:
        st.caption(
            "⏱️ Analysis takes 2–5 min depending on Ollama model speed.  \n"
            "Results are cached — re-running the same input is instant."
        )

    # ── Validation ────────────────────────────────────────────────────────
    if analyze_clicked:
        if not invention.strip():
            st.warning("⚠️ Please enter an invention description before analyzing.")
            return
        if not os.getenv("TAVILY_API_KEY"):
            st.error("❌ TAVILY_API_KEY is not set. Add it to your `.env` file and restart.")
            return

        # ── Run pipeline ──────────────────────────────────────────────────
        with st.spinner("Running PatentGuard AI analysis pipeline…"):
            try:
                results = run_pipeline(
                    invention=invention.strip(),
                    max_patents=config["max_patents"],
                    max_prior_art=config["max_prior_art"],
                    ollama_model=config["ollama_model"],
                )
            except Exception as exc:
                st.error(f"❌ Pipeline error: {exc}")
                return

        st.success("✅ Analysis complete! Scroll down to explore all sections.")
        st.divider()

        # ── Render all output sections ────────────────────────────────────
        render_intelligence_report(results)
        st.divider()
        render_landscape(results)
        st.divider()
        render_draft(results["draft"])
        st.divider()
        render_improvements(results)


if __name__ == "__main__":
    main()
EOF

cat > patentguard-ai/modules/__init__.py << 'EOF'
"""PatentGuard AI – modules package."""
EOF

cat > patentguard-ai/modules/patent_search.py << 'EOF'
"""
modules/patent_search.py
========================
Search global patent databases using the Tavily API.

Databases Covered
-----------------
  • Google Patents  – patents.google.com
  • USPTO           – patents.uspto.gov
  • WIPO PatentScope – patentscope.wipo.int
  • EPO Espacenet   – worldwide.espacenet.com

Usage
-----
    from modules.patent_search import search_patents
    from utils.models import PatentRecord

    records: list[PatentRecord] = search_patents(
        query="non-invasive glucose monitoring wearable",
        max_results=10,
    )
"""

from __future__ import annotations

import logging

from utils.models import PatentRecord
from utils.source_mapper import infer_source
from utils.tavily_client import tavily_search

logger = logging.getLogger(__name__)

PATENT_DOMAINS: list[str] = [
    "patents.google.com",
    "patents.uspto.gov",
    "patentscope.wipo.int",
    "worldwide.espacenet.com",
    "lens.org",
]


def search_patents(
    query: str,
    max_results: int = 10,
) -> list[PatentRecord]:
    """
    Search global patent databases for patents related to the given query.

    Parameters
    ----------
    query : str
        Natural-language description of the invention or search terms.
    max_results : int
        Maximum number of patent records to return (default 10).

    Returns
    -------
    list[PatentRecord]
        Structured patent records sorted by Tavily relevance score.
    """
    raw_results = tavily_search(
        query=f"patent: {query}",
        domains=PATENT_DOMAINS,
        max_results=max_results,
    )

    records = [
        PatentRecord(
            title=item.get("title", "Untitled Patent"),
            abstract=item.get("content", ""),
            url=item.get("url", ""),
            source=infer_source(item.get("url", ""), fallback="Patent DB"),
            record_type="patent",
        )
        for item in raw_results
    ]

    logger.info("search_patents: %d results for query '%s'", len(records), query[:60])
    return records
EOF

cat > patentguard-ai/modules/prior_art_search.py << 'EOF'
"""
modules/prior_art_search.py
============================
Search non-patent literature (NPL) for prior art using the Tavily API.

Sources Covered
---------------
  • arXiv           – scientific preprints
  • GitHub          – open-source repositories
  • IEEE Xplore     – engineering publications
  • Technical blogs – Medium, Towards Data Science, etc.
  • ResearchGate    – academic research

Usage
-----
    from modules.prior_art_search import search_prior_art
    from utils.models import PatentRecord

    records: list[PatentRecord] = search_prior_art(
        query="non-invasive glucose monitoring wearable",
        max_results=10,
    )
"""

from __future__ import annotations

import logging

from utils.models import PatentRecord
from utils.source_mapper import infer_source
from utils.tavily_client import tavily_search

logger = logging.getLogger(__name__)

NPL_DOMAINS: list[str] = [
    "arxiv.org",
    "github.com",
    "ieeexplore.ieee.org",
    "medium.com",
    "towardsdatascience.com",
    "researchgate.net",
    "scholar.google.com",
    "semanticscholar.org",
]


def search_prior_art(
    query: str,
    max_results: int = 10,
) -> list[PatentRecord]:
    """
    Search non-patent literature sources for prior art related to the query.

    Parameters
    ----------
    query : str
        Natural-language description of the invention or key technical terms.
    max_results : int
        Maximum number of prior-art records to return (default 10).

    Returns
    -------
    list[PatentRecord]
        Structured prior-art records (record_type="prior_art") sorted by
        Tavily relevance score.
    """
    raw_results = tavily_search(
        query=query,
        domains=NPL_DOMAINS,
        max_results=max_results,
    )

    records = [
        PatentRecord(
            title=item.get("title", "Untitled"),
            abstract=item.get("content", ""),
            url=item.get("url", ""),
            source=infer_source(item.get("url", ""), fallback="Web"),
            record_type="prior_art",
        )
        for item in raw_results
    ]

    logger.info("search_prior_art: %d results for query '%s'", len(records), query[:60])
    return records
EOF

cat > patentguard-ai/modules/patent_extract.py << 'EOF'
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
EOF

cat > patentguard-ai/modules/embedding_engine.py << 'EOF'
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
EOF

cat > patentguard-ai/modules/similarity_engine.py << 'EOF'
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
EOF

cat > patentguard-ai/modules/landscape_map.py << 'EOF'
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
EOF

cat > patentguard-ai/modules/patent_drafter.py << 'EOF'
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
EOF

cat > patentguard-ai/modules/invention_improver.py << 'EOF'
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
EOF

cat > patentguard-ai/utils/__init__.py << 'EOF'
"""PatentGuard AI – utility package."""
EOF

cat > patentguard-ai/utils/models.py << 'EOF'
"""
utils/models.py
===============
Shared Pydantic data models used across all PatentGuard AI modules.
Single source of truth for every data structure in the prototype.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Core record types ─────────────────────────────────────────────────────

class PatentRecord(BaseModel):
    """A single patent or prior-art record returned by a search module."""

    title: str = "Untitled"
    abstract: str = ""
    url: str = ""
    source: str = ""          # "USPTO" | "EPO" | "WIPO" | "Google Patents" | "arXiv" | "GitHub" | "Blog"
    filing_year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    record_type: str = "patent"   # "patent" | "prior_art"

    @property
    def text_for_embedding(self) -> str:
        """Concatenated text fed to the embedding model."""
        return f"{self.title}. {self.abstract}".strip()

    def to_row(self) -> dict[str, Any]:
        """Flat dict for Streamlit DataFrame rendering."""
        return {
            "Title": self.title[:80],
            "Source": self.source,
            "Year": self.filing_year or "—",
            "URL": self.url,
        }


# ── Similarity & novelty ──────────────────────────────────────────────────

class ScoredRecord(BaseModel):
    """A PatentRecord annotated with its cosine similarity to the invention."""

    record: PatentRecord
    similarity: float = 0.0     # cosine similarity in [0, 1]
    rank: int = 0

    @property
    def similarity_pct(self) -> str:
        return f"{self.similarity * 100:.1f}%"

    def to_row(self) -> dict[str, Any]:
        row = self.record.to_row()
        row["Similarity"] = self.similarity_pct
        row["Rank"] = self.rank
        return row


class NoveltyReport(BaseModel):
    """Output of the similarity engine for one invention analysis."""

    novelty_score: float            # 0–100 (100 = fully novel)
    risk_level: str                 # LOW | MEDIUM | HIGH
    max_similarity: float           # highest cosine score in corpus
    top_matches: list[ScoredRecord] = Field(default_factory=list)
    invention_embedding: list[float] = Field(default_factory=list)
    corpus_embeddings: list[list[float]] = Field(default_factory=list)

    @property
    def risk_color(self) -> str:
        return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(self.risk_level, "⚪")

    @property
    def score_label(self) -> str:
        if self.novelty_score >= 70:
            return "✅ Likely Novel"
        elif self.novelty_score >= 40:
            return "⚠️ Partially Novel"
        return "❌ Likely Anticipated"


# ── Patent draft ──────────────────────────────────────────────────────────

class PatentDraft(BaseModel):
    """Structured patent application draft generated by the LLM."""

    title: str = ""
    background: str = ""
    summary: str = ""
    description: str = ""
    claims: list[str] = Field(default_factory=list)
    abstract: str = ""

    def to_plain_text(self) -> str:
        """Render the full draft as a formatted plain-text document."""
        claims_block = "\n".join(
            f"  {i + 1}. {c}" for i, c in enumerate(self.claims)
        )
        return (
            f"TITLE OF THE INVENTION\n{'─' * 40}\n{self.title}\n\n"
            f"ABSTRACT\n{'─' * 40}\n{self.abstract}\n\n"
            f"BACKGROUND OF THE INVENTION\n{'─' * 40}\n{self.background}\n\n"
            f"SUMMARY OF THE INVENTION\n{'─' * 40}\n{self.summary}\n\n"
            f"DETAILED DESCRIPTION\n{'─' * 40}\n{self.description}\n\n"
            f"CLAIMS\n{'─' * 40}\n{claims_block}\n"
        )


# ── Improvements ──────────────────────────────────────────────────────────

class Improvement(BaseModel):
    """A single actionable novelty-improvement suggestion."""

    category: str       # Technical | Claim Scope | Use Case | Combination | Jurisdictional
    suggestion: str
    rationale: str
    priority: int = 3   # 1 = highest impact
EOF

cat > patentguard-ai/utils/ollama_client.py << 'EOF'
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
EOF

cat > patentguard-ai/utils/source_mapper.py << 'EOF'
"""
utils/source_mapper.py
=======================
Single source of truth for URL-to-source-name mapping.
Used by patent_search, prior_art_search, and patent_extract.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Unified mapping covering all patent databases and NPL sources
DOMAIN_SOURCE_MAP: dict[str, str] = {
    # Patent databases
    "patents.google.com":       "Google Patents",
    "patents.uspto.gov":        "USPTO",
    "patentscope.wipo.int":     "WIPO",
    "worldwide.espacenet.com":  "EPO",
    "lens.org":                 "Lens.org",
    # Non-patent literature
    "arxiv.org":                "arXiv",
    "github.com":               "GitHub",
    "ieeexplore.ieee.org":      "IEEE",
    "medium.com":               "Blog",
    "towardsdatascience.com":   "Blog",
    "researchgate.net":         "ResearchGate",
    "semanticscholar.org":      "Semantic Scholar",
}


def infer_source(url: str, fallback: str = "Unknown") -> str:
    """
    Map a URL's hostname to a human-readable source name.

    Parameters
    ----------
    url : str
        Full URL of the patent or prior-art record.
    fallback : str
        Value returned when no domain matches (default ``"Unknown"``).

    Returns
    -------
    str
        Human-readable source name (e.g. ``"USPTO"``, ``"arXiv"``).
    """
    host = urlparse(url).netloc.lower()
    for domain, name in DOMAIN_SOURCE_MAP.items():
        if domain in host:
            return name
    return fallback
EOF

cat > patentguard-ai/utils/llm_parsing.py << 'EOF'
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
EOF

cat > patentguard-ai/utils/tavily_client.py << 'EOF'
"""
utils/tavily_client.py
=======================
Shared Tavily search wrapper used by patent_search and prior_art_search.
Centralises API key loading, endpoint constant, and error handling.
"""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
TAVILY_ENDPOINT: str = "https://api.tavily.com/search"


def tavily_search(
    query: str,
    domains: list[str],
    max_results: int = 10,
) -> list[dict]:
    """
    Execute a Tavily search and return raw result dicts.

    Parameters
    ----------
    query : str
        Search query string (already formatted by the caller).
    domains : list[str]
        Domain whitelist to restrict Tavily results.
    max_results : int
        Maximum number of results to return (default 10).

    Returns
    -------
    list[dict]
        Raw Tavily result items, or an empty list on any error.

    Raises
    ------
    ValueError
        If ``query`` is empty or ``TAVILY_API_KEY`` is not set.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set. Add it to your .env file.")

    payload: dict = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_domains": domains,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.exceptions.HTTPError as exc:
        logger.error("Tavily HTTP error: %s", exc)
        return []
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        return []
EOF

cat > patentguard-ai/.gitignore << 'EOF'
.env
__pycache__/
*.pyc
*.pyo
.DS_Store
*.egg-info/
.venv/
venv/
EOF

echo ""
echo "✅ PatentGuard AI files created in ./patentguard-ai/"
echo ""
echo "Next steps:"
echo "  cd patentguard-ai"
echo "  cp .env.example .env   # add your TAVILY_API_KEY"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'feat: PatentGuard AI hackathon prototype'"
echo "  git branch -M main"
echo "  git remote add origin https://github.com/joshuapremkumar/patentguard-ai.git"
echo "  git push -u origin main"
echo ""
echo "  # Then run:"
echo "  pip install -r requirements.txt"
echo "  ollama pull llama3"
echo "  streamlit run app.py"
