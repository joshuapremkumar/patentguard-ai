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
