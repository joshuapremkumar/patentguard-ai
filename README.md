# 🛡️ PatentGuard AI

> **AI-powered patent intelligence platform that helps inventors determine whether their ideas are novel.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-black?logo=ollama)](https://ollama.com)
[![Tavily](https://img.shields.io/badge/Tavily-Search%20API-4E79A7)](https://tavily.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

PatentGuard AI is a single-page Streamlit application that empowers inventors, researchers, and IP teams to perform rapid patent due diligence **before** investing in a full patent application. Simply describe your invention in plain English and PatentGuard AI will:

- Search global patent databases and scientific literature in parallel
- Score your invention's novelty against existing prior art using semantic similarity
- Visualise the competitive patent landscape interactively
- Draft a structured patent application using a local LLM
- Suggest targeted improvements to increase novelty and avoid blocking patents

All LLM inference runs **locally via Ollama** — no API keys required for AI generation, no data leaves your machine.

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Global Patent Search** | Queries USPTO, Google Patents, WIPO, EPO, and Lens.org via Tavily |
| 📚 **Prior Art Detection** | Searches arXiv, IEEE, GitHub, ResearchGate, and technical blogs |
| 🧠 **Semantic Similarity** | Embeds all records with `all-MiniLM-L6-v2` (384-dim vectors) |
| 📊 **Novelty Score** | `100 - (max_cosine_similarity x 100)` with HIGH / MEDIUM / LOW risk bands |
| 🗺️ **Patent Landscape Map** | Interactive PCA + K-Means 2D scatter plot (Plotly dark theme) |
| 📝 **AI Patent Draft** | Generates title, abstract, background, summary, description + 5 claims |
| 💡 **Invention Improvements** | 5 actionable suggestions across Technical, Claim Scope, Use Case, Combination, and Jurisdictional categories |
| ⚡ **Parallel Search** | `ThreadPoolExecutor` cuts pipeline time ~50% vs sequential |
| 🔒 **Local LLM** | All drafting and improvement suggestions run on-device via Ollama |

---

## Architecture

```
patentguard-ai/
│
├── app.py                    # Streamlit UI + pipeline orchestration
├── requirements.txt
├── .env.example
│
├── modules/                  # Core pipeline stages
│   ├── patent_search.py      # Search patent databases (Tavily)
│   ├── prior_art_search.py   # Search NPL / prior art (Tavily)
│   ├── patent_extract.py     # Clean, deduplicate, normalise records
│   ├── embedding_engine.py   # SentenceTransformer batch encoding
│   ├── similarity_engine.py  # Cosine similarity + novelty scoring
│   ├── landscape_map.py      # PCA + K-Means + Plotly visualisation
│   ├── patent_drafter.py     # AI patent application draft (Ollama)
│   └── invention_improver.py # AI novelty improvement suggestions (Ollama)
│
└── utils/                    # Shared utilities
    ├── models.py             # Pydantic v2 data models (single source of truth)
    ├── ollama_client.py      # Ollama REST API client
    ├── tavily_client.py      # Centralised Tavily search wrapper
    ├── source_mapper.py      # URL to source name mapping
    └── llm_parsing.py        # JSON fence stripping for LLM output
```

### Pipeline Flow

```
Invention Text
      |
      +--------------------------------------+
      v (parallel)                           v (parallel)
patent_search.py                    prior_art_search.py
      |                                      |
      +--------------+-----------------------+
                     v
             patent_extract.py
             (clean + deduplicate)
                     |
                     v
           embedding_engine.py
           (all-MiniLM-L6-v2)
                     |
                     v
           similarity_engine.py
           (cosine sim + NoveltyReport)
                     |
           +---------+-----------+
           v         v           v
    landscape_map  patent_    invention_
       .py         drafter.py  improver.py
    (Plotly)       (Ollama)    (Ollama)
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A free [Tavily API key](https://tavily.com)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/joshuapremkumar/patentguard-ai.git
cd patentguard-ai

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## Tavily API Setup

PatentGuard AI uses [Tavily](https://tavily.com) to search patent databases and academic literature.

1. Sign up for a free account at https://tavily.com
2. Copy your API key from the dashboard
3. Add it to your `.env` file:

```env
TAVILY_API_KEY=tvly-your-key-here
```

> **Free tier:** 1,000 searches/month — sufficient for ~125 invention analyses.

---

## Ollama Setup

PatentGuard AI uses [Ollama](https://ollama.com) for local LLM inference (patent drafting and improvement suggestions). No data leaves your machine.

```bash
# 1. Install Ollama
# macOS / Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download

# 2. Start the Ollama server
ollama serve

# 3. Pull the default model (llama3 ~4.7GB)
ollama pull llama3

# Optional: pull a faster/lighter model
ollama pull mistral    # ~4.1GB
ollama pull phi3       # ~2.2GB
```

> **Tip:** Select your preferred model from the sidebar in the Streamlit UI. Smaller models like `phi3` are faster but may produce less structured JSON output.

---

## Example Usage

1. **Launch the app:** `streamlit run app.py`
2. **Enter your invention** in the text area, e.g.:
   > *"A wearable patch that uses near-infrared spectroscopy to continuously monitor blood glucose levels without breaking the skin, transmitting readings via Bluetooth to a smartphone app."*
3. **Adjust settings** in the sidebar (max results, Ollama model)
4. **Click "Analyze Invention"** and wait 2-5 minutes
5. **Review results:**
   - 📊 Novelty Score (0-100) + Risk Level
   - 🏆 Top 8 most similar patents with similarity %
   - 🗺️ Interactive 2D patent landscape map
   - 📝 Full AI-generated patent application draft (downloadable)
   - 💡 5 targeted improvement suggestions

### Novelty Score Interpretation

| Score | Risk | Meaning |
|-------|------|---------|
| 70-100 | 🟢 LOW | Likely novel — strong patent potential |
| 40-69 | 🟡 MEDIUM | Partially novel — consider differentiation |
| 0-39 | 🔴 HIGH | Likely anticipated — significant blocking art exists |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **UI** | [Streamlit](https://streamlit.io) 1.35+ |
| **Patent Search** | [Tavily API](https://tavily.com) |
| **Embeddings** | [SentenceTransformers](https://sbert.net) — `all-MiniLM-L6-v2` |
| **Similarity** | NumPy vectorised cosine similarity |
| **Dimensionality Reduction** | scikit-learn PCA |
| **Clustering** | scikit-learn K-Means |
| **Visualisation** | [Plotly](https://plotly.com) |
| **LLM Inference** | [Ollama](https://ollama.com) (llama3, mistral, phi3, gemma2) |
| **Data Models** | [Pydantic v2](https://docs.pydantic.dev) |
| **Concurrency** | `concurrent.futures.ThreadPoolExecutor` |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> **Disclaimer:** PatentGuard AI is a research and productivity tool. It does not constitute legal advice. Always consult a qualified patent attorney before filing a patent application.
