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
