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
