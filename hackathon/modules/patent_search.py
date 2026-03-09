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
