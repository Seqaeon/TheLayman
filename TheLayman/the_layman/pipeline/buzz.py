"""Community buzz scoring for arXiv papers.

Queries two free, no-auth APIs to measure real-world community engagement:

1. Hacker News (Algolia search) — total upvotes + comments across all HN
   stories linking to a paper.
2. Semantic Scholar (public graph API) — citation count + influential
   citation count as a proxy for academic reach.

Usage::

    scores = fetch_buzz_scores(["2401.00001", "2312.99999"])
    # → {"2401.00001": 7.3, "2312.99999": 0.0}  (0–10, log-normalised)
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "TheLayman/0.1 (+https://example.local)"

# Hacker News / Algolia
_HN_BASE = "https://hn.algolia.com/api/v1/search"
_HN_TIMEOUT = 10
_HN_MAX_HITS = 5          # top N HN stories to consider per paper

# Semantic Scholar
_S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_S2_FIELDS = "citationCount,influentialCitationCount"
_S2_TIMEOUT = 10
_S2_RATE_DELAY = 0.15     # ~6 req/s — stays well within the 100 req/min free tier

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 10) -> dict | list | None:
    """Fire a GET request and parse JSON, returning None on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# HN Algolia
# ---------------------------------------------------------------------------

def _hn_score_for_paper(arxiv_id: str) -> int:
    """Return combined HN engagement (points + comments) for a paper.

    Searches for the plain arxiv ID and its full URL to catch both link-posts
    and text posts that embed the URL.
    """
    total = 0
    for query in (arxiv_id, f"arxiv.org/abs/{arxiv_id}"):
        params = urllib.parse.urlencode({
            "query": query,
            "tags": "story",
            "hitsPerPage": _HN_MAX_HITS,
        })
        data = _get_json(f"{_HN_BASE}?{params}", _HN_TIMEOUT)
        if not data or not isinstance(data.get("hits"), list):
            continue
        for hit in data["hits"][:_HN_MAX_HITS]:
            points = hit.get("points") or 0
            comments = hit.get("num_comments") or 0
            total += int(points) + int(comments)
    return total


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _s2_score_for_paper(arxiv_id: str) -> int:
    """Return a raw academic signal from Semantic Scholar.

    Uses citationCount + 2×influentialCitationCount so highly-cited-by-other
    influential-papers gets extra weight.
    """
    url = f"{_S2_BASE}/arXiv:{arxiv_id}?fields={_S2_FIELDS}"
    data = _get_json(url, _S2_TIMEOUT)
    if not data or not isinstance(data, dict):
        return 0
    citations = data.get("citationCount") or 0
    influential = data.get("influentialCitationCount") or 0
    return int(citations) + 2 * int(influential)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_buzz_scores(arxiv_ids: list[str]) -> dict[str, float]:
    """Fetch and log-normalise community buzz for a list of arXiv IDs.

    Returns a dict mapping each arXiv ID to a float in [0, 10].
    Papers with no community signal get 0.0.  All network errors are
    silently swallowed so a bad API response never kills the pipeline.

    Args:
        arxiv_ids: Plain arXiv IDs, e.g. ``["2401.00001", "2312.99999"]``.
                   ``"arxiv:2401.00001"`` prefixes are stripped automatically.
    """
    # Strip "arxiv:" prefix that the pipeline attaches to paper IDs
    clean_ids = [aid.removeprefix("arxiv:") for aid in arxiv_ids]

    raw: dict[str, int] = {}
    total_papers = len(clean_ids)

    for idx, aid in enumerate(clean_ids):
        hn = _hn_score_for_paper(aid)
        time.sleep(_S2_RATE_DELAY)        # gentle rate-limit buffer
        s2 = _s2_score_for_paper(aid)
        raw[aid] = hn + s2
        if (idx + 1) % 20 == 0 or idx == total_papers - 1:
            print(f"  [buzz] scored {idx + 1}/{total_papers} papers …")

    if not raw:
        return {}

    max_raw = max(raw.values()) or 1  # avoid division by zero

    # Log-normalise to [0, 10] so one outlier doesn't dominate
    normalised: dict[str, float] = {}
    for original_id, clean_id in zip(arxiv_ids, clean_ids):
        r = raw.get(clean_id, 0)
        normalised[original_id] = round(math.log1p(r) / math.log1p(max_raw) * 10, 2)

    n_nonzero = sum(1 for v in normalised.values() if v > 0)
    print(f"  [buzz] {n_nonzero}/{total_papers} papers have community signal")
    return normalised
