"""
Google Fact Check Tools API Integration
Queries Google's ClaimReview database to cross-reference news claims.
API key is read from the GOOGLE_FACT_CHECK_API_KEY environment variable.
Get a free key at: https://console.cloud.google.com/apis/credentials
(Enable "Fact Check Tools API" under APIs & Services → Library)
"""

import os
import requests


FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def get_api_key() -> str | None:
    """Return the API key from the environment, or None."""
    return os.environ.get("GOOGLE_FACT_CHECK_API_KEY")


def search_claims(query: str, language_code: str = "en", max_results: int = 5) -> dict:
    """Search Google Fact Check API for claims matching *query*.

    Returns a dict with:
      - "available": bool  – whether the API is configured
      - "claims": list     – fact-check results (may be empty)
      - "verdict": str     – overall summary ("Verified Real", "Likely Fake", etc.)
      - "error": str|None  – error message if the call failed
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "available": False,
            "claims": [],
            "verdict": "Unavailable",
            "error": "GOOGLE_FACT_CHECK_API_KEY not set. "
                     "Get a free key at https://console.cloud.google.com/apis/credentials",
        }

    params = {
        "query": query[:200],
        "languageCode": language_code,
        "pageSize": max_results,
        "key": api_key,
    }

    try:
        resp = requests.get(FACT_CHECK_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {
            "available": True,
            "claims": [],
            "verdict": "Error",
            "error": f"API request failed: {exc}",
        }

    raw_claims = data.get("claims", [])
    if not raw_claims:
        return {
            "available": True,
            "claims": [],
            "verdict": "No fact-checks found",
            "error": None,
        }

    claims = []
    for item in raw_claims:
        for review in item.get("claimReview", []):
            claims.append({
                "claim_text": item.get("text", ""),
                "claimant": item.get("claimant", "Unknown"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "rating": review.get("textualRating", ""),
                "url": review.get("url", ""),
                "review_date": review.get("reviewDate", ""),
                "language": review.get("languageCode", "en"),
            })

    verdict = _derive_verdict(claims)
    return {
        "available": True,
        "claims": claims,
        "verdict": verdict,
        "error": None,
    }


_FALSE_KEYWORDS = {"false", "fake", "pants on fire", "misleading", "incorrect",
                    "mostly false", "not true", "unproven", "fabricated",
                    "scam", "hoax", "conspiracy", "satire", "no evidence"}
_TRUE_KEYWORDS = {"true", "correct", "mostly true", "accurate", "verified",
                  "confirmed", "real", "factual"}


def _derive_verdict(claims: list[dict]) -> str:
    """Summarise multiple fact-check ratings into a single verdict."""
    if not claims:
        return "No fact-checks found"

    false_count = 0
    true_count = 0
    for c in claims:
        rating = c.get("rating", "").lower()
        if any(kw in rating for kw in _FALSE_KEYWORDS):
            false_count += 1
        elif any(kw in rating for kw in _TRUE_KEYWORDS):
            true_count += 1

    total = false_count + true_count
    if total == 0:
        return "Mixed / Inconclusive"
    if false_count > true_count:
        return "Likely Fake"
    if true_count > false_count:
        return "Verified Real"
    return "Mixed / Inconclusive"
