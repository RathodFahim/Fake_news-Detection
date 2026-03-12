"""
Google Gemini AI Analysis for Fake News Detection
Uses multiple Gemini models with automatic fallback when rate limits are hit.

Models (in fallback order):
  1. gemini-2.5-flash          — stable, fast reasoning
  2. gemini-2.5-flash-lite     — stable, cost-efficient
  3. gemini-3-flash-preview    — preview, pro-level intelligence
  4. gemini-3.1-flash-lite-preview — preview, latest workhorse

API key is read from the GEMINI_API_KEY environment variable.
Get a free key at: https://aistudio.google.com/apikey
"""

import os
import json
import re

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

ANALYSIS_PROMPT = """You are an expert fact-checker and media analyst. Analyze the following news text and determine if it is likely REAL or FAKE news.

Evaluate based on:
1. Language patterns (sensationalism, emotional manipulation, clickbait)
2. Logical consistency and coherence
3. Verifiable claims vs unsubstantiated assertions
4. Source credibility indicators
5. Writing quality and professionalism

News text to analyze:
\"\"\"
{text}
\"\"\"

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
  "verdict": "Real" or "Fake" or "Uncertain",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief 2-3 sentence explanation",
  "red_flags": ["list", "of", "specific", "concerns"],
  "credibility_score": 0 to 100
}}"""


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None."""
    return os.environ.get("GEMINI_API_KEY")


def _parse_response(text: str) -> dict:
    """Extract JSON from Gemini's response, handling markdown code blocks."""
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise


def analyze_with_gemini(text: str) -> dict:
    """Analyze news text using Gemini with automatic model fallback.

    Returns a dict with:
      - "available": bool  — whether Gemini is configured
      - "verdict": str     — "Real", "Fake", or "Uncertain"
      - "confidence": float
      - "reasoning": str
      - "red_flags": list[str]
      - "credibility_score": int
      - "model_used": str  — which model produced the result
      - "error": str|None
    """
    if not GENAI_AVAILABLE:
        return _unavailable("google-generativeai package not installed. "
                            "Run: pip install google-generativeai")

    api_key = get_api_key()
    if not api_key:
        return _unavailable(
            "GEMINI_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)
    prompt = ANALYSIS_PROMPT.format(text=text[:3000])

    last_error = None
    for model_id in MODELS:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )

            if not response.text:
                last_error = f"{model_id}: Empty response"
                continue

            parsed = _parse_response(response.text)

            return {
                "available": True,
                "verdict": parsed.get("verdict", "Uncertain"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": parsed.get("reasoning", ""),
                "red_flags": parsed.get("red_flags", []),
                "credibility_score": int(parsed.get("credibility_score", 50)),
                "model_used": model_id,
                "error": None,
            }

        except Exception as exc:
            last_error = f"{model_id}: {exc}"
            continue

    return {
        "available": True,
        "verdict": "Error",
        "confidence": 0.0,
        "reasoning": "",
        "red_flags": [],
        "credibility_score": 50,
        "model_used": None,
        "error": f"All models failed. Last error: {last_error}",
    }


def _unavailable(reason: str) -> dict:
    """Return a standard 'unavailable' response."""
    return {
        "available": False,
        "verdict": "Unavailable",
        "confidence": 0.0,
        "reasoning": "",
        "red_flags": [],
        "credibility_score": 50,
        "model_used": None,
        "error": reason,
    }
