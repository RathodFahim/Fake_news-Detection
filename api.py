"""
Flask REST API for Fake News Detection
Endpoints:
  POST /api/predict       — JSON body { "text": "..." }
  POST /api/predict-url   — JSON body { "url":  "..." }
  GET  /api/health        — health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_training import FakeNewsModel
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)
CORS(app)  # allow browser-extension / cross-origin calls

# Load model once at startup
model = FakeNewsModel("logistic")
model.load()


def _extract_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "app": "Fake News Detector API",
        "endpoints": {
            "POST /api/predict": "Send JSON {\"text\": \"...\"} to classify news",
            "POST /api/predict-url": "Send JSON {\"url\": \"...\"} to scrape & classify",
            "GET /api/health": "Health check",
        },
        "status": "running",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model.is_trained})


@app.route("/api/predict", methods=["POST"])
def predict_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = model.predict(text)
    return jsonify(result)


@app.route("/api/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        article_text = _extract_text(url)
        if not article_text:
            return jsonify({"error": "Could not extract text from URL"}), 422
        result = model.predict(article_text)
        result["extracted_text"] = article_text[:500]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Fake News Detection API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
