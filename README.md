---
title: Fake News Detector API
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# 🛡️ Fake News Detector

An AI-powered platform that detects fake news articles using machine learning. Features an interactive Streamlit dashboard, REST API, and Chrome browser extension.

> **Live demo →** [https://rathodfahim.github.io/Fake_news-Detection/](https://rathodfahim.github.io/Fake_news-Detection/)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📝 **Text Analysis** | Paste any headline or article and get an instant prediction with confidence scores |
| 🔗 **URL Analysis** | Enter a news URL — article text is scraped and analysed automatically |
| 📁 **Batch Analysis** | Upload a CSV file and get predictions for every row with downloadable results |
| 📊 **Model Insights** | Interactive confusion matrix, metrics bar chart, and classification report |
| ⚡ **REST API** | Flask API with `/api/predict` and `/api/predict-url` endpoints |
| 🔌 **Chrome Extension** | Analyse any news page with one click from the browser toolbar |
| 🌐 **GitHub Pages** | Static showcase site with live model metrics |

---

## 📊 Model Performance

Trained on **23,196 articles** from the [FakeNewsNet](https://www.kaggle.com/) dataset with balanced class weights.

| Metric | Score |
|--------|-------|
| Accuracy | 82.1% |
| Precision | 61.6% |
| Recall | 73.9% |
| F1-Score | 67.1% |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/RathodFahim/Fake_news-Detection.git
cd Fake_news-Detection
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model_training.py
```

This trains a Logistic Regression model on `FakeNewsNet.csv` and saves `model.pkl`, `tfidf_vectorizer.pkl`, and `model_metrics.pkl`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

### 4. Start the REST API

```bash
python api.py
# API runs on http://localhost:5000
```

### 5. Load the Chrome extension

1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `browser-extension/` folder
4. Click the extension icon on any news page to analyse it

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn (Logistic Regression, TF-IDF) |
| Dashboard | Streamlit, Plotly |
| API | Flask, Flask-CORS |
| Scraping | Requests, BeautifulSoup4 |
| Extension | Chrome Manifest V3 |
| Data | Pandas, NumPy |
| Deployment | GitHub Pages, GitHub Actions |

---

## 📂 Project Structure

```
.
├── app.py                      # Streamlit web dashboard
├── api.py                      # Flask REST API
├── model_training.py           # Model training & evaluation
├── data_preprocessing.py       # Text cleaning & TF-IDF
├── test_app.py                 # Test suite
├── run_app.py                  # Quick-start script
├── requirements.txt            # Python dependencies
├── FakeNewsNet.csv             # Kaggle dataset (23,196 articles)
├── model.pkl                   # Trained model (generated)
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer (generated)
├── model_metrics.pkl           # Metrics (generated)
├── browser-extension/          # Chrome extension
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── icons/
├── docs/                       # GitHub Pages site
│   ├── index.html
│   └── model_metrics.json
├── .github/workflows/
│   └── pages.yml               # GitHub Pages deployment
└── README.md
```

---

## 🔌 API Endpoints

### `POST /api/predict`
```json
// Request
{ "text": "Government announces new education policy" }

// Response
{
  "prediction": "Real",
  "confidence": 0.804,
  "probabilities": { "Real": 0.804, "Fake": 0.196 }
}
```

### `POST /api/predict-url`
```json
// Request
{ "url": "https://example.com/news-article" }

// Response
{
  "prediction": "Fake",
  "confidence": 0.712,
  "probabilities": { "Real": 0.288, "Fake": 0.712 },
  "extracted_text": "First 500 chars of article…"
}
```

### `GET /api/health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## 🧠 How It Works

1. **Text Cleaning** — lowercase, remove URLs/HTML/special chars, collapse whitespace
2. **Feature Extraction** — TF-IDF vectorisation with 10,000 features and bigrams
3. **Classification** — Logistic Regression with balanced class weights (handles 75/25 class imbalance)
4. **Probability Scoring** — outputs confidence and per-class probabilities

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with ❤️ using Python & Machine Learning.**