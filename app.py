"""
Fake News Detection — Streamlit Web Application
Professional UI with dashboard, single/URL/batch analysis, model insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model_training import FakeNewsModel
from fact_check import search_claims
import requests
from bs4 import BeautifulSoup
import re

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #6c757d;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.result-card {
    padding: 1.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from {opacity:0; transform:translateY(10px)} to {opacity:1; transform:translateY(0)} }

.real-card {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    border-left: 5px solid #28a745;
}
.fake-card {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    border-left: 5px solid #dc3545;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid #e9ecef;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
.metric-label { font-size: 0.85rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
    font-weight: 500;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9ff 0%, #eef1ff 100%);
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    model = FakeNewsModel("logistic")
    model.load()
    return model


def fetch_article_text(url: str) -> str:
    """Scrape readable text from a news URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Clean up
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000] if text else ""


# ── Sidebar ────────────────────────────────────────────────────────────
def render_sidebar(model):
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/news.png", width=70)
        st.markdown("## 🛡️ Fake News Detector")
        st.caption("ML-powered news credibility analysis")
        st.markdown("---")

        if model.metrics:
            st.markdown("### 📊 Model Performance")
            m = model.metrics
            cols = st.columns(2)
            cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
            cols[1].metric("F1-Score", f"{m['f1_score']:.1%}")
            cols = st.columns(2)
            cols[0].metric("Precision", f"{m['precision']:.1%}")
            cols[1].metric("Recall", f"{m['recall']:.1%}")

        st.markdown("---")
        st.markdown("### ⚙️ How It Works")
        st.markdown("""
        1. **Text cleaning** — lowercase, remove noise
        2. **TF-IDF** — 10 000 features, bigrams
        3. **Logistic Regression** — balanced class weights
        4. **Probability** — confidence scoring
        """)
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#adb5bd;font-size:0.8rem'>"
            "Built with ❤️ using Streamlit & scikit-learn<br>"
            "Dataset: FakeNewsNet (23 196 articles)"
            "</div>",
            unsafe_allow_html=True,
        )


# ── Prediction display ────────────────────────────────────────────────
def show_prediction(result):
    label = result["prediction"]
    conf = result["confidence"]
    probs = result["probabilities"]

    css_class = "real-card" if label == "Real" else "fake-card"
    icon = "✅" if label == "Real" else "🚨"
    st.markdown(
        f'<div class="result-card {css_class}">'
        f"{icon} {label.upper()} NEWS &nbsp;—&nbsp; {conf:.1%} confidence"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probs["Fake"] * 100,
            title={"text": "Fake Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#dc3545" if probs["Fake"] > 0.5 else "#28a745"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)

    # Probability bar
    col1, col2 = st.columns(2)
    col1.metric("🟢 Real probability", f"{probs['Real']:.2%}")
    col2.metric("🔴 Fake probability", f"{probs['Fake']:.2%}")


def show_fact_check(text: str):
    """Query the Google Fact Check API and display results."""
    fc = search_claims(text)
    if not fc["available"]:
        st.info(
            "ℹ️ **Google Fact Check API not configured.** "
            "Set the `GOOGLE_FACT_CHECK_API_KEY` environment variable to enable. "
            "[Get a free key →](https://console.cloud.google.com/apis/credentials)"
        )
        return fc
    if fc["error"]:
        st.warning(f"⚠️ Fact-check lookup error: {fc['error']}")
        return fc
    if not fc["claims"]:
        st.info("🔎 No matching fact-checks found in Google's database for this text.")
        return fc

    verdict = fc["verdict"]
    if verdict == "Verified Real":
        st.success(f"✅ **Fact-Check Verdict: {verdict}**")
    elif verdict == "Likely Fake":
        st.error(f"🚨 **Fact-Check Verdict: {verdict}**")
    else:
        st.warning(f"⚠️ **Fact-Check Verdict: {verdict}**")

    with st.expander(f"📋 {len(fc['claims'])} fact-check(s) found", expanded=True):
        for i, claim in enumerate(fc["claims"], 1):
            st.markdown(
                f"**{i}. {claim['claim_text'][:120]}**\n\n"
                f"- **Rating:** {claim['rating']}\n"
                f"- **Publisher:** {claim['publisher']}\n"
                f"- **Source:** [View full review]({claim['url']})"
            )
            if i < len(fc["claims"]):
                st.markdown("---")
    return fc


# ── Main ───────────────────────────────────────────────────────────────
def main():
    model = get_model()
    render_sidebar(model)

    # Header
    st.markdown('<h1 class="main-title">🛡️ Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Analyse news articles in real-time using machine learning</p>',
        unsafe_allow_html=True,
    )

    # Metric cards
    if model.metrics:
        m = model.metrics
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip(
            [c1, c2, c3, c4],
            [("Accuracy", m["accuracy"]), ("Precision", m["precision"]),
             ("Recall", m["recall"]), ("F1-Score", m["f1_score"])],
        ):
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val:.1%}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Text Analysis", "🔗 URL Analysis", "📁 Batch Analysis", "📊 Model Insights"]
    )

    # ── Tab 1: Text ──
    with tab1:
        st.markdown("### Paste or type a news headline / article")
        col_input, col_samples = st.columns([3, 1])
        with col_input:
            user_text = st.text_area(
                "News text",
                height=160,
                placeholder="Enter a news headline or article body here…",
                label_visibility="collapsed",
            )
            analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)
        with col_samples:
            st.markdown("**Quick samples**")
            samples = {
                "✅ Real — Policy": "Government announces new education policy to improve literacy rates across the country",
                "✅ Real — Tech": "Apple reports record quarterly revenue driven by strong iPhone sales in emerging markets",
                "🚨 Fake — Miracle": "Scientists discover miracle cure that pharmaceutical companies are desperately trying to hide from the public",
                "🚨 Fake — Conspiracy": "Breaking: World leaders caught in secret meeting to control global weather patterns using satellites",
            }
            for label, text in samples.items():
                if st.button(label, key=f"sample_{label}", use_container_width=True):
                    st.session_state["fill_text"] = text
                    st.rerun()

        if "fill_text" in st.session_state:
            user_text = st.session_state.pop("fill_text")
            st.info(f"**Sample loaded:** {user_text}")
            result = model.predict(user_text)
            show_prediction(result)
            st.markdown("#### 🌐 Google Fact Check")
            show_fact_check(user_text)
        elif analyse_btn and user_text.strip():
            with st.spinner("Analysing…"):
                result = model.predict(user_text)
            show_prediction(result)
            st.markdown("#### 🌐 Google Fact Check")
            show_fact_check(user_text)
        elif analyse_btn:
            st.warning("⚠️ Please enter some text to analyse.")

    # ── Tab 2: URL ──
    with tab2:
        st.markdown("### Analyse a news article by URL")
        url_input = st.text_input(
            "News URL",
            placeholder="https://example.com/news-article",
            label_visibility="collapsed",
        )
        url_btn = st.button("🌐 Fetch & Analyse", type="primary", use_container_width=True)
        if url_btn and url_input.strip():
            with st.spinner("Fetching article…"):
                try:
                    article_text = fetch_article_text(url_input.strip())
                    if not article_text:
                        st.error("Could not extract text from the URL.")
                    else:
                        st.success(f"Extracted {len(article_text)} characters")
                        with st.expander("📄 Extracted text", expanded=False):
                            st.text(article_text[:2000])
                        result = model.predict(article_text)
                        show_prediction(result)
                        st.markdown("#### 🌐 Google Fact Check")
                        show_fact_check(article_text)
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")
        elif url_btn:
            st.warning("⚠️ Please enter a URL.")

    # ── Tab 3: Batch ──
    with tab3:
        st.markdown("### Upload a CSV for batch analysis")
        st.caption("CSV must have a column named **text** or **title**.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            text_col = "text" if "text" in df.columns else ("title" if "title" in df.columns else None)
            if text_col is None:
                st.error("CSV must contain a **text** or **title** column.")
            else:
                st.success(f"Loaded {len(df)} rows (using column `{text_col}`)")
                if st.button("🔍 Analyse All", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    preds = []
                    for i, text in enumerate(df[text_col]):
                        try:
                            r = model.predict(str(text))
                            preds.append(r)
                        except Exception:
                            preds.append({"prediction": "Error", "confidence": 0, "probabilities": {"Real": 0, "Fake": 0}})
                        progress.progress((i + 1) / len(df))

                    df["Prediction"] = [p["prediction"] for p in preds]
                    df["Confidence"] = [p["confidence"] for p in preds]
                    df["P(Real)"] = [p["probabilities"]["Real"] for p in preds]
                    df["P(Fake)"] = [p["probabilities"]["Fake"] for p in preds]

                    # Summary
                    valid = df[df["Prediction"] != "Error"]
                    rc = (valid["Prediction"] == "Real").sum()
                    fc = (valid["Prediction"] == "Fake").sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total", len(df))
                    c2.metric("🟢 Real", int(rc))
                    c3.metric("🔴 Fake", int(fc))

                    # Pie chart
                    if rc + fc > 0:
                        fig = px.pie(
                            names=["Real", "Fake"], values=[rc, fc],
                            color_discrete_sequence=["#28a745", "#dc3545"],
                            hole=0.4,
                        )
                        fig.update_layout(height=300, margin=dict(t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(df, use_container_width=True)
                    csv_out = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download results", csv_out,
                        file_name="fake_news_results.csv", mime="text/csv",
                    )

    # ── Tab 4: Model Insights ──
    with tab4:
        st.markdown("### Model Performance Insights")
        if model.metrics:
            m = model.metrics
            col_a, col_b = st.columns(2)

            # Confusion matrix
            with col_a:
                cm = np.array(m["confusion_matrix"])
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Real", "Fake"], y=["Real", "Fake"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                )
                fig_cm.update_layout(title="Confusion Matrix", height=400)
                st.plotly_chart(fig_cm, use_container_width=True)

            # Metrics bar chart
            with col_b:
                metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
                metric_vals = [m["accuracy"], m["precision"], m["recall"], m["f1_score"]]
                fig_bar = px.bar(
                    x=metric_names, y=metric_vals,
                    color=metric_names,
                    color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#4facfe"],
                    text=[f"{v:.1%}" for v in metric_vals],
                )
                fig_bar.update_layout(
                    title="Model Metrics", yaxis_range=[0, 1],
                    showlegend=False, height=400,
                )
                fig_bar.update_traces(textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Classification report
            st.markdown("#### Detailed Classification Report")
            cr = m.get("classification_report", {})
            if cr:
                report_data = []
                for cls in ["Real", "Fake"]:
                    if cls in cr:
                        report_data.append({
                            "Class": cls,
                            "Precision": f"{cr[cls]['precision']:.4f}",
                            "Recall": f"{cr[cls]['recall']:.4f}",
                            "F1-Score": f"{cr[cls]['f1-score']:.4f}",
                            "Support": int(cr[cls]["support"]),
                        })
                st.table(pd.DataFrame(report_data))

            st.markdown("#### Training Details")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | Algorithm | Logistic Regression (balanced) |
            | Features  | TF-IDF (10 000, bigrams) |
            | Dataset   | FakeNewsNet — 23 196 articles |
            | Split     | 80% train / 20% test (stratified) |
            | Classes   | Real (75%), Fake (25%) |
            """)
        else:
            st.warning("No model metrics available.")


if __name__ == "__main__":
    main()