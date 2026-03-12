/* Fake News Detector — Extension popup logic */

const $ = (id) => document.getElementById(id);

function getApiUrl() {
  return ($("apiUrl").value || "https://rathodfahim23-fake-news-detector-api.hf.space").replace(/\/+$/, "");
}

function showResult(data) {
  const card = $("resultCard");
  const isFake = data.prediction === "Fake";
  card.className = "result-card " + (isFake ? "fake" : "real");
  $("resultLabel").textContent = (isFake ? "🚨 FAKE" : "✅ REAL") + " NEWS";
  $("resultConf").textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const rp = data.probabilities.Real * 100;
  const fp = data.probabilities.Fake * 100;
  $("realPct").textContent = rp.toFixed(1) + "%";
  $("fakePct").textContent = fp.toFixed(1) + "%";
  $("realBar").style.width = rp + "%";
  $("fakeBar").style.width = fp + "%";

  $("result").classList.add("show");
  $("error").style.display = "none";

  showGeminiAnalysis(data.gemini_analysis);
  showFactCheck(data.fact_check);
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").style.display = "block";
  $("result").classList.remove("show");
  $("geminiSection").style.display = "none";
  $("factCheck").style.display = "none";
}

function showGeminiAnalysis(ga) {
  const container = $("geminiSection");
  const verdict = $("geminiVerdict");
  const details = $("geminiDetails");

  details.innerHTML = "";

  if (!ga) { container.style.display = "none"; return; }

  container.style.display = "block";

  if (!ga.available) {
    verdict.textContent = "Gemini AI Unavailable";
    verdict.className = "fact-verdict verdict-none";
    details.innerHTML = `<div class="fact-claim"><span class="fact-claim-rating">API key not configured</span></div>`;
    return;
  }

  if (ga.error) {
    verdict.textContent = "Gemini AI Error";
    verdict.className = "fact-verdict verdict-mixed";
    details.innerHTML = `<div class="fact-claim"><span class="fact-claim-rating">${esc(ga.error)}</span></div>`;
    return;
  }

  const v = ga.verdict || "Uncertain";
  const cls = v === "Real" ? "verdict-real"
            : v === "Fake" ? "verdict-fake"
            : "verdict-mixed";
  verdict.textContent = `Gemini AI: ${v} — ${(ga.confidence * 100).toFixed(0)}%`;
  verdict.className = `fact-verdict ${cls}`;

  let html = `<div class="fact-claim">
    <div class="fact-claim-text">Credibility: ${ga.credibility_score}/100 · Model: ${esc(ga.model_used || "N/A")}</div>
    ${ga.reasoning ? `<div class="fact-claim-rating">${esc(ga.reasoning)}</div>` : ""}
  </div>`;

  if (ga.red_flags && ga.red_flags.length > 0) {
    html += `<div class="fact-claim">
      <div class="fact-claim-text">🚩 Red Flags</div>
      ${ga.red_flags.map(f => `<div class="fact-claim-rating">• ${esc(f)}</div>`).join("")}
    </div>`;
  }

  details.innerHTML = html;
}

function showFactCheck(fc) {
  const container = $("factCheck");
  const verdict = $("factVerdict");
  const claims = $("factClaims");
  const unavail = $("factUnavailable");

  claims.innerHTML = "";
  unavail.style.display = "none";

  if (!fc) { container.style.display = "none"; return; }

  container.style.display = "block";

  if (!fc.available) {
    verdict.textContent = "Fact Check Unavailable";
    verdict.className = "fact-verdict verdict-none";
    unavail.style.display = "block";
    return;
  }

  if (fc.error) {
    verdict.textContent = "Fact Check Error";
    verdict.className = "fact-verdict verdict-mixed";
    claims.innerHTML = `<div class="fact-claim"><span class="fact-claim-rating">${fc.error}</span></div>`;
    return;
  }

  const v = fc.verdict || "No fact-checks found";
  const cls = v.includes("Real") ? "verdict-real"
            : v.includes("Fake") ? "verdict-fake"
            : v.includes("Mixed") ? "verdict-mixed"
            : "verdict-none";
  verdict.textContent = `Fact Check: ${v}`;
  verdict.className = `fact-verdict ${cls}`;

  if (fc.claims && fc.claims.length > 0) {
    fc.claims.forEach((c) => {
      const div = document.createElement("div");
      div.className = "fact-claim";
      div.innerHTML =
        `<div class="fact-claim-text">${esc(c.claim_text).substring(0, 100)}</div>` +
        `<div class="fact-claim-rating">Rating: ${esc(c.rating)} — ${esc(c.publisher)}</div>` +
        (c.url ? `<a class="fact-claim-link" href="${esc(c.url)}" target="_blank">View review ↗</a>` : "");
      claims.appendChild(div);
    });
  }
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s || "";
  return d.innerHTML;
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  if (loading) btn.dataset.origText = btn.textContent;
  btn.textContent = loading ? "⏳ Analysing…" : btn.dataset.origText;
}

// ── Analyse current page ────────────────────────────────────────────
$("analysePageBtn").addEventListener("click", async () => {
  const btn = $("analysePageBtn");
  setLoading(btn, true);

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // Try URL-based prediction first
    const resp = await fetch(getApiUrl() + "/api/predict-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: tab.url }),
    });

    if (!resp.ok) {
      // Fallback: extract text from the page using content script
      const [{ result: pageText }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => document.body.innerText.substring(0, 5000),
      });

      if (!pageText || pageText.trim().length < 20) {
        throw new Error("Could not extract enough text from this page.");
      }

      const resp2 = await fetch(getApiUrl() + "/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: pageText }),
      });
      const data2 = await resp2.json();
      if (data2.error) throw new Error(data2.error);
      showResult(data2);
    } else {
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      showResult(data);
    }
  } catch (err) {
    showError(err.message || "Failed to connect to API. Is the server running?");
  } finally {
    setLoading(btn, false);
  }
});

// ── Analyse manual text ─────────────────────────────────────────────
$("analyseTextBtn").addEventListener("click", async () => {
  const btn = $("analyseTextBtn");
  const text = $("manualText").value.trim();
  if (!text) {
    showError("Please enter some text to analyse.");
    return;
  }
  setLoading(btn, true);
  try {
    const resp = await fetch(getApiUrl() + "/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    showResult(data);
  } catch (err) {
    showError(err.message || "Failed to connect to API.");
  } finally {
    setLoading(btn, false);
  }
});
