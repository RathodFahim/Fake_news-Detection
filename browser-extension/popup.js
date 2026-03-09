/* Fake News Detector — Extension popup logic */

const $ = (id) => document.getElementById(id);

function getApiUrl() {
  return ($("apiUrl").value || "http://localhost:5000").replace(/\/+$/, "");
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

  showFactCheck(data.fact_check);
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").style.display = "block";
  $("result").classList.remove("show");
  $("factCheck").style.display = "none";
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
