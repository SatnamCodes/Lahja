// Lahja frontend - talks to the FastAPI backend mounted at the same origin.

function confidenceTier(confidence) {
  if (confidence >= 0.6) return "high";
  if (confidence >= 0.3) return "mid";
  return "low";
}

function badgeHTML(confidence, method) {
  const tier = confidenceTier(confidence);
  return `<span class="badge confidence-${tier}"><span class="dot"></span>${method}</span>` +
         `<span class="badge">confidence ${confidence.toFixed(2)}</span>`;
}

function setLoading(form, loading) {
  const btn = form.querySelector(".btn");
  btn.disabled = loading;
  btn.classList.toggle("loading", loading);
}

async function callAPI(path, { method = "POST", body, isForm = false } = {}) {
  const opts = { method };
  if (isForm) {
    opts.body = body;
  } else {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

function showResult(box) {
  box.hidden = false;
}

function showError(box, statusEl, message) {
  box.hidden = false;
  statusEl.innerHTML = "";
  statusEl.insertAdjacentHTML("beforeend", `<span class="error-text">${message}</span>`);
}

// ---------- tabs ----------

const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
    panels.forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

// ---------- speak ----------

const formSpeak = document.getElementById("form-speak");
formSpeak.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = new FormData(formSpeak).get("text").trim();
  if (!text) return;
  const resultBox = document.getElementById("result-speak");
  const statusEl = resultBox.querySelector(".result-status");
  const audioEl = document.getElementById("audio-speak");
  setLoading(formSpeak, true);
  try {
    const data = await callAPI("/api/speak", { body: { text, language: "trp" } });
    statusEl.innerHTML = badgeHTML(data.confidence, data.method);
    audioEl.src = data.audio_url;
    audioEl.hidden = false;
    showResult(resultBox);
    audioEl.play().catch(() => {});
  } catch (err) {
    showError(resultBox, statusEl, err.message);
  } finally {
    setLoading(formSpeak, false);
  }
});

// ---------- translate ----------

const formTranslate = document.getElementById("form-translate");
let translateDir = { src: "trp", tgt: "eng" };

document.querySelectorAll(".dir-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".dir-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    translateDir = { src: btn.dataset.src, tgt: btn.dataset.tgt };
  });
});

formTranslate.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = new FormData(formTranslate).get("text").trim();
  if (!text) return;
  const resultBox = document.getElementById("result-translate");
  const statusEl = resultBox.querySelector(".result-status");
  const textEl = document.getElementById("text-translate");
  setLoading(formTranslate, true);
  try {
    const data = await callAPI("/api/translate", {
      body: { text, source_language: translateDir.src, target_language: translateDir.tgt },
    });
    statusEl.innerHTML = badgeHTML(data.confidence, data.method);
    textEl.textContent = data.translated_text;
    showResult(resultBox);
  } catch (err) {
    showError(resultBox, statusEl, err.message);
  } finally {
    setLoading(formTranslate, false);
  }
});

// ---------- chat ----------

const formChat = document.getElementById("form-chat");
formChat.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = new FormData(formChat).get("text").trim();
  if (!text) return;
  const resultBox = document.getElementById("result-chat");
  const statusEl = resultBox.querySelector(".result-status");
  const answerEl = document.getElementById("text-chat-answer");
  const bridgeEl = document.getElementById("text-chat-bridge");
  setLoading(formChat, true);
  try {
    const data = await callAPI("/api/chat", { body: { text } });
    statusEl.innerHTML = badgeHTML(data.confidence, data.method);
    answerEl.textContent = data.answer;
    bridgeEl.textContent = data.english_bridge;
    showResult(resultBox);
  } catch (err) {
    showError(resultBox, statusEl, err.message);
  } finally {
    setLoading(formChat, false);
  }
});

// ---------- transcribe ----------

const formTranscribe = document.getElementById("form-transcribe");
const fileDrop = document.getElementById("file-drop");
const fileInput = fileDrop.querySelector("input");
const fileDropLabel = fileDrop.querySelector(".file-drop-label");

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    fileDropLabel.textContent = fileInput.files[0].name;
  }
});

formTranscribe.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  const resultBox = document.getElementById("result-transcribe");
  const statusEl = resultBox.querySelector(".result-status");
  const textEl = document.getElementById("text-transcribe");
  const noteEl = document.getElementById("note-transcribe");
  const body = new FormData();
  body.append("audio", file);
  setLoading(formTranscribe, true);
  try {
    const data = await callAPI("/api/transcribe", { body, isForm: true });
    statusEl.innerHTML = badgeHTML(data.confidence, data.method);
    textEl.textContent = data.text;
    noteEl.hidden = data.method !== "phoneme_zero_shot_bridge";
    showResult(resultBox);
  } catch (err) {
    showError(resultBox, statusEl, err.message);
  } finally {
    setLoading(formTranscribe, false);
  }
});
