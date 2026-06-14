/**
 * Tech Stack Recommender — Frontend Controller
 * Handles skill input, autocomplete, presets, API calls, and result rendering.
 */
(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────
  let selectedSkills = [];
  let corpusSkills   = [];
  let acIndex        = -1;

  // ── DOM ────────────────────────────────────────────────────────
  const input         = document.getElementById("skill-input");
  const addBtn        = document.getElementById("add-btn");
  const suggestList   = document.getElementById("suggest-list");
  const chipsArea     = document.getElementById("chips-area");
  const chipsEmpty    = document.getElementById("chips-empty");
  const skillCount    = document.getElementById("skill-count");
  const minPill       = document.getElementById("min-pill");
  const recommendBtn  = document.getElementById("recommend-btn");
  const clearBtn      = document.getElementById("clear-btn");
  const resultsBox    = document.getElementById("results-container");
  const emptyState    = document.getElementById("empty-state");
  const explorerBtn   = document.getElementById("explorer-btn");
  const explorerBody  = document.getElementById("explorer-content");
  const roleGrid      = document.getElementById("role-grid");
  const statSkills    = document.getElementById("stat-skills");
  const statRoles     = document.getElementById("stat-roles");

  // ══════════════════════════════════════════════════════════════
  // INIT
  // ══════════════════════════════════════════════════════════════

  async function init() {
    try {
      const r1 = await fetch("/api/skills");
      const d1 = await r1.json();
      corpusSkills = d1.skills || [];
      statSkills.textContent = corpusSkills.length;
    } catch (_) {}

    try {
      const r2 = await fetch("/api/roles");
      const d2 = await r2.json();
      statRoles.textContent = d2.total || 20;
      buildExplorer(d2.roles || []);
    } catch (_) {}

    bind();
  }

  // ══════════════════════════════════════════════════════════════
  // EVENTS
  // ══════════════════════════════════════════════════════════════

  function bind() {
    input.addEventListener("input", onType);
    input.addEventListener("keydown", onKey);
    input.addEventListener("focus", () => { if (input.value.trim()) onType(); });
    document.addEventListener("click", (e) => {
      if (!e.target.closest("#input-group")) closeSuggest();
    });
    addBtn.addEventListener("click", () => commitInput());
    recommendBtn.addEventListener("click", fetchRecs);
    clearBtn.addEventListener("click", resetAll);

    document.querySelectorAll(".preset-chip").forEach((b) => {
      b.addEventListener("click", () => {
        loadPreset(b.dataset.skills.split(",").map((s) => s.trim()));
      });
    });

    explorerBtn.addEventListener("click", toggleExplorer);
  }

  // ══════════════════════════════════════════════════════════════
  // AUTOCOMPLETE
  // ══════════════════════════════════════════════════════════════

  function onType() {
    const q = input.value.trim().toLowerCase();
    acIndex = -1;
    if (!q) { closeSuggest(); return; }

    const hits = corpusSkills
      .filter((s) => s.toLowerCase().includes(q) && !isDup(s))
      .slice(0, 8);

    if (!hits.length) { closeSuggest(); return; }

    suggestList.innerHTML = hits
      .map((s, i) => `<div class="suggest-item" data-i="${i}" data-val="${s}">${hl(s, q)}</div>`)
      .join("");

    suggestList.classList.add("open");

    suggestList.querySelectorAll(".suggest-item").forEach((el) => {
      el.addEventListener("click", () => {
        addSkill(el.dataset.val);
        input.value = "";
        closeSuggest();
        input.focus();
      });
    });
  }

  function hl(str, q) {
    const i = str.toLowerCase().indexOf(q);
    if (i < 0) return esc(str);
    return esc(str.slice(0, i)) + `<span class="hl">${esc(str.slice(i, i + q.length))}</span>` + esc(str.slice(i + q.length));
  }

  function onKey(e) {
    const items = suggestList.querySelectorAll(".suggest-item");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      acIndex = Math.min(acIndex + 1, items.length - 1);
      markActive(items);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      acIndex = Math.max(acIndex - 1, 0);
      markActive(items);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (acIndex >= 0 && items[acIndex]) {
        addSkill(items[acIndex].dataset.val);
        input.value = "";
        closeSuggest();
      } else {
        commitInput();
      }
    } else if (e.key === "Escape") {
      closeSuggest();
    }
  }

  function markActive(items) {
    items.forEach((el, i) => el.classList.toggle("focused", i === acIndex));
    if (items[acIndex]) items[acIndex].scrollIntoView({ block: "nearest" });
  }

  function closeSuggest() {
    suggestList.classList.remove("open");
    suggestList.innerHTML = "";
    acIndex = -1;
  }

  // ══════════════════════════════════════════════════════════════
  // SKILL MANAGEMENT
  // ══════════════════════════════════════════════════════════════

  function commitInput() {
    const v = input.value.trim();
    if (v) { addSkill(v); input.value = ""; closeSuggest(); input.focus(); }
  }

  function isDup(s) {
    return selectedSkills.some((x) => x.toLowerCase() === s.toLowerCase());
  }

  function addSkill(s) {
    s = s.trim();
    if (!s || isDup(s)) { shakeInput(); return; }
    selectedSkills.push(s);
    renderChips();
    syncUI();
  }

  function removeSkill(i) {
    selectedSkills.splice(i, 1);
    renderChips();
    syncUI();
  }

  function loadPreset(arr) {
    selectedSkills = [...arr];
    renderChips();
    syncUI();
    fetchRecs();
  }

  function resetAll() {
    selectedSkills = [];
    input.value = "";
    renderChips();
    syncUI();
    resultsBox.innerHTML = "";
    resultsBox.appendChild(emptyState);
    emptyState.style.display = "";
  }

  function shakeInput() {
    input.style.animation = "shake 0.3s ease";
    input.addEventListener("animationend", () => { input.style.animation = ""; }, { once: true });
  }

  // ══════════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════════

  function renderChips() {
    chipsArea.querySelectorAll(".chip").forEach((c) => c.remove());
    if (!selectedSkills.length) {
      chipsEmpty.style.display = "";
    } else {
      chipsEmpty.style.display = "none";
      selectedSkills.forEach((s, i) => {
        const el = document.createElement("span");
        el.className = "chip";
        el.innerHTML = `${esc(s)} <button class="chip-x" data-i="${i}">&times;</button>`;
        el.querySelector(".chip-x").addEventListener("click", () => removeSkill(i));
        chipsArea.appendChild(el);
      });
    }
  }

  function syncUI() {
    const n = selectedSkills.length;
    skillCount.textContent = n;
    if (n >= 3) {
      recommendBtn.disabled = false;
      minPill.textContent = "Ready!";
      minPill.className = "min-pill ok";
    } else {
      recommendBtn.disabled = true;
      minPill.textContent = "Min 3 required";
      minPill.className = "min-pill needs";
    }
  }

  // ══════════════════════════════════════════════════════════════
  // API
  // ══════════════════════════════════════════════════════════════

  async function fetchRecs() {
    if (selectedSkills.length < 3) return;
    recommendBtn.classList.add("running");
    recommendBtn.disabled = true;

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ skills: selectedSkills, top_n: 3 }),
      });
      const data = await res.json();

      if (!data.success) { showError(data.error || "Unknown error."); return; }
      if (data.cold_start) { showColdStart(data.message); return; }

      showResults(data.recommendations, data.user_skills);
    } catch (_) {
      showError("Cannot reach the server. Is Flask running?");
    } finally {
      recommendBtn.classList.remove("running");
      syncUI();
    }
  }

  // ══════════════════════════════════════════════════════════════
  // RESULTS
  // ══════════════════════════════════════════════════════════════

  function showResults(recs, userSkills) {
    const uLow = userSkills.map((s) => s.toLowerCase());

    let h = '<div class="result-stack">';
    recs.forEach((r) => {
      const skills = r.skills.split(",").map((s) => s.trim());
      const chips = skills.map((s) => {
        const matched = uLow.some((u) => s.toLowerCase().includes(u) || u.includes(s.toLowerCase()));
        return `<span class="res-chip${matched ? " hit" : ""}">${esc(s)}</span>`;
      }).join("");

      const pct = (r.score * 100).toFixed(1);

      h += `
        <div class="res-card">
          <div class="res-header">
            <div class="res-rank-badge">#${r.rank}</div>
            <div class="res-info">
              <div class="res-role">${esc(r.role)}</div>
              <div class="res-sub">Cosine similarity match</div>
            </div>
            <div class="res-score">
              <div class="score-num">${pct}%</div>
              <div class="score-label">Match</div>
            </div>
          </div>
          <div class="track"><div class="track-fill" data-w="${pct}"></div></div>
          <div class="res-chips">${chips}</div>
        </div>`;
    });
    h += "</div>";

    resultsBox.innerHTML = h;

    requestAnimationFrame(() => {
      setTimeout(() => {
        document.querySelectorAll(".track-fill").forEach((b) => { b.style.width = b.dataset.w + "%"; });
      }, 60);
    });
  }

  function showColdStart(msg) {
    resultsBox.innerHTML = `
      <div class="warn-card cold">
        <div class="warn-glyph">&#9888;</div>
        <h3>Cold Start Detected</h3>
        <p>${esc(msg)}</p>
      </div>`;
  }

  function showError(msg) {
    resultsBox.innerHTML = `
      <div class="warn-card error">
        <div class="warn-glyph">&#10060;</div>
        <h3>Something Went Wrong</h3>
        <p>${esc(msg)}</p>
      </div>`;
  }

  // ══════════════════════════════════════════════════════════════
  // EXPLORER
  // ══════════════════════════════════════════════════════════════

  function toggleExplorer() {
    explorerBtn.classList.toggle("expanded");
    explorerBody.classList.toggle("show");
  }

  function buildExplorer(roles) {
    roleGrid.innerHTML = roles.map((r) => {
      const chips = r.skills.map((s) => `<span class="tiny-chip">${esc(s)}</span>`).join("");
      return `<div class="role-tile"><div class="role-tile-name">${esc(r.role)}</div><div class="role-tile-chips">${chips}</div></div>`;
    }).join("");
  }

  // ══════════════════════════════════════════════════════════════
  // UTIL
  // ══════════════════════════════════════════════════════════════

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  document.addEventListener("DOMContentLoaded", init);
})();
