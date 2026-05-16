/* main.js — shared utilities */

// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(msg, type = "success") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ── Loading overlay ───────────────────────────────────────────────────────────
function showLoading(text = "Processing...") {
  let ov = document.getElementById("loading-overlay");
  if (!ov) {
    ov = document.createElement("div");
    ov.id = "loading-overlay";
    ov.className = "loading-overlay";
    ov.innerHTML = `<div class="loading-spinner"></div>
                    <div class="loading-text" id="loading-text">${text}</div>`;
    document.body.appendChild(ov);
  }
  document.getElementById("loading-text").textContent = text;
  ov.classList.add("show");
}
function hideLoading() {
  const ov = document.getElementById("loading-overlay");
  if (ov) ov.classList.remove("show");
}

// ── Active nav link ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname;
  document.querySelectorAll(".sb-link").forEach(a => {
    a.classList.remove("active");
    const href = a.getAttribute("href");
    if (
      (href === "/" && path === "/") ||
      (href !== "/" && path.startsWith(href))
    ) {
      a.classList.add("active");
    }
  });
});

// ── Format confidence color ───────────────────────────────────────────────────
function sentimentColor(label) {
  return { Positive: "#10b981", Negative: "#ef4444", Neutral: "#f59e0b" }[label] || "#1d4ed8";
}
function sentimentBg(label) {
  return { Positive: "#f0fdf4", Negative: "#fef2f2", Neutral: "#fffbeb" }[label] || "#f8fafc";
}
function sentimentBorder(label) {
  return { Positive: "#6ee7b7", Negative: "#fca5a5", Neutral: "#fcd34d" }[label] || "#e2e8f0";
}

// ── Download helper ───────────────────────────────────────────────────────────
function downloadText(content, filename, mime = "text/plain") {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
function downloadCSV(content, filename) {
  downloadText(content, filename, "text/csv;charset=utf-8;");
}
