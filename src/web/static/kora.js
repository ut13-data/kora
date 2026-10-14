/* KORA shared helpers. Every piece of text from the server is inserted with
   textContent, never innerHTML, so document text can't run as code. */
"use strict";

const K = {};

K.api = async function (path, options = {}) {
  const init = { credentials: "same-origin", headers: {}, ...options };
  if (options.json !== undefined) {
    init.method = options.method || "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.json);
  }
  const res = await fetch(path, init);
  if (res.status === 401 && !path.startsWith("/api/login")) {
    window.location.href = "/login";
    throw new Error("Signed out");
  }
  let body = null;
  try { body = await res.json(); } catch (_) { body = null; }
  if (!res.ok) {
    const err = new Error((body && (body.detail || (body.problems || []).join("; "))) || res.statusText);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
};

/* el("p", {class: "x", text: "..."}, child, child) */
K.el = function (tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
};

K.clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); return node; };

/* What a status means, in plain words. */
K.STATUS = {
  answered: ["Documented", "Every line below is quoted from Nexora's documents or records."],
  partial: ["Partly documented", "Part of this is in the documents. The notes say what is missing."],
  conflict: ["Sources disagree", "The documents say different things. Both sides are shown; KORA doesn't pick one."],
  not_found: ["Not in the knowledge base", "KORA couldn't find this in Nexora's documents."],
};

/* Wrap the part of TEXT that matches QUOTE in a highlighter <mark>, matching on
   letters and digits only (line breaks and **bold** in the source don't matter). */
/* Markdown source -> readable text: no **, no heading hashes, bullets as dots. */
K.readable = (text) => text.replace(/\*\*/g, "").replace(/^#{1,6}\s*/gm, "").replace(/^(\s*)[-*]\s+/gm, "$1\u2022 ");

K.highlighted = function (text, quote) {
  const frag = document.createDocumentFragment();
  const clean = K.readable(text);
  if (!quote) { frag.append(clean); return frag; }
  const keep = [], idx = [];
  for (let i = 0; i < clean.length; i++) {
    const ch = clean[i].toLowerCase();
    if (/[a-z0-9]/.test(ch)) { keep.push(ch); idx.push(i); }
  }
  const needle = quote.toLowerCase().replace(/[^a-z0-9]/g, "");
  const at = needle ? keep.join("").indexOf(needle) : -1;
  if (at < 0) { frag.append(clean); return frag; }
  const start = idx[at], end = idx[at + needle.length - 1] + 1;
  frag.append(clean.slice(0, start), K.el("mark", { class: "hl", text: clean.slice(start, end) }), clean.slice(end));
  return frag;
};

/* ---------- the source panel ---------- */

K.source = {
  panel: () => document.getElementById("source-panel"),
  lastTrigger: null,

  close() {
    const p = K.source.panel();
    if (!p) return;
    p.hidden = true;
    document.querySelectorAll(".cite[aria-expanded='true']").forEach((b) => b.setAttribute("aria-expanded", "false"));
    if (K.source.lastTrigger) K.source.lastTrigger.focus();
  },

  async open(citation, quote, trigger) {
    const p = K.source.panel();
    if (!p) return;
    K.source.lastTrigger = trigger || null;
    document.querySelectorAll(".cite[aria-expanded='true']").forEach((b) => b.setAttribute("aria-expanded", "false"));
    if (trigger) trigger.setAttribute("aria-expanded", "true");
    const title = document.getElementById("source-title");
    const body = K.clear(document.getElementById("source-body"));
    title.textContent = citation.title || "Source";
    body.append(K.el("p", { class: "muted", text: "Opening the source..." }));
    p.hidden = false;
    document.getElementById("source-close").focus();
    try {
      if (citation.kind === "record") await K.source.record(citation, body);
      else if (citation.kind === "header" || citation.kind === "file") await K.source.header(citation, quote, body);
      else await K.source.chunk(citation, quote, body);
    } catch (e) {
      K.clear(body).append(K.el("p", { class: "error", text: e.status === 404 ? "This source isn't available to you." : "The source couldn't be opened: " + e.message }));
    }
  },

  async chunk(c, quote, body) {
    const d = await K.api("/api/chunks/" + encodeURIComponent(c.chunk_id));
    K.clear(body).append(
      K.el("p", { class: "source-meta", text: [d.document_id, d.version && "version " + d.version, d.status, d.effective_date && "effective " + d.effective_date].filter(Boolean).join(", ") }),
      d.section ? K.el("p", { class: "passage-section", text: d.section }) : null,
      K.el("div", { class: "passage" }, K.highlighted(d.text, quote)),
      K.el("p", { class: "small", style: "margin-top:1rem" },
        K.el("a", { href: "/documents/" + encodeURIComponent(d.document_id) + "?chunk=" + encodeURIComponent(d.chunk_id), text: "Open the whole document" })),
    );
    const m = body.querySelector("mark");
    if (m) m.scrollIntoView({ block: "center" });
  },

  async header(c, quote, body) {
    const d = await K.api("/api/citations?id=" + encodeURIComponent(c.chunk_id));
    const list = K.el("div", { class: "passage" });
    const key = (quote || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    for (const line of d.header || []) {
      const plain = line.replace(/\*\*/g, "");
      const hit = key && plain.toLowerCase().replace(/[^a-z0-9]/g, "") === key;
      list.append(hit ? K.el("mark", { class: "hl", text: plain }) : plain, "\n");
    }
    K.clear(body).append(
      K.el("p", { class: "source-meta", text: "Document header of " + (d.document_id || d.file) }),
      list,
      d.document_id ? K.el("p", { class: "small", style: "margin-top:1rem" },
        K.el("a", { href: "/documents/" + encodeURIComponent(d.document_id), text: "Open the whole document" })) : null,
    );
  },

  async record(c, body) {
    const d = await K.api("/api/citations?id=" + encodeURIComponent(c.chunk_id));
    const table = K.el("table", { class: "record-table" });
    for (const [col, val] of Object.entries(d.row || {})) {
      table.append(K.el("tr", { class: col === d.column ? "hl" : null }, K.el("th", { text: col }), K.el("td", { text: val ?? "" })));
    }
    K.clear(body).append(
      K.el("p", { class: "source-meta", text: "Company record: " + d.file + ", row " + d.key }),
      table,
      K.el("p", { class: "small muted", style: "margin-top:1rem", text: "Records come from the company's own tables. Only the columns you may see are shown." }),
    );
  },
};

document.addEventListener("DOMContentLoaded", () => {
  const close = document.getElementById("source-close");
  if (close) close.addEventListener("click", K.source.close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && K.source.panel() && !K.source.panel().hidden) K.source.close(); });
  const out = document.getElementById("sign-out");
  if (out) out.addEventListener("click", async () => {
    await K.api("/api/logout", { method: "POST" });
    window.location.href = "/login";
  });
});
