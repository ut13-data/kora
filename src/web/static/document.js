"use strict";
(async function () {
  const root = document.getElementById("doc");
  const id = root.dataset.document;
  const cited = new URLSearchParams(location.search).get("chunk");
  let d;
  try { d = await K.api("/api/documents/" + encodeURIComponent(id)); }
  catch (e) {
    K.clear(root).append(K.el("h1", { text: "Document not available" }),
      K.el("p", { class: "muted", style: "margin-top:.5rem", text: "It doesn't exist, or it isn't shared with you." }),
      K.el("p", { style: "margin-top:1rem" }, K.el("a", { href: "/ask", text: "Back to Ask" })));
    return;
  }
  const current = d.versions.find((v) => (v.status || "").toLowerCase() === "active") || d.versions[0] || {};
  const older = d.versions.filter((v) => v !== current);
  K.clear(root).append(
    K.el("h1", { text: current.title || id }),
    K.el("p", { class: "doc-meta", text: [id, current.version && "version " + current.version, current.status,
      current.effective_date && "effective " + current.effective_date, current.owner && "owner: " + current.owner].filter(Boolean).join(", ") }),
    older.length ? K.el("p", { class: "small muted", style: "margin-top:-1rem;margin-bottom:1.5rem",
      text: "Also kept for history: " + older.map((v) => `version ${v.version} (${v.status}, ${v.effective_date})`).join("; ") }) : null,
    K.el("div", { class: "doc-passages" }, d.chunks.map((c) => K.el("section", {
      class: "doc-passage" + (c.chunk_id === cited ? " cited" : ""), id: c.chunk_id },
      K.el("p", { class: "passage-section", text: (c.version !== current.version ? `Version ${c.version}: ` : "") + (c.section || "") }),
      K.el("div", { class: "passage", text: K.readable(c.text) })))));
  if (cited) { const node = document.getElementById(cited); if (node) node.scrollIntoView({ block: "start" }); }
})();
