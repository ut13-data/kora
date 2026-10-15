"use strict";
(function () {
  const root = document.getElementById("dash");
  const tip = document.getElementById("tooltip");
  const NS = "http://www.w3.org/2000/svg";
  const STATUS = [                      // stack order = validated adjacent order (see kora.css --st-*)
    ["answered", "Documented"], ["partial", "Partly documented"], ["not_found", "Not found"], ["conflict", "Sources disagree"]];
  const CATEGORY = { approval: "Approvals", escalation: "Escalations", conflict: "Conflicts", gap: "Known gaps",
    version: "Versions and history", document_list: "Which documents", relocation: "Germany relocation",
    communication: "Communication and tone", general: "Other questions" };
  const nice = (s) => CATEGORY[s] || s.replace(/^relationship \(graph\): /, "Relationships: ").replace(/_/g, " ");
  const pct = (x) => (x === null || x === undefined ? "-" : Math.round(x * 100) + "%");
  const ms = (x) => (x === null || x === undefined ? "-" : (x >= 100 ? Math.round(x) : x.toFixed(1)) + " ms");

  /* ---------- tooltip (hover and keyboard focus) ---------- */
  function tipOn(node, text) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("aria-label", text);
    const show = (e) => {
      tip.textContent = text; tip.hidden = false;
      const r = node.getBoundingClientRect();
      const x = e && e.clientX ? e.clientX : r.left + r.width / 2;
      const y = e && e.clientY ? e.clientY : r.top;
      tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, x + 12)) + "px";
      tip.style.top = Math.max(8, y - tip.offsetHeight - 10) + "px";
    };
    node.addEventListener("mousemove", show);
    node.addEventListener("focus", () => show(null));
    node.addEventListener("mouseleave", () => (tip.hidden = true));
    node.addEventListener("blur", () => (tip.hidden = true));
  }
  const svgEl = (tag, attrs = {}) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };

  /* ---------- building blocks ---------- */
  const section = (id, title, definition, ...children) => K.el("section", { class: "dash-section", "aria-labelledby": id },
    K.el("h2", { id, text: title }), definition ? K.el("p", { class: "definition", text: definition }) : null, ...children);
  const tiles = (items) => K.el("div", { class: "tiles" }, items.map(([value, label, note]) =>
    K.el("div", { class: "tile" }, K.el("p", { class: "tile-value", text: value }), K.el("p", { class: "tile-label", text: label }),
      note ? K.el("p", { class: "tile-note", text: note }) : null)));
  const empty = (text) => K.el("p", { class: "empty", text });
  function table(headers, rows, caption) {
    return K.el("table", { class: "dash-table" }, caption ? K.el("caption", { class: "skip", text: caption }) : null,
      K.el("thead", {}, K.el("tr", {}, headers.map((h) => K.el("th", { scope: "col", text: h })))),
      K.el("tbody", {}, rows.map((r) => K.el("tr", {}, r.map((c) => K.el("td", { text: c ?? "", class: String(c ?? "").length > 24 ? "wrap" : null }))))));
  }
  const asTable = (headers, rows, caption) => K.el("details", { class: "table-view" }, K.el("summary", { text: "Show as a table" }), table(headers, rows, caption));

  /* horizontal bars: one series, one colour; the value is written at the end of each bar */
  function hbars(rows, unit, caption) {
    if (!rows.length) return empty("Nothing yet.");
    const max = Math.max(...rows.map((r) => r.value), 1);
    const W = 640, row = 30, labelW = 260, H = rows.length * row + 4;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": caption });
    rows.forEach((r, i) => {
      const y = i * row + 4, w = Math.max(2, ((W - labelW - 70) * r.value) / max);
      const label = svgEl("text", { x: labelW - 10, y: y + 16, "text-anchor": "end", class: "axis-label" });
      label.textContent = r.label.length > 38 ? r.label.slice(0, 37) + "..." : r.label;
      const bar = svgEl("path", { d: `M${labelW},${y + 4} h${w - 4} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h${-(w - 4)} z`, class: "bar" });
      const value = svgEl("text", { x: labelW + w + 6, y: y + 16, class: "value-label" });
      value.textContent = r.value + (unit || "");
      svg.append(label, bar, value);
      tipOn(bar, `${r.label}: ${r.value}${unit || ""}${r.note ? " (" + r.note + ")" : ""}`);
    });
    return svg;
  }

  /* answers per day, stacked by status, with 2px gaps between segments */
  function statusByDay(days) {
    const W = 760, H = 220, left = 36, bottom = 26, top = 8;
    const max = Math.max(...days.map((d) => STATUS.reduce((s, [k]) => s + d[k], 0)), 1);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img",
      "aria-label": "Answers per day by status. The table view below lists the numbers." });
    const plotH = H - bottom - top, step = (W - left) / days.length, bw = Math.min(28, step * 0.6);
    for (const g of [0, 0.5, 1]) {
      const y = top + plotH * (1 - g);
      svg.append(svgEl("line", { x1: left, x2: W, y1: y, y2: y, class: "grid" }));
      const t = svgEl("text", { x: left - 6, y: y + 4, "text-anchor": "end", class: "axis-label" }); t.textContent = Math.round(max * g);
      svg.append(t);
    }
    days.forEach((d, i) => {
      const x = left + i * step + (step - bw) / 2;
      let y = top + plotH;
      for (const [k, name] of STATUS) {
        if (!d[k]) continue;
        const h = (plotH * d[k]) / max;
        const seg = svgEl("rect", { x, y: y - h + 1, width: bw, height: Math.max(1, h - 2), rx: 3, class: "seg st-" + k });
        tipOn(seg, `${d.day}: ${d[k]} ${name.toLowerCase()}`);
        svg.append(seg);
        y -= h;
      }
      if (days.length <= 14 || i % Math.ceil(days.length / 10) === 0) {
        const t = svgEl("text", { x: x + bw / 2, y: H - 8, "text-anchor": "middle", class: "axis-label" }); t.textContent = d.day.slice(5);
        svg.append(t);
      }
    });
    return svg;
  }

  /* median and 95th percentile per day: one axis, one hue, p95 dashed */
  function latencyLine(days) {
    const W = 760, H = 200, left = 46, bottom = 26, top = 10;
    const max = Math.max(...days.map((d) => d.p95 || 0), 1);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img",
      "aria-label": "Answer time per day, median and 95th percentile. The table view below lists the numbers." });
    const plotH = H - bottom - top, x = (i) => left + (days.length === 1 ? (W - left) / 2 : (i * (W - left - 10)) / (days.length - 1));
    const y = (v) => top + plotH * (1 - v / max);
    for (const g of [0, 0.5, 1]) {
      svg.append(svgEl("line", { x1: left, x2: W, y1: y(max * g), y2: y(max * g), class: "grid" }));
      const t = svgEl("text", { x: left - 6, y: y(max * g) + 4, "text-anchor": "end", class: "axis-label" }); t.textContent = Math.round(max * g);
      svg.append(t);
    }
    for (const [key, cls] of [["p95", "line p95"], ["p50", "line p50"]]) {
      if (days.length > 1) svg.append(svgEl("path", { d: days.map((d, i) => `${i ? "L" : "M"}${x(i)},${y(d[key])}`).join(" "), class: cls }));
      days.forEach((d, i) => {
        const dot = svgEl("circle", { cx: x(i), cy: y(d[key]), r: 4.5, class: "dot " + key });
        tipOn(dot, `${d.day}: ${key === "p50" ? "median" : "95th percentile"} ${ms(d[key])}, ${d.answers} answers`);
        svg.append(dot);
      });
    }
    days.forEach((d, i) => {
      if (days.length <= 14 || i % Math.ceil(days.length / 10) === 0) {
        const t = svgEl("text", { x: x(i), y: H - 8, "text-anchor": "middle", class: "axis-label" }); t.textContent = d.day.slice(5);
        svg.append(t);
      }
    });
    return svg;
  }

  const legend = (items) => K.el("ul", { class: "legend" }, items.map(([cls, label]) =>
    K.el("li", {}, K.el("span", { class: "swatch " + cls, "aria-hidden": "true" }), label)));

  /* ---------- the page ---------- */
  function render(d) {
    const L = d.log;
    const when = L.first ? ` from ${String(L.first).slice(0, 10)} to ${String(L.last).slice(0, 10)}` : "";
    document.getElementById("log-line").textContent = L.missing_file ? "No answer log yet: ask KORA something first."
      : `${d.answers} answers${when}. The log has ${L.lines} lines; ${L.skipped_total} could not be read` +
        (L.skipped_total ? ` (${Object.entries(L.skipped).map(([k, v]) => `${v} ${k}`).join(", ")})` : "") +
        `; ${L.repeats} exact repeats counted once.`;
    const D = d.definitions;
    const out = [];

    /* A. answer quality */
    const s = d.status;
    out.push(section("h-quality", "How KORA answered", D.status_share,
      d.answers ? tiles([[pct(s.share.answered), "Documented"], [pct(s.share.partial), "Partly documented"],
        [pct(s.share.not_found), "Not found", "an honest refusal"], [pct(s.share.conflict), "Sources disagree"],
        [String(d.errors.count), "Errors", "crashes, counted separately"]]) : empty("No answers in this period."),
      s.per_day.length > 1 ? K.el("div", {}, legend(STATUS.map(([k, n]) => ["st-" + k, n])), statusByDay(s.per_day),
        asTable(["Day", ...STATUS.map(([, n]) => n)], s.per_day.map((r) => [r.day, ...STATUS.map(([k]) => String(r[k]))]), "Answers per day by status")) : null,
      d.errors.count ? K.el("div", {}, K.el("h3", { text: "Errors" }), K.el("p", { class: "definition", text: D.errors }),
        table(["When (UTC)", "Step", "Error"], d.errors.recent.map((e) => [String(e.at).slice(0, 16).replace("T", " "), e.stage, e.type]))) : null));

    /* B. gaps */
    const g = d.gaps;
    out.push(section("h-gaps", "What KORA couldn't answer", null,
      K.el("h3", { text: "Words the documents never use" }), K.el("p", { class: "definition", text: D.unknown_words }),
      g.unknown_words.length ? hbars(g.unknown_words.map((w) => ({ label: w.word, value: w.times, note: `${w.people} ${w.people === 1 ? "person" : "people"}` })), "x", "Unknown words by number of questions")
        : empty("None in this period."),
      g.name_words_hidden ? K.el("p", { class: "small muted", text: `${g.name_words_hidden} names (people or places asked about) are counted but not listed.` }) : null,
      K.el("p", { class: "small" }, K.el("a", { href: "/admin", text: "Add everyday words in Settings" })),
      K.el("h3", { text: "Questions answered 'not found'" }), K.el("p", { class: "definition", text: D.refused }),
      g.refused.length ? table(["Question", "Times", "People", "Words not in any source"], g.refused.map((r) => [r.question, String(r.times), String(r.people), r.words.join(", ")])) : null,
      K.el("p", { class: "small muted", text: hiddenText(g.refused_hidden, d.filters.min_askers) }),
      K.el("h3", { text: "Questions answered only in part" }), K.el("p", { class: "definition", text: D.partial }),
      g.partial.length ? table(["Question", "Times", "People", "Not covered"], g.partial.map((r) => [r.question, String(r.times), String(r.people), r.words.join(", ")])) : null,
      K.el("p", { class: "small muted", text: hiddenText(g.partial_hidden, d.filters.min_askers) }),
      K.el("h3", { text: "Topics with the weakest evidence" }), K.el("p", { class: "definition", text: D.weak_topics }),
      g.weak_topics.length ? hbars(g.weak_topics.map((t) => ({ label: t.topic.replace(/^TOPIC-/, "").replace(/-/g, " ").toLowerCase(), value: Math.round(t.weak_share * 100), note: `${t.answers} answers` })), "%", "Share of partial or not-found answers per topic")
        : empty("No topic has 3 or more answers yet.")));

    /* C. performance */
    const p = d.performance;
    out.push(section("h-speed", "Speed", D.latency,
      tiles([[ms(p.overall.p50), "Median answer"], [ms(p.overall.p95), "95% of answers within"], [ms(p.overall.max), "Slowest"],
        [p.startup_s.length ? p.startup_s.map((x) => x + " s").join(", ") : "-", "Start-up", "per KORA start"], [String(p.rebuilds), "Answers after a rebuild", "the index was rebuilt first"]]),
      p.per_day.length > 1 ? K.el("div", {}, legend([["ln-p50", "Median"], ["ln-p95", "95th percentile (dashed)"]]), latencyLine(p.per_day),
        asTable(["Day", "Median", "95th percentile", "Answers"], p.per_day.map((r) => [r.day, ms(r.p50), ms(r.p95), String(r.answers)]), "Answer time per day")) : null,
      K.el("h3", { text: "Where the time goes" }), K.el("p", { class: "definition", text: D.steps }),
      hbars(Object.entries(p.steps).map(([k, v]) => ({ label: k.replace(/_/g, " "), value: v })), " ms", "Average milliseconds per step"),
      p.slowest.length ? K.el("div", {}, K.el("h3", { text: "Slowest answers" }),
        table(["When (UTC)", "Time", "Status", "Question"], p.slowest.map((r) => [String(r.at).slice(0, 16).replace("T", " "), ms(r.ms), r.status.replace("_", " "), r.question || "(text hidden)"]))) : null));

    /* D. knowledge quality */
    const k = d.knowledge || {};
    if (k.graph) {
      const t = k.links_by_tier;
      out.push(section("h-knowledge", "Knowledge and relationships", "From the knowledge graph and Phase 3's checks, as you are allowed to see them.",
        tiles([[pct(k.documented_share), "Links that are documented", "quote, header line or company record"],
          [String(k.unverified_evidence.length), "Evidence that no longer checks out"], [String(k.open_conflicts.length), "Open conflicts"],
          [String(k.superseded_documents.length), "Superseded documents"], [String(k.processes_without_owner.length), "Processes without a documented owner"]]),
        K.el("h3", { text: "Documents by department" }),
        hbars(Object.entries(k.documents_by_department).map(([dep, n]) => ({ label: dep, value: n })), "", "Documents per department"),
        K.el("h3", { text: "How KORA knows its links" }), K.el("p", { class: "definition", text: k.definitions.links_by_tier }),
        hbars([["quote", "Quoted sentence"], ["header", "Document header"], ["record", "Company record"], ["mention", "Mentioned together"],
          ["derived", "KORA's own analysis"], ["unverified", "Unverified"]].map(([key, label]) => ({ label, value: t[key] || 0 })), "", "Links by evidence"),
        K.el("h3", { text: "To look at" }),
        table(["Check", "Count", "Which"], [
          ["Processes without a documented owner", String(k.processes_without_owner.length), k.processes_without_owner.join(", ")],
          ["Processes without participating roles", String(k.processes_without_participants.length), k.processes_without_participants.join(", ")],
          ["Roles without a department", String(k.roles_without_department.length), k.roles_without_department.join(", ")],
          ["Links with an unknown shape", String(k.unknown_shapes.length), k.unknown_shapes.slice(0, 5).join("; ")],
          ["Links with a missing end (dangling)", String(k.dangling.length), ""],
          ["Superseded documents", String(k.superseded_documents.length), k.superseded_documents.map((x) => `${x.document_id} (replaced by ${x.superseded_by.join(", ") || "not stated"})`).join("; ")],
          ["Knowledge gaps by severity", String(Object.values(k.gaps_by_severity).reduce((a, b) => a + b, 0)), Object.entries(k.gaps_by_severity).map(([s2, n]) => `${s2}: ${n}`).join(", ")],
        ]),
        k.open_conflicts.length ? K.el("details", { class: "table-view" }, K.el("summary", { text: `Open conflicts (${k.open_conflicts.length})` }),
          table(["Conflict", "Kind", "Documents", "What"], k.open_conflicts.map((c) => [c.conflict_id, c.type.replace(/_/g, " "), c.documents.join(", "), c.detail]))) : null));
    }

    /* E. usage */
    const u = d.usage;
    out.push(section("h-usage", "Usage", D.usage,
      K.el("h3", { text: "Kinds of question" }),
      hbars(Object.entries(u.categories).map(([c, n]) => ({ label: nice(c), value: n })), "", "Answers by kind of question"),
      K.el("h3", { text: "Where questions came from" }),
      hbars(Object.entries(u.channels).map(([c, n]) => ({ label: { ui: "KORA screens", api: "API", cli: "Command line", python: "Python", unknown: "Older log lines" }[c] || c, value: n })), "", "Log records by channel"),
      K.el("h3", { text: "Most cited documents" }), K.el("p", { class: "definition", text: D.cited }),
      hbars(u.cited_documents.map((c) => ({ label: c.document_id, value: c.answers })), "", "Answers citing each document"),
      u.company_record_citations ? K.el("p", { class: "small muted", text: `Company records were cited ${u.company_record_citations} times.` }) : null));

    K.clear(root).append(...out);
  }
  const hiddenText = (n, min) => (n ? `${n} other question${n === 1 ? "" : "s"} counted but not shown: fewer than ${min} different people asked ${n === 1 ? "it" : "each"}, or ${n === 1 ? "it names" : "they name"} a person.` : "");

  /* ---------- filters ---------- */
  async function load() {
    const ch = document.getElementById("f-channel").value, days = document.getElementById("f-days").value;
    const qs = new URLSearchParams(); if (ch) qs.set("channel", ch); if (days) qs.set("days", days);
    try { render(await K.api("/api/admin/dashboard?" + qs)); }
    catch (e) { K.clear(root).append(K.el("p", { class: "error", text: "The dashboard couldn't be loaded: " + e.message })); }
  }
  document.getElementById("f-channel").addEventListener("change", load);
  document.getElementById("f-days").addEventListener("change", load);
  load();
})();
