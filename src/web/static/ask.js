"use strict";
(function () {
  const form = document.getElementById("ask-form");
  const box = document.getElementById("question");
  const thread = document.getElementById("thread");
  const button = document.getElementById("ask-button");
  const KIND = { rule: "Rule", conflict_side: "Says", checklist: "To do", change: "Changed", document: "Document", gap: "Not documented" };

  /* ---------- suggestions, based on who you are ---------- */
  const general = [
    "Which teams are involved in customer onboarding?",
    "Who approves purchases above 10000 euros?",
    "Which channel should I use for a quick question to a colleague?",
    "How quickly do I get my money back after paying for a business trip myself?",
  ];
  async function suggest() {
    const list = document.getElementById("suggestions");
    let mine = [];
    try {
      const me = await K.api("/api/me/context");
      const care = me.germany_care && me.germany_care.eligibility;
      if (care && care.status === "active") mine.push("What should a new employee moving to Germany prepare during their first week?");
      if (me.location && me.location.country) mine.push(`How many days of annual leave do employees in ${me.location.city} get?`);
      const proc = (me.relevant_processes || [])[0];
      if (proc) mine.push(`Who owns the ${proc.name.toLowerCase()} process?`);
    } catch (_) { /* suggestions are optional */ }
    for (const q of [...mine, ...general].slice(0, 5)) {
      list.append(K.el("li", {}, K.el("button", { type: "button", text: q, onclick: () => ask(q) })));
    }
  }

  /* ---------- one answer ---------- */
  function cite(answer, number, quote) {
    const c = answer.citations.find((x) => x.number === number);
    if (!c) return null;
    return K.el("button", {
      class: "cite", type: "button", "aria-expanded": "false", text: String(number),
      "aria-label": `Source ${number}: ${c.title}`,
      onclick: (e) => K.source.open(c, quote, e.currentTarget),
    });
  }

  function quoteBlock(answer, line) {
    return K.el("p", { class: "quote" }, K.el("mark", { text: line.text }), line.citation ? cite(answer, line.citation, line.text) : null);
  }

  function renderLines(answer) {
    const list = K.el("ul", { class: "lines" });
    const graph = answer.decision && answer.decision.graph;
    const grouped = graph && graph.used && graph.kind === "involved";
    let lastGroup = null, possibleShown = false;
    for (const line of answer.lines) {
      if (line.kind === "relation" || line.kind === "possible") {
        let claim = line.detail, group = null;
        if (grouped && line.kind === "relation" && claim.includes(": ")) [group, claim] = [claim.slice(0, claim.indexOf(": ")), claim.slice(claim.indexOf(": ") + 2)];
        if (line.kind === "possible" && !possibleShown) {
          list.append(K.el("li", { class: "possible-head", text: "Possibly related: listed or mentioned together, not stated as a relationship" }));
          possibleShown = true;
        }
        if (group && group !== lastGroup) { list.append(K.el("li", { class: "group-name", text: group })); lastGroup = group; }
        list.append(K.el("li", {}, K.el("p", { class: "claim", text: claim }), quoteBlock(answer, line)));
        continue;
      }
      const label = KIND[line.kind];
      list.append(K.el("li", {},
        label ? K.el("span", { class: "kind-label", text: label }) : null,
        quoteBlock(answer, line),
        line.detail ? K.el("p", { class: "small muted", text: line.detail.replace(/\{ref\}/g, "") }) : null));
    }
    return list;
  }

  function renderAnswer(answer) {
    const [label, meaning] = K.STATUS[answer.status] || [answer.status, ""];
    const node = K.el("div", { class: "answer", "data-status": answer.status },
      K.el("p", { class: "status-line" }, label, K.el("span", { class: "muted", text: " - " + meaning })));
    if (answer.lines.length) node.append(renderLines(answer));
    if (answer.status === "not_found") node.append(K.el("p", { text: "Try other words, or ask your HR Generalist: they are there for questions like this." }));
    if (answer.notes.length) node.append(K.el("ul", { class: "notes", "aria-label": "Notes" }, answer.notes.map((n) => K.el("li", { text: n }))));
    if (answer.citations.length) {
      const heading = answer.status === "not_found" ? "Closest sources (not an answer)" : "Sources";
      node.append(K.el("p", { class: "small muted", style: "margin-top:.9rem", text: heading }),
        K.el("ul", { class: "sources-list" }, answer.citations.map((c) => K.el("li", {},
          K.el("button", { type: "button", text: `[${c.number}] ${c.title}`, onclick: (e) => K.source.open(c, null, e.currentTarget) }),
          " ", c.kind === "record" ? c.section : c.kind === "header" ? "(document header)" :
            [c.version !== "unknown" && "v" + c.version, c.status, c.effective_date !== "unknown" && c.effective_date].filter(Boolean).join(", ")))));
    }
    node.append(K.el("details", { class: "why" }, K.el("summary", { text: "Why this answer" }),
      K.el("ul", {}, answer.why.map((w) => K.el("li", { text: w })), K.el("li", { text: "Time: " + answer.timing_text }))));
    return node;
  }

  /* ---------- asking ---------- */
  async function ask(question) {
    question = question.replace(/\s+/g, " ").trim();
    if (!question) return;
    document.getElementById("intro").hidden = true;
    const turn = K.el("section", { class: "turn" }, K.el("h2", { class: "turn-question", text: question }));
    const waiting = K.el("p", { class: "thinking", text: "Looking through Nexora's documents..." });
    turn.append(waiting);
    thread.append(turn);
    turn.scrollIntoView({ block: "start" });
    button.disabled = true;
    try {
      const answer = await K.api("/api/ask", { json: { question, channel: "ui" } });
      waiting.replaceWith(renderAnswer(answer));
    } catch (e) {
      waiting.replaceWith(K.el("p", { class: "error", text: e.status === 422 ? "Please type a question of up to 500 characters." : "KORA couldn't answer just now: " + e.message }));
    } finally {
      button.disabled = false;
      box.value = "";
      box.style.height = "";
      box.focus();
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); ask(box.value); });
  box.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(box.value); } });
  box.addEventListener("input", () => { box.style.height = "auto"; box.style.height = box.scrollHeight + "px"; });
  const preset = new URLSearchParams(window.location.search).get("q");
  suggest();
  if (preset) ask(preset);
  else box.focus();
})();
