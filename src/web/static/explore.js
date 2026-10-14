"use strict";
(function () {
  const results = document.getElementById("results");
  const box = document.getElementById("search");
  const article = document.getElementById("entity");
  const start = document.getElementById("start");

  const TYPE = { employee: "Person", role: "Role", department: "Department", process: "Process", policy: "Policy",
    product: "Product", document: "Document", location: "Office", topic: "Topic", checklist: "Checklist",
    terminology: "Term", customer: "Customer", audience: "Audience" };

  /* "relationship|direction" -> heading in plain words (in = links pointing at this entity) */
  const HEADINGS = {
    "approves|in": "Who approves it", "approves|out": "What this role approves",
    "owns|in": "Owners", "owns|out": "What it owns",
    "performs|in": "Roles named in its steps", "performs|out": "Process steps it is named in",
    "participates_in|in": "Related roles listed in its header", "participates_in|out": "Processes listing it as a related role",
    "handles_escalations_for|in": "Who handles escalations", "handles_escalations_for|out": "Escalations it handles",
    "uses|in": "Processes that use it", "uses|out": "Products it uses",
    "triggers|in": "What starts it", "triggers|out": "What it starts",
    "governed_by|in": "Processes that list it as related", "governed_by|out": "Related policies",
    "related_to|in": "Listed as related by", "related_to|out": "Lists as related",
    "owned_by|in": "What this department owns", "owned_by|out": "Owning department",
    "headed_by|in": "Department it heads", "headed_by|out": "Head of department",
    "in_department|in": "Roles in this department", "in_department|out": "Department",
    "has_role|in": "People in this role", "has_role|out": "Role",
    "works_in|in": "People", "works_in|out": "Department",
    "located_in|in": "People based here", "located_in|out": "Office",
    "reports_to|in": "Reports", "reports_to|out": "Reports to",
    "documented_by|in": "Describes", "documented_by|out": "Documents",
    "disambiguates|in": "Terms that tell it apart", "disambiguates|out": "Tells apart",
    "defined_in|in": "Terms defined here", "defined_in|out": "Defined in",
    "head_located_in|out": "Head is based in", "present_in|out": "Offices", "present_in|in": "Departments here",
    "has_checklist|out": "Checklists", "has_tone_guidance|out": "Tone guidance",
    "eligible_for|out": "Programmes", "for_audience|out": "Written for",
  };
  const WEAK = "Mentioned together (not a stated relationship)";
  const heading = (l) => !l.documented ? WEAK : (HEADINGS[`${l.relationship}|${l.direction}`] ||
    l.relationship.charAt(0).toUpperCase() + l.relationship.slice(1).replace(/_/g, " "));
  /* most useful first; anything not listed comes after, and "mentioned together" always last */
  const ORDER = ["Owners", "Head of department", "Who approves it", "Roles named in its steps", "What starts it", "What it starts",
    "Products it uses", "Processes that use it", "Related policies", "Owning department", "Department", "Role", "Reports to",
    "Office", "Who handles escalations", "Related roles listed in its header", "What this department owns", "Roles in this department",
    "What it owns", "What this role approves", "Process steps it is named in", "Processes listing it as a related role", "Documents"];
  const rank = (h) => (h === WEAK ? 1e6 : ORDER.includes(h) ? ORDER.indexOf(h) : 500);
  const SHOW = 6;

  /* ---------- search ---------- */
  async function search(q) {
    K.clear(results);
    if (q.trim().length < 2) return;
    const hits = await K.api("/api/graph/search?q=" + encodeURIComponent(q));
    if (!hits.length) { results.append(K.el("li", { class: "muted", text: "Nothing matches. Try part of a name, or a role such as CISO." })); return; }
    for (const h of hits) results.append(K.el("li", {}, K.el("button", { type: "button", onclick: () => show(h.key, true) },
      h.name, " ", K.el("span", { class: "type", text: TYPE[h.type] || h.type }))));
  }
  document.getElementById("search-form").addEventListener("submit", (e) => { e.preventDefault(); search(box.value); });
  let timer = null;
  box.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => search(box.value), 250); });

  /* ---------- a small map: this entity and its direct neighbours ---------- */
  function map(entity, links) {
    const NS = "http://www.w3.org/2000/svg";
    const seen = new Map();
    for (const l of links) {                       // one dot per NAME (a policy, its process and its document
      const k = l.other.name;                      // share a name), documented links first
      if (!seen.has(k)) seen.set(k, l);
      else if (l.documented && !seen.get(k).documented) seen.set(k, l);
    }
    const nodes = [...seen.values()].slice(0, 16);
    const W = 900, H = 380, cx = W / 2, cy = H / 2, R = 150;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "map");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${entity.name} and ${nodes.length} directly linked items. Solid lines are documented links, dashed lines only mentioned together.`);
    const mk = (tag, attrs) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
    const short = (s) => (s.length > 26 ? s.slice(0, 25) + "..." : s);
    nodes.forEach((l, i) => {
      const a = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
      l._x = cx + R * 1.55 * Math.cos(a); l._y = cy + R * Math.sin(a);
      svg.append(mk("line", { x1: cx, y1: cy, x2: l._x, y2: l._y, class: "edge" + (l.documented ? "" : " weak") }));
    });
    nodes.forEach((l) => {
      const link = mk("a", { href: "#", class: "node", tabindex: "0" });
      link.addEventListener("click", (e) => { e.preventDefault(); show(l.other.key, true); });
      const title = mk("title", {}); title.textContent = `${l.other.name} (${TYPE[l.other.type] || l.other.type})`;
      const anchor = l._x < cx - 10 ? "end" : l._x > cx + 10 ? "start" : "middle";
      const dx = anchor === "end" ? -10 : anchor === "start" ? 10 : 0, dy = anchor === "middle" ? (l._y < cy ? -12 : 20) : 4;
      const label = mk("text", { x: l._x + dx, y: l._y + dy, "text-anchor": anchor }); label.textContent = short(l.other.name);
      link.append(title, mk("circle", { cx: l._x, cy: l._y, r: 6 }), label);
      svg.append(link);
    });
    const center = mk("g", { class: "node center" });
    const ct = mk("text", { x: cx, y: cy + 28, "text-anchor": "middle" }); ct.textContent = short(entity.name);
    center.append(mk("circle", { cx, cy, r: 9 }), ct);
    svg.append(center);
    return svg;
  }

  /* ---------- an entity ---------- */
  function evidence(l) {
    const c = l.citation || {};
    if (!c.text) return null;
    const open = { kind: c.kind === "quote" && c.chunk_id ? "chunk" : c.kind, chunk_id: c.chunk_id || citationId(c),
      title: c.document_id || c.file || "Source" };
    return K.el("details", { class: "evidence" }, K.el("summary", { text: "Show evidence" }),
      K.el("p", { class: "quote" }, K.el("mark", { text: c.text }),
        K.el("button", { class: "cite", type: "button", text: "open", "aria-expanded": "false",
          onclick: (e) => K.source.open(open, c.text, e.currentTarget) })));
  }
  function citationId(c) {
    if (c.kind === "header") return `header:${c.file}#${c.field}`;
    if (c.kind === "record") return `record:${c.file}#${c.key_column}=${c.key}#${c.column}`;
    if (c.kind === "quote" && c.file) return `file:${c.file}`;
    return "";
  }

  async function show(key, push) {
    K.clear(results); K.clear(start);
    K.clear(article).append(K.el("p", { class: "muted", text: "Loading..." }));
    let data;
    try { data = await K.api("/api/graph/entity/" + encodeURIComponent(key).replace(/%3A/gi, ":")); }
    catch (e) { K.clear(article).append(K.el("p", { class: "error", text: e.status === 404 ? "This isn't available to you, or doesn't exist." : e.message })); return; }
    if (push) history.pushState({ key }, "", "/explore?key=" + encodeURIComponent(key));
    const e = data.entity;
    const groups = new Map();
    for (const l of data.links) {
      const h = heading(l);
      if (!groups.has(h)) groups.set(h, []);
      groups.get(h).push(l);
    }
    const docLink = e.type === "document" ? K.el("a", { href: "/documents/" + encodeURIComponent(e.id), text: "Read this document" }) : null;
    K.clear(article).append(
      K.el("header", { class: "entity-head" },
        K.el("span", { class: "type", text: TYPE[e.type] || e.type }),
        K.el("h2", { text: e.name }),
        e.aliases.length ? K.el("p", { class: "muted small", text: "Also called: " + e.aliases.join(", ") }) : null,
        docLink),
      data.links.length ? map(e, data.links) : K.el("p", { class: "muted", text: "No verified links to show." }),
      K.el("div", { class: "link-groups" }, [...groups.entries()].sort((a, b) => rank(a[0]) - rank(b[0])).map(([h, list]) => {
        const items = list.map((l) => K.el("li", { class: "link-item" },
          K.el("p", {}, K.el("button", { class: "nav-entity", type: "button", text: l.other.name, onclick: () => show(l.other.key, true) }),
            " ", K.el("span", { class: "type", text: TYPE[l.other.type] || l.other.type })),
          K.el("p", { class: "small", text: l.claim + (l.historical ? " (history)" : "") }),
          K.el("p", { class: "tier" + (l.documented ? "" : " weak"), text: l.tier_label }),
          evidence(l)));
        const ul = K.el("ul", { class: "link-list" }, items.slice(0, SHOW));
        const more = items.length > SHOW ? K.el("button", { class: "link-button small", type: "button", text: `Show ${items.length - SHOW} more`,
          onclick: (e) => { ul.append(...items.slice(SHOW)); e.currentTarget.remove(); } }) : null;
        if (h === WEAK) {
          return K.el("details", { class: "link-group" }, K.el("summary", { text: `${h}: ${list.length}` }),
            K.el("p", { class: "small muted", style: "margin:.4rem 0 .6rem", text: "These only appear in the same text. KORA doesn't treat them as facts." }), ul, more);
        }
        return K.el("section", { class: "link-group" }, K.el("h3", { text: h }), ul, more);
      })));
    article.querySelector("h2").setAttribute("tabindex", "-1");
    article.querySelector("h2").focus();
  }

  window.addEventListener("popstate", () => {
    const key = new URLSearchParams(location.search).get("key");
    if (key) show(key, false); else { K.clear(article); starting(); }
  });

  async function starting() {
    const picks = [["process:PROC-IMPL-001", "Customer Onboarding"], ["process:PROC-HR-001", "Employee Onboarding"]];
    try {
      const me = await K.api("/api/me/context");
      if (me.role) picks.unshift([`role:${me.role.role_id}`, `Your role: ${me.role.name}`]);
      if (me.department) picks.unshift([`department:${me.department.dept_id}`, `Your department: ${me.department.name}`]);
    } catch (_) { /* optional */ }
    K.clear(start).append(K.el("p", { class: "muted small", text: "Or start from:" }),
      K.el("ul", { class: "results" }, picks.map(([k, n]) => K.el("li", {}, K.el("button", { type: "button", text: n, onclick: () => show(k, true) })))));
  }

  const key = new URLSearchParams(location.search).get("key");
  if (key) show(key, false); else starting();
})();
