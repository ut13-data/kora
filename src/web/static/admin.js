"use strict";
(async function () {
  const panel = document.getElementById("panel");
  const message = document.getElementById("message");
  const saveBar = document.getElementById("save-bar");
  let raw, schema, tab = "words", jsonDraft = null;

  try { [raw, schema] = await Promise.all([K.api("/api/admin/config"), K.api("/api/admin/config/schema")]); }
  catch (e) { panel.append(K.el("p", { class: "error", text: "Settings couldn't be loaded: " + e.message })); return; }

  const say = (text, bad, problems) => {
    K.clear(message);
    message.append(K.el("div", { class: "message" + (bad ? " bad" : "") }, K.el("p", { text }),
      problems && problems.length ? K.el("ul", { class: "problems" }, problems.map((p) => K.el("li", { text: p }))) : null));
  };
  const words = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);
  const comment = (obj) => (obj && obj._comment ? K.el("p", { class: "lead", text: obj._comment }) : null);

  /* ---------- tabs ---------- */
  function rows(table, entries, render, addLabel, add) {
    for (const e of entries) table.append(render(e));
    return K.el("button", { class: "button secondary", type: "button", text: addLabel, onclick: add });
  }

  const TABS = {
    words() {
      raw.synonyms = raw.synonyms || {};
      const table = K.el("table", { class: "grid-table" }, K.el("tr", {}, K.el("th", { text: "Everyday word or phrase" }),
        K.el("th", { text: "Words the documents use (comma-separated)" }), K.el("th", {})));
      const entries = Object.entries(raw.synonyms).filter(([k]) => !k.startsWith("_"));
      const render = ([phrase, list]) => {
        const p = K.el("input", { class: "input", value: phrase, "aria-label": "Everyday phrase" });
        const w = K.el("input", { class: "input", value: (list || []).join(", "), "aria-label": `Document words for ${phrase}` });
        const sync = () => { const out = {}; table.querySelectorAll("tr[data-row]").forEach((tr) => {
          const [a, b] = tr.querySelectorAll("input"); if (a.value.trim()) out[a.value.trim()] = words(b.value); });
          for (const [k, v] of Object.entries(raw.synonyms)) if (k.startsWith("_")) out[k] = v;
          raw.synonyms = out; };
        p.addEventListener("change", sync); w.addEventListener("change", sync);
        const tr = K.el("tr", { "data-row": "" }, K.el("td", {}, p), K.el("td", {}, w),
          K.el("td", {}, K.el("button", { class: "link-button", type: "button", text: "Remove", onclick: () => { tr.remove(); sync(); } })));
        return tr;
      };
      const add = rows(table, entries, render, "Add a word", () => { table.append(render(["", []])); table.lastChild.querySelector("input").focus(); });
      return [K.el("p", { class: "lead", text: "Teach KORA everyday wording. When a question uses a phrase on the left, KORA also searches for the words on the right, and says so in the answer. For example, trip means travel." }), table, add];
    },

    thresholds() {
      raw.thresholds = raw.thresholds || {};
      const table = K.el("table", { class: "grid-table" }, K.el("tr", {}, K.el("th", { text: "Setting" }), K.el("th", { text: "Value" }), K.el("th", { text: "Allowed" })));
      for (const [name, s] of Object.entries(schema.thresholds)) {
        const current = raw.thresholds[name] ?? s.default;
        const input = K.el("input", { class: "input", type: "number", value: current, min: s.min, max: s.max,
          step: s.whole_number ? 1 : 0.01, "aria-label": name, style: "max-width:8rem" });
        input.addEventListener("change", () => {
          const v = s.whole_number ? parseInt(input.value, 10) : parseFloat(input.value);
          if (Number.isNaN(v) || v === s.default) delete raw.thresholds[name]; else raw.thresholds[name] = v;
        });
        table.append(K.el("tr", {}, K.el("td", { text: name.replace(/_/g, " ") }), K.el("td", {}, input),
          K.el("td", { class: "range", text: `${s.min} to ${s.max}, default ${s.default}` })));
      }
      return [K.el("p", { class: "lead", text: "How strict KORA is before it answers. Raising a threshold makes KORA refuse more often; lowering it risks weaker answers. Check the evaluation after any change." }), table];
    },

    access() {
      raw.access = raw.access || { rules: {} };
      raw.access.rules = raw.access.rules || {};
      const table = K.el("table", { class: "grid-table" }, K.el("tr", {}, K.el("th", { text: "Document ID (use * for many)" }),
        K.el("th", { text: "Groups allowed to read it (comma-separated)" }), K.el("th", {})));
      const sync = () => { const out = {}; table.querySelectorAll("tr[data-row]").forEach((tr) => {
        const [a, b] = tr.querySelectorAll("input"); if (a.value.trim()) out[a.value.trim()] = words(b.value); });
        for (const [k, v] of Object.entries(raw.access.rules)) if (k.startsWith("_")) out[k] = v;
        raw.access.rules = out; };
      const render = ([pattern, groups]) => {
        const p = K.el("input", { class: "input", value: pattern, "aria-label": "Document pattern" });
        const g = K.el("input", { class: "input", value: (groups || []).join(", "), "aria-label": `Groups for ${pattern}` });
        p.addEventListener("change", sync); g.addEventListener("change", sync);
        const tr = K.el("tr", { "data-row": "" }, K.el("td", {}, p), K.el("td", {}, g),
          K.el("td", {}, K.el("button", { class: "link-button", type: "button", text: "Remove", onclick: () => { tr.remove(); sync(); } })));
        return tr;
      };
      const entries = Object.entries(raw.access.rules).filter(([k]) => !k.startsWith("_"));
      const add = rows(table, entries, render, "Add a rule", () => { table.append(render(["", []])); table.lastChild.querySelector("input").focus(); });
      return [comment(raw.access), K.el("p", { class: "lead", text: "Groups: " + schema.groups_help + ". A document with no rule is open to everyone. A hidden document is never searched, quoted, cited or named for people outside its groups, and admins get no exception." }), table, add];
    },

    people() {
      raw.admins = raw.admins || { employees: [] };
      raw.person_visibility = raw.person_visibility || { colleague_fields: [] };
      const admins = K.el("input", { class: "input", value: (raw.admins.employees || []).join(", "), "aria-label": "Admin employee IDs" });
      admins.addEventListener("change", () => { raw.admins.employees = words(admins.value); });
      const chosen = new Set(raw.person_visibility.colleague_fields || []);
      const boxes = K.el("div", { style: "display:grid;gap:.35rem;margin-top:.5rem" }, schema.employee_columns.map((col) => {
        const cb = K.el("input", { type: "checkbox", id: "col-" + col, checked: chosen.has(col) });
        cb.addEventListener("change", () => { cb.checked ? chosen.add(col) : chosen.delete(col);
          raw.person_visibility.colleague_fields = schema.employee_columns.filter((c) => chosen.has(c)); });
        return K.el("label", { for: "col-" + col, style: "display:flex;gap:.5rem;align-items:center" }, cb, col.replace(/_/g, " "));
      }));
      return [
        K.el("div", { class: "field" }, K.el("label", { text: "Admins (employee IDs, comma-separated)" }), admins,
          K.el("p", { class: "small muted", text: "Admins open this page and the dashboards. You can't remove yourself; ask another admin." })),
        K.el("div", { class: "field", style: "margin-top:1.5rem" }, K.el("label", { text: "What colleagues can see about each other" }),
          comment(raw.person_visibility),
          K.el("p", { class: "small muted", text: "Unticked details are visible only to the person and to admins." }), boxes),
      ];
    },

    json() {
      const area = K.el("textarea", { class: "textarea json-editor", spellcheck: "false", "aria-label": "All settings as JSON" });
      area.value = jsonDraft ?? JSON.stringify(raw, null, 2);
      area.addEventListener("input", () => { jsonDraft = area.value; });
      return [K.el("p", { class: "lead", text: "Everything, including the Germany relocation programme and the communication tone profiles. The same checks run when you save." }), area];
    },

    async history() {
      const list = await K.api("/api/admin/config/backups");
      if (!list.length) return [K.el("p", { class: "lead", text: "No earlier versions yet. Every save keeps the previous settings here." })];
      return [K.el("p", { class: "lead", text: "Restoring an earlier version is a save too, so you can undo it the same way." }),
        K.el("table", { class: "grid-table" }, K.el("tr", {}, K.el("th", { text: "Saved (UTC)" }), K.el("th", {})),
          list.map((b) => K.el("tr", {}, K.el("td", { text: b.saved_at.replace(/^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)(\d\d).*/, "$1-$2-$3 $4:$5:$6") }),
            K.el("td", {}, K.el("button", { class: "link-button", type: "button", text: "Restore this version", onclick: () => restore(b.name) })))))];
    },
  };

  async function showTab(name) {
    if (tab === "json" && name !== "json" && jsonDraft !== null) {
      try { raw = JSON.parse(jsonDraft); jsonDraft = null; }
      catch (e) { say("The JSON isn't valid yet, so the other tabs can't show it: " + e.message, true); return; }
    }
    tab = name;
    document.querySelectorAll("[role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    saveBar.hidden = name === "history";
    K.clear(panel).append(...(await TABS[name]()).filter(Boolean));
  }
  document.querySelectorAll("[role=tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

  function current() {
    if (tab === "json" && jsonDraft !== null) { raw = JSON.parse(jsonDraft); jsonDraft = null; }
    return raw;
  }

  document.getElementById("check").addEventListener("click", async () => {
    try {
      const r = await K.api("/api/admin/config/validate", { json: current() });
      r.valid ? say("These settings are valid. Nothing has been saved yet.") : say("Fix these before saving:", true, r.problems);
    } catch (e) { say(e instanceof SyntaxError ? "The JSON isn't valid: " + e.message : e.message, true); }
  });

  document.getElementById("save").addEventListener("click", async () => {
    try {
      const r = await K.api("/api/admin/config", { method: "PUT", json: current() });
      say("Settings saved. KORA reloaded with them." + (r.backup ? " The previous version is kept under Earlier versions." : ""),
        false, (r.warnings || []).map((w) => "Note: " + w));
      raw = await K.api("/api/admin/config");
      if (tab !== "history") showTab(tab);
    } catch (e) {
      if (e instanceof SyntaxError) say("The JSON isn't valid: " + e.message, true);
      else say("Not saved. Fix these first:", true, (e.body && e.body.problems) || [e.message]);
    }
  });

  async function restore(name) {
    try {
      await K.api("/api/admin/config/restore", { json: { name } });
      raw = await K.api("/api/admin/config");
      say("Earlier version restored. KORA reloaded with it, and the settings it replaced are kept too.");
      showTab("history");
    } catch (e) { say("Not restored: ", true, (e.body && e.body.problems) || [e.message]); }
  }

  showTab("words");
})();
