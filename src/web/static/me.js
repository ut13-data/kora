"use strict";
(async function () {
  const root = document.getElementById("week");
  const STAGE = { pre_start: "Before your first day", week_1: "Your first week", week_2: "Your second week",
    month_1: "Your first month", month_2: "Your second month", month_3: "Your third month", settled: "Your Nexora" };
  const askLink = (q, text) => K.el("a", { href: "/ask?q=" + encodeURIComponent(q), text: text || q });

  let me;
  try { me = await K.api("/api/me/context"); }
  catch (e) { K.clear(root).append(K.el("p", { class: "error", text: "Your page couldn't be loaded: " + e.message })); return; }

  const first = (me.name || "").split(" ")[0];
  const head = K.el("header", { class: "week-head" },
    K.el("h1", { text: me.is_new_joiner ? `${STAGE[me.joining_stage] || "Your Nexora"}, ${first}` : `Your Nexora, ${first}` }),
    me.is_new_joiner ? K.el("p", { class: "muted", text: `Day ${me.day} since you joined on ${me.join_date}. Dates are counted to ${me.as_of}, the demo date set in KORA's configuration.` }) : null);

  const facts = K.el("dl", { class: "facts" },
    K.el("dt", { text: "Role" }), K.el("dd", {}, me.role ? K.el("a", { href: `/explore?key=role:${me.role.role_id}`, text: me.role.name }) : "Not recorded"),
    K.el("dt", { text: "Department" }), K.el("dd", {}, me.department ? K.el("a", { href: `/explore?key=department:${me.department.dept_id}`, text: me.department.name }) : "Not recorded"),
    K.el("dt", { text: "Office" }), K.el("dd", { text: me.location ? `${me.location.city}, ${me.location.country}` : "Not recorded" }),
    K.el("dt", { text: "Manager" }), K.el("dd", {}, me.manager ? K.el("a", { href: `/explore?key=employee:${me.manager.employee_id}`, text: me.manager.name + (me.manager.role ? `, ${me.manager.role}` : "") }) : "No manager recorded"));

  K.clear(root).append(head, facts);

  /* ---------- Germany settling-in (only when the programme applies) ---------- */
  const care = me.germany_care || {};
  if (care.eligibility && care.eligibility.status === "active" && care.plan) {
    const byDay = new Map();
    for (const item of care.plan.items || []) {
      const key = item.due_by_day ? `By day ${item.due_by_day}` : "When it suits you";
      if (!byDay.has(key)) byDay.set(key, []);
      byDay.get(key).push(item);
    }
    const section = K.el("section", { class: "section", "aria-labelledby": "care-h" },
      K.el("h2", { id: "care-h", text: "Settling in to Germany" }),
      K.el("p", { class: "lead", text: "From Nexora's settling-in checklist, at a pace that suits most people. Nothing is ticked until you report it to HR; KORA never marks anything done for you." }),
      (care.plan.items || []).some((it) => it.timing === "past_suggested_pace" && it.status !== "reported_complete")
        ? K.el("p", { style: "margin-bottom:1rem", text: "Some items are past the suggested pace. That's fine: go at your own speed, and ask your HR Generalist if anything is stuck." }) : null,
      care.plan.caveat ? K.el("p", { class: "guidance", style: "margin-bottom:1.25rem", text: care.plan.caveat.replace(/\s*>\s*/g, " ") }) : null);
    for (const [day, items] of byDay) {
      section.append(K.el("div", { class: "day-group" }, K.el("h3", { text: day }), K.el("ul", { class: "checklist" }, items.map((it) => {
        const done = it.status === "reported_complete";
        const meta = [done ? "Reported done" + (it.reported_on ? " on " + it.reported_on : "") : "Not reported yet",
          (it.blocked_by || []).length ? "after " + it.blocked_by.map((b) => b.replace(/^GER-/, "").replace(/-/g, " ").toLowerCase()).join(", ") : null,
          it.commonly_forgotten ? "often forgotten" : null].filter(Boolean).join("; ");
        const contacts = [...(it.contacts_here || []), ...(it.contacts_elsewhere || [])].map((c) => c.name + (c.location_qualifier ? ` (${c.location_qualifier})` : ""));
        return K.el("li", { class: "check-item" },
          K.el("span", { class: "check-box" + (done ? " done" : ""), role: "img", "aria-label": done ? "Reported done" : "Not reported yet" }),
          K.el("div", {}, K.el("p", { class: "check-title", text: it.title }),
            K.el("p", { class: "check-meta", text: meta }),
            contacts.length ? K.el("p", { class: "check-meta", text: "Who can help: " + contacts.join(", ") }) : null,
            it.official_guidance_note ? K.el("p", { class: "guidance", text: it.official_guidance_note }) : null));
      }))));
    }
    section.append(K.el("p", { class: "small muted", style: "margin-top:1rem" }, "Questions about any of these? ",
      askLink("What should a new employee moving to Germany prepare during their first week?", "Ask KORA about your first week")));
    root.append(section);
  }

  /* ---------- working with colleagues ---------- */
  const comm = me.communication;
  if (comm && comm.location_profile) {
    root.append(K.el("section", { class: "section", "aria-labelledby": "comm-h" },
      K.el("h2", { id: "comm-h", text: "Working with colleagues here" }),
      K.el("p", { class: "lead", text: comm.basis ? comm.basis.charAt(0).toUpperCase() + comm.basis.slice(1) + "." : "" }),
      comm.reminder ? K.el("p", { text: comm.reminder }) : null,
      K.el("p", { class: "small", style: "margin-top:.6rem" }, askLink(`What tone should I use when writing to colleagues in ${me.location ? me.location.country : "my office"}?`, "Ask KORA about tone here"))));
  }

  /* ---------- processes and approvals ---------- */
  const procs = me.relevant_processes || [];
  if (procs.length) {
    root.append(K.el("section", { class: "section", "aria-labelledby": "proc-h" },
      K.el("h2", { id: "proc-h", text: "Processes you are part of" }),
      K.el("ul", { class: "plain-list" }, procs.map((p) => K.el("li", {},
        K.el("a", { href: `/explore?key=process:${p.process_id}`, text: p.name }),
        K.el("span", { class: "why-text", text: " - " + (p.why || []).join("; ") }))))));
  }
  const ap = me.approvals || {};
  const approvals = [["You approve", ap.you_approve], ["Your manager approves", ap.your_manager_approves],
    ["You may need approval from", ap.you_may_need_approval_from]].filter(([, list]) => (list || []).length);
  if (approvals.length) {
    root.append(K.el("section", { class: "section", "aria-labelledby": "apr-h" },
      K.el("h2", { id: "apr-h", text: "Approvals that involve you" }),
      approvals.map(([title, list]) => K.el("div", { class: "day-group" }, K.el("h3", { text: title }),
        K.el("ul", { class: "plain-list" }, list.map((r) => K.el("li", {}, K.el("span", { class: "quote" }, K.el("mark", { text: r.statement })),
          K.el("span", { class: "why-text", text: " (" + r.document_id + ")" }))))))));
  }
  const reading = me.reading_list || [];
  if (reading.length) {
    root.append(K.el("section", { class: "section", "aria-labelledby": "read-h" },
      K.el("h2", { id: "read-h", text: "Worth reading" }),
      K.el("ul", { class: "plain-list" }, reading.map((r) => K.el("li", {},
        K.el("a", { href: "/documents/" + encodeURIComponent(r.document_id), text: r.title }),
        K.el("span", { class: "why-text", text: " - " + r.why }))))));
  }
})();
