"use strict";
(async function () {
  const select = document.getElementById("employee");
  const filter = document.getElementById("employee-filter");
  const error = document.getElementById("login-error");
  let people = [];
  try { people = await K.api("/api/login/employees"); }
  catch (e) { error.hidden = false; error.textContent = "The employee list couldn't be loaded. Is KORA running?"; return; }

  function render(term) {
    K.clear(select);
    const t = (term || "").toLowerCase().trim();
    const groups = new Map();
    for (const p of people) {
      const hay = [p.name, p.role, p.department, p.location, p.employee_id].join(" ").toLowerCase();
      if (t && !hay.includes(t)) continue;
      if (!groups.has(p.department)) groups.set(p.department, []);
      groups.get(p.department).push(p);
    }
    for (const [dept, list] of [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
      const g = K.el("optgroup", { label: dept });
      for (const p of list) g.append(K.el("option", { value: p.employee_id, text: `${p.name}, ${p.role} (${p.location})` }));
      select.append(g);
    }
    if (select.options.length) select.selectedIndex = 0;
  }
  render("");
  filter.addEventListener("input", () => render(filter.value));

  document.getElementById("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    error.hidden = true;
    const button = document.getElementById("sign-in");
    if (!select.value) { error.hidden = false; error.textContent = "Choose an employee from the list."; return; }
    button.disabled = true;
    try {
      await K.api("/api/login", { json: { employee_id: select.value, passcode: document.getElementById("passcode").value } });
      window.location.href = "/ask";
    } catch (err) {
      error.hidden = false;
      error.textContent = err.status === 401 ? "That passcode doesn't match. It is set by KORA_DEV_PASSCODE (default: kora-dev)."
        : "Sign-in failed: " + err.message;
      button.disabled = false;
    }
  });
})();
