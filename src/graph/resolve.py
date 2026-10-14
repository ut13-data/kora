"""
Find the entities a question is about: processes, policies, roles,
departments, products and people, by name, alias or ID.

    Resolver(view).find("Which teams are involved in customer onboarding?")
      -> [Match("process:PROC-IMPL-001", "customer onboarding", ...)]

Only entities the USER may see are candidates, so a hidden process can't be
matched (and so can't be answered about, or even shown as "found").

Matching is on word stems ("escalations" = "escalation", "reports" =
"reporting"), longest name first, without overlaps: "Customer Success
Manager" (a role) wins over "Customer Success" (a department).

Names that are too generic to mean one thing on their own are NOT used:
  * a one-word department ("Sales", "Product", "Finance") only counts when
    followed by "department", "team" or "dept" ("the Product department",
    not "a product defect");
  * a process can also be found by ONE distinctive word of its name
    ("incident", "procurement"), but only if no other process, role,
    department or product name uses that word, and it is 5+ letters.
Ambiguity is never resolved by guessing: "onboarding" alone matches nothing,
because Nexora has customer AND employee onboarding.
"""

import re
from dataclasses import dataclass

TYPES = ("process", "policy", "role", "department", "product", "employee")
_SUFFIXES = ("process", "policy", "procedure", "procedures", "policies")
_DEPT_WORDS = ("department", "team", "dept")
# Never used as a one-word name: too common to point at one process.
_STOP = {"and", "the", "of", "for", "a", "an", "to", "in", "on", "with", "new", "internal", "customer",
         "process", "policy", "procedure", "procedures", "policies", "guidelines", "services", "employee",
         "leave", "data", "security", "support", "access", "control", "management", "communication",
         "software", "development", "reporting", "escalation", "onboarding"}
_BRAND = re.compile(r"^[A-Z][a-z]+[A-Z][A-Za-z]*$")      # NexView, NexFlow: a made-up product word


def stem(word: str) -> str:
    w = word.lower()
    for suffix, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not (suffix == "s" and w.endswith("ss")):
            return w[: len(w) - len(suffix)] + keep
    return w


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).lower())]


@dataclass
class Match:
    key: str
    text: str          # the name that matched, as written in the variant
    start: int         # token positions in the question
    end: int

    @property
    def entity_type(self) -> str:
        return self.key.split(":", 1)[0]


class Resolver:
    def __init__(self, view):
        self.view = view
        self.variants: list[tuple[tuple[str, ...], str, str]] = []   # (stems, key, text)
        self._build()

    def _add(self, key: str, text: str, require_after: tuple[str, ...] = ()) -> None:
        stems = tuple(tokens(text))
        if stems:
            self.variants.append((stems + tuple(require_after), key, text))

    def _build(self) -> None:
        entities = [e for t in TYPES for e in self.view.entities(t)]
        process_like = [e for e in entities if e.entity_type in ("process", "policy")
                        and (e.attributes.get("status") or "").lower() != "archived"]
        other_names = [e.name for e in entities if e.entity_type in ("role", "department", "product")]

        for e in entities:
            key = f"{e.entity_type}:{e.entity_id}"
            if e.entity_type in ("process", "policy"):
                if (e.attributes.get("status") or "").lower() == "archived":
                    continue                     # history is the answer layer's job (Phase 4)
                names = {e.name}
                for rel in self.view.hops(key, "documented_by"):
                    doc = self.view.node(rel.node)
                    if doc:
                        names.add(doc.name)
                for n in list(names):
                    words = n.split()
                    if words and words[-1].lower() in _SUFFIXES:
                        names.add(" ".join(words[:-1]))
                for n in names:
                    self._add(key, n)
                self._add(key, e.entity_id)
            elif e.entity_type == "department":
                parts = [p.strip() for p in re.split(r"[/&]", e.name) if p.strip()]
                for n in {e.name, *parts}:
                    if len(tokens(n)) >= 2:
                        self._add(key, n)
                    for word in _DEPT_WORDS:
                        self._add(key, n, (stem(word),))
                self._add(key, e.entity_id)
            elif e.entity_type == "employee":
                if len(e.name.split()) >= 2:
                    self._add(key, e.name)
                self._add(key, e.entity_id)
            else:                                # role, product
                for n in {e.name, *e.aliases, e.entity_id}:
                    self._add(key, n)
                first = e.name.split()[0]
                if e.entity_type == "product" and len(e.name.split()) > 1 and _BRAND.match(first):
                    self._add(key, first)                    # "NexView" for "NexView Analytics", not "Implementation"

        # one distinctive word of a process name
        def name_stems(names):
            return [set(tokens(n)) for n in names]
        other = set().union(*name_stems(other_names)) if other_names else set()
        per_process = {f"{e.entity_type}:{e.entity_id}": set(tokens(e.name)) - {stem(s) for s in _STOP}
                       for e in process_like}
        for key, words in per_process.items():
            for w in words:
                owners = {k for k, ws in per_process.items() if w in ws}
                # a process and its own policy entity (same ID) count as one
                if len({k.split(":", 1)[1] for k in owners}) == 1 and w not in other and len(w) >= 5:
                    self._add(key, w)
        self.variants.sort(key=lambda v: -len(v[0]))

    def find(self, question: str) -> list[Match]:
        q = tokens(question)
        taken = [False] * len(q)
        found: list[Match] = []
        for stems, key, text in self.variants:
            n = len(stems)
            for i in range(len(q) - n + 1):
                if tuple(q[i:i + n]) == stems and not any(taken[i:i + n]):
                    found.append(Match(key, text, i, i + n))
                    for j in range(i, i + n):
                        taken[j] = True
        # one entity per ID: a process and the policy with the same ID -> keep the process
        by_id: dict[str, Match] = {}
        for m in sorted(found, key=lambda m: (m.start, m.entity_type != "process")):
            ident = m.key.split(":", 1)[1]
            if ident not in by_id:
                by_id[ident] = m
        return sorted(by_id.values(), key=lambda m: m.start)
