"""
Small, deterministic text tools shared by the Phase 3 modules.

  blocks()      Markdown -> headings, numbered steps, bullets, paragraphs, table rows, code
  sentences()   paragraph -> sentences
  TermMatcher   find known names (roles, products) in text, longest name first
  durations() / amounts() / percents() / id_refs()   pull out values a rule depends on

No NLP model is involved: every result can be traced to a regular expression.
"""

import re
from dataclasses import dataclass, field

# ---------------- Markdown blocks ----------------

_NUMBERED = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
# '**Kickoff** — text'  or  '**Directness:** text' (colon inside the bold)
_BOLD_TITLE = re.compile(r"^\*\*(.+?)\*\*\s*(?:—|–|--|-|:)\s*(.*)$|^\*\*([^*]+?):\*\*\s*(.*)$")


@dataclass
class Block:
    kind: str                 # heading | numbered | bullet | paragraph | table_row | code
    text: str                 # wrapped lines joined with single spaces
    number: int | None = None
    title: str | None = None  # "Kickoff" in "**Kickoff** — Implementation Consultant reviews..."
    body: str = ""            # text after the title (or the whole text if no title)
    raw_lines: list[str] = field(default_factory=list)


def _finish(block: Block | None, out: list[Block]) -> None:
    if block is None:
        return
    block.text = " ".join(block.text.split())
    match = _BOLD_TITLE.match(block.text) if block.kind != "code" else None
    if match:
        title, body = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        block.title, block.body = title.strip(), body.strip()
    else:
        block.body = block.text
    out.append(block)


def blocks(text: str) -> list[Block]:
    """Split Markdown into blocks, keeping wrapped list items together.

    Works on the ORIGINAL text, where a blank line ends a paragraph and an
    indented line continues the list item above it.
    """
    out: list[Block] = []
    current: Block | None = None
    in_code = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            _finish(current, out)
            current = None
            in_code = not in_code
            continue
        if in_code:
            out.append(Block("code", line, raw_lines=[line]))
            continue
        if not stripped:
            _finish(current, out)
            current = None
            continue
        if stripped.startswith("#"):
            _finish(current, out)
            current = None
            out.append(Block("heading", stripped.lstrip("#").strip(), raw_lines=[line]))
            continue
        if stripped.startswith("|"):
            _finish(current, out)
            current = None
            out.append(Block("table_row", stripped, raw_lines=[line]))
            continue

        numbered = _NUMBERED.match(line)
        bullet = _BULLET.match(line) if not stripped.startswith("**") else None
        if numbered:
            _finish(current, out)
            current = Block("numbered", numbered.group(3), number=int(numbered.group(2)), raw_lines=[line])
        elif bullet:
            _finish(current, out)
            current = Block("bullet", bullet.group(2), raw_lines=[line])
        elif current is not None and current.kind == "paragraph" and _BOLD_TITLE.match(stripped):
            # "**Role** — ..." directly under another paragraph starts a new one.
            _finish(current, out)
            current = Block("paragraph", stripped, raw_lines=[line])
        elif current is not None:
            current.text += " " + stripped
            current.raw_lines.append(line)
        else:
            current = Block("paragraph", stripped, raw_lines=[line])

    _finish(current, out)
    return out


def table_rows(rows: list[Block]) -> list[dict[str, str]]:
    """Consecutive '| a | b |' rows -> list of {header: cell} dicts."""
    cells = [[c.strip() for c in r.text.strip().strip("|").split("|")] for r in rows]
    if len(cells) < 2:
        return []
    header, body = cells[0], [c for c in cells[1:] if not all(set(x) <= set("-: ") for x in c)]
    return [dict(zip(header, row)) for row in body]


def tables(all_blocks: list[Block]) -> list[list[dict[str, str]]]:
    """Every Markdown table in a block list."""
    found, run = [], []
    for b in all_blocks + [Block("end", "")]:
        if b.kind == "table_row":
            run.append(b)
        elif run:
            found.append(table_rows(run))
            run = []
    return [t for t in found if t]


# ---------------- Sentences ----------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(€*])")
_MARKUP = re.compile(r"\*\*|__|`")


def plain(text: str) -> str:
    """Drop bold/code markers and squeeze whitespace."""
    return " ".join(_MARKUP.sub("", text).split())


def sentences(text: str) -> list[str]:
    text = plain(text)
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()] if text else []


def norm_key(text: str) -> str:
    """Letters and digits only, lowercase. Used to find a statement inside a chunk."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


# ---------------- Known-name matching ----------------

@dataclass
class Mention:
    key: str      # entity key, e.g. "role:ROLE-022"
    term: str     # the exact text that matched
    start: int
    end: int


class TermMatcher:
    """Finds known names in text. Longest names are matched first and claim
    their characters, so 'Senior Software Engineer' is never ALSO counted as
    'Software Engineer'. A trailing plural 's' is allowed ('Sales Managers')."""

    def __init__(self) -> None:
        self._terms: list[tuple[str, str, bool]] = []
        self._compiled: list[tuple[re.Pattern, str]] | None = None

    def add(self, term: str, key: str, case_sensitive: bool | None = None) -> None:
        term = term.strip()
        if not term:
            return
        if case_sensitive is None:
            case_sensitive = term.isupper() and len(term) <= 5    # acronyms: 'AE', 'CSM', 'CISO'
        self._terms.append((term, key, case_sensitive))
        self._compiled = None

    def _patterns(self) -> list[tuple[re.Pattern, str]]:
        if self._compiled is None:
            ordered = sorted(set(self._terms), key=lambda t: -len(t[0]))
            self._compiled = [
                (re.compile(rf"(?<![\w-]){re.escape(term)}(?:e?s)?(?![\w-])", 0 if cs else re.I), key)
                for term, key, cs in ordered
            ]
        return self._compiled

    def find(self, text: str) -> list[Mention]:
        taken = [False] * len(text)
        found: list[Mention] = []
        for pattern, key in self._patterns():
            for m in pattern.finditer(text):
                if any(taken[m.start():m.end()]):
                    continue
                for i in range(m.start(), m.end()):
                    taken[i] = True
                found.append(Mention(key, m.group(0), m.start(), m.end()))
        return sorted(found, key=lambda m: m.start)

    def keys(self, text: str) -> list[str]:
        seen, out = set(), []
        for m in self.find(text):
            if m.key not in seen:
                seen.add(m.key)
                out.append(m.key)
        return out

    def mask(self, text: str) -> str:
        """Replace every match with underscores, so later patterns
        (e.g. a bare 'manager') don't fire inside a known role title."""
        chars = list(text)
        for m in self.find(text):
            chars[m.start:m.end] = "_" * (m.end - m.start)
        return "".join(chars)


# ---------------- Values inside rules ----------------

_DURATION = re.compile(
    r"(?:(?:within|in|after|for|every|at least every|more than|exceeding|over|under|up to|than|at)\s+"
    r"(?:the\s+)?(?:first\s+)?)?"
    r"(\d+(?:\s*[–-]\s*\d+)?\+?)\s*"
    r"(business hours?|business days?|working days?|calendar days?|consecutive working days?|days?|hours?|minutes?|weeks?|months?|years?)\b",
    re.I,
)
_AMOUNT = re.compile(r"€\s?\d[\d,.]*(?:\s?(?:–|-|to)\s?€?\s?\d[\d,.]*)?(?:/(?:month|year))?")
_PERCENT = re.compile(r"\d+(?:\s?[–-]\s?\d+)?%")
_IDS = re.compile(r"(?<![\w-])(?:PROC|DOC)-[A-Z0-9]+(?:-[A-Z0-9]+)*(?![\w-])")
_GAP_CODE = re.compile(r"\bKG-\d+\b")


def durations(text: str) -> list[dict]:
    out = []
    for m in _DURATION.finditer(text):
        unit = m.group(2).lower()
        out.append({"text": m.group(0).strip(), "value": m.group(1).replace(" ", ""), "unit": unit})
    return out


def amounts(text: str) -> list[str]:
    """'Above €10,000.' -> ['€10,000'] (a sentence's full stop is not part of the amount)."""
    return [m.group(0).strip().rstrip(".,") for m in _AMOUNT.finditer(text)]


def percents(text: str) -> list[str]:
    return [m.group(0) for m in _PERCENT.finditer(text)]


def id_refs(text: str) -> list[str]:
    seen, out = set(), []
    for m in _IDS.finditer(text):
        if m.group(0) not in seen:
            seen.add(m.group(0))
            out.append(m.group(0))
    return out


def gap_codes(text: str) -> list[str]:
    return sorted(set(_GAP_CODE.findall(text)))
