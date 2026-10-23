"""
Understanding "when" in an employee's reply, with plain rules (no language model).

    parse_when("yesterday", today)        -> ("date", 2025-04-21)
    parse_when("last Monday", today)      -> ("date", the most recent Monday before today)
    parse_when("3 days ago", today)       -> ("date", today - 3 days)
    parse_when("14 April", today)         -> ("date", 2025-04-14)
    parse_when("on the 14th", today)      -> ("date", 2025-04-14)        (this month, if already past)
    parse_when("tomorrow", today)         -> ("future", 2025-04-23)   <- not done yet
    parse_when("last week", today)        -> ("unclear", None)        <- too vague: ask
    parse_when("04/05/2025", today)       -> ("unclear", None)        <- 4 May or 5 April? ask
    parse_when("yes", today)              -> ("none", None)           <- no date mentioned

The rule: understand what is clear, ASK about anything that isn't. A wrong
guess here would put a wrong date in the employee's record.
"""

import re
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}

_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOTTED = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")                 # German style 14.04.2025
_SLASHED = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")                  # ambiguous: ask
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+(\d{{4}}))?")
_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?")
_DAY_ONLY = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b")             # "on the 14th" = this month
_AGO = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week)s?\s+ago\b")
_WEEKDAY = re.compile(rf"\b(?:last\s+|on\s+|this\s+(?:past\s+)?)?({'|'.join(WEEKDAYS)})\b")
_VAGUE = re.compile(r"\b(?:last|this|next|past)\s+(?:week|month|year)\b|\brecently\b|\bearlier\b|\ba while\b|"
                    r"\bsome ?time\b|\bfew days\b|\bthe other day\b|\bnext\b|\bsoon\b|\blater\b")


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _result(d: date | None, today: date):
    if d is None:
        return ("unclear", None)
    return ("future", d) if d > today else ("date", d)


def parse_when(text: str, today: date):
    t = (text or "").lower().strip()

    if "day before yesterday" in t:
        return ("date", today - timedelta(days=2))
    if re.search(r"\byesterday\b", t):
        return ("date", today - timedelta(days=1))
    if re.search(r"\btomorrow\b", t):
        return ("future", today + timedelta(days=1))
    if re.search(r"\b(?:today|this morning|this afternoon|just now|just did)\b", t):
        return ("date", today)

    if m := _ISO.search(t):
        return _result(_safe(int(m[1]), int(m[2]), int(m[3])), today)
    if m := _DOTTED.search(t):
        return _result(_safe(int(m[3]), int(m[2]), int(m[1])), today)
    if _SLASHED.search(t):
        return ("unclear", None)
    for pattern, day_group, month_group in ((_DAY_MONTH, 1, 2), (_MONTH_DAY, 2, 1)):
        if m := pattern.search(t):
            year = int(m[3]) if m[3] else today.year
            return _result(_safe(year, MONTHS[m[month_group]], int(m[day_group])), today)

    if m := _DAY_ONLY.search(t):
        d = _safe(today.year, today.month, int(m[1]))
        return ("date", d) if d and d <= today else ("unclear", None)   # a later day: which month? ask
    if m := _AGO.search(t):
        n = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
        return ("date", today - timedelta(days=n * (7 if m[2] == "week" else 1)))
    if m := _WEEKDAY.search(t):
        if re.search(rf"\bnext\s+{m[1]}\b", t):
            return ("future", None)
        back = (today.weekday() - WEEKDAYS.index(m[1])) % 7 or 7      # the most recent one BEFORE today
        return ("date", today - timedelta(days=back))

    if _VAGUE.search(t) or re.search(r"\d", t):
        return ("unclear", None)
    return ("none", None)
