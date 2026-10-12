"""Name matchers built from discovered entities (titles plus documented aliases only)."""

import re

from src.intelligence.entities import EntityIndex
from src.intelligence.models import PRODUCT, ROLE
from src.intelligence.text import TermMatcher


def role_matcher(index: EntityIndex) -> TermMatcher:
    matcher = TermMatcher()
    for role in index.of_type(ROLE):
        matcher.add(role.name, role.key)
        for alias in role.aliases:
            matcher.add(alias, role.key)
    return matcher


def product_matcher(index: EntityIndex) -> TermMatcher:
    matcher = TermMatcher()
    for product in index.of_type(PRODUCT):
        matcher.add(product.name, product.key)
        for alias in product.aliases:
            matcher.add(alias, product.key, case_sensitive=True)
    return matcher


# Words for "someone's manager" that are NOT a role in roles.csv. Run on text
# where known role titles are already masked, so 'Engineering Manager' never counts.
GENERIC_MANAGER = re.compile(
    r"(?<![\w&])(?:[Dd]irect |[Hh]iring |[Ll]ine |[Aa]pproving |[Rr]elevant )?[Mm]anagers?(?![\w])"
    r"|(?<![\w&])[Dd]epartment heads?(?![\w])"
)


def generic_roles(masked_text: str) -> list[str]:
    seen, out = set(), []
    for m in GENERIC_MANAGER.finditer(masked_text):
        label = m.group(0).strip()
        label = label[0].upper() + label[1:]
        if label.lower() not in seen:
            seen.add(label.lower())
            out.append(label)
    return out
