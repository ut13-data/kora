"""
Phase 5 M4: a quoted line about ANOTHER office is noise.

"How many days of annual leave do Germany-based employees get?" used to also
quote "US-based employees (LOC-AUS): 20 days/year" from the same passage.
Relevant second lines stay (UT: "we want relevant information even if it's
two lines long"); only lines naming a different office, and none of the asked
ones, are dropped.
"""

import pytest

from src.retrieval.answer import KoraAnswerer, location_patterns, places_in
from src.retrieval.config import load_config

ROWS = [{"location_id": "LOC-MUC", "city": "Munich", "country": "Germany"},
        {"location_id": "LOC-BER", "city": "Berlin", "country": "Germany"},
        {"location_id": "LOC-AUS", "city": "Austin", "country": "USA"},
        {"location_id": "LOC-LON", "city": "London", "country": "United Kingdom"},
        {"location_id": "LOC-BLR", "city": "Bengaluru", "country": "India"}]
LOCS = location_patterns(ROWS)


@pytest.mark.parametrize("text,expected", [
    ("Germany-based employees", {"LOC-MUC", "LOC-BER"}),          # a country names all its offices
    ("US-based employees (LOC-AUS)", {"LOC-AUS"}),
    ("employees in the UK", {"LOC-LON"}),
    ("the Munich office", {"LOC-MUC"}),
    ("Please tell us when you are back.", set()),                 # "us" is not the US
    ("German statutory minimum", {"LOC-MUC", "LOC-BER"}),
    ("All employees get 5 sick days.", set()),
])
def test_places_in(text, expected):
    assert places_in(text, LOCS) == expected


@pytest.fixture(scope="module")
def kora(intel):
    return KoraAnswerer.from_intelligence(intel, load_config())


def test_germany_leave_answer_has_no_us_line(kora):
    a = kora.answer("How many days of annual leave do Germany-based employees get?")
    texts = [l.text for l in a.lines]
    assert a.status == "answered" and any("28 days" in t for t in texts)
    assert not any("US-based" in t or "LOC-AUS" in t for t in texts)


def test_question_without_an_office_keeps_every_office(kora):
    a = kora.answer("How many days of annual leave do employees get?")
    offices = set().union(*(places_in(l.text, kora._locations) for l in a.lines)) if a.lines else set()
    assert len(offices) >= 2            # nothing filtered when no office was asked about


def test_lines_naming_no_office_still_stay(kora):
    """The filter only removes lines naming OTHER offices; general lines stay."""
    a = kora.answer("How many days of annual leave do employees in London get?")
    assert a.status == "answered"
    assert all(not places_in(l.text, kora._locations) or "LOC-LON" in places_in(l.text, kora._locations)
               for l in a.lines)
