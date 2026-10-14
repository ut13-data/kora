"""
Phase 4 M1 - Keyword index

TF-IDF (tfidf.py) is good at "which chunks are generally about this topic"
but bad at exact matches: a chunk mentioning "PROC-HR-001" once isn't
meaningfully less relevant than one mentioning it three times, and a rare
ID token can get drowned out by ordinary words elsewhere in the question.

This module is the other kind of search: find any document/process/role/
etc ID-shaped tokens in a question (PROC-HR-001, KG-01, EMP-0085, ...),
then return every chunk whose literal text contains that exact token.
Deterministic, nothing to tune, and exactly right for "which documents
mention PROC-IMPL-001" style questions.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from src.chunking.chunker import Chunk

# ID-shaped tokens: PROC-HR-001, KG-01, EMP-0085, DEPT-CS,
# TOPIC-ACCESS-CONTROL-POLICY, COMM-PROFILE-GERMANY, ...
# Deliberately generic (2+ letters, then one or more -segments) rather than
# enumerating every prefix KORA happens to use today -- new entity types
# should be found without touching this file.
_ID_TOKEN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")

# Plain words worth indexing too: lowercased, letters/digits, 3+ characters.
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def extract_id_tokens(text: str) -> set[str]:
    """ID-shaped tokens found in TEXT AS WRITTEN. Case-sensitive on purpose:
    Nexora's own IDs are always upper-case, so a lowercase word that happens
    to contain a hyphen (like "sign-off") never gets mistaken for one."""
    return set(_ID_TOKEN.findall(text))


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_TOKEN.findall(text)}


@dataclass
class KeywordIndex:
    chunks: dict[str, Chunk]           # chunk_id -> Chunk
    id_index: dict[str, set[str]]      # ID token -> chunk_ids containing it
    word_index: dict[str, set[str]]    # lowercase word -> chunk_ids containing it

    def search_ids(self, ids: set[str]) -> dict[str, set[str]]:
        """For each ID token, the chunk_ids whose text contains it literally.
        An ID with no hits still appears in the result, mapped to an empty
        set -- that absence is itself useful ("nobody mentions KG-05")."""
        return {tid: self.id_index.get(tid, set()) for tid in ids}

    def search_words(self, words: set[str]) -> dict[str, set[str]]:
        """Same shape as search_ids, for plain words."""
        return {w: self.word_index.get(w, set()) for w in words}


def build_keyword_index(chunks: list[Chunk]) -> KeywordIndex:
    id_index: dict[str, set[str]] = defaultdict(set)
    word_index: dict[str, set[str]] = defaultdict(set)
    by_id: dict[str, Chunk] = {}

    for chunk in chunks:
        by_id[chunk.chunk_id] = chunk
        for token in extract_id_tokens(chunk.chunk_text):
            id_index[token].add(chunk.chunk_id)
        for word in _words(chunk.chunk_text):
            word_index[word].add(chunk.chunk_id)

    return KeywordIndex(chunks=by_id, id_index=dict(id_index), word_index=dict(word_index))
