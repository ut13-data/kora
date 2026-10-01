"""
Stage 5: Chunking

Splits a cleaned document's text into smaller, self-contained chunks,
each carrying enough metadata to be useful on its own later.

Strategy: split on paragraph boundaries, then group paragraphs together
up to a target chunk size, with a small overlap between consecutive
chunks so context isn't lost at the boundary.

Does NOT create embeddings or do anything retrieval-related.
"""

import uuid
from dataclasses import dataclass, field

from src.extraction.extractor import ExtractedDocument
from src.metadata.document_metadata import DocumentMetadata

# Target size in characters. Not a hard limit — a single paragraph longer
# than this will still form its own chunk rather than being cut mid-sentence.
DEFAULT_CHUNK_SIZE = 500

# How many characters from the end of one chunk get repeated at the start
# of the next, so a chunk boundary doesn't strand meaning on one side.
DEFAULT_OVERLAP = 50


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_text: str
    chunk_index: int
    document_title: str
    department: str
    document_type: str
    version: str
    status: str


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on blank lines, dropping any empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _group_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Group paragraphs together into chunks up to ~chunk_size characters.

    A single paragraph longer than chunk_size becomes its own chunk rather
    than being cut mid-sentence — this strategy prioritizes not breaking
    meaning over hitting an exact size.
    """
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # Would adding this paragraph push us over the target size?
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current)
            # Start the next chunk with the tail of the previous one, for overlap.
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_document(
    document: ExtractedDocument,
    metadata: DocumentMetadata,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a cleaned document into Chunks, each carrying key metadata."""
    paragraphs = _split_into_paragraphs(document.text)
    text_chunks = _group_paragraphs(paragraphs, chunk_size, overlap)

    chunks = []
    for index, chunk_text in enumerate(text_chunks):
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=metadata.document_id,
            chunk_text=chunk_text,
            chunk_index=index,
            document_title=metadata.title,
            department=metadata.department,
            document_type=metadata.document_type,
            version=metadata.version,
            status=metadata.status,
        )
        chunks.append(chunk)

    return chunks