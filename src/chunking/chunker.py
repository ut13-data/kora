"""
Stage 5: Chunking (structure-aware)

1. Strip the metadata header at the top of the document
   (it's already stored as fields on every chunk).
2. Markdown files: split at headings (#, ##, ###) so each chunk stays
   inside one section, with its heading attached.
3. Sections longer than chunk_size are split further, preferring
   paragraph breaks, then line breaks, then spaces -- never mid-word.
4. Non-Markdown files (PDF, DOCX, TXT) have no headings, so they go
   straight to step 3.

Every chunk records its section heading and a parent_id for the full
section, so later phases can add context or return the whole section
without re-chunking.

Does NOT create embeddings.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.extraction.extractor import ExtractedDocument
from src.ingestion.file_intake import FileType
from src.metadata.document_metadata import DocumentMetadata

# ~1200 characters is roughly 300 tokens: a reasonable starting ceiling.
# Most Nexora sections are shorter, so sections decide the natural size.
DEFAULT_CHUNK_SIZE = 1200

# Only used when ONE section has to be split into several chunks.
# The splitter overlaps on word boundaries, so no mid-word fragments.
DEFAULT_OVERLAP = 100

# Heading levels to split on, and the metadata key the splitter gives each.
_HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Nexora header lines like "**Document ID:** DOC-001".
_HEADER_LINE = re.compile(r"^\*\*[^*]+:\*\*")


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
    section: str     # e.g. "Employee Onboarding Process > Steps" ("" if none)
    parent_id: str   # ID of the full section this chunk belongs to


def _strip_metadata_header(text: str) -> str:
    """Remove the '**Label:** value' block at the very top, plus the '---'
    line that follows it. Text without such a header is returned unchanged."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or _HEADER_LINE.match(lines[i].strip())):
        i += 1
    if i > 0 and i < len(lines) and lines[i].strip() == "---":
        i += 1
    return "\n".join(lines[i:]).strip()


def _is_heading_only(text: str) -> bool:
    """True if every non-empty line is a heading, i.e. a section with no content."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.lstrip().startswith("#") for line in lines)


def _split_into_sections(text: str, is_markdown: bool) -> list[tuple[str, str]]:
    """Return a list of (section_name, section_text)."""
    if not is_markdown:
        return [("", text)]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADINGS, strip_headers=False)

    sections: list[tuple[str, str]] = []
    pending = ""   # heading-only text waiting to be attached to the next section
    for piece in splitter.split_text(text):
        content = piece.page_content.strip()
        if _is_heading_only(content):
            pending += content + "\n\n"
            continue
        # e.g. {"h1": "Employee Onboarding Process", "h2": "Steps"}
        #   -> "Employee Onboarding Process > Steps"
        parts = []
        for _, key in _HEADINGS:
            if key in piece.metadata:
                heading = piece.metadata[key]
                # A heading that wrapped onto two lines leaves an unclosed "(":
                # keep only the part before it.
                if "(" in heading and ")" not in heading:
                    heading = heading.split("(")[0].strip()
                parts.append(heading)
        name = " > ".join(parts)
        sections.append((name, pending + content))
        pending = ""

    if pending:   # a document that ends with a heading and nothing after it
        sections.append(("", pending.strip()))
    return sections


def _id_base(metadata: DocumentMetadata) -> str:
    """Document ID if known, else the filename without extension."""
    if metadata.document_id != "unknown":
        return metadata.document_id
    return Path(metadata.source_filename).stem


def chunk_document(
    document: ExtractedDocument,
    metadata: DocumentMetadata,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a cleaned document into section-aware Chunks."""
    text = _strip_metadata_header(document.text)
    is_markdown = document.file_info.file_type == FileType.MARKDOWN

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],   # paragraph, line, word, last resort
    )

    base = _id_base(metadata)
    chunks: list[Chunk] = []

    for section_index, (section_name, section_text) in enumerate(_split_into_sections(text, is_markdown)):
        parent_id = f"{base}_v{metadata.version}_s{section_index:02d}"

        # Short sections stay whole; long ones are split by size.
        pieces = [section_text] if len(section_text) <= chunk_size else size_splitter.split_text(section_text)

        for piece in pieces:
            if not piece.strip():
                continue
            index = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{base}_v{metadata.version}_{index:03d}",
                document_id=metadata.document_id,
                chunk_text=re.sub(r"[ \t]+\n", "\n", piece).strip(),
                chunk_index=index,
                document_title=metadata.title,
                department=metadata.department,
                document_type=metadata.document_type,
                version=metadata.version,
                status=metadata.status,
                section=section_name,
                parent_id=parent_id,
            ))

    return chunks