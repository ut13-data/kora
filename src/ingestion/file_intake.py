"""
Stage 1: Upload / Input Layer

Responsibility of this module (and ONLY this module, for now):
    - Look at a file on disk.
    - Record basic facts about it (filename, type, size, source, timestamp).
    - Reject file types KORA doesn't support.

This module does NOT read the file's contents. That's Stage 2 (Extraction).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class FileType(Enum):
    """The file types KORA is allowed to ingest in this phase.

    Using an Enum instead of raw strings means a typo like "makdown"
    fails immediately and loudly, instead of silently creating a
    file type that doesn't exist anywhere else in the system.
    """
    MARKDOWN = "markdown"
    CSV = "csv"
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"


# Maps a real file extension (what you see in the filesystem) to our
# internal FileType. Extensions are lowercased before lookup so
# "FILE.PDF" and "file.pdf" behave the same.
EXTENSION_TO_FILETYPE: dict[str, FileType] = {
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".csv": FileType.CSV,
    ".txt": FileType.TXT,
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
}


class UnsupportedFileTypeError(Exception):
    """Raised when a file's extension isn't one KORA knows how to handle.

    A custom exception (rather than a generic ValueError) means later
    code can specifically catch "unsupported file type" problems
    without accidentally catching unrelated bugs.
    """
    pass


@dataclass
class FileInfo:
    """A record of basic facts about one uploaded file.

    This is the ONLY output of Stage 1. Nothing here reads file
    contents — it's purely filesystem metadata.
    """
    filename: str
    file_type: FileType
    size_bytes: int
    source_path: str
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def identify_file(path: Path) -> FileInfo:
    """Inspect a single file on disk and return a FileInfo record.

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        UnsupportedFileTypeError: if the extension isn't in EXTENSION_TO_FILETYPE.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    extension = path.suffix.lower()
    file_type = EXTENSION_TO_FILETYPE.get(extension)

    if file_type is None:
        raise UnsupportedFileTypeError(
            f"'{extension}' is not a supported file type "
            f"(supported: {sorted(e for e in EXTENSION_TO_FILETYPE)})"
        )

    return FileInfo(
        filename=path.name,
        file_type=file_type,
        size_bytes=path.stat().st_size,
        source_path=str(path.resolve()),
    )