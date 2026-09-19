from dataclasses import dataclass
from pathlib import Path

@dataclass
class FileInfo:
    filename: str
    size_bytes: int

p = Path("README.md")
info = FileInfo(filename=p.name, size_bytes=100)  # size_bytes hardcoded for now
print(info)