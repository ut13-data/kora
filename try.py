from pathlib import Path
from src.ingestion.file_intake import identify_file
from src.extraction.extractor import extract_file, ExtractedDocument, ExtractedTable

raw_dir = Path("data/raw")

for file_path in raw_dir.iterdir():
    info = identify_file(file_path)
    result = extract_file(info)

    if isinstance(result, ExtractedDocument):
        print(f"{info.filename} -> {len(result.text)} characters")
        print(result.text[:100])
    elif isinstance(result, ExtractedTable):
        print(f"{info.filename} -> {len(result.rows)} rows")
        print(result.rows[:2])
    print("---")