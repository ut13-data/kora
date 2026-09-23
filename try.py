from pathlib import Path
from src.ingestion.file_intake import identify_file
from src.extraction.extractor import extract_file, ExtractedDocument, ExtractedTable
from src.cleaning.cleaner import clean_extracted
from src.metadata.document_metadata import identify_document_metadata

raw_dir = Path("data/raw")

for file_path in raw_dir.iterdir():
    info = identify_file(file_path)
    extracted = extract_file(info)
    cleaned = clean_extracted(extracted)

    if isinstance(cleaned, ExtractedDocument):
        metadata = identify_document_metadata(cleaned)
        print(f"{info.filename} -> {metadata}")
    elif isinstance(cleaned, ExtractedTable):
        print(f"{info.filename} -> (CSV, no document metadata stage)")
    print("---")