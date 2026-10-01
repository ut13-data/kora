from pathlib import Path
from src.pipeline import process_folder
from src.storage.store import save_result

result = process_folder(Path("data/raw"))
written = save_result(result, Path("data/processed"), Path("data/chunks"))

print(f"Wrote {len(written)} file(s):")
for p in written:
    print(f"  {p}")