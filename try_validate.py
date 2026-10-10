from collections import Counter
from pathlib import Path

from src.pipeline import process_folder
from src.storage.store import save_result

result = process_folder(Path("data/raw"))
save_result(result, Path("data/processed"), Path("data/chunks"))

issues = result.validation_issues
print(f"Validation issues: {len(issues)}")
print("  by severity:", dict(Counter(i.severity.value for i in issues)))
print()
for issue in issues:
    print(f"[{issue.severity.value.upper()}] {issue.message}  (subject: {issue.subject})")