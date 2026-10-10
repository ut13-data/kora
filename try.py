from collections import Counter
from pathlib import Path
from statistics import mean, median

from src.pipeline import process_folder

result = process_folder(Path("data/raw"))
chunks = [c for doc in result.documents for c in doc.chunks]
lengths = [len(c.chunk_text) for c in chunks]

# 1. Overall size picture
print(f"Chunks: {len(chunks)}")
print(f"  min {min(lengths)} | median {median(lengths):.0f} | "
      f"mean {mean(lengths):.0f} | max {max(lengths)} characters")

# 2. How many chunks fall in each size band
print("\nSize bands:")
bands = [("tiny     <150", 0, 150), ("small    150-299", 150, 300),
         ("target   300-649", 300, 650), ("large    650-999", 650, 1000),
         ("oversize 1000+", 1000, 10**9)]
for label, low, high in bands:
    count = sum(1 for n in lengths if low <= n < high)
    print(f"  {label}: {count}")

# 3. Chunks per document
per_doc = Counter(f"{c.document_id} v{c.version}" for c in chunks)
print(f"\nDocuments with only 1 chunk: {sum(1 for n in per_doc.values() if n == 1)}")
print("Most chunks:", per_doc.most_common(5))

# 4. Is chunk 0 mostly the metadata header?
header_first = sum(1 for doc in result.documents
                   if doc.chunks and doc.chunks[0].chunk_text.lstrip().startswith("**Document ID:**"))
print(f"\nDocuments whose first chunk starts with the header: {header_first} of {len(result.documents)}")

# 5. Look at the extremes with your own eyes
by_size = sorted(chunks, key=lambda c: len(c.chunk_text))
print("\n--- 2 smallest chunks ---")
for c in by_size[:2]:
    print(f"[{c.chunk_id}] {len(c.chunk_text)} chars\n{c.chunk_text[:200]!r}\n")
print("--- 2 largest chunks ---")
for c in by_size[-2:]:
    print(f"[{c.chunk_id}] {len(c.chunk_text)} chars\n{c.chunk_text[:300]!r}\n")

# 6. Do chunks now carry their section, and does the key fact stay whole?
sections = {c.parent_id for c in chunks}
print(f"\nSections (parents): {len(sections)}")

print("\n--- Chunk containing '10 working days' ---")
for c in chunks:
    if "10 working days" in c.chunk_text:
        print(f"[{c.chunk_id}] section: {c.section!r} | parent: {c.parent_id}")
        print(c.chunk_text[:300], "\n")