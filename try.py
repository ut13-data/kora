from src.retrieval.answer import KoraAnswerer
kora = KoraAnswerer.load()
print(kora.answer("Are there conflicting instructions about security training?").render())