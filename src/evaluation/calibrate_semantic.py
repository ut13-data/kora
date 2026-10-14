"""
Phase 4 M5b - calibrate the semantic thresholds on the REAL model.

    python -m src.evaluation.calibrate_semantic

Prints a short summary to paste back. It never changes the config.

The thresholds in config (semantic_*) decide when a meaning match is trusted.
Good values depend on the model, so they are measured, not guessed:

  * For answerable questions: the best meaning score among sentences of the
    RIGHT documents (gold) vs among sentences of OTHER documents.
  * For no-answer questions: the best meaning score anywhere -- a meaning
    threshold must stay ABOVE these, or KORA would answer them.
  * For paraphrase words: how well a question word matches its true
    counterpart in the source ("trip" vs "travel") vs unrelated words.

Only development + held-out v1/v2 questions are used (all already seen).
Held-out v3 is NOT touched: it is the final, untouched check for M5b.
"""

import json
from pathlib import Path

import numpy as np

from src.retrieval.answer import GENERIC_WORDS, KoraAnswerer, split_sentences
from src.retrieval.router import normalise

SETS = [Path("data/evaluation/eval_questions.json"), Path("data/evaluation/heldout_questions.json"),
        Path("data/evaluation/heldout_v2_questions.json")]

# Paraphrase pairs from the knowledge base's own wording (question word, source word),
# plus unrelated pairs. Used only to calibrate semantic_word_min.
PARAPHRASES = [("trip", "travel"), ("money", "expenses"), ("refund", "reimbursed"), ("pushed", "deploys"),
               ("doctor", "practitioner"), ("boss", "manager"), ("hire", "employee"), ("laptop", "device")]
UNRELATED = [("pets", "office"), ("canteen", "facilities"), ("stock", "options"), ("bonus", "approval"),
             ("car", "laptop"), ("dress", "address"), ("salary", "sprint"), ("parking", "procurement")]
# Name pairs: for information only. Names are never matched by meaning (a rule,
# not a threshold), because two cities are close in meaning but not the same place.
NAMES = [("paris", "berlin"), ("austin", "munich"), ("london", "bengaluru"), ("tokyo", "berlin")]


def _pct(values, p):
    return round(float(np.percentile(values, p)), 3) if values else None


def main() -> None:
    kora = KoraAnswerer.load()
    sem = kora.retriever.chunks.semantic
    print(f"model: {kora.retriever.semantic_status}")
    if sem is None:
        print("Embeddings are OFF -- nothing to calibrate. Is sentence-transformers installed?")
        return
    min_words = kora.config.t("min_quote_words")
    sentences_by_doc: dict[str, list[str]] = {}
    for c in kora._chunks.values():
        sentences_by_doc.setdefault(c.document_id, []).extend(split_sentences(c.chunk_text, min_words))
    all_docs = list(sentences_by_doc)

    gold_best, other_best, gap, no_answer = [], [], [], []
    for path in SETS:
        for q in json.loads(path.read_text(encoding="utf-8"))["questions"]:
            scores = {d: float(max(sem.sentence_scores(q["question"], s))) if s else 0.0
                      for d, s in sentences_by_doc.items()}
            if q["gold_documents"]:
                g = max(scores.get(d, 0.0) for d in q["gold_documents"])
                excluded = set(q["gold_documents"]) | set(q["acceptable_documents"])
                o = max(v for d, v in scores.items() if d not in excluded)
                gold_best.append(g)
                other_best.append(o)
                gap.append(g - o)
            elif "not_found" in q["expected_status"]:
                top = max(scores.items(), key=lambda kv: kv[1])
                no_answer.append((q["id"], round(top[1], 3), top[0], q["question"]))

    print(f"\nanswerable questions: {len(gold_best)}")
    print(f"  best RIGHT-document sentence: p10 {_pct(gold_best, 10)}  p25 {_pct(gold_best, 25)}  "
          f"median {_pct(gold_best, 50)}")
    print(f"  best OTHER-document sentence: median {_pct(other_best, 50)}  p75 {_pct(other_best, 75)}  "
          f"p90 {_pct(other_best, 90)}")
    print(f"  right minus other: median {_pct(gap, 50)}; right wins in {sum(x > 0 for x in gap)}/{len(gap)}")
    print(f"\nno-answer questions (a meaning threshold must stay ABOVE these):")
    for qid, score, doc, text in no_answer:
        print(f"  {qid} {score}  ({doc})  {text}")
    na_max = max((s for _, s, _, _ in no_answer), default=0.0)

    para = [float(kora._semantic.sentence_scores(a, [b])[0]) for a, b in PARAPHRASES]
    unrel = [float(kora._semantic.sentence_scores(a, [b])[0]) for a, b in UNRELATED]
    print(f"\nword pairs: paraphrase min {min(para):.3f} median {np.median(para):.3f} | "
          f"unrelated max {max(unrel):.3f} median {np.median(unrel):.3f}")
    print("  " + ", ".join(f"{a}/{b} {s:.2f}" for (a, b), s in zip(PARAPHRASES + UNRELATED, para + unrel)))
    names = [float(kora._semantic.sentence_scores(a, [b])[0]) for a, b in NAMES]
    print("  names (blocked by rule, shown for information): "
          + ", ".join(f"{a}/{b} {s:.2f}" for (a, b), s in zip(NAMES, names)))

    # Suggestions (conservative: precision first).
    strong = round(max(na_max + 0.05, _pct(gold_best, 25) or 0), 2)
    quote = round(max(_pct(other_best, 75) or 0, na_max), 2)
    word = round((max(unrel) + min(para)) / 2, 2) if min(para) > max(unrel) else round(max(unrel) + 0.05, 2)
    print("\nsuggested thresholds (precision first):")
    print(f"  semantic_strong_evidence {strong}   (above every no-answer question, near the right-document p25)")
    print(f"  semantic_quote_min       {quote}   (above most wrong-document sentences)")
    print(f"  semantic_word_min        {word}   (between unrelated and paraphrase word pairs)")
    print(f"  current config: strong {kora.config.t('semantic_strong_evidence')}, "
          f"quote {kora.config.t('semantic_quote_min')}, word {kora.config.t('semantic_word_min')}")


if __name__ == "__main__":
    main()
