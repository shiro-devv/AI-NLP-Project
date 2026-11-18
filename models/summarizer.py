"""Extractive summarization: pick the most informative sentences.

Each sentence is treated as a small document, scored with TF-IDF, and the
highest scoring sentences are kept in their original order. The summary is
therefore made only of sentences that already exist in the text.
"""

import math

from models import tfidf
from models.preprocessing import content_tokens, split_sentences, tokenize

RATIO = 0.2  # keep about 20% of the sentences
MIN_SENTENCES = 3
MAX_SENTENCES = 7
LEAD_BONUS = 1.15  # the first sentences usually introduce the topic


def summarize(text):
    """Return the summary sentences, their positions and a compression ratio."""
    sentences = split_sentences(text)
    if not sentences:
        return {"sentences": [], "chosen": [], "summary": [], "compression": 0.0}

    count = max(MIN_SENTENCES, round(len(sentences) * RATIO))
    count = min(count, MAX_SENTENCES, len(sentences))

    vectorizer = tfidf.build_vectorizer()
    try:
        matrix = vectorizer.fit_transform(sentences)
        weights = [float(matrix[i].sum()) for i in range(len(sentences))]
    except ValueError:
        # No usable words (very short text): fall back to sentence length.
        weights = [len(content_tokens(sentence)) for sentence in sentences]

    scores = []
    for index, sentence in enumerate(sentences):
        words = content_tokens(sentence)
        # Divide by length so long sentences do not win just by being long.
        score = weights[index] / math.sqrt(len(words)) if words else 0.0
        scores.append(score * LEAD_BONUS if index < 2 else score)

    # Take the best `count` sentences, then restore the reading order.
    best = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:count]
    chosen = sorted(best)
    summary = [sentences[i] for i in chosen]

    original_words = len(tokenize(text))
    summary_words = len(tokenize(" ".join(summary)))
    compression = summary_words / original_words if original_words else 0.0

    return {
        "sentences": sentences,
        "chosen": chosen,
        "summary": summary,
        "compression": compression,
    }
