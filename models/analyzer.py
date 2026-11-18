"""Runs the full analysis for the selected text files.

The result is a plain dictionary of numbers, words and sentences. It contains
no formatting, so the same data can be shown in the UI and written to a file.
"""

import os
import statistics

from models import bag_of_words, summarizer, tfidf
from models.preprocessing import content_tokens, split_sentences, tokenize

MAX_FILES = 3
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
TERM_LIMIT = 15
PHRASE_LIMIT = 10
SHARED_TERM_LIMIT = 10


def read_text(path):
    """Read a text file, trying common encodings in order."""
    for encoding in ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {os.path.basename(path)}")


def _statistics(text):
    """Basic counts and averages for one document."""
    words = tokenize(text)
    content = content_tokens(text)
    sentences = split_sentences(text)
    unique = set(content)
    longest = max(sentences, key=lambda s: len(tokenize(s))) if sentences else ""

    return {
        "characters": len(text),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "sentences": len(sentences),
        "words": len(words),
        "content_words": len(content),
        "unique_words": len(unique),
        "lexical_diversity": len(unique) / len(content) if content else 0.0,
        "avg_word_length": statistics.fmean(len(w) for w in content) if content else 0.0,
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0.0,
        "longest_sentence": longest,
        "longest_sentence_words": len(tokenize(longest)),
    }


def _phrases(text):
    """Most frequent word pairs and triples."""
    phrases = {}
    for label, ngram in (("bigrams", (2, 2)), ("trigrams", (3, 3))):
        try:
            vectorizer = bag_of_words.build_vectorizer(ngram_range=ngram)
            matrix = vectorizer.fit_transform([text])
            phrases[label] = bag_of_words.top_terms(
                vectorizer, matrix, limit=PHRASE_LIMIT
            )
        except ValueError:
            phrases[label] = []  # text is too short for this phrase length
    return phrases


def _bag_of_words(text):
    """Vocabulary and most frequent single words."""
    try:
        vectorizer = bag_of_words.build_vectorizer()
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return {"vocabulary": 0, "terms": [], "density": 0.0}

    vocabulary = len(vectorizer.get_feature_names_out())
    return {
        "vocabulary": vocabulary,
        "terms": bag_of_words.top_terms(vectorizer, matrix, limit=TERM_LIMIT),
        "density": matrix.nnz / (matrix.shape[0] * vocabulary) if vocabulary else 0.0,
    }


def _tfidf(text):
    """Most heavily weighted terms in this document."""
    try:
        vectorizer = tfidf.build_vectorizer()
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return {"vocabulary": 0, "terms": []}

    return {
        "vocabulary": len(vectorizer.get_feature_names_out()),
        "terms": tfidf.top_terms(vectorizer, matrix, limit=TERM_LIMIT),
    }


def _sentiment(text, sentences, model):
    """Classify the document and each sentence, if a model is available."""
    if model is None or not model.ready:
        return None
    result = model.predict_document(text)
    result["sentences"] = model.predict_sentences(sentences)
    return result


def analyze_document(path, model=None, status=None):
    """Run every analysis step for a single file."""
    name = os.path.basename(path)
    text = read_text(path)
    if not text.strip():
        raise ValueError(f"{name} is empty")

    if status:
        status(f"Analyzing {name}…")

    sentences = split_sentences(text)
    return {
        "name": name,
        "path": path,
        "text": text,
        "statistics": _statistics(text),
        "sentences": sentences,
        "sentence_words": [len(tokenize(s)) for s in sentences],
        "phrases": _phrases(text),
        "bag_of_words": _bag_of_words(text),
        "tfidf": _tfidf(text),
        "summary": summarizer.summarize(text),
        "sentiment": _sentiment(text, sentences, model),
    }


def _compare(documents):
    """Similarity and shared/distinctive terms across two or three documents."""
    if len(documents) < 2:
        return None

    try:
        vectorizer = tfidf.build_vectorizer()
        matrix = vectorizer.fit_transform([doc["text"] for doc in documents])
    except ValueError:
        return None  # the files share no usable words

    names = vectorizer.get_feature_names_out()
    similarity = tfidf.similarity_matrix(matrix)
    weights = matrix.toarray()  # at most 3 rows, so this stays small

    pairs = [
        {
            "left": documents[i]["name"],
            "right": documents[j]["name"],
            "score": float(similarity[i][j]),
        }
        for i in range(len(documents))
        for j in range(i + 1, len(documents))
    ]

    # Terms present in every document, ranked by their average weight.
    shared = []
    for index, term in enumerate(names):
        column = weights[:, index]
        if column.min() > 0:
            shared.append((term, float(column.mean())))
    shared.sort(key=lambda item: item[1], reverse=True)

    # Terms that stand out in one document compared with the others.
    distinctive = []
    for row, document in enumerate(documents):
        best_term, best_score = "", 0.0
        for index, term in enumerate(names):
            value = weights[row][index]
            if value <= 0:
                continue
            score = value / (weights[:, index].mean() + 1e-9)
            if score > best_score:
                best_term, best_score = term, score
        distinctive.append({"name": document["name"], "term": best_term})

    return {
        "pairs": pairs,
        "most_similar": max(pairs, key=lambda pair: pair["score"]),
        "shared_terms": shared[:SHARED_TERM_LIMIT],
        "distinctive": distinctive,
    }


def analyze_files(paths, model=None, status=None):
    """Analyze up to three files and compare them."""
    if not paths:
        raise ValueError("No files selected.")
    if len(paths) > MAX_FILES:
        raise ValueError(f"At most {MAX_FILES} files can be analyzed.")

    documents = [analyze_document(path, model, status) for path in paths]
    return {
        "documents": documents,
        "comparison": _compare(documents),
        "accuracy": model.accuracy if model and model.ready else None,
    }
