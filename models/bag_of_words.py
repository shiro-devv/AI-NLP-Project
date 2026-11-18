"""Bag-of-Words: every document becomes a vector of raw word counts.

A vocabulary is built from the documents, and each document is described by
how often each vocabulary word appears. Word order is ignored, which is why
it is called a "bag" of words.
"""

from sklearn.feature_extraction.text import CountVectorizer

from models.preprocessing import TOKEN_PATTERN, get_stopwords


def build_vectorizer(ngram_range=(1, 1), min_df=1):
    """Create a CountVectorizer using the project's tokenizer settings."""
    return CountVectorizer(
        lowercase=True,
        stop_words=sorted(get_stopwords()),
        token_pattern=TOKEN_PATTERN.pattern,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def top_terms(vectorizer, matrix, row=0, limit=15):
    """Return [(term, count), ...] for one row of the matrix, highest first."""
    names = vectorizer.get_feature_names_out()
    counts = matrix[row].toarray().ravel()
    order = counts.argsort()[::-1][:limit]
    return [(names[i], int(counts[i])) for i in order if counts[i] > 0]
