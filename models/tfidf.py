"""TF-IDF: word counts weighted by how rare a word is across documents.

TF (term frequency) rewards words that appear often in one document, while
IDF (inverse document frequency) penalises words that appear in every
document. The result is a vector that highlights the words that make a
document distinctive.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.preprocessing import TOKEN_PATTERN, get_stopwords


def build_vectorizer(ngram_range=(1, 1), min_df=1):
    """Create a TfidfVectorizer using the project's tokenizer settings."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words=sorted(get_stopwords()),
        token_pattern=TOKEN_PATTERN.pattern,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        norm="l2",
    )


def top_terms(vectorizer, matrix, row=0, limit=15):
    """Return [(term, weight), ...] for one row of the matrix, highest first."""
    names = vectorizer.get_feature_names_out()
    weights = matrix[row].toarray().ravel()
    order = weights.argsort()[::-1][:limit]
    return [(names[i], float(weights[i])) for i in order if weights[i] > 0]


def similarity_matrix(matrix):
    """Cosine similarity between every pair of documents (0 = unrelated, 1 = same)."""
    return cosine_similarity(matrix)
