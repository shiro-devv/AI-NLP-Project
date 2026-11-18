"""Text clean-up helpers shared by every model.

Stopwords and sentence splitting come from NLTK. The packages are downloaded
on first use and then cached by NLTK, so later runs work offline.
"""

import re

import nltk

# A token starts with a letter and may contain letters, apostrophes and
# hyphens, so "don't" and "well-known" stay whole words.
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]+")

RESOURCES = (
    ("stopwords", "corpora/stopwords"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
)

# Negations are kept on purpose: removing them would turn "not good" into
# "good" and give sentiment analysis the opposite meaning.
NEGATIONS = {"no", "nor", "not", "never", "none", "cannot"}

_stopwords = None


def ensure_nltk_data(status=None):
    """Download the NLTK packages used by the app, if they are missing."""
    for package, path in RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            if status:
                status(f"Downloading NLTK resource: {package}…")
            nltk.download(package, quiet=True)


def get_stopwords():
    """Return NLTK's English stopwords without the negations (cached)."""
    global _stopwords
    if _stopwords is None:
        ensure_nltk_data()
        from nltk.corpus import stopwords

        try:
            words = stopwords.words("english")
        except LookupError as exc:
            raise RuntimeError(
                "NLTK stopwords are missing. Run: "
                "python -m nltk.downloader stopwords"
            ) from exc

        _stopwords = frozenset(
            word
            for word in words
            if word not in NEGATIONS and not word.endswith("n't")
        )
    return _stopwords


def split_sentences(text):
    """Split text into sentences, with a regex fallback if NLTK data is missing."""
    try:
        from nltk.tokenize import sent_tokenize

        return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
    except LookupError:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]


def tokenize(text):
    """Return lower-cased words, punctuation and numbers removed."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def content_tokens(text):
    """Tokens with stopwords removed; used for statistics and summaries."""
    stopwords = get_stopwords()
    return [token for token in tokenize(text) if token not in stopwords]
