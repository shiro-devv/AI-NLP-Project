"""Sentiment classification with Naive Bayes on NLTK's movie_reviews corpus.

The same 2,000 labelled reviews are used to train two classifiers so the
report can compare the two representations from the assignment:

  * TF-IDF + Multinomial Naive Bayes   (main prediction)
  * Bag-of-Words + Multinomial Naive Bayes (comparison)

Training takes a few seconds, so the fitted models are cached in data/ and
reused on later runs. The corpus is downloaded once if it is missing.
"""

import os

import joblib
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from models.preprocessing import (
    TOKEN_PATTERN,
    content_tokens,
    ensure_nltk_data,
    get_stopwords,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "data", "sentiment_nb.joblib")
CACHE_VERSION = 1

NGRAM_RANGE = (1, 2)
MIN_DF = 2
ALPHA = 0.3  # smoothing; smaller values trust the training data more
DECISIVE_LIMIT = 8


def _build_vectorizer(kind):
    """Create a vectorizer; both models use identical vocabulary settings."""
    settings = {
        "lowercase": True,
        "stop_words": sorted(get_stopwords()),
        "token_pattern": TOKEN_PATTERN.pattern,
        "ngram_range": NGRAM_RANGE,
        "min_df": MIN_DF,
    }
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **settings)
    return CountVectorizer(**settings)


class SentimentModel:
    """Wraps the two trained classifiers and exposes simple prediction helpers."""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_clf = None
        self.bow_vectorizer = None
        self.bow_clf = None
        self.accuracy = None

    @property
    def ready(self):
        return self.tfidf_clf is not None

    # ---------------------------------------------------------------- training

    def load_or_train(self, status=None):
        """Load the cached models, or train and cache them on first use."""
        if self._load():
            return self

        self._ensure_corpus(status)
        if status:
            status("Training sentiment classifier…")

        documents, labels = [], []
        for label in ("pos", "neg"):
            for file_id in movie_reviews.fileids(label):
                documents.append(movie_reviews.raw(file_id))
                labels.append(label)

        self.tfidf_vectorizer = _build_vectorizer("tfidf")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.tfidf_clf = MultinomialNB(alpha=ALPHA).fit(tfidf_matrix, labels)

        self.bow_vectorizer = _build_vectorizer("bow")
        bow_matrix = self.bow_vectorizer.fit_transform(documents)
        self.bow_clf = MultinomialNB(alpha=ALPHA).fit(bow_matrix, labels)

        # 5-fold cross-validation gives an honest accuracy estimate.
        self.accuracy = float(
            cross_val_score(self.tfidf_clf, tfidf_matrix, labels, cv=5).mean()
        )
        self._save()
        return self

    def _ensure_corpus(self, status):
        """Download the review corpus and sentence tokenizer if needed."""
        ensure_nltk_data(status)
        try:
            nltk.data.find("corpora/movie_reviews")
        except LookupError:
            if status:
                status("Downloading NLTK resource: movie_reviews…")
            nltk.download("movie_reviews", quiet=True)

        try:
            nltk.data.find("corpora/movie_reviews")
        except LookupError as exc:
            raise RuntimeError(
                "The movie_reviews corpus is missing. Run: "
                "python -m nltk.downloader movie_reviews"
            ) from exc

    def _save(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        joblib.dump(
            {
                "version": CACHE_VERSION,
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "tfidf_clf": self.tfidf_clf,
                "bow_vectorizer": self.bow_vectorizer,
                "bow_clf": self.bow_clf,
                "accuracy": self.accuracy,
            },
            CACHE_PATH,
        )

    def _load(self):
        if not os.path.exists(CACHE_PATH):
            return False
        try:
            data = joblib.load(CACHE_PATH)
        except Exception:
            return False  # a corrupt cache is simply rebuilt
        if data.get("version") != CACHE_VERSION:
            return False
        self.tfidf_vectorizer = data["tfidf_vectorizer"]
        self.tfidf_clf = data["tfidf_clf"]
        self.bow_vectorizer = data["bow_vectorizer"]
        self.bow_clf = data["bow_clf"]
        self.accuracy = data["accuracy"]
        return True

    # -------------------------------------------------------------- prediction

    def predict_document(self, text):
        """Classify a whole document and explain the decision."""
        tfidf_label, tfidf_confidence = self._predict_one(
            self.tfidf_vectorizer, self.tfidf_clf, [text]
        )[0]
        bow_label, bow_confidence = self._predict_one(
            self.bow_vectorizer, self.bow_clf, [text]
        )[0]

        return {
            "tfidf_label": tfidf_label,
            "tfidf_confidence": tfidf_confidence,
            "bow_label": bow_label,
            "bow_confidence": bow_confidence,
            "agree": tfidf_label == bow_label,
            "decisive_words": self._decisive_words(text, tfidf_label),
        }

    def predict_sentences(self, sentences):
        """Label every sentence; sentences without known words are 'neutral'."""
        useful = [i for i, s in enumerate(sentences) if content_tokens(s)]
        results = [{"label": "neutral", "confidence": 0.0} for _ in sentences]

        if useful:
            predictions = self._predict_one(
                self.tfidf_vectorizer,
                self.tfidf_clf,
                [sentences[i] for i in useful],
            )
            for index, prediction in zip(useful, predictions):
                results[index] = {
                    "label": prediction[0],
                    "confidence": prediction[1],
                }
        return results

    def _predict_one(self, vectorizer, clf, texts):
        """Return [(label, confidence), ...] for a batch of texts."""
        matrix = vectorizer.transform(texts)
        probabilities = clf.predict_proba(matrix)
        classes = list(clf.classes_)
        results = []
        for row in probabilities:
            best = int(row.argmax())
            results.append((classes[best], float(row[best])))
        return results

    def _decisive_words(self, text, label):
        """Words that pushed this document towards the predicted label."""
        vector = self.tfidf_vectorizer.transform([text])
        names = self.tfidf_vectorizer.get_feature_names_out()
        classes = list(self.tfidf_clf.classes_)
        target = classes.index(label)
        other = 1 - target

        # Positive difference = evidence for `label` over the other class.
        difference = (
            self.tfidf_clf.feature_log_prob_[target]
            - self.tfidf_clf.feature_log_prob_[other]
        )
        row = vector.toarray().ravel()
        scored = [
            (names[i], float(row[i] * difference[i]))
            for i in row.nonzero()[0]
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:DECISIVE_LIMIT]
