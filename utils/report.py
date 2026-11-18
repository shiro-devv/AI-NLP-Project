"""Turns an analysis result into the plain-text report shown and saved by the UI."""

import os
import textwrap
from datetime import datetime

WIDTH = 78
SEPARATOR = "=" * WIDTH
RULE = "-" * WIDTH


def _rule(lines):
    lines.append(RULE)


def _title(lines, text):
    """Write an upper-case section heading followed by a rule."""
    lines.append(text.upper())
    _rule(lines)


def _field(lines, label, value, width=30):
    """Write an aligned "label ...... value" line."""
    dots = "." * max(3, width - len(label))
    lines.append(f"  {label} {dots} {value}")


def _wrapped(lines, text, indent="   ", width=WIDTH):
    """Write text wrapped to the report width."""
    lines.extend(
        textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)
        or [indent]
    )


def _number(value, decimals=1):
    return f"{value:,.{decimals}f}"


def _percent(value):
    return f"{value * 100:.1f}%"


def _document_sections(lines, index, document):
    """Write every section for one analyzed file."""
    lines.append(f"[{index}] {document['name']}")
    _rule(lines)
    lines.append("")

    stats = document["statistics"]
    _title(lines, "1. Text statistics")
    _field(lines, "Characters", f"{stats['characters']:,}")
    _field(lines, "Paragraphs", f"{stats['paragraphs']:,}")
    _field(lines, "Sentences", f"{stats['sentences']:,}")
    _field(lines, "Words (total)", f"{stats['words']:,}")
    _field(lines, "Words (excluding stopwords)", f"{stats['content_words']:,}")
    _field(lines, "Unique words", f"{stats['unique_words']:,}")
    _field(lines, "Lexical diversity", _percent(stats["lexical_diversity"]))
    _field(lines, "Average word length", _number(stats["avg_word_length"]))
    _field(lines, "Average sentence length", f"{_number(stats['avg_sentence_length'])} words")
    _field(lines, "Longest sentence", f"{stats['longest_sentence_words']} words")
    _wrapped(lines, stats["longest_sentence"], indent="    ")
    lines.append("")

    _title(lines, f"2. Sentences ({len(document['sentences'])})")
    for position, (sentence, words) in enumerate(
        zip(document["sentences"], document["sentence_words"]), start=1
    ):
        _wrapped(lines, f"{position}. ({words} words) {sentence}", indent="  ")
    lines.append("")

    _title(lines, "3. Phrases")
    for label, phrase_list in document["phrases"].items():
        lines.append(f"  {label.capitalize()}:")
        if not phrase_list:
            lines.append("    (none found)")
        for phrase, count in phrase_list:
            lines.append(f"    {phrase:<40} {count:>4}x")
    lines.append("")

    bow = document["bag_of_words"]
    _title(lines, "4. Bag-of-Words")
    _field(lines, "Vocabulary size", f"{bow['vocabulary']:,} terms")
    _field(lines, "Matrix shape", f"1 document x {bow['vocabulary']:,} terms")
    _field(lines, "Matrix density", _percent(bow["density"]))
    lines.append("  Top terms by raw count:")
    for term, count in bow["terms"]:
        lines.append(f"    {term:<40} {count:>4}x")
    lines.append("")

    tfidf_data = document["tfidf"]
    _title(lines, "5. TF-IDF")
    _field(lines, "Vocabulary size", f"{tfidf_data['vocabulary']:,} terms")
    lines.append("  Top terms by TF-IDF weight:")
    for term, weight in tfidf_data["terms"]:
        lines.append(f"    {term:<40} {weight:>7.4f}")
    lines.append("")

    summary = document["summary"]
    _title(lines, "6. Summary")
    _field(
        lines,
        "Sentences kept",
        f"{len(summary['summary'])} of {len(summary['sentences'])}",
    )
    _field(lines, "Compression", _percent(summary["compression"]))
    lines.append("  Extractive summary:")
    for position, sentence in enumerate(summary["summary"], start=1):
        _wrapped(lines, f"{position}. {sentence}", indent="    ")
    lines.append("")

    _title(lines, "7. Sentiment analysis")
    sentiment = document["sentiment"]
    if sentiment is None:
        lines.append("  Not available: the classifier could not be loaded.")
        lines.append("  Run: python -m nltk.downloader movie_reviews")
    else:
        _field(
            lines,
            "TF-IDF + Naive Bayes",
            f"{sentiment['tfidf_label'].upper()} ({_percent(sentiment['tfidf_confidence'])})",
        )
        _field(
            lines,
            "Bag-of-Words + Naive Bayes",
            f"{sentiment['bow_label'].upper()} ({_percent(sentiment['bow_confidence'])})",
        )
        _field(lines, "Models agree", "yes" if sentiment["agree"] else "no")
        words = ", ".join(
            f"{term} ({weight:.3f})" for term, weight in sentiment["decisive_words"]
        )
        lines.append("  Decisive words:")
        _wrapped(lines, words or "none", indent="    ")
        lines.append("  Sentence-level labels:")
        for position, (sentence, prediction) in enumerate(
            zip(document["sentences"], sentiment["sentences"]), start=1
        ):
            label = prediction["label"]
            if label == "neutral":
                header = f"  {position:>3}. neutral"
            else:
                header = f"  {position:>3}. {label} ({_percent(prediction['confidence'])})"
            _wrapped(lines, f"{header}  {sentence}", indent="")
    lines.append("")


def _comparison_section(lines, comparison):
    _title(lines, "Cross-file comparison")
    lines.append("  Cosine similarity (0 = unrelated, 1 = identical):")
    for pair in comparison["pairs"]:
        lines.append(f"    {pair['left']} vs {pair['right']} ... {pair['score']:.4f}")
    best = comparison["most_similar"]
    _field(
        lines,
        "Most similar pair",
        f"{best['left']} / {best['right']} ({best['score']:.4f})",
    )

    shared = ", ".join(term for term, _ in comparison["shared_terms"])
    _field(lines, "Shared terms", shared or "none")
    lines.append("  Most distinctive term per file:")
    for item in comparison["distinctive"]:
        lines.append(f"    {item['name']:<30} {item['term'] or 'none'}")
    lines.append("")


def _method_section(lines, accuracy):
    _title(lines, "Method notes")
    _field(lines, "Tokenizer", "lowercase words, punctuation removed")
    _field(lines, "Stopwords", "built-in list, negations kept")
    _field(lines, "Bag-of-Words", "CountVectorizer(ngram_range=(1, 1))")
    _field(lines, "TF-IDF", "TfidfVectorizer(sublinear_tf=True, norm='l2')")
    _field(lines, "Summary", "top TF-IDF sentences, 20% (3-7 sentences)")
    _field(lines, "Sentiment", "MultinomialNB(alpha=0.3) on movie_reviews")
    if accuracy is None:
        _field(lines, "Model accuracy", "not available")
    else:
        _field(lines, "5-fold accuracy", _percent(accuracy))


def build_report(result):
    """Format the whole analysis as one readable text report."""
    lines = [
        SEPARATOR,
        " NLP TEXT ANALYSIS REPORT  |  Bag-of-Words + TF-IDF",
        f" Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f" Files analyzed: {len(result['documents'])}",
        SEPARATOR,
        "",
    ]

    for index, document in enumerate(result["documents"], start=1):
        _document_sections(lines, index, document)

    if result["comparison"]:
        _comparison_section(lines, result["comparison"])

    _method_section(lines, result["accuracy"])
    lines.append(SEPARATOR)
    return "\n".join(lines) + "\n"


def write_report(text, path):
    """Save the report as UTF-8 text."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
