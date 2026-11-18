# NLP Text Analyzer (Bag-of-Words + TF-IDF)

A Tkinter desktop app that analyzes up to three `.txt` files and writes the
whole result to a `.txt` report.

## What it does

For every file:

- **Text statistics** — characters, paragraphs, sentences, words, words
  without stopwords, unique words, lexical diversity, average lengths and the
  longest sentence.
- **Sentences** — every sentence with its word count.
- **Phrases** — the most frequent bigrams and trigrams.
- **Bag-of-Words** — vocabulary size and the most frequent terms.
- **TF-IDF** — the terms with the highest weight in that document.
- **Summary** — extractive: the highest scoring TF-IDF sentences (20% of the
  text, between 3 and 7 sentences), kept in reading order.
- **Sentiment** — Naive Bayes trained on NLTK's `movie_reviews` corpus, shown
  for both TF-IDF and Bag-of-Words features, with the decisive words and a
  label for every sentence.

With two or three files it also adds a cross-file comparison: cosine
similarity, shared terms and the most distinctive term per file.

## Setup

```powershell
python -m venv .env
.\.env\Scripts\Activate.ps1
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab movie_reviews
```

The app also downloads any missing NLTK package on first run, so the last
command is optional. The trained classifier is cached in `data/` after the
first run.

## Run

```powershell
python main.py
```

Pick one to three `.txt` files, press **Analyze**, then **Save report (.txt)**.

## Project layout

```
main.py                     starts the app
ui.py                       Tkinter window, threading and file handling
models/preprocessing.py     NLTK stopwords, sentence splitting, tokenizer
models/bag_of_words.py      CountVectorizer wrapper and top terms
models/tfidf.py             TfidfVectorizer wrapper, top terms, similarity
models/summarizer.py        TF-IDF extractive summary
models/sentiment.py         Naive Bayes training, caching and prediction
models/analyzer.py          runs every step and returns plain data
utils/report.py             formats the data as the .txt report
tests/test_smoke.py         quick checks for the pipeline
samples/                    three sample .txt files
```

## Notes

- Stopwords come from `nltk.corpus.stopwords`, with negations (`not`, `no`,
  `never`, `n't`) kept so sentiment keeps its meaning.
- Only `.txt` input is accepted; at most three files at once.
- The report is written as UTF-8 and contains exactly what the window shows.
- Tests: `python -m unittest discover -s tests`.

### Windows Smart App Control

On machines where Smart App Control is enforcing, `numpy.random`'s compiled
DLL can be blocked, which prevents scikit-learn from importing. The app then
stops with `ImportError: DLL load failed while importing _generator`. Running
on a machine without that restriction (or with Smart App Control disabled)
avoids the problem.
