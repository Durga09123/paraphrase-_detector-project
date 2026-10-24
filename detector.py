"""
detector.py  -  Paraphrase detection logic
Method : WordNet synonym normalisation + TF-IDF + Cosine Similarity
"""
import math
import os
import re
from collections import Counter

import nltk

# --------------------------------------------------------------------------
# NLTK data setup
# 1. First choice: the "nltk_data" folder shipped inside this project
#    (this is what makes it work reliably on Vercel).
# 2. Fallback: download it (to /tmp on Vercel, normal location on your PC).
# Loading is lazy, so any problem is reported on the web page instead of
# crashing the whole function at start-up.
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_DIR = os.path.join(BASE_DIR, "nltk_data")
TMP_DIR = "/tmp/nltk_data"

stop_words = set()
wordnet = None
_ready = False


def _setup():
    global stop_words, wordnet, _ready
    if _ready:
        return

    if os.path.isdir(BUNDLED_DIR):
        nltk.data.path.insert(0, BUNDLED_DIR)

    download_dir = None
    if os.environ.get("VERCEL"):
        os.makedirs(TMP_DIR, exist_ok=True)
        nltk.data.path.insert(0, TMP_DIR)
        download_dir = TMP_DIR

    for resource, package in (("corpora/wordnet", "wordnet"),
                              ("corpora/stopwords", "stopwords")):
        try:
            nltk.data.find(resource)
        except LookupError:
            ok = nltk.download(package, download_dir=download_dir, quiet=True)
            if not ok:
                raise RuntimeError(
                    f"NLTK data '{package}' not found and could not be "
                    f"downloaded. Put it in the project's nltk_data folder."
                )

    from nltk.corpus import stopwords as _sw, wordnet as _wn
    stop_words = set(_sw.words("english"))
    wordnet = _wn
    _ready = True


# Same tokenisation rule that scikit-learn's TfidfVectorizer uses by default
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
def get_synonym(word):
    """Return the first WordNet lemma of the word (or the word itself)."""
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    if synonyms:
        return synonyms[0].replace("_", " ")
    return word


def preprocess_text(text):
    _setup()
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    processed_words = []
    for word in words:
        if word not in stop_words:
            processed_words.append(get_synonym(word))
    return " ".join(processed_words)


# --------------------------------------------------------------------------
# TF-IDF + Cosine similarity (same maths as sklearn's TfidfVectorizer,
# written in plain Python so the app stays small enough for Vercel)
# --------------------------------------------------------------------------
def tfidf_vectors(documents):
    """Return (vocabulary, list of L2-normalised TF-IDF vectors)."""
    tokenised = [TOKEN_PATTERN.findall(d.lower()) for d in documents]
    vocabulary = sorted(set(t for doc in tokenised for t in doc))
    n_docs = len(documents)

    df = {t: sum(1 for doc in tokenised if t in doc) for t in vocabulary}
    # smooth idf:  ln((1 + n) / (1 + df)) + 1
    idf = {t: math.log((1 + n_docs) / (1 + df[t])) + 1 for t in vocabulary}

    vectors = []
    for doc in tokenised:
        counts = Counter(doc)
        vec = [counts.get(t, 0) * idf[t] for t in vocabulary]
        norm = math.sqrt(sum(v * v for v in vec))
        vectors.append([v / norm for v in vec] if norm else vec)
    return vocabulary, vectors


def calculate_similarity(sentence1, sentence2):
    text1 = preprocess_text(sentence1)
    text2 = preprocess_text(sentence2)
    if not text1 or not text2:
        return 0.0, text1, text2
    _, (v1, v2) = tfidf_vectors([text1, text2])
    similarity = sum(a * b for a, b in zip(v1, v2))
    return min(max(similarity, 0.0), 1.0), text1, text2


def detect_paraphrase(sentence1, sentence2, threshold=0.30):
    similarity, text1, text2 = calculate_similarity(sentence1, sentence2)
    result = "Paraphrase" if similarity >= threshold else "Not a Paraphrase"
    return result, similarity, text1, text2
