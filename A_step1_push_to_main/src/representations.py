"""Phase 2 - the single definition of every text representation.

The notebook, the build script (``python -m src.build_representations``) and the
``vectorize()`` interface (``src/vectorize.py``) all import from here, so they cannot
drift apart: same corpus, same tokens, same hyper-parameters.

Design rules (they follow the Assignment 2 brief):
  * The corpus is the Phase 1 output, column ``text_clean``.  We never re-clean it.
  * Phase 1 already tokenised: tokens are separated by single spaces.  Every
    representation therefore splits on whitespace instead of using a library
    tokenizer, so ``not``, ``!``, ``<name>`` and emoji survive exactly as cleaned.
  * Everything is *fitted on the training split only*.  Validation/test comments are
    only ever transformed, so Phase 3 evaluation stays leak-free.
  * CBOW, Skip-gram and fastText share identical hyper-parameters; only the training
    scheme (and, for fastText, the sub-word information) differs.
  * Training is deterministic (fixed seed, one worker, process-independent hashing).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "goemotions_clean.csv"
MODELS_DIR = ROOT / "models"

FIT_SPLIT = "train"
SEED = 42

MODEL_FILES = {
    "bow": "bow.joblib",
    "tfidf": "tfidf.joblib",
    "cbow": "word2vec_cbow.kv",
    "skipgram": "word2vec_skipgram.kv",
    "fasttext": "fasttext.kv",
}
METHODS = tuple(MODEL_FILES)

def load_clean_corpus(path: Path = CLEAN_CSV) -> pd.DataFrame:
    """The cleaned dataset written by Phase 1 (``python -m src.run_cleaning``)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Build the Phase 1 output first:  python -m src.run_cleaning"
        )
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def training_texts(df: pd.DataFrame | None = None, split: str = FIT_SPLIT) -> list[str]:
    """``text_clean`` of the fitting split, as a list of strings."""
    df = load_clean_corpus() if df is None else df
    return df.loc[df["split"] == split, "text_clean"].tolist()


def whitespace_tokenizer(text: str) -> list[str]:
    """Phase 1 output is already tokenised and space-separated."""
    return text.split()


def tokenize_corpus(texts) -> list[list[str]]:
    """List of token lists - the 'sentences' format gensim expects."""
    return [whitespace_tokenizer(t) for t in texts]


def make_count_vectorizer(ngram_range=(1, 1), min_df=1, **kwargs):
    """Bag of Words / n-gram counts.

    ``lowercase=False`` (already lower-cased by Phase 1) and ``token_pattern=None``
    (we supply our own tokenizer) keep the vocabulary identical to Phase 1's tokens.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(tokenizer=whitespace_tokenizer, token_pattern=None, lowercase=False,
                           ngram_range=ngram_range, min_df=min_df, dtype=np.int32, **kwargs)


def make_tfidf_vectorizer(ngram_range=(1, 1), min_df=1, **kwargs):
    """TF-IDF with scikit-learn's defaults: smooth idf, sublinear_tf off, L2 row norm."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(tokenizer=whitespace_tokenizer, token_pattern=None, lowercase=False,
                           ngram_range=ngram_range, min_df=min_df, **kwargs)


BOW_FINAL = dict(ngram_range=(1, 1), min_df=1)
TFIDF_FINAL = dict(ngram_range=(1, 2), min_df=3)


def slim(vectorizer):
    """Drop ``stop_words_`` (every pruned n-gram) so the pickle stays small."""
    if hasattr(vectorizer, "stop_words_"):
        del vectorizer.stop_words_
    return vectorizer


def sparsity_pct(matrix) -> float:
    """Percentage of zero entries of a (sparse) matrix."""
    rows, cols = matrix.shape
    return 100.0 * (1.0 - matrix.nnz / (rows * cols))


def stable_hash(key) -> int:
    """Process-independent hash (Python's ``hash`` changes between runs)."""
    return zlib.crc32(str(key).encode("utf-8"))


EMBED_PARAMS = dict(
    vector_size=100,
    window=5,
    min_count=5,
    epochs=30,
    negative=5,
    workers=1,
    seed=SEED,
    hashfxn=stable_hash,
)
FASTTEXT_EXTRA = dict(min_n=3, max_n=6, bucket=50_000)


def train_word2vec(sentences, sg: int):
    """sg=0 -> CBOW (context -> centre), sg=1 -> Skip-gram (centre -> context)."""
    from gensim.models import Word2Vec
    return Word2Vec(sentences=sentences, sg=sg, **EMBED_PARAMS)


def train_fasttext(sentences):
    """fastText trained with the Skip-gram scheme, so that the only difference from the
    Skip-gram Word2Vec model is the use of character n-grams."""
    from gensim.models import FastText
    return FastText(sentences=sentences, sg=1, **EMBED_PARAMS, **FASTTEXT_EXTRA)


def save_vectors(keyed_vectors, path: Path) -> None:
    """Save only the vectors (no training state) as ONE file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keyed_vectors.save(str(path), sep_limit=10**10)


def load_vectors(path: Path):
    from gensim.models import KeyedVectors
    return KeyedVectors.load(str(path))


def document_vector(tokens: list[str], keyed_vectors, dim: int | None = None) -> np.ndarray:
    """Mean of the word vectors of a document.

    * Word2Vec: words missing from its vocabulary are skipped (they have no vector).
    * fastText: every word has a vector (built from its character n-grams).
    * A document with no usable word gets the zero vector.
    """
    dim = dim or keyed_vectors.vector_size
    is_fasttext = hasattr(keyed_vectors, "vectors_ngrams")
    vectors = [keyed_vectors[w] for w in tokens if is_fasttext or w in keyed_vectors.key_to_index]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)
