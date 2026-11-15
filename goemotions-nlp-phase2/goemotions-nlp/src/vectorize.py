"""Phase 2 / Part D2 - the interface Phase 3 calls.

    from src.vectorize import vectorize, vectorize_many

    vectorize("I LOVE this game!!! :D", method="tfidf")      # one raw comment -> one vector
    vectorize_many(list_of_raw_comments, method="fasttext")   # many comments -> one matrix

Methods (``METHODS``): ``bow``, ``tfidf``, ``cbow``, ``skipgram``, ``fasttext``.

What comes back
    bow, tfidf        scipy.sparse CSR row (1 x V)   - or a dense 1-D array with ``dense=True``
    cbow, skipgram,   dense float32 1-D array of 100 numbers: the MEAN of the vectors of the
    fasttext          words in the document (Word2Vec skips words it never learned; fastText
                      builds a vector for every word from its character n-grams; a document
                      with no usable word gets the zero vector)

How it works
    raw text --(Phase 1 cleaning: ``cleaning.clean_text``, the very same code that produced
    ``text_clean``)--> tokens --(saved vectorizer / saved embedding)--> vector.

Nothing is ever trained here: models are loaded from ``models/`` once and cached.  If they
are missing the functions raise an error that says how to build them - they do not train.
Pass ``clean=False`` if the text is already Phase 1 ``text_clean`` (much faster for a whole
dataset, because cleaning needs spaCy).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import joblib
import numpy as np
import scipy.sparse as sp

from src import cleaning
from src import representations as R

METHODS = R.METHODS
_COUNT_BASED = ("bow", "tfidf")


@lru_cache(maxsize=None)
def load_model(method: str):
    """Load one saved representation (cached; never trains)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
    path = R.MODELS_DIR / R.MODEL_FILES[method]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Create the saved models once with:  python -m src.build_representations"
        )
    return joblib.load(path) if method in _COUNT_BASED else R.load_vectors(path)


def vector_size(method: str) -> int:
    """Length of the vectors returned for ``method``."""
    model = load_model(method)
    return len(model.vocabulary_) if method in _COUNT_BASED else model.vector_size


def _clean(documents: list[str], clean: bool) -> list[str]:
    if not clean:
        return [str(d) for d in documents]
    return cleaning.clean_corpus(documents)["text_clean"].tolist()


def vectorize_many(documents: Iterable[str], method: str = "tfidf", *, clean: bool = True):
    """Vectors for many documents: sparse CSR matrix (count-based) or ndarray (embeddings)."""
    docs = list(documents)
    model = load_model(method)
    texts = _clean(docs, clean)
    if method in _COUNT_BASED:
        return model.transform(texts)
    if not texts:
        return np.zeros((0, model.vector_size), dtype=np.float32)
    return np.vstack([R.document_vector(R.whitespace_tokenizer(t), model) for t in texts])


def vectorize(document: str, method: str = "tfidf", *, dense: bool = False, clean: bool = True):
    """Vector of ONE raw document under the chosen representation (see module docstring)."""
    matrix = vectorize_many([document], method, clean=clean)
    if method in _COUNT_BASED:
        return matrix.toarray().ravel() if dense else matrix
    return matrix[0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vectorize one document with a saved representation")
    parser.add_argument("text")
    parser.add_argument("--method", default="tfidf", choices=METHODS)
    args = parser.parse_args()
    vec = vectorize(args.text, args.method)
    if sp.issparse(vec):
        print(f"{args.method}: sparse {vec.shape}, {vec.nnz} non-zero entries")
    else:
        print(f"{args.method}: dense {vec.shape}, first values {np.round(vec[:5], 3)}")
