"""Phase 2 / Part D1 - fit every representation ONCE and save it to ``models/``.

    python -m src.build_representations                 # build all five
    python -m src.build_representations --only tfidf    # rebuild just one

Output (all small enough for a normal git commit and for GitHub's 25 MB web upload):

    models/bow.joblib              CountVectorizer, unigrams                (sparse BoW)
    models/tfidf.joblib            TfidfVectorizer, unigrams + bigrams      (sparse TF-IDF)
    models/word2vec_cbow.kv        Word2Vec, CBOW (sg=0)                    (100-d vectors)
    models/word2vec_skipgram.kv    Word2Vec, Skip-gram (sg=1)               (100-d vectors)
    models/fasttext.kv             fastText, Skip-gram scheme + sub-words   (100-d + n-gram vectors)
    models/manifest.json           what was built, from which corpus, with which settings

Everything is fitted on the TRAINING split only and is deterministic: running this script
twice gives identical models.  Phase 3 never trains - it only loads these files
through ``src/vectorize.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time

import joblib

from src import representations as R

ALL = R.METHODS
FILES = R.MODEL_FILES


def _versions() -> dict:
    import gensim, numpy, scipy, sklearn
    return {"python": platform.python_version(), "gensim": gensim.__version__,
            "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__}


def _readable(params: dict) -> dict:
    """JSON-friendly copy of a hyper-parameter dict (drops the hash function object)."""
    return {k: (v if isinstance(v, (int, float, str, tuple, list)) else getattr(v, "__name__", str(v)))
            for k, v in params.items()}


def build(only=ALL) -> dict:
    texts = R.training_texts()
    sentences = R.tokenize_corpus(texts)
    manifest_path = R.MODELS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}
    manifest.update({
        "fit_split": R.FIT_SPLIT,
        "corpus": {
            "source": "data/processed/goemotions_clean.csv (column text_clean)",
            "documents": len(texts),
            "tokens": sum(len(s) for s in sentences),
            "sha1_of_texts": hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest(),
        },
        "versions": _versions(),
    })
    R.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in only:
        start = time.time()
        path = R.MODELS_DIR / FILES[name]
        if name == "bow":
            vec = R.slim(R.make_count_vectorizer(**R.BOW_FINAL).fit(texts))
            joblib.dump(vec, path)
            info = {"type": "CountVectorizer", "params": _readable(R.BOW_FINAL),
                    "features": len(vec.vocabulary_)}
        elif name == "tfidf":
            vec = R.slim(R.make_tfidf_vectorizer(**R.TFIDF_FINAL).fit(texts))
            joblib.dump(vec, path)
            info = {"type": "TfidfVectorizer", "params": _readable(R.TFIDF_FINAL),
                    "features": len(vec.vocabulary_)}
        elif name in ("cbow", "skipgram"):
            model = R.train_word2vec(sentences, sg=0 if name == "cbow" else 1)
            R.save_vectors(model.wv, path)
            info = {"type": "Word2Vec", "sg": model.sg, "params": _readable(R.EMBED_PARAMS),
                    "vocabulary": len(model.wv), "dimensions": model.wv.vector_size}
        else:
            model = R.train_fasttext(sentences)
            R.save_vectors(model.wv, path)
            info = {"type": "FastText", "sg": model.sg,
                    "params": _readable({**R.EMBED_PARAMS, **R.FASTTEXT_EXTRA}),
                    "vocabulary": len(model.wv), "dimensions": model.wv.vector_size}
        info.update(file=f"models/{path.name}", size_mb=round(path.stat().st_size / 1e6, 2))
        manifest["models"][name] = info
        print(f"  {name:9s} -> {path.name:22s} {info['size_mb']:6.2f} MB   ({time.time() - start:4.1f} s)")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest  -> {manifest_path.name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=ALL, default=list(ALL),
                        help="build only these representations (default: all)")
    args = parser.parse_args()
    corpus = R.training_texts()
    print(f"Fitting on the '{R.FIT_SPLIT}' split: {len(corpus):,} documents")
    build(args.only)
    print("Done. Load with:  from src.vectorize import vectorize")


if __name__ == "__main__":
    main()
