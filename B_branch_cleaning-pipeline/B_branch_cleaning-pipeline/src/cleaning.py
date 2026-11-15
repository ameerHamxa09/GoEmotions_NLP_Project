"""Reusable text-cleaning pipeline for GoEmotions (Assignment 1, Task 4).

Two output levels, produced by ONE code path so results are identical everywhere:

* ``text_norm``  - normalised but still natural text (placeholders, quotes, case and
                   character noise handled).  Intended for BERT-style models, which
                   were pre-trained on natural text and should not see lemmatised or
                   stop-word-stripped input.
* ``text_clean`` - ``text_norm`` + tokenisation, lemmatisation and a *custom*
                   stop-word filter.  Intended for bag-of-words / TF-IDF models.

Public API
----------
clean_text(text, stage="full")      -> str
clean_corpus(texts)                 -> DataFrame[text_norm, text_clean]
extract_signals(text)               -> dict   (information that lower-casing destroys)
NORMALIZATION_STEPS                 -> ordered (name, function) pairs, used by the notebook
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from typing import Callable, Iterable

import pandas as pd

SPACY_MODEL = "en_core_web_sm"

# Tokens that are kept verbatim all the way through the pipeline.
PLACEHOLDERS = frozenset({"<name>", "<religion>", "<subreddit>", "<user>", "<num>", "<url>",
                          "<smile>", "<sad>", "<heart>"})
SIGNAL_TOKENS = frozenset({"!", "?", "..."})  # punctuation that carries emotion

# --------------------------------------------------------------------------- regexes
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
SUBREDDIT_RE = re.compile(r"(?<![\w/])/?r/\w+", re.IGNORECASE)   # r/hockey, /r/hockey
USER_RE = re.compile(r"(?<![\w/])/?u/\w+", re.IGNORECASE)        # u/someone
NAME_RE = re.compile(r"\[NAME\]")                                  # dataset anonymisation
RELIGION_RE = re.compile(r"\[RELIGION\]")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# ASCII emoticons.  The (?![A-Za-z]) guard stops ':please' / ':Dumb' from matching.
SMILE_RE = re.compile(r"[:;=][-^o']?[)DPp]+(?![A-Za-z])")
SAD_RE = re.compile(r"[:;=][-^o']?\(+")
HEART_RE = re.compile(r"<3+")
ELONGATION_RE = re.compile(r"([a-z])\1{2,}")                       # sooooo -> soo
REPEATED_PUNCT_RE = re.compile(r"([!?])\1+")                       # !!! -> !
ELLIPSIS_RE = re.compile(r"\.{2,}")                                # .. / .... -> ' ... '
TILDE_RE = re.compile(r"~+")                                       # markdown ~~strikethrough~~, 'no~'
DANGLING_HYPHEN_RE = re.compile(r"(?<![\w>])-+|-+(?![\w<])")        # 'no-', '- item', not 'well-known'
MISSING_SPACE_RE = re.compile(r"(?<=[a-z])([.,;:!?])(?=[a-z])")    # 'vomitting,can't' -> 'vomitting, can't'
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")
SKIN_TONE_RE = re.compile("[\U0001F3FB-\U0001F3FF]")
BRAILLE_RE = re.compile("[\u2800-\u28FF]")                         # braille ASCII art
CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
WHITESPACE_RE = re.compile(r"\s+")

_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u2032": "'",   # curly / modifier apostrophes
    "\u201c": '"', "\u201d": '"', "\u201e": '"',                  # curly double quotes
    "\u2013": " - ", "\u2014": " - ",                             # en / em dash
})
# Unicode categories removed outright: combining marks (Zalgo text, variation
# selectors), format chars (zero-width joiner, bidi marks) and control chars.
_DROP_CATEGORIES = {"Mn", "Me", "Cf", "Cc"}


# ------------------------------------------------------------------ normalisation steps
def step_unicode(text: str) -> str:
    """NFKC-normalise, then drop invisible / combining / art characters.

    NFKC (not plain NFC) also folds compatibility forms such as full-width '!' '?'
    and the single-character ellipsis, which both occur in this corpus.
    """
    text = unicodedata.normalize("NFKC", text)
    text = SKIN_TONE_RE.sub("", text)
    text = BRAILLE_RE.sub(" ", text)
    return "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in _DROP_CATEGORIES
    )


def step_html_and_quotes(text: str) -> str:
    """Unescape HTML entities and map curly quotes / dashes to ASCII."""
    return html.unescape(text).translate(_QUOTE_MAP)


def step_mask_entities(text: str) -> str:
    """Replace entities that carry no emotion (but would bloat the vocabulary)
    with a small set of placeholder tokens, and map ASCII emoticons to <smile>/<sad>/<heart>.  Must run BEFORE case folding because
    the dataset's own placeholders are upper-case ([NAME], [RELIGION])."""
    text = URL_RE.sub(" <url> ", text)
    text = HEART_RE.sub(" <heart> ", text)      # before digits/angle brackets are touched
    text = SMILE_RE.sub(" <smile> ", text)      # emoticons are emotion signal: keep them
    text = SAD_RE.sub(" <sad> ", text)
    text = SUBREDDIT_RE.sub(" <subreddit> ", text)
    text = USER_RE.sub(" <user> ", text)
    text = NAME_RE.sub(" <name> ", text)
    text = RELIGION_RE.sub(" <religion> ", text)
    return NUMBER_RE.sub(" <num> ", text)


def step_casefold(text: str) -> str:
    return text.lower()


def step_squeeze(text: str) -> str:
    """Collapse expressive repetition, repair glued punctuation, isolate emoji.

    'sooooo' -> 'soo' (keeps a trace of emphasis but one spelling per word),
    '!!!' -> '!', '....' -> ' ... ', every emoji becomes its own token.
    Punctuation glued to the next word ('vomitting,can't', 'future...don't') gets a
    space: otherwise the tokenizer keeps '...don't' as ONE token and the negation is lost.
    """
    text = ELONGATION_RE.sub(r"\1\1", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = TILDE_RE.sub(" ", text)
    text = DANGLING_HYPHEN_RE.sub(" ", text)
    text = ELLIPSIS_RE.sub(" ... ", text)
    text = MISSING_SPACE_RE.sub(r"\1 ", text)
    return EMOJI_RE.sub(lambda m: f" {m.group()} ", text)


def step_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


NORMALIZATION_STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("1. unicode (NFKC, strip invisible/combining chars)", step_unicode),
    ("2. HTML entities + curly quotes/dashes", step_html_and_quotes),
    ("3. mask names/religion/urls/subreddits/numbers, map emoticons", step_mask_entities),
    ("4. case folding", step_casefold),
    ("5. squeeze elongation/punctuation, fix glued punctuation, isolate emoji", step_squeeze),
    ("6. whitespace", step_whitespace),
]


def _as_str(text) -> str:
    if isinstance(text, str):
        return text
    return "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)


def normalize_text(text) -> str:
    """Steps 1-6: the ``text_norm`` level."""
    text = _as_str(text)
    for _, func in NORMALIZATION_STEPS:
        text = func(text)
    return text


# ------------------------------------------------------------------------ stop words
# Words that a stock stop-word list removes but that carry emotional meaning in
# short comments.  Justified with data in the notebook (section 4.4).
NEGATIONS = frozenset({"no", "nor", "not", "never", "none", "nobody", "nothing",
                       "neither", "nowhere", "cannot"})
DEGREE_WORDS = frozenset({"very", "too", "so", "more", "most", "just", "only", "such"})
PERSONAL_PRONOUNS = frozenset({"i", "me", "my", "myself", "we", "our", "ourselves",
                               "you", "your", "yours", "yourself"})
QUESTION_WORDS = frozenset({"what", "why", "how", "who", "whom", "when", "where", "which"})
KEEP_WORDS = NEGATIONS | DEGREE_WORDS | PERSONAL_PRONOUNS | QUESTION_WORDS | {"should"}


def ensure_nltk_stopwords() -> None:
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


@lru_cache(maxsize=1)
def default_stopwords() -> frozenset[str]:
    """NLTK's stock English list (what most tutorials use)."""
    ensure_nltk_stopwords()
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def custom_stopwords() -> frozenset[str]:
    """NLTK's list minus the emotionally informative words in ``KEEP_WORDS``."""
    return frozenset(default_stopwords() - KEEP_WORDS)


# ------------------------------------------------------------------- tokenise + lemmatise
@lru_cache(maxsize=1)
def get_nlp():
    """spaCy pipeline with only what we need: tokenizer, tagger, lemmatizer.

    The tagger gives the POS tags the rule-based lemmatizer relies on; the parser and
    NER are disabled for speed.  Placeholder tokens are registered as special cases so
    '<name>' stays one token instead of '<', 'name', '>'.
    """
    import spacy
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except OSError as exc:  # model not installed
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' missing. Run: python -m spacy download {SPACY_MODEL}"
        ) from exc
    for token in PLACEHOLDERS:
        nlp.tokenizer.add_special_case(token, [{"ORTH": token}])
    return nlp


def _doc_to_clean(doc) -> str:
    stop = custom_stopwords()
    kept: list[str] = []
    fallback: list[str] = []          # alphabetic lemmas before stop-word filtering
    for tok in doc:
        text = tok.text
        if tok.is_space:
            continue
        if text in PLACEHOLDERS or text in SIGNAL_TOKENS or EMOJI_RE.fullmatch(text):
            kept.append(text)
            fallback.append(text)
            continue
        lemma = tok.lemma_.lower()    # "n't" -> "not", "did" -> "do", "annoyed" -> "annoy"
        if not lemma.isalpha():       # punctuation, hyphens, stray symbols, contraction stubs
            continue
        fallback.append(lemma)
        if tok.lower_ in stop or lemma in stop:
            continue
        kept.append(lemma)
    # A comment made only of stop words ("it's", "so it is") must not become empty.
    result = " ".join(kept if kept else fallback)
    if not result:  # symbols glued to words, e.g. ">pri*toiletries*": salvage the letters
        result = " ".join(re.findall(r"[a-z]+", doc.text))
    return result


def clean_corpus(texts: Iterable, batch_size: int = 512, n_process: int = 1) -> pd.DataFrame:
    """Clean many comments.  Returns a DataFrame with ``text_norm`` and ``text_clean``."""
    index = texts.index if isinstance(texts, pd.Series) else None
    normed = [normalize_text(t) for t in texts]
    docs = get_nlp().pipe(normed, batch_size=batch_size, n_process=n_process)
    cleaned = [_doc_to_clean(d) for d in docs]
    return pd.DataFrame({"text_norm": normed, "text_clean": cleaned}, index=index)


def clean_text(text, stage: str = "full") -> str:
    """Clean ONE comment with exactly the same code path as ``clean_corpus``.

    stage="normalized" -> text_norm,  stage="full" -> text_clean.
    """
    if stage == "normalized":
        return normalize_text(text)
    if stage != "full":
        raise ValueError("stage must be 'normalized' or 'full'")
    return clean_corpus([text])["text_clean"].iloc[0]


# --------------------------------------------------------------------------- signals
def extract_signals(text) -> dict[str, int]:
    """Information that case folding / squeezing destroys, measured on the RAW text.

    caps_words : words of 2+ capital letters ('WHY', 'OK'); dataset placeholders such
                 as [NAME] are removed first so they are not counted as shouting.
    exclaims / questions : number of '!' and '?' characters before collapsing.
    """
    text = unicodedata.normalize("NFKC", _as_str(text))
    text = RELIGION_RE.sub(" ", NAME_RE.sub(" ", text))
    return {
        "caps_words": len(CAPS_WORD_RE.findall(text)),
        "exclaims": text.count("!"),
        "questions": text.count("?"),
    }


def vocabulary(texts: Iterable[str], lowercase: bool = False):
    """Whitespace-token frequency Counter (used for the before/after comparison)."""
    from collections import Counter
    counter: Counter = Counter()
    for t in texts:
        counter.update((t.lower() if lowercase else t).split())
    return counter
