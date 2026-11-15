"""Task 5 - end-to-end runner: raw GoEmotions in, cleaned dataset out.

    python -m src.run_cleaning                 # downloads raw data if missing
    python -m src.run_cleaning --n-process 4   # faster on multi-core machines

Outputs (both git-ignored / regenerable except the small stats file):
    data/processed/goemotions_clean.csv   one row per comment, original split kept
    reports/cleaning_stats.json           vocabulary / length numbers for the report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.cleaning import clean_corpus, extract_signals, vocabulary
from src.load_data import DEFAULT_RAW_DIR, ROOT, load_goemotions

OUT_COLUMNS = [
    "comment_id", "split", "text", "text_norm", "text_clean",
    "label_ids", "label_names", "caps_words", "exclaims", "questions",
]


def build_clean_dataset(raw_dir: Path = DEFAULT_RAW_DIR, n_process: int = 1) -> pd.DataFrame:
    df = load_goemotions(raw_dir)
    signals = pd.DataFrame([extract_signals(t) for t in df["text"]], index=df.index)
    cleaned = clean_corpus(df["text"], n_process=n_process)
    return pd.concat([df, cleaned, signals], axis=1)[OUT_COLUMNS]


def compute_stats(out: pd.DataFrame) -> dict:
    before_cased = vocabulary(out["text"])
    before_lower = vocabulary(out["text"], lowercase=True)
    after = vocabulary(out["text_clean"])
    words_before = out["text"].str.split().str.len()
    words_after = out["text_clean"].str.split().str.len()
    return {
        "rows": len(out),
        "rows_per_split": out["split"].value_counts().to_dict(),
        "vocab_raw_cased": len(before_cased),
        "vocab_raw_lowercased": len(before_lower),
        "vocab_clean": len(after),
        "vocab_reduction_pct_vs_raw_cased": round(100 * (1 - len(after) / len(before_cased)), 1),
        "tokens_raw": int(sum(before_cased.values())),
        "tokens_clean": int(sum(after.values())),
        "mean_words_raw": round(float(words_before.mean()), 2),
        "mean_words_clean": round(float(words_after.mean()), 2),
        "empty_after_cleaning": int((out["text_clean"].str.len() == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "goemotions_clean.csv")
    parser.add_argument("--stats", type=Path, default=ROOT / "reports" / "cleaning_stats.json")
    parser.add_argument("--n-process", type=int, default=1,
                        help="spaCy worker processes (keep 1 on Windows/low-RAM machines)")
    args = parser.parse_args()

    start = time.time()
    out = build_clean_dataset(args.raw_dir, args.n_process)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    stats = compute_stats(out)
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cleaned {len(out):,} comments in {time.time() - start:.0f}s")
    print(f"  saved  -> {args.out}")
    print(f"  stats  -> {args.stats}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
