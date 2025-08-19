"""
Textual EDA for CFPB consumer-complaint narratives
Converted directly from eda_cfpb_text.ipynb

Steps:
- Load complaint narratives parquet
- Basic length stats and histogram
- Simple token snapshot
- Sentence embeddings + t-SNE visualization
- VADER sentiment distribution
- Save cleaned features + sentiment
"""

from pathlib import Path
import re
import warnings

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import torch
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)


# ────────────────── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
REPORT_FIGS = BASE_DIR / "reports" / "figures"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)

PARQUET_FILE = DATA_PROCESSED / "cfpb.parquet"  # produced by data_ingestion.py
FEATURES_OUT = DATA_PROCESSED / "cfpb_text_feats.parquet"


# ────────────────── Helpers ────────────────────────────────────────────────
def clean_text(series: pl.Series) -> list[str]:
    """
    Light text cleaning for complaint narratives.
    - Remove placeholder 'XXXX'
    - Drop digits and punctuation
    - Collapse whitespace
    """
    patt_x = re.compile(r"\b[x]+\b", flags=re.I)
    patt_nond = re.compile(r"[^a-z\s]+")
    cleaned: list[str] = []
    for s in series:
        if s is None:
            cleaned.append("")
            continue
        s = patt_x.sub(" ", s.lower())
        s = patt_nond.sub(" ", s)
        cleaned.append(" ".join(s.split()))
    return cleaned


def plot_hist(data, bins: int, xlabel: str, ylabel: str, title: str, save_path: Path) -> None:
    """Utility for saving simple histograms."""
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, log=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ────────────────── Main ───────────────────────────────────────────────────
def main() -> None:
    # Load parquet
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {PARQUET_FILE}")

    df = pl.read_parquet(PARQUET_FILE)
    print(f"Loaded {df.height:,} rows  |  {df.width} columns")

    # Pick narrative column and filter
    text_col = "narrative" if "narrative" in df.columns else \
        [c for c in df.columns if "narrative" in c.lower()][0]
    df = df.rename({text_col: "narrative"})
    df = df.filter(pl.col("narrative").is_not_null() &
                   (pl.col("narrative").str.len_bytes() > 0))
    print(f"Narratives retained: {df.height:,}\n")

    # Length stats
    lengths = df.with_columns(
        pl.col("narrative").str.len_bytes().alias("char_len")
    )["char_len"]
    print("Length summary:")
    print(lengths.describe(), "\n")

    plot_hist(
        lengths.to_list(),
        bins=60,
        xlabel="Narrative length (chars)",
        ylabel="Count (log scale)",
        title="Fig T1. Histogram of Narrative Lengths",
        save_path=REPORT_FIGS / "T1_length_hist.png",
    )

    # Token snapshot
    cleaned_sample = clean_text(df.sample(n=100_000, seed=42)["narrative"])
    vec = CountVectorizer(
        stop_words="english",
        max_features=25,
        token_pattern=r"(?u)\\b[a-z]{2,}\\b",
    )
    X = vec.fit_transform(cleaned_sample)
    totals = X.sum(axis=0).A1
    top25 = sorted(
        zip(vec.get_feature_names_out(), totals),
        key=lambda t: t[1],
        reverse=True,
    )
    print("Top-25 tokens after cleaning:")
    for w, c in top25:
        print(f"{w:<12} {c:,}")
    print()

    # Embeddings + t-SNE
    sample_size = 25_000
    sample_df = df.sample(n=min(sample_size, df.height), seed=42)
    texts_clean = clean_text(sample_df["narrative"])
    labels = sample_df["Product"]

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(texts_clean):,} narratives on {device} …")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    emb = model.encode(
        texts_clean,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device,
    )

    print("Running t-SNE …")
    tsne = TSNE(
        n_components=2, perplexity=30,
        n_iter=1_000, init="random",
        random_state=42, verbose=1,
    )
    coords = tsne.fit_transform(emb)

    palette = {p: i for i, p in enumerate(labels.unique().to_list()[:10])}
    plt.figure(figsize=(6, 6))
    plt.scatter(
        coords[:, 0], coords[:, 1],
        s=5, alpha=0.6,
        c=[palette.get(p, 0) for p in labels], cmap="tab10"
    )
    plt.xticks([]); plt.yticks([])
    plt.title(f"Fig T2. t-SNE of {len(texts_clean):,} Complaint Embeddings")
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "T2_tsne_embeddings.png", dpi=150)
    plt.show()

    # VADER sentiment
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    sent_scores = np.array([sia.polarity_scores(t)["compound"] for t in texts_clean])

    plot_hist(
        sent_scores,
        bins=40,
        xlabel="Compound sentiment (−1 … +1)",
        ylabel="Count",
        title=f"VADER Sentiment for {len(sent_scores):,} Narratives",
        save_path=REPORT_FIGS / "vader_sentiment.png",
    )

    print("Sentiment percentiles:")
    for p in (5, 25, 50, 75, 95):
        print(f"{p:>2}th : {np.percentile(sent_scores, p): .3f}")

    # Save features
    (
        pl.DataFrame(sent_scores, schema=["sentiment"])
          .with_columns(pl.Series("narrative_clean", texts_clean))
          .with_row_count(name="row_id")
    ).write_parquet(FEATURES_OUT)

    print(f"\nCleaned text + sentiment saved to {FEATURES_OUT.name}")


if __name__ == "__main__":
    main()


















