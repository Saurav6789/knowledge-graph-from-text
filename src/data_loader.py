"""
Stage 1: Data Acquisition
=========================
Downloads the BBC News Full-Text dataset (Greene & Cunningham, 2006) -
2,225 real news articles across 5 categories (business, entertainment,
politics, sport, tech). This is a standard, widely cited NLP benchmark
corpus, used here as a realistic stand-in for "real-world unstructured
text" (news articles, reports, press releases) that a Knowledge Graph
pipeline would typically ingest.

The loader is source-agnostic: swap `source_url` in config.yaml for
any other CSV/text corpus (internal reports, Wikipedia dumps, PDFs
converted to text, etc.) and the rest of the pipeline is unaffected,
as long as the output has `category` and `text` columns (category is
optional metadata only).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from src.utils import resolve_path

logger = logging.getLogger("kg_pipeline.data_loader")


def _download_csv(url: str, dest: Path, timeout: int = 30) -> None:
    logger.info(f"Downloading dataset from {url}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info(f"Saved raw dataset to {dest} ({len(response.content)/1024:.1f} KB)")


def download_dataset(config: Dict[str, Any]) -> Path:
    """Download the raw dataset if it isn't already cached locally."""
    data_cfg = config["data"]
    raw_path = resolve_path(data_cfg["raw_path"])

    if raw_path.exists() and raw_path.stat().st_size > 0:
        logger.info(f"Raw dataset already present at {raw_path}, skipping download.")
        return raw_path

    try:
        _download_csv(data_cfg["source_url"], raw_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Primary source failed ({exc}); trying fallback URL.")
        _download_csv(data_cfg["fallback_url"], raw_path)

    return raw_path


def load_articles(config: Dict[str, Any], raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the CSV into a DataFrame, apply category filtering and sampling
    as configured, and return a clean DataFrame with columns:
    [doc_id, category, text].
    """
    data_cfg = config["data"]
    raw_path = raw_path or resolve_path(data_cfg["raw_path"])

    # The BBC CSV mirrors contain a stray non-UTF8 byte (a mis-encoded
    # pound sign, £) in a handful of articles, so we read with latin-1
    # which never raises on arbitrary byte values.
    df = pd.read_csv(raw_path, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise the two known column naming schemes to (category, text)
    rename_map = {"news": "text", "type": "category", "articles": "text"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "text" not in df.columns:
        raise ValueError(
            f"Expected a 'text' column in the dataset, found columns: {list(df.columns)}"
        )
    if "category" not in df.columns:
        df["category"] = "uncategorized"

    # Repair the common mis-decoded pound-sign artifact (Â£ -> £)
    df["text"] = df["text"].astype(str).str.replace("Â£", "£", regex=False)

    df = df.dropna(subset=["text"]).reset_index(drop=True)

    categories = data_cfg.get("categories") or []
    if categories:
        df = df[df["category"].isin(categories)].reset_index(drop=True)
        logger.info(f"Filtered to categories {categories}: {len(df)} articles remain.")

    sample_size = data_cfg.get("sample_size")
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=data_cfg.get("random_seed", 42))
        df = df.reset_index(drop=True)
        logger.info(f"Sub-sampled dataset down to {sample_size} articles for a fast run.")

    df.insert(0, "doc_id", [f"doc_{i:05d}" for i in range(len(df))])
    logger.info(f"Loaded {len(df)} articles. Category distribution:\n{df['category'].value_counts()}")
    return df[["doc_id", "category", "text"]]


def get_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Convenience one-shot: download (if needed) + load + filter/sample."""
    raw_path = download_dataset(config)
    return load_articles(config, raw_path)


if __name__ == "__main__":
    import sys

    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)
    articles = get_dataset(cfg)
    print(articles.head())
    print(f"\nTotal articles loaded: {len(articles)}", file=sys.stderr)
