"""
Stage 2: Text Preprocessing
============================
The raw BBC dataset text is lightly cleaned (whitespace, encoding
artifacts) and segmented into sentences using spaCy's sentencizer.
We deliberately do NOT lowercase or strip punctuation aggressively
here, because Named Entity Recognition and dependency parsing (the
next stages) rely heavily on capitalization and sentence structure.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import pandas as pd
import spacy
from spacy.language import Language

logger = logging.getLogger("kg_pipeline.preprocessing")

_WHITESPACE_RE = re.compile(r"\s+")
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str, max_chars: int = 20000) -> str:
    """Normalise whitespace and strip control characters. Keeps
    sentence-ending punctuation intact since it drives segmentation."""
    if not isinstance(text, str):
        return ""
    text = _CTRL_CHARS_RE.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


def load_spacy_sentencizer(model_name: str) -> Language:
    """Load a lightweight spaCy pipeline for sentence splitting only
    (used before the full NLP pass to keep this stage fast/independent)."""
    nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def segment_into_sentences(nlp: Language, text: str, min_tokens: int = 4) -> List[str]:
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s.split()) >= min_tokens:
            sentences.append(s)
    return sentences


def preprocess_articles(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Clean each article's text and split into sentence-level rows.

    Returns a long-format DataFrame with columns:
    [doc_id, category, sent_id, sentence]
    """
    prep_cfg = config["preprocessing"]
    nlp_cfg = config["nlp"]

    nlp = load_spacy_sentencizer(nlp_cfg["spacy_model"])

    records = []
    for _, row in df.iterrows():
        cleaned = clean_text(row["text"], max_chars=prep_cfg["max_doc_chars"])
        sentences = segment_into_sentences(nlp, cleaned, prep_cfg["min_sentence_length"])
        for i, sent in enumerate(sentences):
            records.append(
                {
                    "doc_id": row["doc_id"],
                    "category": row["category"],
                    "sent_id": f"{row['doc_id']}_s{i:03d}",
                    "sentence": sent,
                }
            )

    sent_df = pd.DataFrame(records)
    logger.info(
        f"Preprocessed {len(df)} documents into {len(sent_df)} sentences "
        f"(avg {len(sent_df) / max(len(df), 1):.1f} sentences/doc)."
    )
    return sent_df


if __name__ == "__main__":
    from src.data_loader import get_dataset
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)
    articles = get_dataset(cfg)
    sentences = preprocess_articles(articles, cfg)
    print(sentences.head(10))
