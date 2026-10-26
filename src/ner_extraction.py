"""
Stage 3: Named Entity Recognition & Lightweight Coreference Resolution
=======================================================================
Runs spaCy's full pipeline (tagger, parser, NER) over every sentence,
keeps entities of the configured types, and normalises entity surface
forms (e.g. "Apple", "Apple Inc.", "apple" -> one canonical node).

Coreference: full neural coreference resolution (e.g. neuralcoref /
coreferee) is heavyweight and brittle to install across platforms, so
for tutorial purposes we implement a transparent, explainable heuristic
that is enough to noticeably improve graph connectivity:
  - Pronouns ("it", "he", "she", "they") are resolved to the nearest
    preceding entity of a compatible type within a look-back window
    of sentences (within the same document).
This is intentionally simple so viewers can read the code and see
exactly what a coreference pass is doing before graduating to a full
neural coreference library.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from tqdm import tqdm

logger = logging.getLogger("kg_pipeline.ner_extraction")

PRONOUN_MAP = {
    "he": "PERSON", "him": "PERSON", "his": "PERSON",
    "she": "PERSON", "her": "PERSON", "hers": "PERSON",
    "they": {"PERSON", "ORG", "NORP", "GPE"}, "them": {"PERSON", "ORG", "NORP", "GPE"},
    "their": {"PERSON", "ORG", "NORP", "GPE"},
    "it": {"ORG", "PRODUCT", "GPE", "LOC", "FAC"}, "its": {"ORG", "PRODUCT", "GPE", "LOC", "FAC"},
}


@dataclass
class EntityMention:
    doc_id: str
    sent_id: str
    text: str
    canonical: str
    label: str
    start_char: int
    end_char: int
    resolved_from_pronoun: bool = False


def normalize_entity_text(text: str) -> str:
    """Canonicalise an entity surface form: strip leading articles/
    possessives and excess whitespace, keep original casing for
    display but use a lowercased key for matching elsewhere."""
    text = text.strip().strip("'\"")
    for article in ("The ", "the ", "A ", "a ", "An ", "an "):
        if text.startswith(article):
            text = text[len(article):]
    text = text.replace("’s", "").replace("'s", "")
    return " ".join(text.split())


def load_full_pipeline(model_name: str) -> Language:
    return spacy.load(model_name)


def _find_pronoun_antecedent(
    pronoun_lower: str,
    recent_entities: List[EntityMention],
) -> Optional[EntityMention]:
    allowed = PRONOUN_MAP.get(pronoun_lower)
    if allowed is None:
        return None
    if isinstance(allowed, str):
        allowed = {allowed}
    for ent in reversed(recent_entities):
        if ent.label in allowed:
            return ent
    return None


def extract_entities(
    sent_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Run NER + coreference heuristics over every sentence.

    Returns a DataFrame of entity mentions:
    [doc_id, sent_id, text, canonical, label, start_char, end_char, resolved_from_pronoun]
    """
    nlp_cfg = config["nlp"]
    keep_labels = set(nlp_cfg["keep_entity_labels"])
    coref_cfg = nlp_cfg.get("coreference", {})
    coref_enabled = coref_cfg.get("enabled", True)
    lookback = coref_cfg.get("lookback_sentences", 3)

    nlp = load_full_pipeline(nlp_cfg["spacy_model"])

    all_mentions: List[EntityMention] = []

    for doc_id, group in tqdm(sent_df.groupby("doc_id"), desc="NER per document"):
        recent_entities: List[EntityMention] = []
        sent_window: List[List[EntityMention]] = []

        for _, row in group.iterrows():
            doc: Doc = nlp(row["sentence"])
            sentence_mentions: List[EntityMention] = []

            for ent in doc.ents:
                if ent.label_ not in keep_labels:
                    continue
                canonical = normalize_entity_text(ent.text)
                if len(canonical) < 2:
                    continue
                mention = EntityMention(
                    doc_id=doc_id,
                    sent_id=row["sent_id"],
                    text=ent.text,
                    canonical=canonical,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
                sentence_mentions.append(mention)
                all_mentions.append(mention)

            if coref_enabled:
                for token in doc:
                    if token.pos_ == "PRON" and token.lower_ in PRONOUN_MAP:
                        antecedent = _find_pronoun_antecedent(token.lower_, recent_entities)
                        if antecedent is not None:
                            resolved = EntityMention(
                                doc_id=doc_id,
                                sent_id=row["sent_id"],
                                text=token.text,
                                canonical=antecedent.canonical,
                                label=antecedent.label,
                                start_char=token.idx,
                                end_char=token.idx + len(token.text),
                                resolved_from_pronoun=True,
                            )
                            sentence_mentions.append(resolved)
                            all_mentions.append(resolved)

            recent_entities.extend(sentence_mentions)
            sent_window.append(sentence_mentions)
            if len(sent_window) > lookback:
                dropped = sent_window.pop(0)
                for d in dropped:
                    if d in recent_entities:
                        recent_entities.remove(d)

    mentions_df = pd.DataFrame([m.__dict__ for m in all_mentions])
    logger.info(
        f"Extracted {len(mentions_df)} entity mentions "
        f"({mentions_df['canonical'].nunique() if not mentions_df.empty else 0} unique canonical entities), "
        f"{mentions_df['resolved_from_pronoun'].sum() if not mentions_df.empty else 0} resolved via pronoun coreference."
    )
    return mentions_df


if __name__ == "__main__":
    from src.data_loader import get_dataset
    from src.preprocessing import preprocess_articles
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)
    articles = get_dataset(cfg)
    sentences = preprocess_articles(articles, cfg)
    mentions = extract_entities(sentences.head(200), cfg)
    print(mentions.head(20))
    print(mentions["label"].value_counts())
