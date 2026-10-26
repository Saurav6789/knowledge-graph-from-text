"""
Stage 4: Relation Extraction (Subject - Predicate - Object triples)
=====================================================================
This is the heart of turning unstructured text into a graph: for every
sentence we walk spaCy's dependency parse tree to pull out
(subject, relation, object) triples, e.g.:

    "Parmalat collapsed in 2003 after fraud was uncovered."
    -> (Parmalat, collapsed in, 2003)

Approach ("dependency" method):
  1. Find the sentence's ROOT verb (or any verb with a nominal subject).
  2. Walk left from the verb to its nsubj/nsubjpass child -> the subject.
  3. Walk right from the verb to its dobj/attr/pobj (via prep) child -> the object.
  4. The relation label is the verb lemma, extended with any attached
     preposition/particle (e.g. "agreed to", "invested in") so the edge
     reads naturally.
  5. Subject/object spans are expanded to their full noun chunk so
     "the Italian food group" stays together instead of just "group".
  6. Each triple's subject/object are matched back to the nearest
     overlapping entity mention (from ner_extraction) when possible,
     so the resulting graph nodes are the *canonical* entities rather
     than raw noun-chunk text - this is what keeps the graph from
     fragmenting into hundreds of near-duplicate nodes.

This dependency-based method is intentionally transparent and fully
inspectable (no black-box model) which makes it ideal for a tutorial:
viewers can print doc.sents and doc[i].dep_ side by side with what the
function extracts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm

from src.ner_extraction import normalize_entity_text

logger = logging.getLogger("kg_pipeline.relation_extraction")

SUBJECT_DEPS = {"nsubj", "nsubjpass"}
OBJECT_DEPS = {"dobj", "attr", "oprd", "dative"}
PREP_OBJECT_DEPS = {"pobj"}
PASSIVE_DEPS = {"nsubjpass"}

# Bare pronouns/determiners that were NOT resolved by the coreference
# step (Stage 3 only resolves pronouns to entities for the NER mention
# list, not inline within the dependency tree here) make for
# uninformative, noisy graph nodes like "it -> use -> Mobile". Rather
# than presenting these as if they were real entities, we drop triples
# whose subject or object is *entirely* one of these tokens.
PRONOUN_STOPWORDS = {
    "it", "he", "she", "they", "them", "this", "that", "these", "those",
    "who", "which", "what", "i", "we", "you", "its", "his", "her", "their",
    "our", "your", "one", "some", "others", "himself", "herself", "itself",
    "themselves",
}


@dataclass
class Triple:
    doc_id: str
    sent_id: str
    subject: str
    relation: str
    object: str
    sentence: str


def _expand_to_span(token: Token) -> str:
    """Expand a token to its enclosing noun chunk (if any) else its
    subtree of compound/amod/det children, for a more readable node
    label than a single bare token."""
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return normalize_entity_text(chunk.text)
    subtree_tokens = [t for t in token.subtree if t.dep_ in {"compound", "amod", "det"} or t == token]
    subtree_tokens = sorted(subtree_tokens, key=lambda t: t.i)
    text = " ".join(t.text for t in subtree_tokens) if subtree_tokens else token.text
    return normalize_entity_text(text)


def _build_relation_label(verb: Token, max_tokens: int = 6) -> str:
    """Compose a relation phrase from the verb lemma plus any attached
    particle/preposition (e.g. 'agree' + 'to' -> 'agree to')."""
    parts = [verb.lemma_]
    for child in verb.children:
        if child.dep_ in {"prt", "prep"} and child.i > verb.i:
            parts.append(child.text.lower())
            break  # only take the first attached preposition
    if verb.dep_ in PASSIVE_DEPS or any(c.dep_ == "auxpass" for c in verb.children):
        parts.insert(0, "was")
    return " ".join(parts)[: max_tokens * 15]


def _extract_triples_from_sentence(doc: Doc) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []

    for token in doc:
        if token.pos_ != "VERB" and token.pos_ != "AUX":
            continue

        subject_tok: Optional[Token] = None
        object_tok: Optional[Token] = None

        for child in token.children:
            if child.dep_ in SUBJECT_DEPS and subject_tok is None:
                subject_tok = child
            elif child.dep_ in OBJECT_DEPS and object_tok is None:
                object_tok = child
            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ in PREP_OBJECT_DEPS and object_tok is None:
                        object_tok = grandchild

        if subject_tok is None:
            continue

        # Fall back: look for an object via a prepositional phrase attached
        # directly to the verb even if we already scanned children once.
        if object_tok is None:
            for child in token.children:
                if child.dep_ == "prep":
                    pobjs = [gc for gc in child.children if gc.dep_ == "pobj"]
                    if pobjs:
                        object_tok = pobjs[0]
                        break

        if object_tok is None:
            continue

        subject_text = _expand_to_span(subject_tok)
        object_text = _expand_to_span(object_tok)
        relation = _build_relation_label(token)

        if not subject_text or not object_text or subject_text.lower() == object_text.lower():
            continue
        if subject_text.lower() in PRONOUN_STOPWORDS or object_text.lower() in PRONOUN_STOPWORDS:
            continue
        if not any(ch.isalpha() for ch in subject_text) or not any(ch.isalpha() for ch in object_text):
            continue

        triples.append((subject_text, relation, object_text))

    return triples


def extract_relations(
    sent_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Run dependency-based triple extraction over every sentence.

    Returns a DataFrame: [doc_id, sent_id, subject, relation, object, sentence]
    """
    nlp_cfg = config["nlp"]
    rel_cfg = config["relation_extraction"]
    nlp: Language = spacy.load(nlp_cfg["spacy_model"])

    drop_stopword_only = rel_cfg.get("drop_stopword_only_relations", True)
    max_rel_tokens = rel_cfg.get("max_relation_tokens", 6)

    all_triples: List[Triple] = []

    for _, row in tqdm(sent_df.iterrows(), total=len(sent_df), desc="Relation extraction"):
        doc = nlp(row["sentence"])
        for subj, rel, obj in _extract_triples_from_sentence(doc):
            if drop_stopword_only and rel.strip() in {"be", "have", "do"}:
                continue
            all_triples.append(
                Triple(
                    doc_id=row["doc_id"],
                    sent_id=row["sent_id"],
                    subject=subj,
                    relation=rel,
                    object=obj,
                    sentence=row["sentence"],
                )
            )

    triples_df = pd.DataFrame([t.__dict__ for t in all_triples])
    logger.info(f"Extracted {len(triples_df)} raw (subject, relation, object) triples.")
    return triples_df


if __name__ == "__main__":
    from src.data_loader import get_dataset
    from src.preprocessing import preprocess_articles
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)
    articles = get_dataset(cfg)
    sentences = preprocess_articles(articles, cfg)
    triples = extract_relations(sentences.head(200), cfg)
    pd.set_option("display.max_colwidth", 40)
    print(triples.head(25)[["subject", "relation", "object"]])
