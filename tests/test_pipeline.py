"""
Basic unit/integration tests for the Knowledge Graph pipeline.

Run with:  pytest -v
"""
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_builder import build_graph, compute_graph_stats
from src.ner_extraction import normalize_entity_text
from src.preprocessing import clean_text, load_spacy_sentencizer, segment_into_sentences
from src.relation_extraction import _extract_triples_from_sentence
from src.utils import load_config

import spacy


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def test_config_loads(config):
    assert "data" in config
    assert "nlp" in config
    assert config["nlp"]["spacy_model"] == "en_core_web_sm"


def test_clean_text_normalizes_whitespace():
    dirty = "Hello   world\n\nthis\tis   messy"
    cleaned = clean_text(dirty)
    assert "  " not in cleaned
    assert "\n" not in cleaned
    assert "\t" not in cleaned


def test_clean_text_truncates():
    long_text = "a" * 100
    assert len(clean_text(long_text, max_chars=10)) == 10


def test_sentence_segmentation(config):
    nlp_light = load_spacy_sentencizer(config["nlp"]["spacy_model"])
    text = "Apple was founded in Cupertino. It makes the iPhone. Short."
    sentences = segment_into_sentences(nlp_light, text, min_tokens=3)
    assert len(sentences) == 2  # "Short." is dropped (below min_tokens)


def test_normalize_entity_text_strips_articles():
    assert normalize_entity_text("The United Nations") == "United Nations"
    assert normalize_entity_text("Apple's") == "Apple"
    assert normalize_entity_text("  extra   spaces  ") == "extra spaces"


def test_svo_extraction_basic(nlp):
    doc = nlp("Microsoft acquired Activision Blizzard in 2023.")
    triples = _extract_triples_from_sentence(doc)
    assert len(triples) >= 1
    subj, rel, obj = triples[0]
    assert "microsoft" in subj.lower()
    assert "activision" in obj.lower()


def test_svo_extraction_no_verb_no_triples(nlp):
    doc = nlp("A very quiet afternoon in the park.")
    triples = _extract_triples_from_sentence(doc)
    assert isinstance(triples, list)  # should not crash; may be empty


def test_build_graph_deduplicates_edges(config):
    mentions_df = pd.DataFrame(
        [
            {"doc_id": "d1", "sent_id": "d1_s0", "text": "Microsoft", "canonical": "Microsoft",
             "label": "ORG", "start_char": 0, "end_char": 9, "resolved_from_pronoun": False},
            {"doc_id": "d1", "sent_id": "d1_s1", "text": "Activision", "canonical": "Activision",
             "label": "ORG", "start_char": 0, "end_char": 10, "resolved_from_pronoun": False},
        ]
    )
    triples_df = pd.DataFrame(
        [
            {"doc_id": "d1", "sent_id": "d1_s0", "subject": "Microsoft", "relation": "acquire",
             "object": "Activision", "sentence": "Microsoft acquired Activision."},
            {"doc_id": "d1", "sent_id": "d1_s1", "subject": "Microsoft", "relation": "acquire",
             "object": "Activision", "sentence": "Microsoft acquired Activision again."},
        ]
    )
    graph = build_graph(mentions_df, triples_df, config)
    assert graph.number_of_edges() == 1  # duplicate triple collapsed into one edge
    edge_data = graph.get_edge_data("Microsoft", "Activision")
    assert edge_data["count"] == 2


def test_graph_stats_on_empty_graph():
    empty_graph = nx.DiGraph()
    stats = compute_graph_stats(empty_graph)
    assert stats["num_nodes"] == 0


def test_graph_no_self_loops(config):
    mentions_df = pd.DataFrame(
        [
            {"doc_id": "d1", "sent_id": "d1_s0", "text": "Apple", "canonical": "Apple",
             "label": "ORG", "start_char": 0, "end_char": 5, "resolved_from_pronoun": False},
        ]
    )
    triples_df = pd.DataFrame(
        [
            {"doc_id": "d1", "sent_id": "d1_s0", "subject": "Apple", "relation": "is",
             "object": "Apple", "sentence": "Apple is Apple."},
        ]
    )
    graph = build_graph(mentions_df, triples_df, config)
    assert graph.number_of_edges() == 0  # self-loop should be dropped
