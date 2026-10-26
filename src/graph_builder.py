"""
Stage 5: Knowledge Graph Construction
=======================================
Fuses the entity mentions (Stage 3) and the (subject, relation, object)
triples (Stage 4) into a single deduplicated graph:

  - Nodes = canonical entities (falls back to the raw noun-chunk text
    if a triple's subject/object doesn't overlap any recognised named
    entity - this keeps useful relations like "the company / launched
    / a product" even when spaCy's NER missed a span).
  - Edges = relations, aggregated so that repeated (subject, relation,
    object) triples across the corpus become ONE edge with a `count`
    weight and a `sentences` provenance list, instead of duplicate edges.
  - Node attributes: entity label (PERSON/ORG/...), mention frequency,
    a `sources` set of doc_ids it appeared in.

The result is exported to GraphML/GEXF (Gephi/Cytoscape-ready) and can
optionally be pushed into a live Neo4j instance for Cypher querying.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import pandas as pd

from src.utils import resolve_path

logger = logging.getLogger("kg_pipeline.graph_builder")


def _build_canonical_lookup(mentions_df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Map lowercased mention text -> (canonical form, most common label).
    Used to resolve a relation-extraction subject/object string to the
    entity graph already built from NER, so we don't create duplicate
    nodes like 'Parmalat' (from NER) and 'parmalat' (from a triple)."""
    lookup: Dict[str, Tuple[str, Counter]] = {}
    for _, row in mentions_df.iterrows():
        key = row["canonical"].lower()
        if key not in lookup:
            lookup[key] = [row["canonical"], Counter()]
        lookup[key][1][row["label"]] += 1

    resolved = {}
    for key, (canonical, label_counts) in lookup.items():
        resolved[key] = (canonical, label_counts.most_common(1)[0][0])
    return resolved


def _build_word_index(lookup: Dict[str, Tuple[str, str]]) -> Dict[str, set]:
    """Inverted index: word -> set of lookup keys containing that word.
    Lets us shortlist candidates instead of scanning every known entity
    for every triple (which would be O(triples x entities))."""
    index: Dict[str, set] = defaultdict(set)
    for k in lookup:
        for word in k.split():
            if len(word) >= 3:
                index[word].add(k)
    return index


_WORD_BOUNDARY_CACHE: Dict[str, "re.Pattern"] = {}


def _whole_word_contains(shorter: str, longer: str) -> bool:
    if len(shorter) < 3:
        return False
    pattern = _WORD_BOUNDARY_CACHE.get(shorter)
    if pattern is None:
        pattern = re.compile(r"(?<!\w)" + re.escape(shorter) + r"(?!\w)")
        _WORD_BOUNDARY_CACHE[shorter] = pattern
    return pattern.search(longer) is not None


def _resolve_node(
    text: str,
    lookup: Dict[str, Tuple[str, str]],
    word_index: Dict[str, set],
) -> Tuple[str, str]:
    """Try exact match, then a WORD-BOUNDARY-safe partial match (scoped
    to a small candidate shortlist via an inverted word index), against
    the NER canonical lookup. Falls back to the raw text with label
    'MISC'.

    NOTE: matching must respect word boundaries - a naive `k in key`
    substring check would wrongly link an entity like "IE" to any word
    that merely *contains* the letters "ie" (e.g. "field", "believe"),
    fabricating false hub nodes. We only accept a partial match when
    the shorter phrase appears as a complete sequence of whole words
    inside the longer one, and only for candidates with >= 3 characters
    to avoid short acronyms swallowing unrelated text.
    """
    key = text.lower()
    if key in lookup:
        return lookup[key]

    candidates: set = set()
    for word in key.split():
        if len(word) >= 3 and word in word_index:
            candidates |= word_index[word]

    for k in candidates:
        if _whole_word_contains(k, key) or _whole_word_contains(key, k):
            return lookup[k]
    return text, "MISC"


def build_graph(
    mentions_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    config: Dict[str, Any],
) -> nx.DiGraph:
    """Construct the aggregated, deduplicated knowledge graph."""
    graph_cfg = config["graph"]
    min_freq = graph_cfg.get("min_entity_frequency", 1)

    lookup = _build_canonical_lookup(mentions_df) if not mentions_df.empty else {}
    word_index = _build_word_index(lookup)

    G = nx.DiGraph()

    # First pass: register every NER entity as a node (even if it never
    # appears as a triple subject/object) so the graph reflects the full
    # set of entities discussed in the corpus.
    entity_freq: Counter = Counter(mentions_df["canonical"]) if not mentions_df.empty else Counter()
    entity_docs: Dict[str, set] = defaultdict(set)
    entity_label: Dict[str, str] = {}
    if not mentions_df.empty:
        for _, row in mentions_df.iterrows():
            entity_docs[row["canonical"]].add(row["doc_id"])
            entity_label[row["canonical"]] = row["label"]

    for canonical, freq in entity_freq.items():
        if freq < min_freq:
            continue
        G.add_node(
            canonical,
            label=entity_label.get(canonical, "MISC"),
            frequency=int(freq),
            num_source_docs=len(entity_docs[canonical]),
        )

    # Second pass: add/aggregate edges from triples, resolving each
    # subject/object to a canonical entity where possible.
    edge_agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for _, row in triples_df.iterrows():
        subj_canon, subj_label = _resolve_node(row["subject"], lookup, word_index)
        obj_canon, obj_label = _resolve_node(row["object"], lookup, word_index)
        relation = row["relation"].strip()

        if not subj_canon or not obj_canon or subj_canon.lower() == obj_canon.lower():
            continue

        edge_key = (subj_canon, relation, obj_canon)
        if edge_key not in edge_agg:
            edge_agg[edge_key] = {
                "count": 0,
                "doc_ids": set(),
                "example_sentence": row["sentence"],
                "subj_label": subj_label,
                "obj_label": obj_label,
            }
        edge_agg[edge_key]["count"] += 1
        edge_agg[edge_key]["doc_ids"].add(row["doc_id"])

        for node, lbl in ((subj_canon, subj_label), (obj_canon, obj_label)):
            if not G.has_node(node):
                G.add_node(node, label=lbl, frequency=1, num_source_docs=1)

    for (subj, relation, obj), data in edge_agg.items():
        G.add_edge(
            subj,
            obj,
            relation=relation,
            count=data["count"],
            num_source_docs=len(data["doc_ids"]),
            example_sentence=data["example_sentence"][:280],
        )

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    logger.info(
        f"Built knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(removed {len(isolated)} isolated nodes)."
    )
    return G


def compute_graph_stats(G: nx.DiGraph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"num_nodes": 0, "num_edges": 0}

    undirected = G.to_undirected()
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: -x[1])[:15]
    components = list(nx.connected_components(undirected))

    label_counts = Counter(nx.get_node_attributes(G, "label").values())
    relation_counts = Counter(nx.get_edge_attributes(G, "relation").values())

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_connected_components": len(components),
        "largest_component_size": max((len(c) for c in components), default=0),
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "density": round(nx.density(G), 5),
        "top_entities_by_degree_centrality": [
            {"entity": n, "score": round(s, 4), "label": G.nodes[n].get("label")} for n, s in top_nodes
        ],
        "entity_label_distribution": dict(label_counts.most_common()),
        "top_relations": dict(relation_counts.most_common(15)),
    }
    return stats


def export_graph(G: nx.DiGraph, config: Dict[str, Any]) -> None:
    out_cfg = config["output"]

    graphml_path = resolve_path(out_cfg["graphml_path"])
    gexf_path = resolve_path(out_cfg["gexf_path"])
    triples_csv_path = resolve_path(out_cfg["triples_csv_path"])
    stats_path = resolve_path(out_cfg["stats_json_path"])

    G_export = G.copy()
    for _, _, data in G_export.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, set):
                data[k] = ", ".join(sorted(v))

    nx.write_graphml(G_export, graphml_path)
    nx.write_gexf(G_export, gexf_path)
    logger.info(f"Exported graph to {graphml_path} and {gexf_path}")

    rows = [
        {"subject": u, "relation": data.get("relation"), "object": v, "count": data.get("count")}
        for u, v, data in G.edges(data=True)
    ]
    pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(triples_csv_path, index=False)
    logger.info(f"Exported {len(rows)} deduplicated triples to {triples_csv_path}")

    stats = compute_graph_stats(G)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Exported graph statistics to {stats_path}")


def push_to_neo4j(G: nx.DiGraph, config: Dict[str, Any]) -> None:
    """Optional: load the graph into a running Neo4j instance so it can
    be explored with Cypher / Neo4j Browser / Bloom."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError(
            "The 'neo4j' package is required for --backend neo4j. Install with `pip install neo4j`."
        ) from exc

    neo_cfg = config["graph"]["neo4j"]
    driver = GraphDatabase.driver(neo_cfg["uri"], auth=(neo_cfg["user"], neo_cfg["password"]))

    with driver.session(database=neo_cfg.get("database", "neo4j")) as session:
        if neo_cfg.get("clear_before_load", True):
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing Neo4j graph.")

        for node, data in G.nodes(data=True):
            session.run(
                "MERGE (e:Entity {name: $name}) SET e.label = $label, e.frequency = $frequency",
                name=node,
                label=data.get("label", "MISC"),
                frequency=data.get("frequency", 0),
            )

        for u, v, data in G.edges(data=True):
            session.run(
                """
                MATCH (a:Entity {name: $u}), (b:Entity {name: $v})
                MERGE (a)-[r:RELATION {type: $relation}]->(b)
                SET r.count = $count
                """,
                u=u,
                v=v,
                relation=data.get("relation", "related_to"),
                count=data.get("count", 1),
            )
    driver.close()
    logger.info(
        f"Pushed {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to Neo4j at {neo_cfg['uri']}."
    )


if __name__ == "__main__":
    from src.data_loader import get_dataset
    from src.ner_extraction import extract_entities
    from src.preprocessing import preprocess_articles
    from src.relation_extraction import extract_relations
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)
    articles = get_dataset(cfg)
    sentences = preprocess_articles(articles, cfg)
    mentions = extract_entities(sentences, cfg)
    triples = extract_relations(sentences, cfg)
    graph = build_graph(mentions, triples, cfg)
    export_graph(graph, cfg)
    print(json.dumps(compute_graph_stats(graph), indent=2))
