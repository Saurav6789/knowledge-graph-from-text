#!/usr/bin/env python3
"""
End-to-End Knowledge Graph Pipeline
=====================================
Orchestrates all six stages:
  1. Data acquisition          (src/data_loader.py)
  2. Text preprocessing        (src/preprocessing.py)
  3. NER + coreference         (src/ner_extraction.py)
  4. Relation extraction       (src/relation_extraction.py)
  5. Graph construction        (src/graph_builder.py)
  6. Visualization             (src/visualize.py)

Usage:
    python main.py                          # run with config.yaml defaults
    python main.py --sample-size 50         # quick smoke-test run
    python main.py --categories tech sport  # restrict to categories
    python main.py --backend neo4j          # also push graph into Neo4j
    python main.py --skip-download          # reuse cached raw CSV

See README.md for full setup and usage instructions.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data_loader import download_dataset, load_articles
from src.graph_builder import build_graph, compute_graph_stats, export_graph, push_to_neo4j
from src.ner_extraction import extract_entities
from src.preprocessing import preprocess_articles
from src.relation_extraction import extract_relations
from src.utils import load_config, setup_logging
from src.visualize import render_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Knowledge Graph from unstructured text.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--sample-size", type=int, default=None, help="Override number of articles to process.")
    parser.add_argument("--categories", nargs="*", default=None, help="Override which BBC categories to include.")
    parser.add_argument("--backend", choices=["networkx", "neo4j"], default=None, help="Override graph backend.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse the cached raw dataset if present.")
    parser.add_argument("--no-viz", action="store_true", help="Skip generating the interactive HTML visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.sample_size is not None:
        config["data"]["sample_size"] = args.sample_size
    if args.categories is not None:
        config["data"]["categories"] = args.categories
    if args.backend is not None:
        config["graph"]["backend"] = args.backend

    logger = setup_logging(config)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("STAGE 1/6 - Data acquisition")
    raw_path = download_dataset(config) if not args.skip_download else Path(
        config["data"]["raw_path"]
    )
    articles = load_articles(config, raw_path if raw_path.is_absolute() else None)

    logger.info("=" * 70)
    logger.info("STAGE 2/6 - Preprocessing & sentence segmentation")
    sentences = preprocess_articles(articles, config)

    logger.info("=" * 70)
    logger.info("STAGE 3/6 - Named entity recognition & coreference resolution")
    mentions = extract_entities(sentences, config)

    logger.info("=" * 70)
    logger.info("STAGE 4/6 - Relation extraction (SVO triples)")
    triples = extract_relations(sentences, config)

    logger.info("=" * 70)
    logger.info("STAGE 5/6 - Knowledge graph construction")
    graph = build_graph(mentions, triples, config)
    export_graph(graph, config)

    if config["graph"]["backend"] == "neo4j":
        logger.info("Pushing graph into Neo4j ...")
        push_to_neo4j(graph, config)

    if not args.no_viz:
        logger.info("=" * 70)
        logger.info("STAGE 6/6 - Interactive visualization")
        html_path = render_html(graph, config)
        logger.info(f"Open this file in your browser: {html_path}")

    elapsed = time.time() - t0
    stats = compute_graph_stats(graph)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Articles processed : {len(articles)}")
    logger.info(f"  Sentences parsed   : {len(sentences)}")
    logger.info(f"  Entity mentions    : {len(mentions)}")
    logger.info(f"  Raw triples        : {len(triples)}")
    logger.info(f"  Graph nodes/edges  : {stats['num_nodes']} / {stats['num_edges']}")
    logger.info(f"  Total time         : {elapsed:.1f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
