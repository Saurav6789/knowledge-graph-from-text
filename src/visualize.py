"""
Stage 6: Visualization
========================
Renders the knowledge graph as a self-contained, interactive HTML file
(drag nodes, zoom, hover for details) using PyVis, which wraps the
vis-network.js JavaScript library. Great for screen-recording a demo:
open the .html file in any browser, no server required.

Nodes are colour-coded by entity type and sized by degree (how many
relations they participate in) so hubs are immediately visible.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import networkx as nx
from pyvis.network import Network

from src.utils import resolve_path

logger = logging.getLogger("kg_pipeline.visualize")

LABEL_COLORS = {
    "PERSON": "#4C72B0",
    "ORG": "#DD8452",
    "GPE": "#55A868",
    "LOC": "#55A868",
    "NORP": "#C44E52",
    "PRODUCT": "#8172B2",
    "EVENT": "#937860",
    "FAC": "#DA8BC3",
    "MONEY": "#8C8C8C",
    "DATE": "#CCB974",
    "MISC": "#64B5CD",
}


def render_html(G: nx.DiGraph, config: Dict[str, Any]) -> str:
    graph_cfg = config["graph"]
    out_cfg = config["output"]
    max_nodes = graph_cfg.get("max_nodes_for_viz", 400)

    # If the graph is large, keep only the highest-degree nodes so the
    # browser rendering stays smooth and readable.
    if G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        keep = {n for n, _ in sorted(degrees.items(), key=lambda x: -x[1])[:max_nodes]}
        G = G.subgraph(keep).copy()
        logger.info(f"Graph pruned to top {max_nodes} nodes by degree for visualization.")

    net = Network(
        height="850px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#0f1117",
        font_color="#f5f5f5",
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-2500, central_gravity=0.15, spring_length=150, spring_strength=0.03)

    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    for node, data in G.nodes(data=True):
        label = data.get("label", "MISC")
        color = LABEL_COLORS.get(label, LABEL_COLORS["MISC"])
        size = 12 + 28 * (degrees.get(node, 1) / max_degree)
        title = (
            f"<b>{node}</b><br>Type: {label}<br>Mentions: {data.get('frequency', '?')}"
            f"<br>Appears in {data.get('num_source_docs', '?')} document(s)<br>Degree: {degrees.get(node, 0)}"
        )
        net.add_node(node, label=node, title=title, color=color, size=size)

    for u, v, data in G.edges(data=True):
        title = f"{data.get('relation')} (seen {data.get('count', 1)}x)<br>e.g. \"{data.get('example_sentence', '')}\""
        net.add_edge(
            u,
            v,
            label=data.get("relation", ""),
            title=title,
            value=data.get("count", 1),
            arrows="to",
            color="#5c5f66",
        )

    net.set_options(
        """
        {
          "edges": {"font": {"size": 10, "color": "#cccccc", "strokeWidth": 0}, "smooth": {"type": "continuous"}},
          "nodes": {"font": {"size": 14}},
          "interaction": {"hover": true, "tooltipDelay": 100},
          "physics": {"stabilization": {"iterations": 150}}
        }
        """
    )

    html_path = resolve_path(out_cfg["html_viz_path"])
    net.write_html(str(html_path), notebook=False, open_browser=False)
    logger.info(f"Interactive visualization written to {html_path}")
    return str(html_path)


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg)
    graph_path = resolve_path(cfg["output"]["graphml_path"])
    G = nx.read_graphml(graph_path)
    render_html(G, cfg)
