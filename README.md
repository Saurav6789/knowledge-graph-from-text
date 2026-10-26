# Developing a Knowledge Graph from Unstructured Text Using NLP (Python)

An end-to-end, production-shaped pipeline that turns raw, unstructured
text (news articles, reports, transcripts, ...) into a queryable,
visualizable **Knowledge Graph** — entities as nodes, relationships as
edges — using classic NLP techniques (Named Entity Recognition,
dependency parsing, coreference resolution) with **spaCy**,
**NetworkX**, and **PyVis**.

Built as the companion repo for a YouTube tutorial. Every stage is a
separate, independently-runnable, well-commented module so you can
show each concept on screen in isolation before wiring it all together.

```
Unstructured Text  ──►  Sentences  ──►  Entities + Coreference  ──►  Relations (SVO)  ──►  Knowledge Graph  ──►  Interactive Viz
    (raw .csv)          (spaCy)         (spaCy NER + heuristic)     (dependency parse)      (NetworkX)          (PyVis HTML)
```

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Pre-conditions](#pre-conditions)
5. [Setup](#setup)
6. [Running the pipeline](#running-the-pipeline)
7. [Understanding the output](#understanding-the-output)
8. [Configuration reference](#configuration-reference)
9. [Optional: loading into Neo4j](#optional-loading-into-neo4j)
10. [Using your own text data](#using-your-own-text-data)
11. [Running tests](#running-tests)
12. [Project structure](#project-structure)
13. [Known limitations & tutorial talking points](#known-limitations--tutorial-talking-points)
14. [Troubleshooting](#troubleshooting)
15. [License](#license)

---

## What this project does

Given a folder of unstructured documents, the pipeline:

1. **Ingests** the raw text.
2. **Cleans & segments** it into sentences.
3. **Extracts named entities** (people, organizations, places, dates,
   money, products, ...) and resolves simple pronouns ("it", "they",
   "he") back to the entity they refer to.
4. **Extracts relationships** between entities by walking each
   sentence's grammatical dependency tree to find
   `(subject, relation, object)` triples — e.g.
   `(Microsoft, acquire, Activision Blizzard)`.
5. **Builds a deduplicated knowledge graph**: repeated mentions of the
   same entity collapse into one node; repeated relations collapse
   into one weighted edge.
6. **Exports** the graph as GraphML/GEXF (for Gephi/Cytoscape),
   CSV triples, JSON stats, and a self-contained **interactive HTML
   visualization** you can open in any browser — no server required.
7. *(Optional)* **Loads the graph into Neo4j** for Cypher querying.

---

## Dataset

This repo ships pointed at the **BBC News Full-Text dataset**
(Greene, D. & Cunningham, P., 2006), a well-known, widely-cited
benchmark corpus in the NLP community — **2,225 real news articles**
across 5 categories (business, entertainment, politics, sport, tech).
It's a realistic stand-in for the kind of "real-world unstructured
text" (press releases, articles, reports) a Knowledge Graph pipeline
is typically built for, and it's large enough to produce a genuinely
interesting, densely-connected graph rather than a toy example.

The loader downloads it automatically on first run from a public
GitHub mirror — no manual download, no Kaggle account needed. See
[Using your own text data](#using-your-own-text-data) to point the
pipeline at a different corpus (your own documents, Wikipedia dumps,
PDFs converted to text, etc.) — nothing else in the pipeline needs to
change.

> By default, `config.yaml` restricts the run to the **business** and
> **tech** categories and samples **150 articles** so the tutorial run
> finishes in under a minute. Set `sample_size: null` and
> `categories: []` in `config.yaml` (or pass `--sample-size 0
> --categories` flags — see below) to process the **full** 2,225-article
> corpus, which takes roughly 10–20 minutes on a laptop CPU.

---

## Architecture

```
project/
│
├── src/
│   ├── data_loader.py        # Stage 1: download + load + filter/sample the dataset
│   ├── preprocessing.py      # Stage 2: text cleaning + sentence segmentation
│   ├── ner_extraction.py     # Stage 3: NER + heuristic pronoun coreference resolution
│   ├── relation_extraction.py# Stage 4: dependency-parse based SVO triple extraction
│   ├── graph_builder.py      # Stage 5: entity+triple fusion, dedup, NetworkX graph, exports
│   ├── visualize.py          # Stage 6: interactive PyVis HTML rendering
│   └── utils.py              # config loading, logging, path helpers
│
├── main.py                   # CLI orchestrator - runs all 6 stages end to end
├── config.yaml               # every tunable parameter lives here
├── pyproject.toml / uv.lock  # uv project definition + locked dependencies
├── requirements.txt          # pip alternative
│
├── notebooks/
│   └── knowledge_graph_walkthrough.ipynb   # stage-by-stage interactive walkthrough
│
├── tests/
│   └── test_pipeline.py      # pytest unit/integration tests
│
└── data/
    ├── raw/                  # downloaded dataset cache
    ├── processed/            # (reserved for cached intermediate artifacts)
    └── output/                # graph exports, visualization, logs (generated)
```

Each `src/*.py` module has a `if __name__ == "__main__":` block, so you
can run any single stage on its own, e.g.:

```bash
python -m src.data_loader          # just test data acquisition
python -m src.ner_extraction       # just test NER + coreference
```

This is designed so you can screen-record each stage in isolation for
the tutorial before showing `main.py` tie everything together.

---

## Pre-conditions

- **Python 3.11 or 3.12** (developed & tested on 3.12; pinned in
  `.python-version`). Python 3.13+ is not supported by the pinned
  spaCy 3.7 release.
- **[uv](https://docs.astral.sh/uv/)** (recommended) or plain `pip`.
- **~500 MB free disk space** (spaCy model + dataset + graph exports).
- **Internet access** on first run, to download:
  - the BBC News dataset (~5 MB, from GitHub), and
  - the spaCy English model (`en_core_web_sm`, ~13 MB).
  After the first run, both are cached locally and no internet is
  required.
- **(Optional)** A running **Neo4j** instance (Desktop or Docker) if
  you want to push the graph into Neo4j instead of/in addition to the
  NetworkX/HTML output. Not required for the default run.
- No GPU required — everything runs on CPU in well under a minute for
  the default 150-article sample.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Saurav6789/knowledge-graph-from-text.git
cd knowledge-graph-from-text
```

### Option A — uv (recommended)

The project ships a `pyproject.toml` and `uv.lock`, so a single command
creates the virtual environment, installs every pinned dependency
(including the `en_core_web_sm` spaCy model and `pytest`):

```bash
uv sync
```

Then prefix commands with `uv run`, e.g. `uv run python main.py` or
`uv run pytest`. Skip ahead to [step 5](#5-verify-the-install).

### Option B — pip

#### 2. Create a virtual environment (strongly recommended)

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

> If your network blocks the spaCy CLI downloader, install the model
> directly as a wheel instead:
> ```bash
> pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
> ```

### 5. Verify the install

```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK:', nlp('Hello world.').ents)"
```

You're ready to run the pipeline.

---

## Running the pipeline

### Quick start (recommended first run)

```bash
python main.py
```

This uses the defaults in `config.yaml`: 150 articles from the
`business` and `tech` categories. On a typical laptop this finishes in
**30–60 seconds** and prints a stage-by-stage log, ending with:

```
PIPELINE COMPLETE
  Articles processed : 150
  Sentences parsed   : 2858
  Entity mentions    : 6206
  Raw triples        : 1651
  Graph nodes/edges  : 2185 / 1639
  Total time         : 44.0s
```

Then open **`data/output/knowledge_graph.html`** in your browser to
explore the interactive graph.

### Common variations

```bash
# Fast smoke-test run (great for a live tutorial demo)
python main.py --sample-size 30

# Process the FULL 2,225-article dataset, all 5 categories
python main.py --sample-size 0 --categories

# Restrict to specific categories
python main.py --categories sport politics

# Reuse the already-downloaded raw CSV (skip the network call)
python main.py --skip-download

# Skip generating the (slower) HTML visualization
python main.py --no-viz

# Push the resulting graph into a running Neo4j instance
python main.py --backend neo4j
```

Run `python main.py --help` for the full list of flags.

### Step-by-step, in a notebook

For the tutorial, `notebooks/knowledge_graph_walkthrough.ipynb` runs
every stage one cell at a time with intermediate DataFrames printed
out, an entity-type bar chart, a live dependency-parse demo on a
sentence you can edit, and graph-querying examples with NetworkX.

```bash
jupyter notebook notebooks/knowledge_graph_walkthrough.ipynb
```

---

## Understanding the output

After a run, `data/output/` contains:

| File | What it is |
|---|---|
| `knowledge_graph.html` | **Open this first.** Self-contained interactive graph — drag nodes, zoom, hover for entity/relation details. |
| `knowledge_graph.graphml` | Standard graph format, importable into **Gephi**, **Cytoscape**, or **yEd** for advanced layout/analysis. |
| `knowledge_graph.gexf` | Same graph, GEXF format (also Gephi-native). |
| `triples.csv` | Every deduplicated `(subject, relation, object, count)` edge — easy to open in Excel/Pandas. |
| `graph_stats.json` | Node/edge counts, density, top entities by degree centrality, entity-type distribution, most common relation types. |
| `pipeline.log` | Full run log (also printed to console). |

Node colors in the HTML visualization are entity-type coded:
`PERSON` (blue), `ORG` (orange), `GPE`/`LOC` (green), `NORP` (red),
`PRODUCT` (purple), `MONEY`/`DATE` (grey/gold), etc. Node size scales
with how many relations that entity participates in (its degree) —
the biggest nodes are your graph's "hubs."

---

## Configuration reference

Every tunable parameter lives in **`config.yaml`** so you rarely need
to touch the code. Key sections:

```yaml
data:
  sample_size: 150          # null/0 = full corpus; int = fast demo subset
  categories: ["business", "tech"]   # [] = all 5 categories

nlp:
  keep_entity_labels: [...]   # which spaCy NER labels become graph nodes
  coreference:
    enabled: true
    lookback_sentences: 3     # how far back to search for a pronoun's antecedent

relation_extraction:
  drop_stopword_only_relations: true   # drop uninformative "be/have/do" relations

graph:
  backend: "networkx"        # or "neo4j"
  min_entity_frequency: 1    # prune entities mentioned fewer than N times
  max_nodes_for_viz: 400     # cap HTML viz to top-N nodes by degree (perf)

output:
  html_viz_path: "data/output/knowledge_graph.html"
  # ...other export paths
```

CLI flags (`--sample-size`, `--categories`, `--backend`) override the
corresponding config values for a single run without editing the file.

---

## Optional: loading into Neo4j

If you'd like to explore the graph with **Cypher** queries or **Neo4j
Bloom**:

1. Start a Neo4j instance (Desktop app, or Docker: `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5`).
2. Update `graph.neo4j` credentials in `config.yaml` if they differ
   from the defaults (`bolt://localhost:7687`, `neo4j`/`password`).
3. Run:
   ```bash
   python main.py --backend neo4j
   ```
4. Open Neo4j Browser (`http://localhost:7474`) and try:
   ```cypher
   MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100
   ```

`push_to_neo4j()` in `src/graph_builder.py` uses `MERGE` so re-running
is idempotent (or clears the graph first if `clear_before_load: true`).

---

## Using your own text data

The pipeline only requires a CSV with a `text` column (a `category`
column is optional metadata). To point it at your own corpus:

1. Replace `data.source_url` in `config.yaml` with a URL to your CSV,
   **or** just drop your file at `data/raw/bbc-text.csv` (the path in
   `data.raw_path`) and run with `--skip-download`.
2. Everything downstream (preprocessing, NER, relation extraction,
   graph building, visualization) works unchanged.

This also works well with:
- Wikipedia article dumps (via the `wikipedia` package — dump to CSV first).
- PDF/Word reports converted to plain text.
- Scraped news/blog articles.
- Meeting transcripts or support tickets.

---

## Running tests

```bash
pip install pytest   # pip users; uv sync installs it automatically
pytest tests/ -v
# or, with uv:
uv run pytest -v
```

The suite covers: config loading, text cleaning, sentence segmentation,
entity-text normalization, SVO triple extraction on hand-written
sentences, edge deduplication, self-loop prevention, and empty-graph
handling.

---

## Project structure

```
knowledge-graph-from-text/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .python-version
├── requirements.txt
├── config.yaml
├── main.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ner_extraction.py
│   ├── relation_extraction.py
│   ├── graph_builder.py
│   └── visualize.py
├── notebooks/
│   └── knowledge_graph_walkthrough.ipynb
├── tests/
│   └── test_pipeline.py
└── data/
    ├── raw/         (gitignored, populated on first run)
    ├── processed/   (gitignored)
    └── output/      (gitignored, populated on each run)
```

---

## Known limitations & tutorial talking points

This project intentionally uses **transparent, inspectable, rule-based**
methods rather than black-box neural relation extractors — great for
teaching, with some honest trade-offs worth calling out on camera:

- **Coreference resolution is heuristic**, not a full neural coreference
  model (e.g. spaCy's `coreferee` or AllenNLP). It resolves simple
  pronouns to the nearest matching preceding entity within a sentence
  window — good enough to visibly improve graph connectivity, but it
  will make mistakes on complex, multi-entity paragraphs. A great "next
  step" upgrade to discuss with viewers.
- **Relation extraction is a dependency-parse heuristic (OpenIE-lite)**,
  not a fine-tuned relation-classification model. It produces some noisy
  or overly generic relations (e.g. "be in", "make") alongside good ones
  — this is normal for rule-based extraction and a good opportunity to
  discuss upgrading to a transformer-based relation extraction model
  (e.g. REBEL) or an LLM-based extraction prompt as a v2.
- **Entity resolution across surface forms** (e.g. "Apple" vs. "Apple
  Inc." vs. "the Cupertino company") is handled with simple canonical-
  form + word-boundary matching, not full entity linking to a knowledge
  base like Wikidata. This is where tools like `spaCy-entity-linker` or
  a proper Entity Linking model would go in a production system.
- The BBC dataset is **English-only** and from 2004–2005 — swap in a
  multilingual spaCy model (e.g. `xx_ent_wiki_sm`) for other languages.

These are all called out in code comments at the relevant spots in
`src/ner_extraction.py` and `src/relation_extraction.py` for easy
reference while recording.

---

## Troubleshooting

**`OSError: [E050] Can't find model 'en_core_web_sm'`**
Run `python -m spacy download en_core_web_sm` (see Setup, step 4).

**Dataset download fails / `requests.exceptions.ConnectionError`**
Check your internet connection, or manually download the CSV from
<https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv>
and save it to `data/raw/bbc-text.csv`, then run with `--skip-download`.

**Pipeline runs but the graph looks empty / very small**
Check `data.categories` and `data.sample_size` in `config.yaml` — an
overly narrow filter combined with a small sample can leave few
qualifying entities. Try `--sample-size 150 --categories` (empty =
all categories).

**`ModuleNotFoundError: No module named 'src'`**
Run commands from the project root (where `main.py` lives), not from
inside `src/`.

**Neo4j connection refused**
Confirm Neo4j is running and reachable at the URI in
`config.yaml -> graph.neo4j.uri`, and that the `neo4j` Python driver is
installed (`pip install neo4j`).

**HTML visualization is slow/cluttered on a full-corpus run**
Lower `graph.max_nodes_for_viz` in `config.yaml` (default 400) to
prune to fewer, higher-degree nodes.

---

## License

MIT — see [LICENSE](LICENSE). The BBC News dataset is provided by
Greene & Cunningham (2006) for research/educational use; see the
[original dataset page](http://mlg.ucd.ie/datasets/bbc.html) for
citation details if you use it in published work.
