# RAG Pipeline — Web Retrieval System

A modular, production-ready pipeline that searches the web, crawls pages,
extracts content, and stores it in a local vector database for RAG retrieval.

```
search → crawl → extract → chunk → embed → store → retrieve
```

---

## Project Structure

```
rag_pipeline/
├── config.py            # Central config (loaded from .env)
├── crawler.py           # Stage 1a+1b: Web search (Tavily/LangSearch) and crawling
├── extractor.py         # Stage 2–4: Content extraction, structuring, and chunking
├── embedder.py          # Stage 5+6: Embedding & ChromaDB storage
├── retriever.py         # Query interface
├── pipeline.py          # Orchestrator + CLI
├── modal_pipeline.py    # Optional: Modal.com for remote scaling
└── .env.example
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on sentence-transformers**: The model (`all-MiniLM-L6-v2`, ~80 MB)
> downloads automatically on first run. It runs fully on CPU.

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
#   TAVILY_API_KEY=...
#   FIRECRAWL_API_KEY=...      (optional — for JS-heavy sites)
#   LANGSEARCH_API_KEY=...     (optional — fallback search)
```

---

## Usage

### Ingest content for a topic

```bash
python pipeline.py ingest "Python async programming best practices"
```

Options:
```
--max-results N    Number of search results to use as seeds (default: 10)
--max-pages N      Total pages to crawl (default: 30)
--max-depth N      Link-follow depth (default: 2)
--firecrawl        Use Firecrawl for JS-heavy sites
--no-save          Don't write documents.jsonl
```

Example — crawl documentation site deeply:
```bash
python pipeline.py ingest "FastAPI middleware" --max-depth 3 --max-pages 50
```

Example — JS-heavy site via Firecrawl:
```bash
python pipeline.py ingest "React Server Components" --firecrawl
```

---

### Query the knowledge base

```bash
python pipeline.py query "how do I handle rate limits in async Python?"
```

Options:
```
--top-k N          Number of results (default: 6)
--min-score F      Minimum similarity score, 0–1 (default: 0.30)
--domain DOMAIN    Filter to specific domain (e.g. docs.python.org)
--json             Output raw JSON instead of formatted context
```

Example — domain-specific query:
```bash
python pipeline.py query "dependency injection" --domain fastapi.tiangolo.com
```

---

### Check collection stats

```bash
python pipeline.py stats
```

---

## Use as a Python module

```python
from pipeline import run_ingest, run_query

# Ingest
summary = run_ingest("machine learning deployment strategies", max_search_results=5)

# Query
results = run_query("how to serve ML models in production?", top_k=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['title']}")
    print(f"  {r['url']}")
    print(f"  {r['text'][:200]}…\n")
```

---

## Using with an LLM

```python
from pipeline import run_query
from retriever import retrieve, format_context

# Get formatted context block ready for LLM prompt injection
results = retrieve("what are the best chunking strategies for RAG?")
context = format_context(results)

# Use context in your LLM call
prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question: What are the best chunking strategies for RAG?
Answer:"""

# Pass `prompt` to your LLM of choice (OpenAI, Anthropic, etc.)
```

---

## Optional: Scale with Modal

For large ingestion jobs (hundreds of URLs, scheduled pipelines):

```bash
pip install modal
modal token new                  # authenticate once

# Deploy
modal deploy modal_pipeline.py

# Run remote ingest
modal run modal_pipeline.py::ingest_remote --query "your topic"
```

Add your API keys to Modal Secrets (dashboard → Secrets → Create secret group
named `rag-api-keys`) with the same variable names as in `.env`.

---

## Configuration Reference

All settings can be overridden in `.env`:

| Variable             | Default             | Description                          |
|----------------------|---------------------|--------------------------------------|
| `TAVILY_API_KEY`     | —                   | Required (or LANGSEARCH_API_KEY)     |
| `FIRECRAWL_API_KEY`  | —                   | Optional, for JS-heavy sites         |
| `LANGSEARCH_API_KEY` | —                   | Optional, fallback search            |
| `CHROMA_DB_PATH`     | `./chroma_db`       | Where ChromaDB persists data         |
| `EMBEDDING_MODEL`    | `all-MiniLM-L6-v2`  | Sentence-transformers model name     |
| `CHUNK_SIZE`         | `500`               | Characters per chunk                 |
| `CHUNK_OVERLAP`      | `75`                | Overlap between consecutive chunks   |
| `CRAWL_MAX_DEPTH`    | `2`                 | Link-follow depth from seed URLs     |
| `CRAWL_MAX_PAGES`    | `30`                | Hard cap on total pages crawled      |
| `RETRIEVAL_TOP_K`    | `6`                 | Chunks returned per query            |
| `RETRIEVAL_MIN_SCORE`| `0.30`              | Minimum cosine similarity threshold  |

---

## Troubleshooting

**Empty results after ingest**
- Check `pipeline.log` for extraction failures
- Lower `RETRIEVAL_MIN_SCORE` to `0.20` and retry the query
- Inspect `documents.jsonl` to verify content was extracted

**Short content / mostly navigation text**
- Add `--firecrawl` flag for sites that require JS rendering
- Check if the site requires authentication (pipeline won't handle login walls)

**Slow embedding**
- Normal on first run (model download). Subsequent runs are faster.
- Reduce `EMBEDDING_BATCH_SIZE` in config.py if you hit memory issues

**ChromaDB errors on re-ingest**
- The pipeline uses `upsert` so re-running is safe and updates existing chunks
- To start fresh: `rm -rf ./chroma_db`
