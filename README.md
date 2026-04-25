# RAG CLI

Local-first RAG assistant for terminal workflows, with a modular web-ingestion pipeline.

This project combines:
- An interactive chat CLI powered by Ollama and ChromaDB
- Hybrid retrieval (dense + BM25-style sparse)
- Optional live web grounding through multiple providers
- A standalone search/crawl/extract/embed pipeline for building local knowledge bases

## Why This Project

RAG systems often fail in one of two ways: they are either fast but shallow, or powerful but fragile.
This codebase focuses on practical balance:
- Strong local-first behavior for speed and privacy
- Optional web augmentation when recency is needed
- Retrieval quality upgrades that improve answer grounding
- Operational tooling for memory, docs cleanup, stress testing, and runtime monitoring

## Architecture

Two primary execution paths live in the same repository.

### 1) Chat Path (rag.py)

query -> retrieve local docs/memory -> optional web context -> compose prompt -> Ollama response

### 2) Ingestion Path (pipeline.py)

search -> crawl -> extract -> chunk -> embed -> store -> retrieve

## Core Features

- Terminal chat assistant with local vector memory
- File ingestion from local sources (PDF/text/markdown and directories)
- Web tools inside chat:
    - /web for forced web search
    - /fetch for full-page text retrieval
    - /weather and /cve helper commands
    - /dns and provider strategy tooling
- Docs maintenance commands (list, prune suspicious chunks, clear index)
- Memory controls (list/clear)
- Session export (markdown/json)
- Resource monitor controls during chat
- Stress profile runner for local file QA
- Optional Modal workflow via modal_pipeline.py

## Tech Stack

- Python 3.12+
- Ollama for generation/embedding serving
- ChromaDB for persistent vector storage
- Sentence Transformers for local embeddings
- Tavily, SerpAPI, LangSearch, Jina, Firecrawl integrations
- Rich terminal UI

## Project Layout

- rag.py: Main CLI assistant (chat, ingest, docs, memory, stress, tools)
- pipeline.py: Modular ingestion/retrieval pipeline CLI
- config.py: Pipeline config loaded from .env
- crawler.py: Search + crawl stage
- extractor.py: Content extraction and chunking pipeline stages
- embedder.py: Embedding + Chroma storage
- retriever.py: Retrieval formatting and ranking output
- query_router.py: Query routing and web/local decision heuristics
- runtime_features.py: Runtime caching/session helpers
- modal_pipeline.py: Optional remote execution path
- .env.example: Safe environment template (no secrets)

## Quick Start

### 1. Create environment and install dependencies

Using uv (recommended):

```bash
uv sync
```

Using venv + pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit .env with your local keys and runtime preferences.

### 3. Start the assistant

```bash
uv run python rag.py chat
```

Shortcut (chat is default):

```bash
uv run python rag.py
```

## CLI Usage

### Main assistant (rag.py)

```bash
uv run python rag.py chat
uv run python rag.py ingest <file_or_dir>
uv run python rag.py docs list
uv run python rag.py docs prune
uv run python rag.py docs clear
uv run python rag.py memory list
uv run python rag.py memory clear
uv run python rag.py sources clean <file1> <file2>
uv run python rag.py stress <file>
```

### Chat slash commands

- /web <query>: force web search
- /fetch <url>: fetch page text
- /weather <city>: weather lookup
- /cve <CVE-ID>: NVD lookup
- /dns <domain>: DNS recon helper
- /strategy <query>: inspect provider strategy
- /providers: provider readiness view
- /monitor <cmd>: runtime monitor control
- /export <md|json>: export session
- /clear: clear terminal
- /help: command reference

### Pipeline mode (pipeline.py)

```bash
python pipeline.py ingest "your topic"
python pipeline.py query "your question"
python pipeline.py stats
```

Useful ingest flags:

- --max-results: number of seed results
- --max-pages: crawl cap
- --max-depth: link depth
- --firecrawl: prefer JS-friendly crawl path
- --no-save: skip documents.jsonl output

## Python API Examples

```python
from pipeline import run_ingest, run_query

summary = run_ingest("machine learning deployment strategies", max_search_results=5)
results = run_query("how to serve ML models in production?", top_k=5)

for r in results:
        print(r["score"], r["title"], r["url"])
```

## Configuration Notes

The repository includes many env-tunable knobs. Key groups:

- Model + generation:
    - OLLAMA_MODEL
    - OLLAMA_CHAT_NUM_CTX
    - OLLAMA_CHAT_NUM_PREDICT
    - OLLAMA_SHOW_THINKING
- Retrieval + chunking:
    - EMBED_MODEL
    - CHUNK_SIZE / CHUNK_OVERLAP
    - TOP_K_DOCS / DOC_MIN_SCORE
- Web routing + providers:
    - AUTO_WEB_FALLBACK_ON_EMPTY_DOCS
    - WEB_SEARCH_PROVIDER
    - TAVILY_API_KEY / SERPAPI_API_KEY / LANGSEARCH_API_KEY / JINA_API_KEY / FIRECRAWL_API_KEY

Use separate CHROMA_DB_PATH values when changing embedding models so vector spaces do not mix.

## Safety and Publishing

- Never commit real .env secrets
- Keep local DB/index artifacts out of git
- Rotate keys immediately if a credential was ever committed
- Use .env.example as the only shared configuration template

## Troubleshooting

### No results or weak answers

- Re-ingest your source files
- Raise top-k or lower minimum score
- Confirm your embedding model and Chroma path are consistent

### Web features not working

- Check API keys in .env
- Verify provider selection variables
- Use /providers and /strategy in chat for diagnostics

### Slow first run

- Embedding models may download on first execution
- Keep Ollama model warm using existing keep-alive settings

## Optional Remote Scaling

If you need remote ingestion execution, use modal_pipeline.py with Modal.

## License

See LICENSE in this repository if present.
