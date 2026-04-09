# RAG CLI Features and Pipeline Reference

This document describes all major features currently implemented in this project, including both user-facing and internal/stability features.

## 1. High-level Architecture

The CLI runtime is split across:

- rag.py: Main app runtime and orchestration.
- runtime_features.py: Reusable runtime helpers for cache and session export.

Core flow:

1. Read config from environment and .env.
2. Load embedding model.
3. Initialize Chroma collections for documents and memory.
4. Enter chat loop.
5. Parse command or auto-detect tools.
6. Retrieve memory/docs and optional web/tool context.
7. Build prompt and generate with Ollama.
8. Apply safety/quality post-checks.
9. Save memory and session data.

## 2. User-facing Features

### 2.1 Chat and Document RAG

- Ingest local files (pdf, txt, md) into Chroma.
- Retrieve top relevant chunks for each query.
- Retrieve relevant prior memory turns.

### 2.2 Tool Integrations

- Web providers: Tavily, SerpAPI, LangSearch, Jina, Firecrawl.
- URL fetch via Jina Reader.
- Weather lookup via OpenWeatherMap.
- CVE lookup via NVD API.
- DNS recon via HackerTarget.

### 2.3 Provider Selection and Strategy

- /web lets you force web lookup.
- Provider choice is non-interactive by default.
- The runtime uses a combined pipeline: Tavily + LangSearch + Jina + Firecrawl, and also SerpAPI when key is configured.
- /strategy <query> previews selected providers and rationale.
- /providers shows readiness/default state for each provider.

Strategy types:

- Default strategy: fixed combined provider pipeline with key-aware availability checks.

### 2.4 Citation-aware Web Answers

When web data is used, the system prepares evidence tags and asks the model to cite them, for example:

- [web:tavily#1]
- [web:firecrawl#1]

If a web-grounded answer is returned without citations, a second pass is triggered to add the missing tags.

### 2.5 Capability Question Handling

Meta capability questions (for example, asking if the assistant can use internet tools) are answered directly without launching web search.

### 2.6 Session Export

- /export md exports the current session to markdown.
- /export json exports the current session to JSON.

Export files are written under:

- ~/.rag-cli/exports

### 2.7 Monitoring

- /monitor status, on, off, live on, live off, reset.
- Tracks process CPU and memory and system-level CPU/RAM.

### 2.8 Tool Usage Q&A Inside Chat

- The system prompt includes an explicit command usage guide.
- You can ask natural language questions about commands, for example:
	- "What does /web do?"
	- "How do I use /export?"
	- "What does /query do?"
- For "/query", the assistant explains that this CLI does not define a /query command and suggests /web <query> for web lookups.
- This behavior is available directly in chat without running any command first.

Implementation note (latest):

- Command-help questions now use a dedicated intent path (`is_command_help_question`) and direct responder (`command_help_response`) before auto-tool detection.
- This prevents accidental web-tool invocation for prompts like "what does /strategy tool do".
- `/query` is treated as a user alias mistake and mapped to guidance for `/web <query>`.

## 3. Internal and Stability Features

These are not primary user-facing controls but improve reliability:

### 3.1 Web Response Cache

- Implemented in runtime_features.py via WebSearchCache.
- Reduces repeated provider calls for recent identical query+provider combinations.
- Controlled by WEB_CACHE_TTL_SEC.
- Backed by JSON file under ~/.rag-cli.

### 3.2 Prompt Safety and Data Trust Model

- Retrieved docs are marked as untrusted.
- Prompt-injection-like chunks can be filtered.
- Real-time tool outputs are marked as authoritative for freshness.

### 3.3 Refusal Recovery

If the model incorrectly denies internet/web access while tool outputs are present, the app automatically regenerates with stricter grounding instructions.

### 3.4 Memory Condensation

Saved conversational memory is condensed to keep embeddings compact and relevant.

### 3.5 Resource Cleanup Hooks

Periodic garbage collection and optional GPU cache cleanup reduce long-session memory growth.

### 3.6 System-level Tool Usage Guide Injection

- A dedicated TOOL_USAGE_GUIDE string is injected into the base system prompt.
- This keeps command usage explanations stable even when retrieved context is noisy.

## 4. Commands Summary

- chat: start interactive chat.
- ingest <paths>: index files and folders.
- docs list|prune|clear: inspect and manage document index.
- memory list|clear: inspect and manage memory.
- sources clean <files>: clean source markdown files.

In-chat commands:

- /web <query>
- /fetch <url>
- /weather <city>
- /cve <CVE-ID>
- /dns <domain>
- /strategy <query>
- /providers
- /export <md|json>
- /help
- /monitor <cmd>
- /clear
- exit or quit

## 5. Relevant Environment Variables

Tool/provider behavior:

- WEB_SEARCH_PROVIDER
- WEB_SEARCH_PICK_MODE
- WEB_SEARCH_DEFAULT_PROVIDERS
- WEB_SEARCH_MAX_PROVIDERS
- WEB_CACHE_TTL_SEC
- FIRECRAWL_BASE_URL
- FIRECRAWL_MAX_URLS

Provider keys:

- TAVILY_API_KEY
- SERPAPI_API_KEY
- LANGSEARCH_API_KEY
- JINA_API_KEY
- FIRECRAWL_API_KEY

Other integrations:

- OPENWEATHER_API_KEY
- NVD_API_KEY

Generation/retrieval:

- OLLAMA_CHAT_NUM_PREDICT
- OLLAMA_CHAT_TEMPERATURE
- USE_OLLAMA_EMBED
- AUTO_WEB_FALLBACK_ON_EMPTY_DOCS
- AUTO_WEB_MIN_QUERY_WORDS

Monitoring:

- RESOURCE_MONITOR_ENABLED
- RESOURCE_MONITOR_LIVE
- RESOURCE_MONITOR_INTERVAL_SEC

## 6. File Responsibilities

rag.py includes:

- config/env loading
- embeddings + retrieval
- tool invocation
- prompt building
- chat loop and command router
- post-generation guardrails

runtime_features.py includes:

- WebSearchCache: JSON-backed query/provider cache
- SessionRecorder: turn collection and md/json export

## 7. Recommended Operating Pattern

1. Keep default combined pipeline enabled for broad coverage.
2. Use /strategy before /web for important tasks.
3. Use /export json for machine-readable audit logs.
4. Use /providers to quickly diagnose missing API key issues.
5. Run docs prune occasionally if your sources are untrusted.

## 8. Change Diary

### 2026-04-09: doc/cli: document web pipeline and stream chat with quieter logs

Commit message:

Document web pipeline and stream chat with quieter logs

- Add a dedicated README describing the modular web retrieval pipeline (config, crawler, extractor, embedder, retriever, pipeline, modal_pipeline).
- Introduce a shared config.py with environment-driven settings for the modular pipeline, including a larger EMBEDDING_BATCH_SIZE=128 to speed up batch embedding.
- Switch rag.py chat loop to call generate_chat_response with streaming enabled so answers appear progressively instead of in a single block.
- Simplify post-generation passes by removing the extra rewrite calls and keeping only a lightweight Sources footer driven by web evidence tags.
- Tighten log configuration in rag.py so huggingface_hub HTTP warnings (including transient 503 HEAD requests) are suppressed in interactive CLI runs while other libraries remain at WARNING level.

### 2026-04-07: feat(web): add standalone web_search.py pipeline for Tavily+LangSearch+Jina discovery and Firecrawl enrichment

Commit message:

feat(web): add standalone web_search.py pipeline for Tavily+LangSearch+Jina discovery and Firecrawl enrichment

- create new `web_search.py` as an independent experimentation pipeline for web retrieval and LLM-ready context packaging
- implement staged retrieval design in one file:
	1) discovery stage using Tavily, LangSearch, and Jina Search
	2) URL dedupe/ranking stage with provider weighting and recency keyword bonus
	3) enrichment stage using Firecrawl scrape per top URL (with Jina Reader fallback)
	4) LLM-ready packaging stage with compact evidence blocks and source citations
- include robust URL extraction/normalization utilities and provider-level graceful degradation when keys are missing
- add CLI interface: `--query`, `--max-urls`, `--json-out`
- output both human-readable summary and a machine-consumable JSON payload containing `llm_ready_context`
- run end-to-end validation with real query (`latest AI model release updates`) and confirm discovered/enriched output is produced

Integration strategy encoded in the script:

- Use search providers for breadth (fresh link discovery) and crawler/reader providers for depth (clean page text).
- Keep providers decoupled at function level and integrate through a shared URL pipeline.
- Feed the LLM only ranked, trimmed evidence blocks with explicit source IDs to reduce context noise and hallucination risk.

### 2026-04-07: fix(help): restore in-chat tool help for patterns like /cve /help and /help /cve

Commit message:

fix(help): restore in-chat tool help for patterns like /cve /help and /help /cve

- add dedicated help-argument detector for command arguments (`help`, `/help`, `-h`, `--help`, `?`)
- add normalized command-usage formatter that maps command tokens to COMMAND_USAGE entries
- handle `/<tool> /help` (and equivalent) before tool execution so help is shown instead of calling the tool API
- enhance `/help` to support command-specific usage when an argument is provided (for example `/help /cve`)
- preserve existing global help panel behavior when `/help` is called without arguments
- validate rag.py after changes (no syntax/parser errors; only optional BM25 unresolved import warning remains)
- verify in live chat run: `/cve /help`, `/help /cve`, `/web /help`, and `/help` all return expected guidance

### 2026-04-07: fix(runtime): resolve rag.py startup crash and improve /web grounding quality

Commit message:

fix(runtime): resolve rag.py startup crash and improve /web grounding quality

- reproduce startup failure by running `uv run python rag.py chat` and capture traceback
- fix crash in rag.py caused by invalid Chroma type hints (`chromadb.PersistentClient | None`) at import time
- switch Chroma typing to `ClientAPI` (`from chromadb.api import ClientAPI`) for runtime-safe annotations
- verify app launches successfully after fix and chat loop is reachable
- validate slash web tooling with live run (`/providers`, `/strategy`, `/web`) to check retrieval behavior
- improve explicit `/web` behavior by skipping local doc/memory retrieval to avoid unrelated context contamination
- confirm explicit `/web` now uses tool results only (no local-memory bleed into answer)
- run static diagnostics on rag.py after patch (no syntax/type parse errors apart from optional BM25 import warning)
- note current web quality constraints: missing SERPAPI/LANGSEARCH/FIRECRAWL keys and missing `rank_bm25` package reduce breadth and hybrid quality

### 2026-04-07: chore(git): correct .gitignore configuration for Python and local runtime artifacts

Commit message:

chore(git): correct .gitignore configuration for Python and local runtime artifacts

- expand .gitignore from minimal entries to a complete project-safe baseline
- keep local secret handling by preserving .env in ignored files
- ignore Python cache/bytecode artifacts (__pycache__/, *.py[cod], *$py.class)
- ignore local virtual environments (.venv/, venv/, env/, ENV/)
- ignore build/package artifacts (build/, dist/, *.egg-info/, .eggs/)
- ignore common tool caches (.pytest_cache/, .mypy_cache/, .ruff_cache/, .coverage*, htmlcov/)
- ignore local logs and editor/OS metadata (*.log, .vscode/, .idea/, .DS_Store, Thumbs.db)
- preserve existing project-specific memory file ignores (memory_1.md, memory_2.md)
- validate updated .gitignore for syntax/errors (no issues)

### 2026-04-07: fix(chat): harden slash-command routing for /help, /strategy, and related in-chat commands

Commit message:

fix(chat): harden slash-command routing for /help, /strategy, and related in-chat commands

- replace prefix-only command parsing with regex-based slash parser that handles mixed whitespace/case consistently
- keep the known command set explicit (/web, /fetch, /weather, /cve, /dns, /monitor, /strategy, /providers, /export, /help)
- add explicit unknown slash-command handling so unsupported inputs do not fall through into normal Q&A
- preserve existing command behavior and tool orchestration for supported commands
- validate rag.py with static error checks (no errors)
- verify in a real chat run that /help and /strategy execute, and unknown slash commands return guidance to use /help

### 2026-04-07: fix(cli): restore visible RAG documentation/help output in CLI mode

Commit message:

fix(cli): restore visible RAG documentation/help output in CLI mode

- print startup RAG CLI summary panel on chat launch (model, docs/memory counts, command list)
- print missing-key warnings directly to CLI instead of storing them only in hidden history
- print direct assistant responses for meta capability questions and command-help questions
- restore visible outputs for /monitor, /strategy, /providers, and /export commands in CLI
- route /help to show_help_panel() so the full RAG documentation/help section appears again
- keep history recording behavior intact for session export and memory persistence
- validate rag.py after edits (no static errors)

### 2026-04-07: feat(cli): remove TUI runtime and enforce CLI-only chat flow

Commit message:

feat(cli): remove TUI runtime and enforce CLI-only chat flow

- remove TUI-only rendering path from rag.py by deleting render_tui_frame
- remove USE_TUI feature flag and all conditional TUI branches from chat loop
- remove unused Rich TUI imports (Markdown, Layout, Align)
- switch startup session label from RAG TUI started to RAG CLI started
- keep retrieval, tool orchestration, and grounding logic unchanged
- keep assistant output consistent through direct CLI printing
- validate rag.py with static error checks (no errors)
- verify no remaining TUI symbols (USE_TUI, render_tui_frame, Layout, Align, RAG TUI)

### 2026-04-07: Rolled Back Heavy Search Stack + Default Combined Web Pipeline

Problem:

- The prior setup was too heavy for local web-search runtime and required manual provider selection.

What changed:

- Reverted the project back to the previous lightweight commit baseline.
- Removed SearXNG/Redis Docker image dependencies from the active flow.
- Updated web search routing to a non-interactive default pipeline:
	- Tavily discovery
	- LangSearch structured summaries
	- SerpAPI discovery (only when `SERPAPI_API_KEY` is present)
	- Jina search context
	- Firecrawl page extraction over discovered URLs
- Added Firecrawl crawl orchestration that consumes URLs discovered from Tavily/SerpAPI/LangSearch/Jina outputs.
- Updated `.env` defaults and provider descriptions to match this pipeline.

Outcome:

- Web search now runs with merged multi-source output by default without provider selection prompts.
- SerpAPI remains integrated but is automatically skipped until its API key is configured.

### 2026-04-06: Command-Usage Q&A Reliability Fix

Problem:

- Asking usage questions such as "what does /strategy tool do" still triggered web provider selection due to broad auto-web heuristics.

What changed:

- Added command-help intent classifier and response generator in `rag.py`.
- Added explicit command usage map for all supported commands.
- Added an early chat-loop branch to answer command-usage queries directly (without retrieval/tool calls).
- Updated `should_web_search` to avoid triggering web for command-usage questions.

Outcome:

- Usage/help questions now return immediate in-chat explanations.
- Unwanted tool prompts for command-help questions are avoided.

### 2026-04-06: Git Branch Setup + TUI Migration (experiment)

Problem:

- Needed a safer workflow for advanced changes and a richer terminal experience than plain sequential CLI output.

What changed:

- Initialized Git in this project and created branch workflow:
	- `main`
	- `experiment` (created from `main` baseline)
- Migrated chat interaction to a TUI-style loop in `rag.py`:
	- Added structured dashboard rendering (`render_tui_frame`) using Rich Layout.
	- Added conversation transcript panel with rolling history.
	- Added status/footer panel for prompts and generation state.
	- Switched generation to non-streaming per turn for cleaner panel updates.
	- Converted command/status feedback into transcript entries for consistent TUI output.

Outcome:

- The app now behaves like a terminal UI (dashboard + conversation history + status footer) rather than plain scrolling logs.
- Branch `experiment` is now the active place for iterative advanced UX changes.

### 2026-04-06: Auto-Web Fallback Regression Fix (TUI QA)

Problem:

- During TUI QA, generic prompts could still trigger provider selection because `AUTO_WEB_FALLBACK_ON_EMPTY_DOCS` was activating even when query intent was not web-related.

What changed:

- Updated fallback condition in `rag.py` to require `should_web_search(cleaned_query)` before auto web fallback can run.

Outcome:

- Generic/non-web prompts no longer get hijacked by web-provider selection when local docs are empty.
- Web fallback remains active for true web-intent queries.

### 2026-04-08: Gitignore Cleanup for Local Artifacts

Problem:

- The repository ignore rules were missing several common local/temp file patterns that can appear during development.

What changed:

- Updated `.gitignore` with additional non-source patterns:
	- temp/backup files (`*.tmp`, `*.temp`, `*.bak`, `*.orig`)
	- python runtime metadata (`.python-version`)
	- notebook checkpoints (`.ipynb_checkpoints/`)
	- editor swap files (`*.swp`, `*.swo`, `*~`)

Outcome:

- Fewer accidental local artifacts will show up in git status.
- Source files and project docs remain unaffected.

### 2026-04-08: Web Pipeline Memory Stabilization

Problem:

- Web/news queries caused sharp RAM growth when large search + crawl payloads were passed through the pipeline and combined with a large model context window.

What changed:

- Reduced default runtime memory pressure for web-heavy turns in `rag.py`:
	- lowered default `OLLAMA_CHAT_NUM_CTX` to 8192
	- added `OLLAMA_CHAT_NUM_CTX_WEB` with lower web-turn context (default 6144)
	- reduced default provider fan-out to `tavily,firecrawl`
	- reduced default `FIRECRAWL_MAX_URLS` to 2
- Added strict payload caps:
	- `FIRECRAWL_EXTRACT_MAX_CHARS` for per-URL extracted content
	- clipped provider outputs before aggregation
	- cache now stores compact web evidence instead of large raw merged blocks
	- web cache entries are bounded with `WEB_CACHE_MAX_VALUE_CHARS`
- Added in-memory retention limits to prevent growth over long sessions:
	- `SessionRecorder(max_turns=...)` now trims old turns
	- conversation history stores clipped text per turn

Outcome:

- Web search turns now allocate less Python-side memory and send a smaller prompt to Ollama.
- The runtime avoids unbounded growth in session/cache structures during long chats.
- News/current-events queries should no longer cause runaway RAM spikes as quickly as before.
