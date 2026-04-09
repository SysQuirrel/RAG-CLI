"""
RAG CLI — Ollama + ChromaDB + persistent memory + API integrations
Usage:
  uv run python rag.py ingest <file_or_dir>   # index PDFs / text files
  uv run python rag.py chat                    # start chat session
  uv run python rag.py memory list             # show saved memory
  uv run python rag.py memory clear            # wipe memory
  uv run python rag.py docs list               # show indexed documents
  uv run python rag.py docs clear              # wipe document index

Chat commands:
  /web <query>         force Tavily web search
  /weather <city>      get weather for any location
  /cve <CVE-ID>        look up a CVE (e.g. /cve CVE-2024-1234)
  /dns <domain>        DNS recon via HackerTarget
  /shodan <ip>         Shodan host lookup
  /clear               clear the screen
  exit / quit          exit the chat
"""

import sys
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, cast

import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama
from pypdf import PdfReader
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# ── config — edit these ───────────────────────────────────────────────────────
DATA_DIR         = Path.home() / ".rag-cli"
CHROMA_DIR       = DATA_DIR / "chroma"
EMBED_MODEL      = "all-MiniLM-L6-v2"   # ~90 MB, downloaded once from HuggingFace
USE_OLLAMA_EMBED = False                 # True = use nomic-embed-text via Ollama instead
OLLAMA_MODEL     = "qwen2.5-coder:3b"   # change to your active model
TOP_K_DOCS       = 3                    # chunks retrieved per query
MEMORY_TURNS     = 4                    # past turns injected as context
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 80

# ── API keys — set via environment variables ──────────────────────────────────
# export TAVILY_API_KEY="tvly-..."
# export OPENWEATHER_API_KEY="..."
# export NVD_API_KEY="..."          (optional — raises NVD rate limit)
# export SHODAN_API_KEY="..."
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NVD_API_KEY         = os.getenv("NVD_API_KEY", "")
SHODAN_API_KEY      = os.getenv("SHODAN_API_KEY", "")

DATA_DIR.mkdir(parents=True, exist_ok=True)
console = Console()

# ── embedder ──────────────────────────────────────────────────────────────────
class OllamaEmbedder:
    def encode(self, texts: list[str], **kwargs):
        return [ollama.embed(model="nomic-embed-text", input=t)["embeddings"][0] for t in texts]

def load_embedder():
    if USE_OLLAMA_EMBED:
        console.print("[dim]Embedder: nomic-embed-text (Ollama)[/dim]")
        return OllamaEmbedder()
    with console.status("[dim]Loading embedding model...[/dim]", spinner="dots"):
        m = SentenceTransformer(EMBED_MODEL)
    return m

# ── chroma ────────────────────────────────────────────────────────────────────
def get_chroma():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

def get_collections(client):
    return (
        client.get_or_create_collection("documents"),
        client.get_or_create_collection("memory"),
    )

# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[dict]:
    chunks, start, idx = [], 0, 0
    text = text.replace("\x00", "")
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source, "idx": idx})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def chunk_id(source: str, idx: int) -> str:
    return f"{hashlib.md5(source.encode()).hexdigest()[:8]}-{idx}"

# ── file readers ──────────────────────────────────────────────────────────────
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        r = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in r.pages)
    return path.read_text(errors="replace")

# ── ingest ────────────────────────────────────────────────────────────────────
def ingest(paths: list[str]):
    embedder    = load_embedder()
    client      = get_chroma()
    docs_col, _ = get_collections(client)
    files: list[Path] = []
    for p in paths:
        fp = Path(p)
        if fp.is_dir():
            for ext in ("*.pdf", "*.txt", "*.md"):
                files.extend(fp.rglob(ext))
        elif fp.is_file():
            files.append(fp)
        else:
            console.print(f"[yellow]Skipping (not found): {p}[/yellow]")
    if not files:
        console.print("[red]No files found.[/red]")
        return
    total = 0
    for fp in files:
        console.print(f"[cyan]Ingesting:[/cyan] {fp.name}", end=" ")
        try:
            chunks = chunk_text(read_file(fp), str(fp))
            if not chunks:
                console.print("[yellow](empty)[/yellow]")
                continue
            texts  = [c["text"] for c in chunks]
            ids    = [chunk_id(c["source"], c["idx"]) for c in chunks]
            metas: list[dict[str, Any]] = [{"source": c["source"], "idx": c["idx"]} for c in chunks]
            with console.status("embedding...", spinner="dots"):
                embeds = embedder.encode(texts, show_progress_bar=False)
                if not isinstance(embeds, list):
                    embeds = embeds.tolist()
            docs_col.upsert(
                ids=ids,
                embeddings=embeds,
                documents=texts,
                metadatas=cast(Any, metas),
            )
            console.print(f"[green]{len(chunks)} chunks[/green]")
            total += len(chunks)
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    console.print(f"\n[bold green]Done. {total} chunks indexed.[/bold green]")

# ── retrieval ─────────────────────────────────────────────────────────────────
def _to_list(embeds) -> list:
    return embeds if isinstance(embeds, list) else embeds.tolist()

def retrieve_docs(query: str, embedder, docs_col) -> list[dict]:
    if docs_col.count() == 0:
        return []
    q_emb = _to_list(embedder.encode([query]))
    res   = docs_col.query(
        query_embeddings=q_emb,
        n_results=min(TOP_K_DOCS, docs_col.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "source": meta.get("source", "?"), "score": 1 - dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]

def retrieve_memory(query: str, embedder, memory_col) -> list[dict]:
    if memory_col.count() == 0:
        return []
    q_emb = _to_list(embedder.encode([query]))
    res   = memory_col.query(
        query_embeddings=q_emb,
        n_results=min(MEMORY_TURNS, memory_col.count()),
        include=["documents", "metadatas"],
    )
    turns = [
        {"text": doc, "ts": meta.get("ts", "")}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]
    return sorted(turns, key=lambda x: x["ts"])

def save_memory(user_msg: str, assistant_msg: str, embedder, memory_col):
    ts   = datetime.now().isoformat()
    text = f"User: {user_msg}\nAssistant: {assistant_msg}"
    mid  = f"mem-{hashlib.md5(ts.encode()).hexdigest()[:12]}"
    emb  = _to_list(embedder.encode([text]))
    memory_col.add(ids=[mid], embeddings=emb, documents=[text], metadatas=[{"ts": ts}])

# ══════════════════════════════════════════════════════════════════════════════
# API INTEGRATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Tavily web search ──────────────────────────────────────────────────────
WEB_TRIGGERS = [
    "latest", "recent", "2024", "2025", "2026", "news", "today",
    "current", "new exploit", "poc", "patch", "writeup",
]

def should_web_search(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in WEB_TRIGGERS)

def tool_web_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return "[Tavily] No API key set. Run: export TAVILY_API_KEY=tvly-..."
    try:
        from tavily import TavilyClient
        res      = TavilyClient(api_key=TAVILY_API_KEY).search(
            query, max_results=3, search_depth="basic"
        )
        snippets = [r.get("content", "") for r in res.get("results", [])]
        return "\n\n".join(snippets[:3]) or "No results."
    except Exception as e:
        return f"[Tavily error] {e}"

# ── 2. OpenWeatherMap ─────────────────────────────────────────────────────────
WEATHER_TRIGGERS = ["weather", "temperature", "forecast", "raining", "humid", "climate"]

def should_weather(query: str) -> tuple[bool, str]:
    q = query.lower()
    if not any(t in q for t in WEATHER_TRIGGERS):
        return False, ""
    m = re.search(r"(?:weather|forecast|temperature)\s+(?:in|at|for)?\s+([a-zA-Z ,]+)", q)
    loc = m.group(1).strip().rstrip("?.,") if m else ""
    return True, loc

def tool_weather(location: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "[Weather] No API key set. Run: export OPENWEATHER_API_KEY=..."
    if not location:
        return "[Weather] Could not detect location. Use /weather <city>."
    try:
        r    = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=8,
        )
        data = r.json()
        if r.status_code != 200:
            return f"[Weather] {data.get('message', 'Unknown error')} (city: {location})"
        desc    = data["weather"][0]["description"].capitalize()
        temp    = data["main"]["temp"]
        feels   = data["main"]["feels_like"]
        humid   = data["main"]["humidity"]
        wind    = data["wind"]["speed"]
        city    = data["name"]
        country = data["sys"]["country"]
        return (
            f"Weather in {city}, {country}:\n"
            f"  Condition   : {desc}\n"
            f"  Temperature : {temp}°C (feels like {feels}°C)\n"
            f"  Humidity    : {humid}%\n"
            f"  Wind        : {wind} m/s"
        )
    except Exception as e:
        return f"[Weather error] {e}"

# ── 3. NVD CVE lookup ─────────────────────────────────────────────────────────
CVE_PATTERN = re.compile(r"cve-\d{4}-\d+", re.IGNORECASE)

def should_cve(query: str) -> tuple[bool, str]:
    m = CVE_PATTERN.search(query)
    return (True, m.group(0).upper()) if m else (False, "")

def tool_cve(cve_id: str) -> str:
    try:
        headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
        r       = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            params={"cveId": cve_id},
            headers=headers,
            timeout=10,
        )
        data  = r.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"[CVE] {cve_id} not found in NVD."
        cve   = vulns[0]["cve"]
        desc  = next(
            (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
            "No description."
        )
        metrics = cve.get("metrics", {})
        score, sev = "N/A", "N/A"
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                cvss  = metrics[key][0].get("cvssData", {})
                score = cvss.get("baseScore", "N/A")
                sev   = cvss.get("baseSeverity", cvss.get("vectorString", "N/A"))
                break
        published = cve.get("published", "?")[:10]
        refs      = [ref["url"] for ref in cve.get("references", [])[:3]]
        return (
            f"{cve_id}  (published {published})\n"
            f"  CVSS Score  : {score} [{sev}]\n"
            f"  Description : {desc[:400]}{'...' if len(desc) > 400 else ''}\n"
            f"  References  :\n    " + "\n    ".join(refs)
        )
    except Exception as e:
        return f"[CVE error] {e}"

# ── 4. HackerTarget DNS recon ─────────────────────────────────────────────────
DNS_TRIGGERS  = ["dns", "subdomain", "recon", "nslookup", "mx record", "resolve", "nameserver"]
DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
)

def should_dns(query: str) -> tuple[bool, str]:
    q = query.lower()
    if not any(t in q for t in DNS_TRIGGERS):
        return False, ""
    m = DOMAIN_PATTERN.search(query)
    return (True, m.group(0)) if m else (False, "")

def tool_dns(domain: str) -> str:
    try:
        endpoints = {
            "DNS Lookup" : f"https://api.hackertarget.com/dnslookup/?q={domain}",
            "Subdomains" : f"https://api.hackertarget.com/hostsearch/?q={domain}",
            "Reverse DNS": f"https://api.hackertarget.com/reversedns/?q={domain}",
        }
        out = []
        for label, url in endpoints.items():
            r    = requests.get(url, timeout=8)
            text = r.text.strip()
            if "error" in text.lower() and len(text) < 100:
                out.append(f"{label}:\n  [rate limited: {text}]")
            else:
                lines = text.splitlines()[:12]
                out.append(f"{label}:\n  " + "\n  ".join(lines))
        return "\n\n".join(out)
    except Exception as e:
        return f"[DNS error] {e}"

# ── 5. Shodan host lookup ─────────────────────────────────────────────────────
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

def tool_shodan(ip: str) -> str:
    if not SHODAN_API_KEY:
        return "[Shodan] No API key set. Run: export SHODAN_API_KEY=..."
    try:
        r = requests.get(
            f"https://api.shodan.io/shodan/host/{ip}",
            params={"key": SHODAN_API_KEY},
            timeout=10,
        )
        if r.status_code == 404:
            return f"[Shodan] No data for {ip}."
        if r.status_code != 200:
            return f"[Shodan] HTTP {r.status_code}: {r.text[:200]}"
        data      = r.json()
        ports     = data.get("ports", [])
        vulns     = list(data.get("vulns", {}).keys())
        hostnames = data.get("hostnames", [])
        lines     = [
            f"Shodan report for {ip}",
            f"  Organisation : {data.get('org', 'Unknown')} / {data.get('isp', 'Unknown')}",
            f"  Location     : {data.get('city', '?')}, {data.get('country_name', '?')}",
            f"  OS           : {data.get('os', 'Unknown')}",
            f"  Open ports   : {', '.join(str(p) for p in ports[:20]) or 'None'}",
            f"  Hostnames    : {', '.join(hostnames[:5]) or 'None'}",
            f"  Tags         : {', '.join(data.get('tags', [])) or 'None'}",
        ]
        if vulns:
            lines.append(f"  CVEs flagged : {', '.join(vulns[:10])}")
        return "\n".join(lines)
    except Exception as e:
        return f"[Shodan error] {e}"

# ══════════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def parse_command(query: str) -> tuple:
    """Return (cmd, arg, cleaned_query) for explicit /commands."""
    q = query.strip()
    for cmd in ("/web", "/weather", "/cve", "/dns", "/shodan"):
        if q.lower().startswith(cmd + " ") or q.lower() == cmd:
            arg = q[len(cmd):].strip()
            return cmd[1:], arg, arg
    return None, "", q

def auto_detect_tools(query: str) -> dict:
    """Return {tool: arg} for tools that should fire automatically."""
    triggered = {}
    if TAVILY_API_KEY and should_web_search(query):
        triggered["web"] = query
    hit, loc = should_weather(query)
    if hit and OPENWEATHER_API_KEY and loc:
        triggered["weather"] = loc
    hit, cve_id = should_cve(query)
    if hit:
        triggered["cve"] = cve_id
    hit, domain = should_dns(query)
    if hit and domain:
        triggered["dns"] = domain
    # Shodan never auto-triggers — always requires /shodan <ip>
    return triggered

# ── prompt builder ────────────────────────────────────────────────────────────
def build_prompt(query: str, doc_chunks: list, mem_turns: list, tool_results: dict) -> str:
    parts = []
    if mem_turns:
        parts.append("=== Relevant past conversation ===")
        parts.extend(t["text"] for t in mem_turns)
    if doc_chunks:
        parts.append("=== Retrieved document context ===")
        for i, c in enumerate(doc_chunks, 1):
            parts.append(f"[{i}] ({Path(c['source']).name})\n{c['text']}")
    for label, content in tool_results.items():
        parts.append(f"=== {label} ===")
        parts.append(content)
    system = (
        "You are a helpful assistant specialising in cybersecurity, "
        "bug bounty hunting, and CTFs. Use the context below to answer "
        "accurately. If the context is not relevant, use your own knowledge."
    )
    if parts:
        return f"{system}\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {query}\nAnswer:"
    return f"{system}\n\nQuestion: {query}\nAnswer:"

# ── chat loop ─────────────────────────────────────────────────────────────────
def chat():
    embedder             = load_embedder()
    client               = get_chroma()
    docs_col, memory_col = get_collections(client)

    keys = (
        f"Tavily:[{'green' if TAVILY_API_KEY else 'red'}]{'✓' if TAVILY_API_KEY else '✗'}[/]  "
        f"Weather:[{'green' if OPENWEATHER_API_KEY else 'red'}]{'✓' if OPENWEATHER_API_KEY else '✗'}[/]  "
        f"NVD:[{'green' if NVD_API_KEY else 'yellow'}]{'✓' if NVD_API_KEY else 'no key (slower)'}[/]  "
        f"Shodan:[{'green' if SHODAN_API_KEY else 'red'}]{'✓' if SHODAN_API_KEY else '✗'}[/]"
    )
    console.print(Panel(
        f"[bold cyan]RAG CLI[/bold cyan]\n"
        f"Model: [yellow]{OLLAMA_MODEL}[/yellow]  |  "
        f"Docs: [green]{docs_col.count()}[/green]  |  "
        f"Memory: [green]{memory_col.count()}[/green]\n"
        f"{keys}\n"
        f"[dim]/web · /weather <city> · /cve <id> · /dns <domain> · /shodan <ip> · exit[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            raw = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            console.print("[dim]Bye.[/dim]")
            break
        if raw.lower() == "/clear":
            console.clear()
            continue

        cmd, arg, query = parse_command(raw)
        tool_results: dict[str, str] = {}
        doc_chunks: list = []
        mem_turns:  list = []

        DIRECT_DISPLAY = {"weather", "cve", "dns", "shodan"}

        if cmd:
            # explicit command
            with console.status(f"[dim]Running /{cmd}...[/dim]", spinner="dots"):
                if cmd == "web":
                    tool_results["Web search"] = tool_web_search(arg)
                elif cmd == "weather":
                    tool_results["Weather"] = tool_weather(arg)
                elif cmd == "cve":
                    tool_results["CVE lookup"] = tool_cve(arg)
                elif cmd == "dns":
                    tool_results["DNS recon"] = tool_dns(arg)
                elif cmd == "shodan":
                    tool_results["Shodan"] = tool_shodan(arg)
            with console.status("[dim]Retrieving docs...[/dim]", spinner="dots"):
                doc_chunks = retrieve_docs(query or arg, embedder, docs_col)
                mem_turns  = retrieve_memory(query or arg, embedder, memory_col)
        else:
            # natural language — retrieve + auto-detect
            with console.status("[dim]Retrieving context...[/dim]", spinner="dots"):
                doc_chunks = retrieve_docs(query, embedder, docs_col)
                mem_turns  = retrieve_memory(query, embedder, memory_col)
                auto       = auto_detect_tools(query)
            for tool, targ in auto.items():
                with console.status(f"[dim]Fetching {tool}...[/dim]", spinner="dots"):
                    if tool == "web":
                        tool_results["Web search"] = tool_web_search(targ)
                    elif tool == "weather":
                        tool_results["Weather"] = tool_weather(targ)
                    elif tool == "cve":
                        tool_results["CVE lookup"] = tool_cve(targ)
                    elif tool == "dns":
                        tool_results["DNS recon"] = tool_dns(targ)

        # show source indicators
        if doc_chunks:
            srcs = list(dict.fromkeys(Path(c["source"]).name for c in doc_chunks))
            console.print(f"[dim]  Docs: {', '.join(srcs)}[/dim]")
        for label in tool_results:
            console.print(f"[dim]  + {label}[/dim]")

        # for direct-display tools, print raw output first then ask model for insight
        llm_tool_results = {}
        if cmd and cmd in DIRECT_DISPLAY:
            raw_output = list(tool_results.values())[0]
            label      = list(tool_results.keys())[0]
            console.print(f"\n[bold cyan]{label}[/bold cyan]")
            console.print(raw_output)
            insight_query = f"Given this data, give a concise security or operational insight:\n{raw_output}"
            prompt        = build_prompt(insight_query, doc_chunks, mem_turns, {})
        else:
            llm_tool_results = tool_results
            prompt           = build_prompt(query, doc_chunks, mem_turns, llm_tool_results)

        # generate
        console.print("\n[bold green]Assistant[/bold green]")
        full_response = ""
        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.3, "num_ctx": 1024},
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_response += token
                print(token, end="", flush=True)
            print()
        except Exception as e:
            console.print(f"[red]Ollama error: {e}[/red]")
            console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
            continue

        save_memory(raw, full_response, embedder, memory_col)

# ── management ────────────────────────────────────────────────────────────────
def memory_list():
    client        = get_chroma()
    _, memory_col = get_collections(client)
    count         = memory_col.count()
    if count == 0:
        console.print("[dim]No memory stored yet.[/dim]")
        return
    results = memory_col.get(include=["documents", "metadatas"], limit=20)
    table   = Table(title=f"Memory ({count} turns)", show_lines=True)
    table.add_column("Time", style="dim", width=20)
    table.add_column("Content")
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    paired = sorted(
        zip(documents, metadatas),
        key=lambda x: str((x[1] or {}).get("ts", "")),
    )
    for doc, meta in paired[-20:]:
        meta_map = meta or {}
        doc_text = str(doc)
        table.add_row(
            str(meta_map.get("ts", "?"))[:19],
            doc_text[:180] + ("…" if len(doc_text) > 180 else ""),
        )
    console.print(table)

def memory_clear():
    get_chroma().delete_collection("memory")
    console.print("[green]Memory cleared.[/green]")

def docs_list():
    client      = get_chroma()
    docs_col, _ = get_collections(client)
    count       = docs_col.count()
    if count == 0:
        console.print("[dim]No documents indexed yet.[/dim]")
        return
    results = docs_col.get(include=["metadatas"], limit=1000)
    sources: dict[str, int] = {}
    for meta in (results.get("metadatas") or []):
        src = Path(str((meta or {}).get("source", "?"))).name
        sources[src] = sources.get(src, 0) + 1
    table = Table(title=f"Indexed documents ({count} chunks)")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for src, cnt in sorted(sources.items()):
        table.add_row(src, str(cnt))
    console.print(table)

def docs_clear():
    get_chroma().delete_collection("documents")
    console.print("[green]Document index cleared.[/green]")

# ── entrypoint ────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    if not args or args[0] == "chat":
        chat()
    elif args[0] == "ingest":
        ingest(args[1:]) if len(args) > 1 else console.print("[red]Usage: ingest <file_or_dir>[/red]")
    elif args[0] == "memory":
        {"list": memory_list, "clear": memory_clear}.get(
            args[1] if len(args) > 1 else "list",
            lambda: console.print("[red]Usage: memory list | clear[/red]"),
        )()
    elif args[0] == "docs":
        {"list": docs_list, "clear": docs_clear}.get(
            args[1] if len(args) > 1 else "list",
            lambda: console.print("[red]Usage: docs list | clear[/red]"),
        )()
    else:
        console.print(__doc__)

if __name__ == "__main__":
    main()