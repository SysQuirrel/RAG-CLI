import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CacheEntry:
    value: str
    ts_epoch: float


class WebSearchCache:
    """Small JSON-backed cache to reduce repeated external tool calls."""

    def __init__(
        self,
        cache_path: Path,
        ttl_sec: int = 900,
        max_entries: int = 300,
        max_value_chars: int = 12000,
    ):
        self.cache_path = cache_path
        self.ttl_sec = max(60, int(ttl_sec))
        self.max_entries = max(50, int(max_entries))
        self.max_value_chars = max(1000, int(max_value_chars))
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            self._store = {}
            return
        try:
            raw = self.cache_path.read_text(encoding="utf-8", errors="replace")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                self._store = parsed
            else:
                self._store = {}
        except Exception:
            self._store = {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._store, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _normalize_key(self, query: str, providers: list[str]) -> str:
        q = " ".join(query.lower().strip().split())
        p = ",".join(p.lower().strip() for p in providers)
        digest = hashlib.md5(f"{p}::{q}".encode("utf-8", errors="replace")).hexdigest()
        return digest

    def get(self, query: str, providers: list[str], now_epoch: float) -> str | None:
        key = self._normalize_key(query, providers)
        item = self._store.get(key)
        if not isinstance(item, dict):
            return None
        ts = float(item.get("ts_epoch", 0.0))
        if ts <= 0 or (now_epoch - ts) > self.ttl_sec:
            return None
        value = item.get("value", "")
        return value if isinstance(value, str) else None

    def set(self, query: str, providers: list[str], value: str, now_epoch: float) -> None:
        key = self._normalize_key(query, providers)
        if isinstance(value, str) and len(value) > self.max_value_chars:
            value = value[: self.max_value_chars]
        self._store[key] = {
            "query": query,
            "providers": providers,
            "value": value,
            "ts_epoch": float(now_epoch),
        }
        if len(self._store) > self.max_entries:
            sorted_items = sorted(
                self._store.items(),
                key=lambda kv: float(kv[1].get("ts_epoch", 0.0)),
                reverse=True,
            )
            self._store = dict(sorted_items[: self.max_entries])
        self._save()


class SessionRecorder:
    """Capture turns and export in json or markdown."""

    def __init__(self, max_turns: int = 250) -> None:
        self.started_at = utc_now_iso()
        self.max_turns = max(20, int(max_turns))
        self.turns: list[dict[str, Any]] = []

    def add_turn(
        self,
        user_text: str,
        assistant_text: str,
        tools: list[str] | None = None,
        gen_time_sec: float | None = None,
    ) -> None:
        self.turns.append(
            {
                "ts": utc_now_iso(),
                "user": user_text,
                "assistant": assistant_text,
                "tools": tools or [],
                "gen_time_sec": gen_time_sec,
            }
        )
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def to_json_text(self) -> str:
        payload = {
            "started_at": self.started_at,
            "turn_count": len(self.turns),
            "turns": self.turns,
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def to_markdown_text(self) -> str:
        lines: list[str] = []
        lines.append("# Chat Session Export")
        lines.append("")
        lines.append(f"Started: {self.started_at}")
        lines.append(f"Turns: {len(self.turns)}")
        lines.append("")
        for i, turn in enumerate(self.turns, start=1):
            lines.append(f"## Turn {i}")
            lines.append("")
            lines.append(f"Timestamp: {turn.get('ts', '')}")
            tools = turn.get("tools") or []
            lines.append(f"Tools: {', '.join(tools) if tools else 'none'}")
            gt = turn.get("gen_time_sec")
            if isinstance(gt, (int, float)):
                lines.append(f"Generation time: {gt:.2f}s")
            lines.append("")
            lines.append("### User")
            lines.append("")
            lines.append(str(turn.get("user", "")))
            lines.append("")
            lines.append("### Assistant")
            lines.append("")
            lines.append(str(turn.get("assistant", "")))
            lines.append("")
        return "\n".join(lines)

    def export(self, out_dir: Path, fmt: str = "md") -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "json" if fmt.lower() == "json" else "md"
        target = out_dir / f"session_{stamp}.{ext}"
        if ext == "json":
            target.write_text(self.to_json_text(), encoding="utf-8")
        else:
            target.write_text(self.to_markdown_text(), encoding="utf-8")
        return target
