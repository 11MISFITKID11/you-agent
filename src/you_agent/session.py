"""树形 JSONL 会话 —— 对照 Pi 的 packages/coding-agent/src/core/session-manager.ts。

关键设计（对照 pi session-format v3）：
- 文件首行是 SessionHeader（type=session, version, id, cwd）；
- 其余每行一个条目，带 id（8 位 hex）与 parentId，构成树：从历史节点分叉不丢数据；
- 追加即落盘（每行 open/append，崩溃安全）。
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .types import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

_MSG_CLS = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AssistantMessage,
    "toolResult": ToolResultMessage,
}


def _escaped(cwd: Path) -> str:
    return re.sub(r"[/\\:]", "-", str(cwd))


def _session_dir_for(cwd: Path, home: Path | None = None) -> Path:
    base = (home or Path.home()) / ".you" / "sessions"
    return base / f"--{_escaped(cwd)}--"


def list_sessions(cwd: Path | None = None, home: Path | None = None) -> list[Path]:
    """列出某工作目录的历史会话文件，按修改时间新→旧。"""
    d = _session_dir_for(Path(cwd) if cwd else Path.cwd(), home)
    if not d.exists():
        return []
    return sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


class SessionManager:
    def __init__(self, path: Path | None = None, cwd: Path | None = None):
        self.cwd = cwd or Path.cwd()
        if path is None:
            sid = uuid.uuid4().hex[:12]
            ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            path = _session_dir_for(self.cwd) / f"{ts}_{sid}.jsonl"
        self.path = path
        self.id = uuid.uuid4().hex[:12]
        self.entries: list[dict] = []
        self.leaf: str | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "session", "version": 3, "id": self.id, "cwd": str(self.cwd)},
                               ensure_ascii=False) + "\n")

    @classmethod
    def open(cls, path: Path) -> SessionManager:
        """打开既有会话文件（--continue/--resume 用）：追加写入同一文件，延续树结构。"""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].strip():
            raise ValueError(f"会话文件为空：{path}")
        header = json.loads(lines[0])
        if header.get("type") != "session":
            raise ValueError(f"不是有效的会话文件（首行应为 session header）：{path}")
        sm = cls.__new__(cls)
        sm.path = Path(path)
        sm.id = header.get("id", "")
        sm.cwd = Path(header.get("cwd", "."))
        sm.entries = [json.loads(l) for l in lines[1:] if l.strip()]
        sm.leaf = sm.entries[-1]["id"] if sm.entries else None
        return sm

    @staticmethod
    def _eid() -> str:
        return uuid.uuid4().hex[:8]

    def append_raw(self, payload: dict) -> str:
        """追加任意条目（消息 / 检查点 / 未来扩展的自定义状态），返回条目 id。"""
        entry = {"type": "entry", "id": self._eid(), "parentId": self.leaf, **payload}
        self.entries.append(entry)
        self.leaf = entry["id"]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]

    def append(self, message: Message) -> str:
        return self.append_raw({"message": message.model_dump()})

    def append_checkpoint(self, ref: str, label: str) -> str:
        """记录一个文件状态检查点（对照 Pi 的 CustomEntry：扩展状态进会话树）。"""
        return self.append_raw({"checkpoint": {"ref": ref, "label": label}})

    def checkpoints(self) -> list[dict]:
        return [e for e in self.entries if "checkpoint" in e]

    def checkout(self, entry_id: str) -> None:
        """把活动叶子切回历史节点：后续 append 从该节点分叉，旧分支保留（树的分支）。"""
        for e in self.entries:
            if e["id"] == entry_id:
                self.leaf = entry_id
                return
        raise KeyError(f"会话条目不存在：{entry_id}")

    def entry_ids(self) -> list[str]:
        return [e["id"] for e in self.entries]

    def messages_up_to(self, entry_id: str) -> list[Message]:
        """从树根沿 parent 链走到 entry_id，重建该分支的消息序列（回滚后重建 agent 上下文用）。"""
        by_id = {e["id"]: e for e in self.entries}
        dumps: list[dict] = []
        cur = by_id.get(entry_id)
        while cur is not None:
            if "message" in cur:
                dumps.append(cur["message"])
            cur = by_id.get(cur.get("parentId"))
        dumps.reverse()
        return [_MSG_CLS[d["type"]].model_validate(d) for d in dumps]
