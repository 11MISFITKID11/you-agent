"""仓库符号地图 —— 对照 Aider 的 repo map / Cursor 的代码库索引。

两条实现路径：
1. universal-ctags（`--output-format=json`，全语言、含签名）——本机装了就用；
2. 纯 Python 正则扫描回退（.py/.js/.ts/.mjs 的 def/class/function/const）——免安装，
   保证工具在任何环境可用。

排序（对照 Aider 基于引用图的 PageRank，MVP 简化为"出现度"）：符号名在仓库源码中
出现的文件数越多排越前；query 关键词命中的符号与文件额外加成。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from .types import BaseTool, ToolResult

MAX_OUTPUT = 12_000
MAX_SCAN_FILES = 400
SCAN_EXTS = {".py", ".js", ".ts", ".mjs", ".cjs"}
SKIP_DIRS = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".you", ".pi"}

_PY_FUNC = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.MULTILINE)
_PY_CLASS = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
_JS_FUNC = re.compile(r"\bfunction\s+(\w+)")
_JS_CLASS = re.compile(r"\bclass\s+(\w+)")
_JS_CONST = re.compile(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>")


def _iter_source_files(root: Path, cap: int = MAX_SCAN_FILES):
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SCAN_EXTS:
            yield p
            n += 1
            if n >= cap:
                return


def _scan(root: Path) -> list[dict]:
    """纯 Python 扫描回退：抽取 py/js/ts 的函数与类定义。"""
    out: list[dict] = []
    for p in _iter_source_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(root))

        def line_of(pos: int, _text: str = text) -> int:  # 默认参绑定当前迭代的 text，避免 B023
            return _text.count("\n", 0, pos) + 1

        for m in _PY_FUNC.finditer(text):
            out.append({"name": m.group(1), "kind": "func", "file": rel, "line": line_of(m.start())})
        for m in _PY_CLASS.finditer(text):
            out.append({"name": m.group(1), "kind": "class", "file": rel, "line": line_of(m.start())})
        for m in _JS_FUNC.finditer(text):
            out.append({"name": m.group(1), "kind": "func", "file": rel, "line": line_of(m.start())})
        for m in _JS_CLASS.finditer(text):
            out.append({"name": m.group(1), "kind": "class", "file": rel, "line": line_of(m.start())})
        for m in _JS_CONST.finditer(text):
            out.append({"name": m.group(1), "kind": "arrow", "file": rel, "line": line_of(m.start())})
    return out


def _ctags(root: Path) -> list[dict] | None:
    """ctags 路径：全语言 + 签名。不可用时返回 None 走扫描回退。"""
    if shutil.which("ctags") is None:
        return None
    try:
        r = subprocess.run(
            ["ctags", "--output-format=json", "--fields=+n", "-R", "."],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return None
    if r.returncode != 0:
        return None
    out: list[dict] = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("_type") == "ptag":
            continue
        name, path = d.get("name"), d.get("path")
        if name and path:
            out.append({"name": name, "kind": d.get("kind") or "?", "file": path,
                        "line": d.get("line") or 0})
    return out or None


def _rank(symbols: list[dict], root: Path, query: str | None, top: int) -> list[dict]:
    """出现度排序：符号名出现在多少个源文件里（含定义处）；query 命中加成。"""
    file_texts: list[tuple[str, str]] = []
    for p in _iter_source_files(root):
        try:
            file_texts.append((str(p.relative_to(root)), p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue

    def ref_files_count(name: str) -> int:
        return sum(1 for _, text in file_texts if name in text)

    q = (query or "").lower()
    scored: list[tuple[int, dict]] = []
    for s in symbols[:2000]:  # 超大仓库截断，MVP 已知限制
        score = ref_files_count(s["name"])
        if q:
            if q in s["name"].lower():
                score += 50
            if q in s["file"].lower():
                score += 10
        scored.append((score, s))
    scored.sort(key=lambda x: (-x[0], x[1]["file"], x[1]["name"]))
    return [s for _, s in scored[:top]]


class RepoMapArgs(BaseModel):
    query: str | None = Field(default=None, description="关注的关键词，用于排序加成（如模块名/功能词）")
    max_symbols: int = Field(default=40, ge=5, le=200)


class RepoMapTool(BaseTool):
    name = "repo_map"
    description = (
        "生成仓库符号地图：按被引用程度排序，输出 函数/类 的种类、名称与 文件:行号。"
        "先看地图再决定读哪些文件，比逐个 grep 更省上下文。可选 query 关键词加成排序。"
    )
    args_model = RepoMapArgs

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else Path.cwd()
        self._cache: list[dict] | None = None

    def _symbols(self) -> list[dict]:
        if self._cache is None:
            self._cache = _ctags(self.root) or _scan(self.root)
        return self._cache

    async def execute(self, tool_call_id: str, params: dict) -> ToolResult:
        syms = self._symbols()
        if not syms:
            return ToolResult(content="未解析出任何符号（空仓库，或没有受支持语言的文件）")
        ranked = _rank(syms, self.root, params["query"], params["max_symbols"])
        lines = [f'{s["kind"]:6} {s["name"]}  ({s["file"]}:{s["line"]})' for s in ranked]
        body = "\n".join(lines)
        if len(body) > MAX_OUTPUT:
            body = body[:MAX_OUTPUT] + "\n...[超限截断，可减小 max_symbols]"
        return ToolResult(content=f"{body}\n\n(共 {len(syms)} 个符号，展示前 {len(ranked)} 个)")
