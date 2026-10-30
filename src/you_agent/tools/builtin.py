"""八大内置工具 —— 对照 Pi 的 packages/coding-agent/src/core/tools/。

read / bash / powershell / edit / write / grep / find / ls。
设计要点（对照 Pi 源码策略）：
- 输出统一截断（对照 bash.ts 的超限处理，简化为字符截断 + 提示）；
- edit 用"旧串唯一匹配"语义（对照 edit.ts：模型对行号记忆不可靠，上下文里的代码才是事实）；
- 大文件/二进制/常见垃圾目录一律跳过。
"""

from __future__ import annotations

import asyncio
import json
import re
from html import unescape
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..types import BaseTool, ToolResult

MAX_OUTPUT = 30_000
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".you", ".pi"}


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[输出超限已截断，原始 {len(text)} 字符]"


# ---------- read ----------


class ReadArgs(BaseModel):
    path: str = Field(description="文件路径")
    offset: int = Field(default=0, ge=0, description="起始行（0 基）")
    limit: int = Field(default=2000, ge=1, description="最多读取行数")


class ReadTool(BaseTool):
    name = "read"
    description = "读取文本文件内容，输出带行号。大文件自动截断，可用 offset/limit 分段读取。"
    args_model = ReadArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        p = Path(params["path"])
        if not p.exists():
            return ToolResult(content=f"文件不存在：{p}", is_error=True)
        if p.is_dir():
            return ToolResult(content=f"是目录不是文件：{p}（列目录请用 ls）", is_error=True)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        off, lim = params["offset"], params["limit"]
        sel = lines[off : off + lim]
        body = "\n".join(f"{off + i + 1}\t{line}" for i, line in enumerate(sel))
        if off + lim < len(lines):
            body += f"\n...[共 {len(lines)} 行已截断，可用 offset={off + lim} 继续读取]"
        return ToolResult(content=_clip(body) or "(空文件)")


# ---------- bash / powershell ----------


class BashArgs(BaseModel):
    command: str = Field(description="要执行的 shell 命令")
    timeout: int = Field(default=60, ge=1, le=600, description="前台执行的超时秒数")
    run_in_background: bool = Field(
        default=False,
        description="后台运行并立即返回任务 id（长命令推荐）；之后用 jobs 工具轮询输出",
    )


class BashTool(BaseTool):
    execution_mode = "sequential"  # 写操作：强制所在批次串行，避免并发写竞争
    name = "bash"
    description = (
        "在当前工作目录执行 shell 命令并返回合并输出（Windows 下为 cmd）。"
        "适合构建、运行测试、git 等操作；文件检索优先用 grep/find/ls。"
    )
    args_model = BashArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        if params.get("run_in_background"):
            from ..jobs import JOB_MANAGER

            try:
                jid = JOB_MANAGER.start(params["command"], cwd=str(Path.cwd()),
                                        timeout_hint=params["timeout"])
            except RuntimeError as e:
                return ToolResult(content=str(e), is_error=True)
            return ToolResult(
                content=f"已后台启动 [{jid}] {params['command']}\n"
                        f"用 jobs 工具（action=check&id={jid}）轮询输出"
            )
        try:
            proc = await asyncio.create_subprocess_shell(
                params["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=params["timeout"])
        except TimeoutError:
            return ToolResult(
                content=f"命令超时（>{params['timeout']}s）：{params['command']}", is_error=True
            )
        text = out.decode("utf-8", errors="replace")
        body = f"$ {params['command']}\n{text}\n(exit={proc.returncode})"
        return ToolResult(content=_clip(body))


class PowerShellArgs(BaseModel):
    command: str = Field(description="要执行的 PowerShell 命令")
    timeout: int = Field(default=60, ge=1, le=600, description="超时秒数")


class PowerShellTool(BaseTool):
    execution_mode = "sequential"  # 写操作：强制所在批次串行，避免并发写竞争
    name = "powershell"
    description = "以 PowerShell 执行命令（Windows 专用），返回合并输出。适合需要 PowerShell 能力的操作。"
    args_model = PowerShellArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", params["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=params["timeout"])
        except TimeoutError:
            return ToolResult(
                content=f"命令超时（>{params['timeout']}s）：{params['command']}", is_error=True
            )
        except FileNotFoundError:
            return ToolResult(content="当前环境没有 powershell 可执行文件", is_error=True)
        text = out.decode("utf-8", errors="replace")
        return ToolResult(content=_clip(f"PS> {params['command']}\n{text}\n(exit={proc.returncode})"))


# ---------- write / edit ----------


class WriteArgs(BaseModel):
    path: str = Field(description="目标文件路径")
    content: str = Field(description="完整文件内容")


class WriteTool(BaseTool):
    execution_mode = "sequential"  # 写操作：强制所在批次串行，避免并发写竞争
    name = "write"
    description = "写入整个文件（覆盖已有内容）。新建文件或整体重写时使用；局部修改请用 edit。"
    args_model = WriteArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        p = Path(params["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(params["content"], encoding="utf-8")
        return ToolResult(content=f"已写入 {p}（{len(params['content'])} 字符）")


class EditArgs(BaseModel):
    path: str = Field(description="目标文件路径")
    old_text: str = Field(description="要被替换的原文片段，必须在文件中唯一")
    new_text: str = Field(description="替换后的内容")


class EditTool(BaseTool):
    execution_mode = "sequential"  # 写操作：强制所在批次串行，避免并发写竞争
    name = "edit"
    description = (
        "精确字符串替换编辑：old_text 需与文件内容一致且唯一出现（行尾空白差异可容错）。"
        "失败时会返回出现次数或最接近内容的行号上下文，据此修正后重试。"
    )
    args_model = EditArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        p = Path(params["path"])
        if not p.exists():
            return ToolResult(content=f"文件不存在：{p}", is_error=True)
        raw = p.read_text(encoding="utf-8", errors="replace")
        old, new = params["old_text"], params["new_text"]
        had_trailing_newline = raw.endswith(("\n", "\r\n"))
        lines = raw.splitlines()

        # 第一遍：精确匹配
        n = raw.count(old)
        if n == 1:
            p.write_text(raw.replace(old, new, 1), encoding="utf-8")
            return ToolResult(content=f"已编辑 {p}")

        # 第二遍：容错匹配（对照竞品的归一化策略——模型常在行尾空白上出错）
        if n == 0:
            old_lines = old.rstrip("\n").splitlines()
            hits = _find_normalized_window(lines, old_lines)
            if len(hits) == 1:
                i = hits[0]
                lines[i : i + len(old_lines)] = new.rstrip("\n").splitlines()
                body = "\n".join(lines) + ("\n" if had_trailing_newline else "")
                p.write_text(body, encoding="utf-8")
                return ToolResult(content=f"已编辑 {p}（行尾空白差异已容错，命中第 {i + 1} 行起）")
            if len(hits) > 1:
                return ToolResult(
                    content=(
                        f"old_text 容错匹配到 {len(hits)} 处（起始行："
                        f"{', '.join(str(h + 1) for h in hits[:5])}），不唯一。"
                        "请扩大上下文使其唯一后重试"
                    ),
                    is_error=True,
                )
            return ToolResult(
                content="old_text 在文件中未找到。最接近的位置如下（请以其实际内容重试）：\n"
                        + _nearest_context(lines, old_lines),
                is_error=True,
            )

        return ToolResult(
            content=f"old_text 出现 {n} 次，不唯一。请扩大上下文使其唯一后重试", is_error=True
        )


def _norm(line: str) -> str:
    return line.rstrip()


def _find_normalized_window(lines: list[str], old_lines: list[str]) -> list[int]:
    """逐行 rstrip 归一化后找匹配窗口，返回全部起始行下标。"""
    m = len(old_lines)
    if m == 0 or m > len(lines):
        return []
    tgt = [_norm(l) for l in old_lines]
    return [
        i for i in range(len(lines) - m + 1)
        if all(_norm(lines[i + k]) == tgt[k] for k in range(m))
    ]


def _nearest_context(lines: list[str], old_lines: list[str], radius: int = 2) -> str:
    """零命中时，用 difflib 找与 old_text 首行最相似的行，返回带行号的上下文。"""
    import difflib

    probe = _norm(old_lines[0]) if old_lines else ""
    best_i, best_ratio = 0, 0.0
    for i, line in enumerate(lines):
        r = difflib.SequenceMatcher(None, probe, _norm(line)).ratio()
        if r > best_ratio:
            best_i, best_ratio = i, r
    lo, hi = max(0, best_i - radius), min(len(lines), best_i + radius + 1)
    shown = "\n".join(f"{i + 1}\t{lines[i]}" for i in range(lo, hi))
    return f"最接近的行（第 {best_i + 1} 行附近，相似度 {best_ratio:.0%}）：\n{shown}"


# ---------- todo 任务清单（对照 Claude Code 的 TodoWrite：长任务防丢步骤） ----------


class TodoItem(BaseModel):
    content: str = Field(description="步骤描述")
    status: str = Field(default="pending", description="pending | in_progress | done")


class TodoArgs(BaseModel):
    action: Literal["list", "replace"] = Field(default="list", description="list 查看；replace 全量替换")
    todos: list[TodoItem] | None = Field(default=None, description="action=replace 时的新清单（全量）")


def _todo_path() -> Path:
    return Path.cwd() / ".you-todo.json"


def _render_todos(items: list[TodoItem]) -> str:
    marks = {"done": "[done]", "in_progress": "[进行中]", "pending": "[待办]"}
    return "\n".join(f"{marks.get(i.status, '?')} {i.content}" for i in items) or "(清单为空)"


class TodoTool(BaseTool):
    execution_mode = "sequential"
    name = "todo"
    description = (
        "维护本次会话的任务清单（存储于 .you-todo.json）。长任务开始前先写入分步清单；"
        "每完成一步用 action=replace 更新全量状态（进行中的那步标 in_progress）。"
        "action=list 查看当前清单。"
    )
    args_model = TodoArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        if params["action"] == "list":
            if not _todo_path().exists():
                return ToolResult(content="(清单为空)")
            items = [TodoItem.model_validate(d) for d in json.loads(_todo_path().read_text(encoding="utf-8"))]
            return ToolResult(content=_render_todos(items))
        # 入参可能是 dict（model_dump 后）或 TodoItem，统一归一
        items = [i if isinstance(i, TodoItem) else TodoItem.model_validate(i) for i in (params.get("todos") or [])]
        _todo_path().write_text(
            json.dumps([i.model_dump() for i in items], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ToolResult(content="清单已更新：\n" + _render_todos(items))


# ---------- web_fetch（抓取网页转纯文本） ----------


class WebFetchArgs(BaseModel):
    url: str = Field(description="http/https 网址")
    max_chars: int = Field(default=20_000, ge=200, le=100_000, description="返回文本上限")


def html_to_text(page: str) -> str:
    """极简 HTML 转文本：去 script/style、去标签、还原实体、压缩空白。"""
    page = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", page)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    text = re.sub(r"\s+", " ", unescape(page)).strip()
    if title:
        text = f"[标题] {unescape(title.group(1)).strip()}\n{text}"
    return text


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = (
        "抓取 http/https 网页并转为纯文本（去标签），按 max_chars 截断。"
        "适合查阅在线文档与资料；不支持搜索，只抓取给定 URL。"
    )
    args_model = WebFetchArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        url = params["url"].strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(content="仅支持 http/https URL", is_error=True)
        import httpx

        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url)
        except Exception as e:  # noqa: BLE001 网络错误转工具结果
            return ToolResult(content=f"请求失败：{type(e).__name__}: {e}", is_error=True)
        if resp.status_code >= 400:
            return ToolResult(content=f"HTTP {resp.status_code}: {url}", is_error=True)
        text = resp.text
        if "html" in resp.headers.get("content-type", "").lower():
            text = html_to_text(text)
        if len(text) > params["max_chars"]:
            text = text[: params["max_chars"]] + f"\n...[超限截断，原始 {len(text)} 字符]"
        return ToolResult(content=text or "(空响应)")


# ---------- grep / find / ls ----------


class GrepArgs(BaseModel):
    pattern: str = Field(description="正则表达式")
    path: str = Field(default=".", description="搜索根目录")
    glob: str | None = Field(default=None, description="文件名过滤 glob，如 *.py")
    max_results: int = Field(default=100, ge=1, le=1000)


class GrepTool(BaseTool):
    name = "grep"
    description = "在文件内容中做正则搜索，返回 文件:行号:匹配行。默认跳过二进制与垃圾目录。"
    args_model = GrepArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        try:
            rx = re.compile(params["pattern"])
        except re.error as e:
            return ToolResult(content=f"正则表达式无效：{e}", is_error=True)
        root = Path(params["path"])
        if not root.exists():
            return ToolResult(content=f"目录不存在：{root}", is_error=True)
        hits: list[str] = []
        files = root.rglob(params["glob"]) if params["glob"] else root.rglob("*")
        for f in files:
            if not f.is_file() or any(part in SKIP_DIRS for part in f.parts):
                continue
            try:
                if f.stat().st_size > 1_000_000:
                    continue
                blob = f.read_bytes()
                if b"\x00" in blob[:4096]:
                    continue
                for i, line in enumerate(blob.decode("utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f}:{i}:{line.strip()[:200]}")
                        if len(hits) >= params["max_results"]:
                            break
            except OSError:
                continue
            if len(hits) >= params["max_results"]:
                break
        if not hits:
            return ToolResult(content="无匹配结果")
        return ToolResult(content=_clip("\n".join(hits)))


class FindArgs(BaseModel):
    pattern: str = Field(description="glob 文件名模式，如 **/*.py 或 *.json")
    path: str = Field(default=".", description="搜索根目录")


class FindTool(BaseTool):
    name = "find"
    description = "按 glob 模式查找文件名（如 **/*.py），返回匹配路径列表。"
    args_model = FindArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        root = Path(params["path"])
        if not root.exists():
            return ToolResult(content=f"目录不存在：{root}", is_error=True)
        found = [
            str(f)
            for f in sorted(root.glob(params["pattern"]))
            if not any(part in SKIP_DIRS for part in f.parts)
        ][:200]
        return ToolResult(content=_clip("\n".join(found) or "无匹配文件"))


class LsArgs(BaseModel):
    path: str = Field(default=".", description="要列出的目录")


class LsTool(BaseTool):
    name = "ls"
    description = "列出目录内容（含隐藏文件，字母序，目录带 / 标记）。"
    args_model = LsArgs

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        root = Path(params["path"])
        if not root.exists():
            return ToolResult(content=f"目录不存在：{root}", is_error=True)
        rows = [f"{e.name}{'/' if e.is_dir() else ''}" for e in sorted(root.iterdir())]
        return ToolResult(content=_clip("\n".join(rows) or "(空目录)"))


def default_tools() -> list[BaseTool]:
    from ..jobs import JobsTool
    from ..repo_map import RepoMapTool

    return [ReadTool(), BashTool(), WriteTool(), EditTool(), GrepTool(), FindTool(), LsTool(),
            PowerShellTool(), TodoTool(), WebFetchTool(), RepoMapTool(), JobsTool()]


def read_only_tools() -> list[BaseTool]:
    from ..repo_map import RepoMapTool

    return [ReadTool(), GrepTool(), FindTool(), LsTool(), RepoMapTool()]
