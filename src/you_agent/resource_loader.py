"""项目上下文文件发现 —— 对照 Pi 的 resource-loader.ts 与 AGENTS.md 开放规范。

规则（对照 Pi 的候选顺序，简化）：
- 从 cwd 向上遍历祖先目录，每个目录取第一个命中的 AGENTS.md（兼容 CLAUDE.md，仅当
  该目录没有 AGENTS.md 时生效）；
- 再叠加全局 ~/.you/agent/AGENTS.md；
- cwd 优先于祖先（最具体的放最前），全局放最后；
- 单文件截断 8000 字符，总截断 24000 字符。
"""

from __future__ import annotations

from pathlib import Path

CANDIDATE_NAMES = ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")
MAX_FILE_CHARS = 8_000
MAX_TOTAL_CHARS = 24_000


def _first_marker(d: Path) -> Path | None:
    for name in CANDIDATE_NAMES:
        p = d / name
        if p.is_file():
            return p
    return None


def load_project_context(cwd: Path | None = None, home: Path | None = None) -> str:
    """返回拼装好的项目上下文文本（无文件时为空串）。"""
    cwd = Path(cwd) if cwd else Path.cwd()
    home = Path(home) if home else Path.home()

    found: list[Path] = []
    for d in (cwd, *cwd.parents):
        p = _first_marker(d)
        if p is not None and p not in found:
            found.append(p)
    g = home / ".you" / "agent" / "AGENTS.md"
    if g.is_file() and g not in found:
        found.append(g)

    parts: list[str] = []
    total = 0
    for p in found:
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n...[文件过长已截断]"
        if total + len(text) > MAX_TOTAL_CHARS:
            text = text[: max(0, MAX_TOTAL_CHARS - total)] + "\n...[总长超限已截断]"
        parts.append(f'<project_context path="{p}">\n{text}\n</project_context>')
        total += len(text)
        if total >= MAX_TOTAL_CHARS:
            break
    return "\n\n".join(parts)
