"""技能加载 —— Agent Skills 规范（agentskills.io，与 Pi/Claude 兼容）。

渐进式上下文（对照 Pi skills.ts）：只把 name + description 注入系统提示词索引，
正文由模型判断任务匹配后用 read 工具按需读取——几十个技能也不会撑爆窗口。

目录：项目 .you/skills/<技能名>/SKILL.md 与全局 ~/.you/skills/<技能名>/SKILL.md；
同名时项目级覆盖全局。frontmatter 仅解析扁平的 name/description（MVP 不引 pyyaml）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def _parse_frontmatter(text: str) -> dict[str, str]:
    """解析 YAML frontmatter 的扁平键值（name/description 足够）。"""
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line and not line.startswith((" ", "\t")):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _load_from_dir(d: Path) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    if not d.is_dir():
        return skills
    for sub in sorted(d.iterdir()):
        md = sub / "SKILL.md"
        if not (sub.is_dir() and md.is_file()):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        name = fm.get("name") or sub.name
        desc = fm.get("description", "").strip()
        if len(desc) > 1024:
            desc = desc[:1024] + "…"
        skills[name] = Skill(name=name, description=desc, path=md)
    return skills


def load_skills(cwd: Path | None = None, home: Path | None = None) -> list[Skill]:
    cwd = Path(cwd) if cwd else Path.cwd()
    home = Path(home) if home else Path.home()
    merged: dict[str, Skill] = {}
    merged.update(_load_from_dir(home / ".you" / "skills"))  # 全局先入
    merged.update(_load_from_dir(cwd / ".you" / "skills"))  # 项目覆盖全局
    return sorted(merged.values(), key=lambda s: s.name)


def skills_section(skills: list[Skill]) -> str:
    lines = [f"- {s.name}: {s.description or '(无描述)'}  [正文: {s.path}]" for s in skills]
    return (
        "<skills>\n可用技能（任务匹配时先用 read 读取对应 SKILL.md 全文，然后按其指引执行）：\n"
        + "\n".join(lines)
        + "\n</skills>"
    )
