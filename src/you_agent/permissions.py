"""权限闸门与审计 —— 对照岗位篇方案④：ask / allow-all / readonly 三档 + 规则 + 审计流水。

设计：
- 三档模式（--permission）：ask（交互逐项确认，默认于 REPL）/ allow-all（print/评测默认，
  但 deny 规则仍生效）/ readonly（只允许只读工具）；
- 规则文件 .you/permissions.json：{"allow": ["read", "bash:git *"], "deny": ["bash:rm *"]}，
  "工具名" 匹配该工具全部调用，"工具名:pattern" 对主参数做 fnmatch
  （bash/powershell 匹配命令，read/write/edit 等匹配路径）；deny 永远优先于 allow；
- ask 档的三种回答：y 本次允许 / always 本会话不再问（写入内存 allow）/ n 拒绝
  （拒绝原因回传模型，它会换安全方案——约束下的自主性）；
- 审计：每次判定追加一行 JSONL 到 .you-audit.jsonl（时间/工具/参数摘要/结论）。
"""

from __future__ import annotations

import fnmatch
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

from .types import ToolCall

READONLY_TOOL_NAMES = {"read", "grep", "find", "ls", "repo_map", "web_fetch"}

_ARG_KEY = {
    "bash": "command",
    "powershell": "command",
    "read": "path",
    "write": "path",
    "edit": "path",
    "grep": "path",
    "find": "path",
    "ls": "path",
}


def _arg_of(c: ToolCall) -> str:
    key = _ARG_KEY.get(c.name)
    v = c.arguments.get(key) if key else None
    return str(v) if v is not None else json.dumps(c.arguments, ensure_ascii=False)[:200]


def _rule_matches(rule: str, c: ToolCall) -> bool:
    if ":" not in rule:
        return rule == c.name
    name, _, pattern = rule.partition(":")
    return name == c.name and fnmatch.fnmatch(_arg_of(c), pattern)


class PermissionGate:
    """挂在 Hooks.before_tool_call 上：返回 {"block": True, "reason": ...} 即阻断。"""

    def __init__(
        self,
        mode: str = "ask",  # ask | allow-all | readonly
        rules_path: Path | None = None,
        audit_path: Path | None = None,
        interactive: bool = False,
        ask_fn=None,  # (prompt) -> "y" | "always" | "n"，测试可注入；默认用 rich Prompt
    ):
        if mode not in ("ask", "allow-all", "readonly"):
            raise ValueError(f"未知权限模式：{mode}")
        self.mode = mode
        self.interactive = interactive
        self.ask_fn = ask_fn or self._default_ask
        self.audit_path = audit_path or (Path.cwd() / ".you-audit.jsonl")
        self.allow: list[str] = []
        self.deny: list[str] = []
        rp = rules_path if rules_path is not None else Path.cwd() / ".you" / "permissions.json"
        if rp.exists():
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                self.allow = [str(r) for r in data.get("allow", [])]
                self.deny = [str(r) for r in data.get("deny", [])]
            except (json.JSONDecodeError, OSError):
                pass  # 规则文件坏了不影响启动，只是规则不生效

    # ---------- 判定 ----------

    def decide(self, c: ToolCall) -> str:
        """返回 allow | deny | ask。deny 规则永远优先（安全默认）。"""
        if any(_rule_matches(r, c) for r in self.deny):
            return "deny"
        if self.mode == "readonly":
            return "allow" if c.name in READONLY_TOOL_NAMES else "deny"
        if any(_rule_matches(r, c) for r in self.allow):
            return "allow"
        if self.mode == "allow-all":
            return "allow"
        return "ask"

    async def __call__(self, c: ToolCall) -> dict | None:
        verdict = self.decide(c)
        if verdict == "ask":
            if self.interactive and self.mode == "ask":
                verdict = await self._ask_and_learn(c)
            else:
                # fail-safe：无界面无法逐项确认时，ask 视为拒绝（headless 请显式 --permission allow-all）
                verdict = "deny"
        self._audit(c, verdict)
        if verdict == "deny":
            reason = (
                "权限闸门拒绝（无界面无法确认，ask 视为拒绝；"
                "如需放行可在 .you/permissions.json 加 allow 规则或用 --permission allow-all）"
                if self.mode == "ask" and not self.interactive
                else f"权限闸门拒绝（模式 {self.mode}；如需放行可在 .you/permissions.json 加 allow 规则）"
            )
            return {"block": True, "reason": reason}
        return None

    async def _ask_and_learn(self, c: ToolCall) -> str:
        if self.ask_fn is None:
            return "deny"
        answer = self.ask_fn(f"允许执行 {c.name}({_arg_of(c)[:120]}) 吗？")
        if inspect.isawaitable(answer):  # TUI 弹窗等异步确认
            answer = await answer
        if answer == "always":
            self.allow.append(c.name)  # 本会话该工具全部放行
            return "allow"
        return "allow" if answer == "y" else "deny"

    @staticmethod
    def _default_ask(prompt: str) -> str:
        from rich.prompt import Prompt

        return Prompt.ask(f"[yellow]{prompt}[/]", choices=["y", "always", "n"], default="n")

    # ---------- 审计 ----------

    def _audit(self, c: ToolCall, verdict: str) -> None:
        try:
            line = json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(timespec="seconds"),
                    "tool": c.name,
                    "arg": _arg_of(c)[:200],
                    "verdict": verdict,
                    "mode": self.mode,
                },
                ensure_ascii=False,
            )
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # 审计失败不阻塞执行
