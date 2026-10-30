"""P0 特性测试：AGENTS.md 加载 / 会话恢复 / edit 容错 / 溢出自愈。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from you_agent.compaction import Compactor
from you_agent.loop import Agent, Hooks
from you_agent.providers import StreamEvent, is_context_overflow
from you_agent.resource_loader import load_project_context
from you_agent.session import SessionManager, list_sessions
from you_agent.tools.builtin import EditTool
from you_agent.types import (
    AssistantMessage,
    SystemMessage,
    TextContent,
    UserMessage,
)

# ---------- AGENTS.md 项目上下文 ----------


def test_agents_md_discovery(tmp_path):
    (tmp_path / "AGENTS.md").write_text("项目规范A", encoding="utf-8")
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    ctx = load_project_context(cwd=sub, home=tmp_path / "home")
    assert "项目规范A" in ctx

    # 全局 ~/.you/agent/AGENTS.md 叠加
    g = tmp_path / "home" / ".you" / "agent"
    g.mkdir(parents=True)
    (g / "AGENTS.md").write_text("全局规范G", encoding="utf-8")
    assert "全局规范G" in load_project_context(cwd=sub, home=tmp_path / "home")

    # CLAUDE.md 兼容：没有 AGENTS.md 的目录生效
    d2 = tmp_path / "only-claude"
    d2.mkdir()
    (d2 / "CLAUDE.md").write_text("claude规范", encoding="utf-8")
    assert "claude规范" in load_project_context(cwd=d2, home=tmp_path / "home")

    # 同目录同时存在时 AGENTS.md 优先于 CLAUDE.md
    d4 = tmp_path / "both"
    d4.mkdir()
    (d4 / "CLAUDE.md").write_text("claude规范", encoding="utf-8")
    (d4 / "AGENTS.md").write_text("agents优先", encoding="utf-8")
    ctx3 = load_project_context(cwd=d4, home=tmp_path / "home")
    assert "agents优先" in ctx3 and "claude规范" not in ctx3


# ---------- 会话恢复 ----------


def test_session_open_and_continue_tree(tmp_path):
    s1 = SessionManager(path=tmp_path / "s.jsonl", cwd=tmp_path)
    s1.append(SystemMessage(content="sys"))
    s1.append(UserMessage(content="第一问"))
    s1.append(AssistantMessage(content=[TextContent(text="第一答")]))
    p = s1.path
    del s1

    s2 = SessionManager.open(p)
    assert s2.leaf is not None and len(s2.entries) == 3
    msgs = s2.messages_up_to(s2.leaf)
    assert [m.content if m.type != "assistant" else m.text for m in msgs] == ["sys", "第一问", "第一答"]

    # 恢复后继续追加：写同一文件，延续树
    rid = s2.append(UserMessage(content="第二问"))
    assert s2.leaf == rid and len(s2.entries) == 4
    assert SessionManager.open(p).entries.__len__() == 4


def test_list_sessions_order(tmp_path):
    home = tmp_path / "home"
    d = home / ".you" / "sessions" / "--x--"
    d.mkdir(parents=True)
    (d / "old.jsonl").write_text("{}", encoding="utf-8")
    time.sleep(0.05)
    (d / "new.jsonl").write_text("{}", encoding="utf-8")
    assert [p.name for p in list_sessions(cwd=Path("x"), home=home)] == ["new.jsonl", "old.jsonl"]
    assert list_sessions(cwd=Path("nowhere"), home=home) == []


# ---------- edit 容错 ----------


def test_edit_fuzzy_trailing_whitespace(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def foo():\n    return 1   \n\ndef bar():\n    return 2\n", encoding="utf-8")
    res = asyncio.run(EditTool().execute(
        "t", {"path": str(f), "old_text": "    return 1\n", "new_text": "    return 42\n"}))
    assert not res.is_error and "容错" in res.content
    body = f.read_text(encoding="utf-8")
    assert "return 42" in body and "return 1" not in body.replace("return 42", "")


def test_edit_zero_match_returns_nearest_context(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n", encoding="utf-8")
    res = asyncio.run(EditTool().execute(
        "t", {"path": str(f), "old_text": "def alpah():\n    return 99\n", "new_text": "x"}))
    assert res.is_error and "最接近" in res.content and "alpha" in res.content


def test_edit_multiple_match_still_errors(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("same\nsame\n", encoding="utf-8")
    res = asyncio.run(EditTool().execute(
        "t", {"path": str(f), "old_text": "same", "new_text": "diff"}))
    assert res.is_error and "2 次" in res.content


# ---------- 溢出自愈 ----------


class OverflowThenOK:
    """第 1 次调用抛溢出错误，之后每次返回一条预设消息。"""

    def __init__(self):
        self.i = 0

    async def __call__(self, model, messages, tools=None):
        self.i += 1
        if self.i == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'maximum context length is 4096 tokens, "
                "however you requested 5000 tokens'}}"
            )
        yield StreamEvent(
            type="message_end",
            message=AssistantMessage(content=[TextContent(text=f"resp{self.i}")]),
        )


def _hist() -> list:
    msgs: list = [SystemMessage(content="s")]
    for i in range(4):
        msgs.append(UserMessage(content=f"任务{i} " + "x" * 500))
        msgs.append(AssistantMessage(content=[TextContent(text="答" * 100)]))
    return msgs


def test_overflow_auto_compact_retry():
    stub = OverflowThenOK()
    agent = Agent(
        stream_fn=stub, tools=[], system_prompt="s", model="faux", hooks=Hooks(),
        compactor=Compactor(stream_fn=stub, model="faux", context_window=1_000_000),
    )
    agent.messages = _hist()  # 远未达内部阈值：验证 force 路径

    final = asyncio.run(agent.run("继续"))

    # 调用序列：1 溢出 -> 2 压缩摘要 -> 3 重试回答
    assert stub.i == 3
    assert final.text == "resp3"
    assert any(m.type == "system" and "早期对话摘要" in m.content for m in agent.messages)


def test_overflow_unrelated_error_propagates():
    class ConnErr:
        async def __call__(self, model, messages, tools=None):
            raise RuntimeError("connection error: Connection reset")
            yield  # pragma: no cover

    agent = Agent(stream_fn=ConnErr(), tools=[], system_prompt="s", model="faux")
    agent.messages = _hist()
    try:
        asyncio.run(agent.run("x"))
        raised = False
    except RuntimeError as e:
        raised = "connection error" in str(e)
    assert raised  # 非溢出错误不被吞掉，走友好报错路径


def test_is_context_overflow_patterns():
    assert is_context_overflow(RuntimeError("maximum context length is 4096"))
    assert is_context_overflow(ValueError("prompt is too long: 91234 tokens"))
    assert is_context_overflow(Exception("This model's maximum context length is 128000 tokens"))
    assert not is_context_overflow(RuntimeError("connection error"))
