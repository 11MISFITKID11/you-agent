"""P1 特性测试：Skills 加载 / todo 工具 / 并行工具执行 / MCP 桥接 / web_fetch / 自动窗口。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from you_agent.loop import Agent, Hooks
from you_agent.mcp_client import McpTool, load_mcp_config
from you_agent.providers import FauxStream, model_context_window
from you_agent.skills import load_skills, skills_section
from you_agent.tools.builtin import TodoTool, WebFetchTool, html_to_text
from you_agent.types import BaseTool, ToolResult

# ---------- Skills ----------


def _mk_skill(d: Path, name: str, desc: str):
    sub = d / name
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n正文内容", encoding="utf-8"
    )


def test_skills_loading_and_precedence(tmp_path):
    proj = tmp_path / "proj"
    home = tmp_path / "home"
    _mk_skill(home / ".you" / "skills", "global-skill", "全局技能")
    _mk_skill(proj / ".you" / "skills", "proj-skill", "项目技能")
    _mk_skill(proj / ".you" / "skills", "global-skill", "项目覆盖版")  # 同名覆盖全局

    skills = load_skills(cwd=proj, home=home)
    by_name = {s.name: s for s in skills}
    assert set(by_name) == {"global-skill", "proj-skill"}
    assert by_name["global-skill"].description == "项目覆盖版"

    section = skills_section(skills)
    assert section.startswith("<skills>") and "proj-skill: 项目技能" in section
    assert "SKILL.md" in section  # 索引里带正文路径，供模型 read


def test_skills_empty(tmp_path):
    assert load_skills(cwd=tmp_path / "x", home=tmp_path / "h") == []


# ---------- todo 工具 ----------


def test_todo_tool_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool = TodoTool()
    res = asyncio.run(tool.execute("t", {"action": "list"}))
    assert res.content == "(清单为空)"

    res2 = asyncio.run(tool.execute("t", {
        "action": "replace",
        "todos": [
            {"content": "读需求", "status": "done"},
            {"content": "改代码", "status": "in_progress"},
            {"content": "跑测试", "status": "pending"},
        ],
    }))
    assert not res2.is_error and "[进行中] 改代码" in res2.content
    saved = json.loads((tmp_path / ".you-todo.json").read_text(encoding="utf-8"))
    assert len(saved) == 3 and saved[0]["status"] == "done"

    res3 = asyncio.run(tool.execute("t", {"action": "list"}))
    assert "[done] 读需求" in res3.content


# ---------- 并行工具执行 ----------


class _Empty(BaseModel):
    pass


class SlowTool(BaseTool):
    args_model = _Empty

    def __init__(self, name: str, mode: str):
        self.name = name
        self.description = "慢工具"
        self.execution_mode = mode

    async def execute(self, tool_call_id, params):
        await asyncio.sleep(0.08)
        return ToolResult(content=f"{self.name} done")


def _batch_agent(tools, calls):
    from you_agent.types import AssistantMessage, TextContent, ToolCall

    script = [
        AssistantMessage(content=[ToolCall(id=f"c{i}", name=n, arguments={}) for i, n in enumerate(calls)],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="ok")]),
    ]
    events = []
    agent = Agent(stream_fn=FauxStream(script), tools=tools, system_prompt="t",
                  model="faux", hooks=Hooks(on_event=events.append))
    return agent, events


def test_parallel_batch_overlaps():
    agent, events = _batch_agent(
        [SlowTool("a", "parallel"), SlowTool("b", "parallel")], ["a", "b"]
    )
    asyncio.run(agent.run("go"))
    starts = [e.tool_call.name for e in events if e.type == "tool_execution_start"]
    ends = [e.tool_call.name for e in events if e.type == "tool_execution_end"]
    # 并行：两个 start 都在任一 end 之前
    seq = [(e.type, e.tool_call.name) for e in events if e.type.startswith("tool_execution")]
    first_end = next(i for i, (t, _) in enumerate(seq) if t == "tool_execution_end")
    assert all(t == "tool_execution_start" for t, _ in seq[:first_end])
    assert sorted(starts) == ["a", "b"] and sorted(ends) == ["a", "b"]


def test_sequential_batch_no_overlap():
    agent, events = _batch_agent(
        [SlowTool("a", "parallel"), SlowTool("w", "sequential")], ["a", "w"]
    )
    asyncio.run(agent.run("go"))
    seq = [(e.type, e.tool_call.name) for e in events if e.type.startswith("tool_execution")]
    assert seq == [
        ("tool_execution_start", "a"), ("tool_execution_end", "a"),
        ("tool_execution_start", "w"), ("tool_execution_end", "w"),
    ]  # 批内含写工具 -> 整批串行


# ---------- MCP 桥接（用假 session，不启动真实服务器） ----------


class _FakeMcpSession:
    async def list_tools(self):
        from mcp.types import ListToolsResult
        from mcp.types import Tool as MTool

        return ListToolsResult(tools=[MTool(
            name="echo",
            description="回声工具",
            inputSchema={"type": "object", "properties": {"msg": {"type": "string"}}},
        )])

    async def call_tool(self, name, arguments):
        from mcp.types import CallToolResult, TextContent

        return CallToolResult(content=[TextContent(type="text", text=f"回声:{arguments.get('msg')}")])


def test_mcp_tool_bridge():
    session = _FakeMcpSession()
    tools = None

    async def build():
        nonlocal tools
        listing = await session.list_tools()
        tools = [McpTool("test", td, session) for td in listing.tools]

    asyncio.run(build())
    t = tools[0]
    assert t.name == "mcp_test_echo"
    assert t.schema()["function"]["parameters"]["properties"]["msg"]["type"] == "string"  # schema 透传

    res = asyncio.run(t.execute("c1", {"msg": "你好"}))
    assert res.content == "回声:你好" and not res.is_error


def test_mcp_tool_in_agent_roundtrip():
    session = _FakeMcpSession()
    from you_agent.types import AssistantMessage, TextContent, ToolCall

    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="mcp_test_echo", arguments={"msg": "hi"})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="done")]),
    ]
    agent = Agent(stream_fn=FauxStream(script), tools=[], system_prompt="t", model="faux")

    async def main():
        listing = await session.list_tools()
        agent.add_tools([McpTool("test", td, session) for td in listing.tools])
        return await agent.run("调用它")

    final = asyncio.run(main())
    assert final.text == "done"
    results = [m for m in agent.messages if m.type == "toolResult"]
    assert results and results[0].content == "回声:hi" and not results[0].is_error


def test_load_mcp_config(tmp_path):
    f = tmp_path / "mcp.json"
    f.write_text('{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "x"]}}}', encoding="utf-8")
    assert load_mcp_config(f) == {"fs": {"command": "npx", "args": ["-y", "x"]}}
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    try:
        load_mcp_config(bad)
        raised = False
    except TypeError:
        raised = True
    assert raised


# ---------- web_fetch ----------


def test_html_to_text():
    page = ("<html><head><title>文档</title><style>x{}</style>"
            "<script>evil()</script></head><body><h1>你好 &amp; 世界</h1><p>内容</p></body></html>")
    text = html_to_text(page)
    assert text.startswith("[标题] 文档")
    assert "你好 & 世界" in text and "evil" not in text and "x{}" not in text


def test_web_fetch_rejects_bad_scheme(tmp_path):
    res = asyncio.run(WebFetchTool().execute("t", {"url": "ftp://example.com", "max_chars": 1000}))
    assert res.is_error and "http" in res.content


# ---------- 自动上下文窗口 ----------


def test_model_context_window():
    w = model_context_window("deepseek/deepseek-chat")
    assert w is None or w >= 8_000  # 目录命中则合理，未收录则 None（回退路径）
    assert model_context_window("totally/unknown-model-xyz") is None
