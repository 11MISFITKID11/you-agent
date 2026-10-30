"""内核冒烟测试：全部用 FauxStream 脚本化假模型驱动，无需任何 API Key。

覆盖 Pi 复刻清单中最关键的四条语义：
1. 工具调用往返（决策 -> 执行 -> 回填 -> 最终回答）；
2. 截断防御（stop_reason=length 时工具不执行，错误回填，且不再发起后续调用）；
3. before_tool_call 闸门阻断；
4. 参数校验失败转错误结果（模型可自行补救）。
"""

from __future__ import annotations

import asyncio

from you_agent.loop import Agent, Hooks
from you_agent.providers import FauxStream
from you_agent.tools.builtin import EditTool, ReadTool, WriteTool
from you_agent.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage


def _make_agent(script, tools=None, hooks=None):
    faux = FauxStream(script)
    events = []
    h = hooks or Hooks()
    h.on_event = events.append
    agent = Agent(
        stream_fn=faux,
        tools=tools or [ReadTool(), WriteTool(), EditTool()],
        system_prompt="test",
        model="faux",
        hooks=h,
    )
    return agent, faux, events


def test_tool_roundtrip(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("Hello you-agent", encoding="utf-8")
    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="read", arguments={"path": str(f)})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="文件内容是 Hello you-agent")]),
    ]
    agent, faux, events = _make_agent(script)
    final = asyncio.run(agent.run("读一下这个文件"))

    assert "Hello you-agent" in final.text
    assert isinstance(agent.messages[-1], AssistantMessage)
    assert any(isinstance(m, ToolResultMessage) and not m.is_error for m in agent.messages)
    # 第二次调用时上下文里必须已有工具结果（回填生效）
    assert any(m.type == "toolResult" for m in faux.calls[1])
    assert any(e.type == "tool_execution_end" and not e.is_error for e in events)


def test_truncation_defense(tmp_path):
    """stop_reason=length 且带工具调用 -> 不执行、错误回填、不再有第二次 LLM 调用。"""
    script = [
        AssistantMessage(
            content=[TextContent(text="截断的输出..."), ToolCall(id="c9", name="read", arguments={"path": "x"})],
            stop_reason="length",
        )
    ]
    agent, faux, events = _make_agent(script)
    asyncio.run(agent.run("触发截断"))

    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert results and results[0].is_error and "截断" in results[0].content
    assert len(faux.calls) == 1  # 没有后续 LLM 调用
    assert not any(e.type == "tool_execution_start" for e in events)  # 工具确实没被执行


def test_before_tool_call_block(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="read", arguments={"path": str(f)})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="好吧，被拦了")]),
    ]
    hooks = Hooks(before_tool_call=lambda c: {"block": True, "reason": "测试阻断"})
    agent, faux, _ = _make_agent(script, hooks=hooks)
    asyncio.run(agent.run("读文件"))

    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert results and results[0].is_error and "阻断" in results[0].content
    # 闸门在工具执行前生效：faux 收到的第二轮上下文里工具结果是阻断说明
    second_ctx = faux.calls[1]
    tr = next(m for m in second_ctx if m.type == "toolResult")
    assert "阻断" in tr.content


def test_invalid_params_feed_back(tmp_path):
    """参数不符合 schema（read 的 offset 传了字符串数字）-> pydantic 宽松纠正后正常执行。"""
    f = tmp_path / "n.txt"
    f.write_text("line", encoding="utf-8")
    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="read",
                                           arguments={"path": str(f), "offset": "1", "limit": "5"})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="ok")]),
    ]
    agent, _, _ = _make_agent(script)
    asyncio.run(agent.run("x"))
    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert results and not results[0].is_error  # "1"/"5" 被宽松校验纠正为 int


def test_unknown_tool(tmp_path):
    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="nope", arguments={})], stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="了解了")]),
    ]
    agent, _, _ = _make_agent(script)
    asyncio.run(agent.run("x"))
    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert results and results[0].is_error and "未知工具" in results[0].content
