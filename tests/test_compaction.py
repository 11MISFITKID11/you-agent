"""压缩管线测试 —— 对照 Pi compaction.ts 的语义：估算 / 切割点配对完整性 / 摘要替换。"""

from __future__ import annotations

import asyncio

from you_agent.compaction import Compactor, estimate_tokens, find_cut_point
from you_agent.loop import Agent, Hooks
from you_agent.providers import FauxStream
from you_agent.types import (
    AssistantMessage,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _history(blocks: int = 8) -> list:
    msgs: list = [SystemMessage(content="sys")]
    for i in range(blocks):
        msgs.append(UserMessage(content=f"任务{i} " + "x" * 400))
        msgs.append(AssistantMessage(content=[
            TextContent(text="回答" * 100),
            ToolCall(id=f"c{i}", name="read", arguments={"path": f"f{i}.py"}),
        ], stop_reason="tool_calls"))
        msgs.append(ToolResultMessage(tool_call_id=f"c{i}", tool_name="read", content="y" * 400))
        msgs.append(AssistantMessage(content=[TextContent(text="总结" * 50)]))
    return msgs


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 400) == 100          # ASCII 约 4 字符 / token
    assert estimate_tokens("中" * 100) == 100         # CJK 约 1 字符 / token
    assert estimate_tokens("中" * 100 + "x" * 400) == 200


def test_find_cut_point_lands_on_user_boundary():
    msgs = _history()
    cut = find_cut_point(msgs, keep_tokens=2000)
    assert cut is not None
    assert msgs[cut].type == "user"
    # 配对完整性：被压缩段里的每个 toolResult，其对应的 assistant（含 toolCall）也必须在 cut 之前
    ids_in_seg = set()
    for m in msgs[:cut]:
        if m.type == "assistant":
            ids_in_seg |= {c.id for c in m.tool_calls}
    for m in msgs[:cut]:
        if m.type == "toolResult":
            assert m.tool_call_id in ids_in_seg
    # 保留尾部里不得出现"孤儿" toolResult（对应的 assistant 被切走了）
    tail_ids = set()
    for m in msgs[cut:]:
        if m.type == "assistant":
            tail_ids |= {c.id for c in m.tool_calls}
    for m in msgs[cut:]:
        if m.type == "toolResult":
            assert m.tool_call_id in tail_ids


def test_find_cut_point_none_when_fits():
    msgs = [SystemMessage(content="s"), UserMessage(content="hi"), AssistantMessage(content=[TextContent(text="ok")])]
    assert find_cut_point(msgs, keep_tokens=10_000) is None


def test_compact_end_to_end():
    # faux 脚本：第 1 条给压缩摘要用，第 2 条给压缩后的正常回答
    script = [
        AssistantMessage(content=[TextContent(text="用户先后要求处理多个文件，已完成 f0-f5 的审阅，未做修改。")]),
        AssistantMessage(content=[TextContent(text="好的，继续。")]),
    ]
    faux = FauxStream(script)
    events = []
    agent = Agent(
        stream_fn=faux, tools=[], system_prompt="sys", model="faux",
        hooks=Hooks(on_event=events.append),
        compactor=Compactor(stream_fn=faux, model="faux", context_window=3000),
    )
    agent.messages = _history()  # 预置超长历史（估算约 4000+ tokens > 3000*0.8）

    final = asyncio.run(agent.run("继续"))

    # 摘要替换生效：早期消息消失，出现摘要 system 消息
    rendered = [m for m in agent.messages]
    assert any(m.type == "system" and "早期对话摘要" in m.content for m in rendered)
    assert not any(m.type == "user" and "任务0" in m.content for m in rendered)
    assert agent.messages[0].content == "sys"  # 领头系统提示词保留
    # faux 被调了两次：压缩摘要一次 + 正常回答一次
    assert len(faux.calls) == 2
    assert faux.calls[0][0].content.startswith("你是对话压缩器")  # 第一次调用的系统提示词是压缩指令
    assert "继续" in final.text
    assert any(e.type == "compaction" for e in events)
    # 压缩后不再触发
    assert not agent.compactor.should_compact(agent.messages)


def test_no_compact_below_threshold():
    faux = FauxStream([])
    agent = Agent(stream_fn=faux, tools=[], system_prompt="s", model="faux",
                  compactor=Compactor(stream_fn=faux, model="faux", context_window=1_000_000))
    agent.messages = _history(blocks=2)
    assert not agent.compactor.should_compact(agent.messages)
