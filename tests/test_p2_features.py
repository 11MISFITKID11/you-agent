"""P2 特性测试：权限闸门（规则/ask/审计/fail-safe）与 steering（插话注入/revive）。"""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel

from you_agent.loop import Agent, Hooks
from you_agent.permissions import PermissionGate
from you_agent.providers import FauxStream
from you_agent.types import (
    AssistantMessage,
    BaseTool,
    TextContent,
    ToolCall,
    ToolResult,
)


def _call(name, **args):
    return ToolCall(id="c1", name=name, arguments=args)


# ---------- 权限：规则 ----------


def _gate(tmp_path, mode="ask", **kw):
    return PermissionGate(mode=mode, rules_path=tmp_path / "perm.json",
                          audit_path=tmp_path / "audit.jsonl", **kw)


def test_permission_rules_deny_over_allow(tmp_path):
    (tmp_path / "perm.json").write_text(json.dumps({
        "allow": ["read", "bash:git *"],
        "deny": ["bash:rm *", "write"],
    }), encoding="utf-8")
    g = _gate(tmp_path)
    assert g.decide(_call("read", path="a.py")) == "allow"
    assert g.decide(_call("bash", command="git status")) == "allow"
    assert g.decide(_call("bash", command="rm -rf /")) == "deny"       # deny 优先
    assert g.decide(_call("write", path="x")) == "deny"                # 精确工具名
    assert g.decide(_call("bash", command="npm test")) == "ask"        # 未匹配 -> ask


def test_permission_readonly_mode(tmp_path):
    g = _gate(tmp_path, mode="readonly")
    assert g.decide(_call("read", path="a")) == "allow"
    assert g.decide(_call("repo_map")) == "allow"
    assert g.decide(_call("edit", path="a")) == "deny"


def test_permission_ask_flow_and_always(tmp_path):
    g = _gate(tmp_path, mode="ask", interactive=True)
    answers = iter(["y", "always", "n"])

    async def ask(prompt):  # 覆盖异步 ask_fn（TUI 弹窗路径）
        return next(answers)

    g.ask_fn = ask

    assert asyncio.run(g(_call("write", path="a.txt"))) is None         # y -> 本次放行
    assert asyncio.run(g(_call("write", path="b.txt"))) is None         # always -> 该工具进 allow
    assert g.decide(_call("write", path="c.txt")) == "allow"            # 不再问
    blocked = asyncio.run(g(_call("edit", path="d.txt")))
    assert blocked and blocked["block"]                                  # n -> 拒绝


def test_permission_headless_ask_fails_safe(tmp_path):
    g = _gate(tmp_path, mode="ask", interactive=False)                  # print/评测场景
    blocked = asyncio.run(g(_call("bash", command="git push")))
    assert blocked and "无界面" in blocked["reason"]                     # fail-safe：拒而不是放


def test_permission_audit_log(tmp_path):
    g = _gate(tmp_path, mode="allow-all")
    asyncio.run(g(_call("read", path="x")))
    asyncio.run(g(_call("write", path="y")))
    lines = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["tool"] for l in lines] == ["read", "write"]
    assert all(l["verdict"] == "allow" for l in lines)


# ---------- steering：运行中插话 ----------


class _Empty(BaseModel):
    pass


class SteerTool(BaseTool):
    name = "steer_now"
    description = "测试用：往 agent 塞一条插话"
    args_model = _Empty

    def __init__(self, agent_ref, text):
        self.agent_ref = agent_ref
        self.text = text

    async def execute(self, tool_call_id, params):
        self.agent_ref.steer(self.text)
        return ToolResult(content="steered")


def test_steering_injected_next_turn():
    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="steer_now", arguments={})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="收到插话")]),
    ]
    faux = FauxStream(script)
    agent = Agent(stream_fn=faux, tools=[], system_prompt="t", model="faux")
    agent.all_tools["steer_now"] = SteerTool(agent, "换个思路，只做分析")
    agent.tools["steer_now"] = agent.all_tools["steer_now"]

    final = asyncio.run(agent.run("开始"))

    assert final.text == "收到插话"
    assert any(m.type == "user" and m.content == "换个思路，只做分析" for m in agent.messages)
    # faux 第二次调用收到的上下文里已含插话（下一轮生效）
    second_ctx = faux.calls[1]
    assert any(m.type == "user" and m.content == "换个思路，只做分析" for m in second_ctx)


def test_steering_revives_run():
    """模型给出无工具调用的收尾回答后，新到的插话把 run 复活一轮。"""
    script = [
        AssistantMessage(content=[TextContent(text="初次回答")]),
        AssistantMessage(content=[TextContent(text="追答")]),
    ]
    faux = FauxStream(script)
    agent_holder: dict = {}

    def on_msg(ev):
        # 在"初次回答"落地的瞬间注入插话（此刻停止判定尚未发生，时序确定）
        if ev.type == "message_end" and ev.message is not None and ev.message.text == "初次回答":
            agent_holder["agent"].steer("还有个问题：继续说")

    agent = Agent(stream_fn=faux, tools=[], system_prompt="t", model="faux",
                  hooks=Hooks(on_event=on_msg))
    agent_holder["agent"] = agent

    final = asyncio.run(agent.run("你好"))

    assert len(faux.calls) == 2                      # 本该 1 轮结束，被插话复活
    assert final.text == "追答"
    assert any(m.type == "user" and m.content == "还有个问题：继续说" for m in agent.messages)


def test_steering_event_persisted(tmp_path):
    from you_agent.cli import Runtime  # 复用真实事件处理（含 user_message 持久化）
    from you_agent.govern import CostGuard
    from you_agent.session import SessionManager

    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="steer_now", arguments={})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="好")]),
    ]
    agent = Agent(stream_fn=FauxStream(script), tools=[], system_prompt="t", model="faux")
    tool = SteerTool(agent, "插话内容X")
    agent.all_tools[tool.name] = tool
    agent.tools[tool.name] = tool

    session = SessionManager(path=tmp_path / "s.jsonl")
    events = []
    rt = Runtime.__new__(Runtime)
    rt.agent = agent
    rt.session = session
    rt.guard = CostGuard()
    rt.printer = lambda ev: events.append(ev.type)
    rt._system_saved = True
    rt.turns = 0
    rt.tool_calls = 0
    agent.hooks.on_event = rt._on_event

    asyncio.run(agent.run("go"))
    assert "user_message" in events  # 插话作为事件发出
    assert "插话内容X" in session.path.read_text(encoding="utf-8")  # 已落进会话树
