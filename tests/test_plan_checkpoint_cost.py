"""方案书三件核心能力的测试：计划模式（只读强制）/ 中止机制 / 成本护栏 / 检查点 / 会话分支重建。"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest
from pydantic import BaseModel

from you_agent.checkpoint import CheckpointManager
from you_agent.govern import CostGuard
from you_agent.loop import Agent, Hooks
from you_agent.providers import FauxStream
from you_agent.session import SessionManager
from you_agent.tools.builtin import EditTool, ReadTool, WriteTool, read_only_tools
from you_agent.types import (
    AssistantMessage,
    BaseTool,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _make_agent(script, tools=None, hooks=None):
    faux = FauxStream(script)
    h = hooks or Hooks()
    agent = Agent(stream_fn=faux, tools=tools or [ReadTool(), WriteTool(), EditTool()],
                  system_prompt="t", model="faux", hooks=h)
    return agent, faux


# ---------- 计划模式：只读强制 ----------


def test_plan_mode_readonly_enforcement(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    script = [
        AssistantMessage(content=[
            ToolCall(id="c1", name="read", arguments={"path": str(f)}),
            ToolCall(id="c2", name="write", arguments={"path": str(tmp_path / "x.txt"), "content": "x"}),
        ], stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="done")]),
    ]
    agent, _ = _make_agent(script)
    agent.set_active_tools([t.name for t in read_only_tools()])  # /plan 做的事

    asyncio.run(agent.run("调研一下"))

    results = {m.tool_name: m for m in agent.messages if isinstance(m, ToolResultMessage)}
    assert not results["read"].is_error            # 只读工具放行
    assert results["write"].is_error               # 写工具被挡在活动集之外
    assert "未知工具" in results["write"].content
    agent.set_active_tools(None)                   # /act 做的事：恢复全量
    assert set(agent.tools) == {"read", "write", "edit"}


# ---------- 中止机制（护栏暂停的挂点） ----------


class _EmptyArgs(BaseModel):
    pass


class AbortTool(BaseTool):
    name = "abort_now"
    description = "测试用：请求中止循环"
    args_model = _EmptyArgs

    def __init__(self, agent_ref):
        self.agent_ref = agent_ref

    async def execute(self, tool_call_id, params):
        self.agent_ref.abort()
        return ToolResult(content="abort requested")


def test_abort_stops_between_turns():
    script = [
        AssistantMessage(content=[ToolCall(id="c1", name="abort_now", arguments={})],
                         stop_reason="tool_calls"),
        AssistantMessage(content=[TextContent(text="不应被调用到")]),
    ]
    agent, faux = _make_agent(script, tools=[])  # 工具稍后手动注入（需要 agent 引用）
    agent.all_tools["abort_now"] = AbortTool(agent)
    agent.tools["abort_now"] = agent.all_tools["abort_now"]

    final = asyncio.run(agent.run("go"))

    assert final.stop_reason == "aborted"
    assert len(faux.calls) == 1  # 第二轮 LLM 调用被拦下


# ---------- 成本护栏 ----------


def _msg(i, o, model="deepseek/deepseek-chat"):
    return AssistantMessage(usage=Usage(input_tokens=i, output_tokens=o), model=model)


def test_cost_guard_levels_and_cost():
    g = CostGuard(warn=100, pause=200, stop=300)
    assert g.level == "ok"
    g.on_message_end(_msg(60, 50))                 # 110
    assert g.level == "warn"
    g.on_message_end(_msg(50, 50))                 # 210
    assert g.level == "pause"
    assert g.should_abort()
    g.on_message_end(_msg(50, 50))                 # 310
    assert g.level == "stop"
    # 成本估算：deepseek-chat 输入 0.27 $/M、输出 1.10 $/M
    # 累计输入 160、输出 150 -> 160*0.27/1e6 + 150*1.10/1e6
    assert g.est_cost_usd() == pytest.approx(160 * 0.27 / 1e6 + 150 * 1.10 / 1e6)


# ---------- 检查点（git） ----------


git_missing = shutil.which("git") is None


@pytest.mark.skipif(git_missing, reason="git 不可用")
def test_checkpoint_roundtrip(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-qm", "init")

    cp = CheckpointManager(repo_dir=tmp_path)
    assert cp.available
    (tmp_path / "a.txt").write_text("v2", encoding="utf-8")  # 用户未提交的改动也是快照的一部分
    ref = cp.create(label="before run")
    assert ref and ref.startswith("refs/you-cp/")

    (tmp_path / "a.txt").write_text("v3-agent-edit", encoding="utf-8")  # 模拟 agent 改了文件
    ok, _ = cp.restore(ref)
    assert ok
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "v2"  # 回到运行前快照


def test_checkpoint_unavailable_outside_git(tmp_path):
    cp = CheckpointManager(repo_dir=tmp_path)
    assert not cp.available
    assert cp.create() is None


# ---------- 会话分支重建（/rewind 的数据基础） ----------


def test_session_branch_and_rebuild(tmp_path):
    s = SessionManager(path=tmp_path / "s.jsonl")
    s.append(SystemMessage(content="sys"))
    s.append(UserMessage(content="q1"))
    aid = s.append(AssistantMessage(content=[TextContent(text="a1")]))
    s.append(UserMessage(content="q2-bad-branch"))      # 这条走错了
    s.checkout(aid)                                      # 回到 a1 分叉
    bid = s.append(UserMessage(content="q3-good"))

    last = s.entries[-1]
    assert last["parentId"] == aid                       # 树：从 a1 分叉，旧分支保留
    msgs = s.messages_up_to(bid)
    assert [m.content if m.type != "assistant" else m.text for m in msgs] == ["sys", "q1", "a1", "q3-good"]
    assert all(m.type != "user" or m.content != "q2-bad-branch" for m in msgs)
