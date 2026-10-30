"""Agent 循环 —— 对照 Pi 的 packages/agent/src/agent-loop.ts。

复刻的关键语义：
- 双层循环：内层"还有工具调用就继续"；外层由 finish_turn 钩子决定是否再跑一轮；
- 截断防御：stop_reason == "length" 时工具调用一律不执行，以错误结果回填
  （对照 failToolCallsFromTruncatedMessage——截断时参数 JSON 可能不完整）；
- 五个钩子：before_tool_call（可阻断）、after_tool_call（可改写结果）、
  transform_context（LLM 调用前裁剪，compaction 的挂点）、prepare_request、finish_turn；
- 工具执行异常/参数校验失败一律转为 is_error 结果回填，让模型自行补救。
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Literal

from .providers import is_context_overflow
from .types import (
    AgentEvent,
    AssistantMessage,
    Message,
    StreamEvent,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)

BlockDecision = dict | None  # {"block": True, "reason": "..."} 表示阻断


class Hooks:
    def __init__(
        self,
        before_tool_call: Callable[[ToolCall], BlockDecision] | None = None,
        after_tool_call: Callable[[ToolCall, ToolResult], ToolResult] | None = None,
        transform_context: Callable[[list[Message]], list[Message]] | None = None,
        prepare_request: Callable[[list[Message]], list[Message]] | None = None,
        finish_turn: Callable[[], Literal["continue"] | None] | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
    ):
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.transform_context = transform_context
        self.prepare_request = prepare_request
        self.finish_turn = finish_turn
        self.on_event = on_event


class Agent:
    """有状态包装：消息历史 + 工具表 + 循环（对照 pi agent.ts 的 Agent 类）。"""

    def __init__(
        self,
        stream_fn: Callable[..., AsyncIterator[StreamEvent]],
        tools: Sequence[Tool],
        system_prompt: str,
        model: str = "deepseek/deepseek-chat",
        hooks: Hooks | None = None,
        max_turns: int = 40,
        compactor=None,  # Compactor（鸭子类型：should_compact/compact），避免内核依赖压缩模块
    ):
        self.stream_fn = stream_fn
        self.all_tools: dict[str, Tool] = {t.name: t for t in tools}
        self.tools: dict[str, Tool] = dict(self.all_tools)
        self.model = model
        self.hooks = hooks or Hooks()
        self.messages: list[Message] = [SystemMessage(content=system_prompt)]
        self.max_turns = max_turns
        self.compactor = compactor
        self._abort = False
        self._steering: deque[UserMessage] = deque()  # 运行中插话队列（对照 Pi PendingMessageQueue）

    def set_active_tools(self, names: list[str] | None = None) -> None:
        """切换活动工具集（names=None 恢复全量）。计划模式的只读强制就靠它。"""
        if names is None:
            self.tools = dict(self.all_tools)
        else:
            self.tools = {n: self.all_tools[n] for n in names if n in self.all_tools}

    def add_tools(self, tools: Sequence[Tool]) -> None:
        """运行期注册新工具（MCP 桥接、扩展注入用），同时进全量表与活动表。"""
        for t in tools:
            self.all_tools[t.name] = t
            self.tools[t.name] = t

    def abort(self) -> None:
        """请求中止：当前工具批次执行完后，循环在下一轮开始前停下（对照护栏暂停语义）。"""
        self._abort = True

    def steer(self, text: str) -> None:
        """运行中插话（对照 Pi 的 steering）：下一轮 LLM 调用前注入；若 run 已到末轮则复活。"""
        self._steering.append(UserMessage(content=text))

    def _drain_steering(self) -> None:
        while self._steering:
            m = self._steering.popleft()
            self.messages.append(m)
            self.emit("user_message", message=m)

    def emit(self, type_: str, **kw) -> None:
        if self.hooks.on_event:
            self.hooks.on_event(AgentEvent(type=type_, **kw))

    async def run(self, user_input: str) -> AssistantMessage:
        self._abort = False
        self.messages.append(UserMessage(content=user_input))
        return await self._loop()

    async def _loop(self) -> AssistantMessage:
        self.emit("agent_start")
        final = AssistantMessage()
        turn = 0
        while True:
            if self._abort:
                final = AssistantMessage(
                    content=[TextContent(text="[you-agent] 收到中止请求，本轮已停止。上下文完整保留，可继续对话。")],
                    stop_reason="aborted",
                )
                self.messages.append(final)
                self.emit("message_end", message=final)
                break
            turn += 1
            if turn > self.max_turns:
                final = AssistantMessage(
                    content=[TextContent(text=f"[you-agent] 达到最大轮数 {self.max_turns}，已强制停止。")],
                    stop_reason="error",
                )
                self.messages.append(final)
                self.emit("message_end", message=final)
                break
            self.emit("turn_start")

            # steering：用户在运行中的插话在下一轮 LLM 调用前注入（对照 Pi PendingMessageQueue）
            self._drain_steering()

            # 压缩检查点（对照 Pi prepareNextTurn 里的 compaction 时机）：每次 LLM 调用前判断
            if self.compactor is not None and self.compactor.should_compact(self.messages):
                try:
                    await self.compactor.compact(self)
                except Exception:  # noqa: BLE001, S110 压缩失败不阻塞主流程，下一轮还会再试
                    pass

            ctx = list(self.messages)
            if self.hooks.transform_context:
                ctx = self.hooks.transform_context(ctx)
            if self.hooks.prepare_request:
                ctx = self.hooks.prepare_request(ctx)

            self.emit("message_start")
            msg: AssistantMessage | None = None
            for llm_attempt in (1, 2):  # 溢出自愈：第一次溢出 -> 强制压缩 -> 重试一次（对照 Pi 溢出恢复）
                try:
                    async for ev in self.stream_fn(self.model, ctx, list(self.tools.values())):
                        if ev.type == "message_end":
                            msg = ev.message
                        elif ev.type == "text_delta":
                            self.emit("message_update", delta=ev.delta, kind="text")
                        elif ev.type == "thinking_delta":
                            self.emit("message_update", delta=ev.delta, kind="thinking")
                        elif ev.type == "tool_call_delta":
                            self.emit("message_update", delta=ev.delta, kind="tool")
                    break
                except Exception as e:
                    if (
                        llm_attempt == 1
                        and self.compactor is not None
                        and is_context_overflow(e)
                        and await self.compactor.compact(self, force=True)
                    ):
                        ctx = list(self.messages)  # 压缩后重建请求上下文
                        continue
                    raise
            if msg is None:
                msg = AssistantMessage(
                    content=[TextContent(text="[you-agent] 供应商未返回有效消息")], stop_reason="error"
                )
            self.messages.append(msg)
            self.emit("message_end", message=msg)

            calls = msg.tool_calls
            # 截断防御：输出被 max_tokens 截断时，工具调用不执行（参数可能不完整）
            if msg.stop_reason == "length" and calls:
                for c in calls:
                    r = ToolResultMessage(
                        tool_call_id=c.id,
                        tool_name=c.name,
                        content="响应因达到 max_tokens 被截断，工具调用未执行。请精简输出或分步执行后重试。",
                        is_error=True,
                    )
                    self.messages.append(r)
                    self.emit("tool_execution_end", tool_call=c, is_error=True)
                final = msg
                self.emit("turn_end")
                break

            if not calls:
                final = msg
                self.emit("turn_end")
                if self._steering:  # 插话到达时即使模型想结束也复活一轮（Pi 的 steering 语义）
                    continue
                if self.hooks.finish_turn and self.hooks.finish_turn() == "continue":
                    continue
                break

            # 批量执行（对照 Pi executeToolCallsSequential/Parallel）：
            # 批次里含 sequential 工具（写类）时整批串行避免竞争；否则只读工具并发
            modes = [getattr(self.tools.get(c.name), "execution_mode", "parallel") for c in calls]
            if len(calls) == 1 or any(m == "sequential" for m in modes):
                for c in calls:
                    r = await self._exec(c)
                    self.messages.append(r)
                    self.emit("tool_result", message=r)
                    self.emit("tool_execution_end", tool_call=c, is_error=r.is_error)
            else:
                results = await asyncio.gather(*(self._exec(c) for c in calls))
                for c, r in zip(calls, results, strict=True):  # gather 保序，回填配对不乱
                    self.messages.append(r)
                    self.emit("tool_result", message=r)
                    self.emit("tool_execution_end", tool_call=c, is_error=r.is_error)
            self.emit("turn_end")
        self.emit("agent_end")
        return final

    async def _exec(self, c: ToolCall) -> ToolResultMessage:
        self.emit("tool_execution_start", tool_call=c)
        if self.hooks.before_tool_call:
            decision = self.hooks.before_tool_call(c)
            if inspect.isawaitable(decision):  # TUI 弹窗确认等异步闸门
                decision = await decision
            if decision and decision.get("block"):
                return ToolResultMessage(
                    tool_call_id=c.id,
                    tool_name=c.name,
                    content=f"该调用被闸门阻断：{decision.get('reason', '')}",
                    is_error=True,
                )
        tool = self.tools.get(c.name)
        if tool is None:
            return ToolResultMessage(
                tool_call_id=c.id,
                tool_name=c.name,
                content=f"未知工具：{c.name}。可用工具：{', '.join(sorted(self.tools))}",
                is_error=True,
            )
        try:
            # pydantic 宽松校验 + 类型纠正（对照 pi validation.ts）
            params = tool.args_model.model_validate(c.arguments).model_dump()
        except Exception as e:  # noqa: BLE001 参数不合法转错误结果，让模型自行修正
            return ToolResultMessage(
                tool_call_id=c.id, tool_name=c.name, content=f"参数校验失败：{e}", is_error=True
            )
        try:
            result = await tool.execute(c.id, params)
        except Exception as e:  # noqa: BLE001 工具异常同样转错误结果回填
            result = ToolResult(content=f"工具执行异常：{type(e).__name__}: {e}", is_error=True)
        if self.hooks.after_tool_call:
            result = self.hooks.after_tool_call(c, result)
        return ToolResultMessage(
            tool_call_id=c.id, tool_name=c.name, content=result.content, is_error=result.is_error
        )
