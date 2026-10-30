"""供应商适配层 —— 对照 Pi 的 packages/ai。

LiteLLM 负责 100+ 家供应商的协议归一（对照 pi-ai 的 42 家 / 10 种 wire 协议），
本模块只做两件事：
1. 把统一消息模型转成 LiteLLM/OpenAI 格式（to_litellm_messages）；
2. 把 LiteLLM 流式 chunk 归一成 StreamEvent，并在流结束时给出完整 AssistantMessage
   （工具参数的增量累积在这里完成，对照 pi streaming-json.ts 的职责）。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Sequence

# 直接用本地模型价格表：跳过 litellm 启动时对 github 价格文件的联网拉取（必须在 import litellm 前设置）
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    StreamEvent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    Usage,
)

litellm.suppress_debug_info = True


def to_litellm_messages(messages: Sequence[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.type == "system":
            out.append({"role": "system", "content": m.content})
        elif m.type == "user":
            out.append({"role": "user", "content": m.content})
        elif m.type == "assistant":
            text = m.text
            calls = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
                }
                for c in m.tool_calls
            ]
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif m.type == "toolResult":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


async def litellm_stream(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[Tool] | None = None,
    api_base: str | None = None,
) -> AsyncIterator[StreamEvent]:
    """统一流函数：吃统一消息，吐 StreamEvent，最后一条是 message_end（完整消息）。"""
    started = time.time()
    kwargs: dict = {
        "model": model,
        "messages": to_litellm_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = [t.schema() for t in tools]
    if api_base:
        kwargs["api_base"] = api_base

    resp = await litellm.acompletion(**kwargs)

    text_parts: list[str] = []
    think_parts: list[str] = []
    tool_acc: dict[int, dict] = {}  # 流式 index -> {id, name, args}
    usage = Usage()
    finish = "stop"

    async for chunk in resp:
        u = getattr(chunk, "usage", None)
        if u is not None:
            usage = Usage(
                input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
            )
        for choice in chunk.choices or []:
            d = choice.delta
            if d is None:
                continue
            reasoning = getattr(d, "reasoning_content", None)  # DeepSeek reasoner 等的思考流
            if reasoning:
                think_parts.append(reasoning)
                yield StreamEvent(type="thinking_delta", delta=reasoning)
            if d.content:
                text_parts.append(d.content)
                yield StreamEvent(type="text_delta", delta=d.content)
            for tc in d.tool_calls or []:
                slot = tool_acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
                yield StreamEvent(type="tool_call_delta", delta=slot["name"])
            if choice.finish_reason:
                finish = choice.finish_reason

    content: list[ContentBlock] = []
    if think_parts:
        content.append(ThinkingContent(thinking="".join(think_parts)))
    if text_parts:
        content.append(TextContent(text="".join(text_parts)))
    for idx in sorted(tool_acc):
        slot = tool_acc[idx]
        try:
            args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            args = {}  # 参数 JSON 不完整：置空并继续，由循环层的校验兜底
        content.append(ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"], arguments=args))

    stop = {"tool_calls": "tool_calls", "length": "length", "stop": "stop"}.get(finish, "stop")
    msg = AssistantMessage(
        content=content,
        stop_reason=stop,
        usage=usage,
        model=model,
        duration_ms=(time.time() - started) * 1000,
    )
    yield StreamEvent(type="message_end", message=msg)


class FauxStream:
    """脚本化假模型（对照 Pi 的 faux provider）：按脚本顺序吐预设消息，用于无 Key 测试。

    每次被调用消费脚本中的一条；calls 记录每次请求收到的上下文，便于测试断言。
    """

    def __init__(self, script: list[AssistantMessage]):
        self.script = list(script)
        self.i = 0
        self.calls: list[list[Message]] = []

    async def __call__(
        self,
        model: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        if self.i >= len(self.script):
            raise RuntimeError("faux 脚本已耗尽")
        msg = self.script[self.i]
        self.i += 1
        for block in msg.content:
            if isinstance(block, TextContent):
                for k in range(0, len(block.text), 40):
                    yield StreamEvent(type="text_delta", delta=block.text[k : k + 40])
            elif isinstance(block, ToolCall):
                yield StreamEvent(type="tool_call_delta", delta=block.name)
        yield StreamEvent(type="message_end", message=msg.model_copy(deep=True))


# 上下文溢出报错文案匹配（对照 Pi overflow.ts 思路，MVP 覆盖常见供应商的错误形态）
_OVERFLOW_PATTERNS = (
    "maximum context length",
    "context_length_exceeded",
    "context length exceeded",
    "prompt is too long",
    "context window is too long",
    "input length and `max_tokens` exceed",
    "exceeds the context window",
    "too many tokens",
)


def is_context_overflow(e: BaseException) -> bool:
    s = str(e).lower()
    return any(p in s for p in _OVERFLOW_PATTERNS)


def model_context_window(model: str) -> int | None:
    """从 litellm 模型目录查询上下文窗口；查询不到返回 None（调用方自行回退）。"""
    try:
        from litellm import get_max_tokens

        v = get_max_tokens(model)
        return int(v) if v else None
    except Exception:  # noqa: BLE001 目录未收录/离线均可回退
        return None
