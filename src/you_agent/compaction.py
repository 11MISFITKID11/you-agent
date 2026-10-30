"""上下文压缩管线 —— 对照 Pi 的 packages/coding-agent/src/core/compaction/compaction.ts。

四步：估算（estimate）→ 判断（should_compact）→ 找切割点（find_cut_point）→ 摘要替换（compact）。

关键约束（面试考点）：
- 切割点必须落在 UserMessage 边界——保证 AssistantMessage 里的工具调用与其
  ToolResultMessage 的配对不被切开（切散会被供应商 API 直接拒掉）；
- 领头的系统提示词（含计划模式注入）永远保留，只压缩中段历史；
- 摘要本身由同一个模型生成（自压缩），替换后的树形会话里另存 compact 条目可回溯。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .types import AssistantMessage, Message, SystemMessage, UserMessage

SUMMARIZE_PROMPT = (
    "你是对话压缩器。把下面的编码智能体对话历史压缩成一段简洁摘要，必须保留："
    "任务目标、已完成的改动（含文件路径）、关键决策与原因、未完成事项、重要的错误与修复。"
    "直接输出摘要正文，不要任何前后缀。"
)


def estimate_tokens(text: str) -> int:
    """免 tokenizer 的启发式估算：CJK 按每字符约 1 token，其余按约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_message_tokens(m: Message) -> int:
    return estimate_tokens(json.dumps(m.model_dump(), ensure_ascii=False))


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _leading_system_count(messages: Sequence[Message]) -> int:
    n = 0
    for m in messages:
        if m.type == "system":
            n += 1
        else:
            break
    return n


def find_cut_point(messages: Sequence[Message], keep_tokens: int) -> int | None:
    """返回切割下标 cut：messages[cut:] 为保留的近期尾部，且 messages[cut] 必须是 UserMessage。

    返回 None 表示没有可行的切割点（尾部已超预算 / 边界推进到了末尾）。
    """
    prefix = _leading_system_count(messages)
    ests = [estimate_message_tokens(m) for m in messages]

    cut = len(messages)
    suffix = 0
    for i in range(len(messages) - 1, -1, -1):
        suffix += ests[i]
        if suffix > keep_tokens:
            cut = i + 1
            break
    else:
        return None  # 整段都在保留预算内，无需压缩

    cut = max(cut, prefix + 1)  # 至少要有一条非系统消息可被压缩
    while cut < len(messages) and messages[cut].type != "user":
        cut += 1  # 推进到 UserMessage 边界，保证工具调用/结果配对不被切开
    if cut >= len(messages):
        return None
    return cut


class Compactor:
    """挂在 Agent 循环里：每次 LLM 调用前检查，超阈值就把中段历史替换为模型生成的摘要。"""

    def __init__(
        self,
        stream_fn,
        model: str,
        context_window: int = 64_000,
        threshold: float = 0.8,
        keep_fraction: float = 0.4,
        min_gain_fraction: float = 0.2,
    ):
        self.stream_fn = stream_fn
        self.model = model
        self.context_window = context_window
        self.threshold = threshold
        self.keep_fraction = keep_fraction
        self.min_gain_fraction = min_gain_fraction

    def should_compact(self, messages: Sequence[Message]) -> bool:
        return estimate_messages_tokens(messages) >= int(self.context_window * self.threshold)

    async def compact(self, agent, force: bool = False) -> bool:
        """执行压缩并整体替换 agent.messages。返回 False 表示本次未压缩。

        force=True 供溢出自愈路径使用：跳过阈值与最小收益检查（能减一点是一点）。
        """
        msgs = agent.messages
        if not force and not self.should_compact(msgs):
            return False
        if force:
            # 溢出自愈：不知道供应商真实窗口，只保留当前估算量的 30%，保证有切割点且必然减量
            keep = max(500, int(estimate_messages_tokens(msgs) * 0.3))
        else:
            keep = int(self.context_window * self.keep_fraction)
        cut = find_cut_point(msgs, keep)
        if cut is None:
            return False
        prefix = _leading_system_count(msgs)
        seg = msgs[prefix:cut]
        gain = estimate_messages_tokens(seg)
        if not force and gain < int(self.context_window * self.min_gain_fraction):
            return False  # 可压的量太小，不值得多花一次摘要调用

        transcript = _render(seg)
        summary: AssistantMessage | None = None
        async for ev in self.stream_fn(
            self.model,
            [SystemMessage(content=SUMMARIZE_PROMPT), UserMessage(content=transcript)],
            [],
        ):
            if ev.type == "message_end":
                summary = ev.message
        if summary is None or not summary.text:
            return False

        before = estimate_messages_tokens(msgs)
        marker = SystemMessage(content=f"[早期对话摘要 | 覆盖 {len(seg)} 条消息]\n{summary.text}")
        agent.messages = msgs[:prefix] + [marker] + msgs[cut:]
        after = estimate_messages_tokens(agent.messages)
        agent.emit(
            "compaction",
            payload={
                "removed": len(seg),
                "before_tokens": before,
                "after_tokens": after,
                "summary_head": summary.text[:120],
            },
        )
        return True


def _render(messages: Sequence[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        if m.type == "system":
            lines.append(f"[system] {m.content}")
        elif m.type == "user":
            lines.append(f"[user] {m.content}")
        elif m.type == "assistant":
            for b in m.content:
                if b.type == "text":
                    lines.append(f"[assistant] {b.text}")
                elif b.type == "toolCall":
                    lines.append(f"[assistant 调用工具] {b.name}({json.dumps(b.arguments, ensure_ascii=False)})")
        elif m.type == "toolResult":
            flag = "错误" if m.is_error else "结果"
            lines.append(f"[工具{flag}] {m.tool_name}: {m.content[:2000]}")
    return "\n".join(lines)
