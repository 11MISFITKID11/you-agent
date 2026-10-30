"""成本护栏 —— 对照 WidelCode 的单次运行成本上限（三档：提醒 → 暂停 → 停止）。

数据源：AssistantMessage.usage（LiteLLM 从供应商流里归一的 token 用量）。
挂点：cli 的事件钩子在 message_end 时累计；循环层在轮与轮之间检查 abort。
"""

from __future__ import annotations

from .types import AssistantMessage

# 参考单价（美元 / 百万 token，2026-10 参考值，仅用于展示估算，以官网为准）
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-chat": (0.27, 1.10),
    "deepseek/deepseek-reasoner": (0.55, 2.19),
    "faux": (0.0, 0.0),
}


class CostGuard:
    """会话级用量护栏。阈值单位：累计 token（输入+输出）。"""

    def __init__(self, warn: int = 200_000, pause: int = 500_000, stop: int = 1_000_000):
        self.warn, self.pause, self.stop = warn, pause, stop
        self.session_tokens = 0
        self.per_model: dict[str, list[int]] = {}  # model -> [input, output]

    def on_message_end(self, message: AssistantMessage) -> None:
        u = message.usage
        self.session_tokens += u.input_tokens + u.output_tokens
        slot = self.per_model.setdefault(message.model or "?", [0, 0])
        slot[0] += u.input_tokens
        slot[1] += u.output_tokens

    @property
    def level(self) -> str:
        if self.session_tokens >= self.stop:
            return "stop"
        if self.session_tokens >= self.pause:
            return "pause"
        if self.session_tokens >= self.warn:
            return "warn"
        return "ok"

    def should_abort(self) -> bool:
        """达到 pause 档即请求中止：当前工具批次完成后停下，上下文保留，用户可选择继续。"""
        return self.session_tokens >= self.pause

    def est_cost_usd(self) -> float:
        total = 0.0
        for model, (i, o) in self.per_model.items():
            pi, po = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
            total += (i * pi + o * po) / 1_000_000
        return total

    def summary(self) -> str:
        rows = [f"{m}: 输入 {i:,} / 输出 {o:,}" for m, (i, o) in self.per_model.items()]
        return (
            f"会话累计 {self.session_tokens:,} tokens（阈值 提醒 {self.warn:,} / "
            f"暂停 {self.pause:,} / 停止 {self.stop:,}），当前档位 [{self.level}]，"
            f"估算成本 ${self.est_cost_usd():.4f}"
            + (("\n  " + "\n  ".join(rows)) if rows else "")
        )
