"""消息模型与工具协议 —— 对照 Pi 的 packages/ai/src/types.ts 与 packages/agent/src/types.ts。

设计对齐点：
- 消息四角色 System/User/Assistant/ToolResult，Assistant 内容为块列表（文本/思考/工具调用）；
- 工具 = name + description + pydantic 参数模型（model_json_schema() 即 JSON Schema，
  pydantic 宽松校验自动完成"模型输出 '5' -> 5"的类型纠正，对照 Pi validation.ts）；
- 事件流分两层：StreamEvent（供应商流）与 AgentEvent（循环生命周期），对照 Pi 的
  AssistantMessageEvent / AgentEvent。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------- 消息模型 ----------


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


StopReason = Literal["stop", "length", "tool_calls", "error", "aborted"]


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolCall(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


ContentBlock = TextContent | ThinkingContent | ToolCall


class SystemMessage(BaseModel):
    type: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    type: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    type: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    stop_reason: StopReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    duration_ms: float = 0.0

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextContent))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]


class ToolResultMessage(BaseModel):
    type: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage

# ---------- 工具协议（对照 Pi 的 AgentTool） ----------


class ToolResult(BaseModel):
    content: str
    is_error: bool = False


class BaseTool:
    """内置工具基类：参数模型用 pydantic 定义，schema 自动导出，校验自动完成。

    execution_mode（对照 Pi 的 ToolExecutionMode）："parallel" 可与同批只读工具并发；
    "sequential" 强制所在批次串行（写类工具应声明，避免并发写竞争）。
    """

    name: str = ""
    description: str = ""
    args_model: type[BaseModel]
    execution_mode: str = "parallel"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


Tool = BaseTool  # 类型别名：文档与签名里用 Tool 指代任何工具实现


# ---------- 事件 ----------


class StreamEvent(BaseModel):
    """供应商流事件（对照 Pi 的 AssistantMessageEvent，取其常用子集）。"""

    type: Literal["text_delta", "thinking_delta", "tool_call_delta", "message_end"]
    delta: str = ""
    message: AssistantMessage | None = None


class AgentEvent(BaseModel):
    """循环生命周期事件（对照 Pi 的 AgentEvent）。message_update 的 kind 区分增量类型。"""

    type: str  # agent_start/end, turn_start/end, message_start, message_update, message_end, tool_result, tool_execution_start/end, compaction
    delta: str = ""
    kind: str = ""  # text | thinking | tool（仅 message_update 使用）
    message: Message | None = None  # Assistant/ToolResult/用户插话消息都可能出现
    tool_call: ToolCall | None = None
    is_error: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)  # compaction 等自定义事件的载荷
