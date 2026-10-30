"""MCP 客户端 —— 把 stdio MCP 服务器的工具桥接为本地工具（对照 Pi/Claude Code 的 MCP 接入）。

配置文件（mcp.json 或 .you/mcp.json，也可 --mcp 显式指定）：
  {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}

桥接规则：
- 远端工具重命名为 mcp_<服务器名>_<工具名>（防跨服务器重名）；
- schema 用服务器返回的 inputSchema 原样透传，不做本地 pydantic 校验
  （参数合法性由服务器判定，_AnyArgs 仅做透传容器）；
- 服务器进程生命周期挂在 AsyncExitStack 上：会话结束自动关闭。
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .types import BaseTool, ToolResult


class _AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class McpTool(BaseTool):
    """远端 MCP 工具的本地代理。"""

    def __init__(self, server: str, tool_def, session):
        self.server = server
        self.remote_name = tool_def.name
        self.session = session
        self.name = f"mcp_{server}_{tool_def.name}"
        desc = getattr(tool_def, "description", "") or tool_def.name
        self.description = f"[MCP:{server}] {desc}"[:800]
        # mcp SDK 的 pydantic 字段是蛇形 input_schema，JSON 层才用驼峰 inputSchema
        self._input_schema = (
            getattr(tool_def, "input_schema", None)
            or getattr(tool_def, "inputSchema", None)
            or {"type": "object", "properties": {}}
        )
        self.args_model = _AnyArgs

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._input_schema,
            },
        }

    async def execute(self, tool_call_id: str, params: dict) -> ToolResult:
        try:
            res = await asyncio.wait_for(
                self.session.call_tool(self.remote_name, arguments=params), timeout=120
            )
        except Exception as e:  # noqa: BLE001 远端服务器错误转工具结果，让模型自行处理
            return ToolResult(
                content=f"MCP 调用失败（{self.server}/{self.remote_name}）："
                        f"{type(e).__name__}: {e}",
                is_error=True,
            )
        texts = [getattr(b, "text", "") for b in (getattr(res, "content", None) or [])]
        body = "\n".join(t for t in texts if t) or "(服务器无文本输出)"
        return ToolResult(content=body, is_error=bool(getattr(res, "isError", False)))


def load_mcp_config(path: Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        raise TypeError('mcp.json 格式应为 {"mcpServers": {名称: {command, args?, env?}}}')
    return servers


def find_mcp_config(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit
    for cand in (Path.cwd() / ".you" / "mcp.json", Path.cwd() / "mcp.json"):
        if cand.exists():
            return cand
    return None


async def attach_mcp_tools(
    agent, stack: AsyncExitStack, servers: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """逐个启动 stdio 服务器并桥接其工具。返回 (成功描述列表, 失败原因列表)。"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    attached: list[str] = []
    errors: list[str] = []
    for name, cfg in servers.items():
        try:
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env")
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            for td in listing.tools:
                agent.add_tools([McpTool(name, td, session)])
            attached.append(f"{name}({len(listing.tools)} 个工具)")
        except Exception as e:  # noqa: BLE001 单服务器失败不影响其余
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return attached, errors
