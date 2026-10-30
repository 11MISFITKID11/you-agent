"""后台任务管理 —— 对照 Claude Code 的后台 shell：长命令不阻塞 agent 循环。

用法：bash 工具传 run_in_background=true 立即拿任务 id；用 jobs 工具
list / check / kill 轮询。输出落系统临时文件 you-job-<id>.log，进程退出后句柄关闭。
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .types import BaseTool, ToolResult


class JobManager:
    def __init__(self, keep: int = 20):
        self.keep = keep
        self.jobs: dict[str, dict] = {}

    def start(self, command: str, cwd: str | None = None, timeout_hint: int = 600) -> str:
        jid = uuid.uuid4().hex[:8]
        out_path = Path(tempfile.gettempdir()) / f"you-job-{jid}.log"
        f = out_path.open("wb")
        try:
            proc = subprocess.Popen(
                command, shell=True, cwd=cwd or None,
                stdout=f, stderr=subprocess.STDOUT,
            )
        except OSError as e:
            f.close()
            raise RuntimeError(f"后台命令启动失败：{e}") from e
        self.jobs[jid] = {
            "command": command, "proc": proc, "out": out_path, "fh": f,
            "started": time.time(), "timeout_hint": timeout_hint,
        }
        self._trim()
        return jid

    def _sweep(self) -> None:
        for info in self.jobs.values():
            if info["proc"].poll() is not None and not info["fh"].closed:
                info["fh"].close()

    def status(self, jid: str | None = None) -> list[dict]:
        self._sweep()
        if jid is not None:
            items = [(jid, self.jobs[jid])] if jid in self.jobs else []
        else:
            items = list(self.jobs.items())
        return [
            {
                "id": j,
                "command": i["command"],
                "running": i["proc"].poll() is None,
                "exit": i["proc"].poll(),
                "seconds": round(time.time() - i["started"], 1),
            }
            for j, i in items
        ]

    def tail(self, jid: str, max_chars: int = 2000) -> str:
        self._sweep()
        info = self.jobs.get(jid)
        if info is None:
            return ""
        try:
            text = info["out"].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def kill(self, jid: str) -> bool:
        info = self.jobs.get(jid)
        if info is None or info["proc"].poll() is not None:
            return False
        info["proc"].terminate()
        return True

    def _trim(self) -> None:
        finished = [j for j, i in self.jobs.items() if i["proc"].poll() is not None]
        while len(self.jobs) > self.keep and finished:
            j = finished.pop(0)
            info = self.jobs.pop(j)
            if not info["fh"].closed:
                info["fh"].close()


JOB_MANAGER = JobManager()


class JobsArgs(BaseModel):
    action: Literal["list", "check", "kill"] = Field(default="list")
    id: str | None = Field(default=None, description="check/kill 需要的任务 id")


class JobsTool(BaseTool):
    execution_mode = "sequential"
    name = "jobs"
    description = (
        "查看/管理后台任务：action=list 列出全部；action=check&id=… 返回该任务状态与输出尾部；"
        "action=kill&id=… 终止运行中的任务。长命令（构建/安装/测试）建议 bash 传 "
        "run_in_background=true 后用本工具轮询，而不是傻等。"
    )
    args_model = JobsArgs

    async def execute(self, tool_call_id: str, params: dict) -> ToolResult:
        m = JOB_MANAGER

        def _state(r: dict) -> str:
            return "运行中" if r["running"] else f"已结束(exit={r['exit']})"

        if params["action"] == "list":
            rows = m.status()
            if not rows:
                return ToolResult(content="（没有后台任务）")
            body = "\n".join(
                f"[{r['id']}] {_state(r)} {r['seconds']}s {r['command'][:80]}" for r in rows
            )
            return ToolResult(content=body)
        jid = params.get("id") or ""
        if jid not in m.jobs:
            return ToolResult(content=f"任务不存在：{jid}", is_error=True)
        if params["action"] == "kill":
            ok = m.kill(jid)
            return ToolResult(content="已发送终止信号" if ok else "任务已结束，无需终止",
                              is_error=not ok)
        st = m.status(jid)[0]
        tail = m.tail(jid) or "(暂无输出)"
        return ToolResult(
            content=f"[{jid}] {_state(st)} {st['seconds']}s\n"
                    f"命令：{st['command']}\n输出尾部：\n{tail}"
        )
