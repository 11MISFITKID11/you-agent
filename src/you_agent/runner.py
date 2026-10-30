"""任务执行与评测 —— 对照方案书方案⑤：任务集 / 隔离执行 / 脚本判定 / 报告 / 对比。

范式来自 SWE-bench：每个任务 = 提示词 + 判定命令（退出码 0 即通过），在仓库的
隔离副本里执行（git 仓库用 worktree，否则 copytree），互不污染。
执行器可注入（executor 参数），测试里用假执行器离线验证整条评测链路。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

Executor = Callable[[Path, str], Awaitable[tuple[int, str]]]

# Windows 下子进程默认 GBK 输出，判定脚本里的中文/符号会解码崩；统一强制 UTF-8
_UTF8_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


@dataclass
class Task:
    id: str
    prompt: str
    check: str  # 在隔离目录执行的判定命令，退出码 0 = 通过
    setup: str | None = None  # 可选：agent 执行前跑的准备命令
    tags: list[str] = field(default_factory=list)  # 能力维度标签（报告按此分组）

    @staticmethod
    def load_jsonl(path: Path | str) -> list[Task]:
        tasks = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tasks.append(Task(**json.loads(line)))
        return tasks


@dataclass
class TaskResult:
    task: Task
    passed: bool
    agent_exit: int
    check_output: str = ""
    duration_s: float = 0.0
    workdir: str = ""
    diff: str = ""
    error: str = ""
    stats: dict = field(default_factory=dict)  # agent 效率指标：turns/tool_calls/tokens


async def cli_executor(workdir: Path, prompt: str, model: str, task_timeout: int = 300) -> tuple[int, str]:
    """真实执行器：在 workdir 里起一个 you print 模式子进程跑任务，并让它落一份效率指标。"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "you_agent.cli", "-p", prompt, "--no-session", "--model", model,
        "--stats-file", "you-stats.json",
        cwd=str(workdir),
        env=_UTF8_ENV,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=task_timeout)
    except TimeoutError:
        proc.kill()
        return 124, "任务超时被终止"
    return proc.returncode or 0, out.decode("utf-8", "replace")[-4000:]


class TaskRunner:
    def __init__(
        self,
        model: str = "deepseek/deepseek-chat",
        executor: Executor | None = None,
        keep: bool = False,
        task_timeout: int = 300,
    ):
        self.model = model
        self.keep = keep
        self.task_timeout = task_timeout
        self.executor: Executor = executor or (
            lambda w, p: cli_executor(w, p, model, task_timeout)
        )

    def _isolate(self, src: Path, tag: str) -> Path:
        """给任务一个隔离副本：git 仓库用 worktree，否则 copytree。"""
        base = Path(tempfile.mkdtemp(prefix=f"you-eval-{tag}-"))
        if (src / ".git").exists():
            r = subprocess.run(
                ["git", "worktree", "add", "--detach", str(base), "HEAD"],
                cwd=src, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if r.returncode == 0:
                return base
        shutil.copytree(src, base, dirs_exist_ok=True)
        return base

    def _cleanup(self, src: Path, workdir: Path) -> None:
        if self.keep:
            return
        if (workdir / ".git").exists() and (src / ".git").exists():
            subprocess.run(["git", "worktree", "remove", "--force", str(workdir)],
                           cwd=src, capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)

    async def run_task(self, task: Task, src_repo: Path) -> TaskResult:
        workdir = self._isolate(src_repo, task.id)
        started = time.time()
        try:
            if task.setup:
                subprocess.run(task.setup, shell=True, cwd=workdir, capture_output=True, env=_UTF8_ENV)
            agent_exit, _out = await self.executor(workdir, task.prompt)
            check = subprocess.run(task.check, shell=True, cwd=workdir,
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", env=_UTF8_ENV)
            diff = ""
            if (workdir / ".git").exists():
                d = subprocess.run(["git", "diff"], cwd=workdir, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                diff = (d.stdout or "")[:8000]
            stats: dict = {}
            stats_path = workdir / "you-stats.json"
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    stats = {}
            return TaskResult(
                task=task,
                passed=(agent_exit == 0 and check.returncode == 0),
                agent_exit=agent_exit,
                check_output=(check.stdout + check.stderr)[-500:],
                duration_s=round(time.time() - started, 1),
                workdir=str(workdir),
                diff=diff,
                stats=stats,
            )
        except Exception as e:  # noqa: BLE001 评测 harness 不因单任务崩溃而中断
            return TaskResult(task=task, passed=False, agent_exit=-1, error=f"{type(e).__name__}: {e}",
                              duration_s=round(time.time() - started, 1), workdir=str(workdir))
        finally:
            self._cleanup(src_repo, workdir)


async def run_eval(
    tasks: list[Task], src_repo: Path, jobs: int = 1, **runner_kwargs
) -> list[TaskResult]:
    runner = TaskRunner(**runner_kwargs)
    sem = asyncio.Semaphore(max(1, jobs))

    async def one(t: Task) -> TaskResult:
        async with sem:
            return await runner.run_task(t, src_repo)

    return list(await asyncio.gather(*(one(t) for t in tasks)))


# ---------- 报告与对比 ----------


def summarize(results: list[TaskResult], model: str) -> dict:
    passed = sum(1 for r in results if r.passed)
    # 能力维度分组（对照方案书：评测不只通过率，还要按维度看短板）
    dims: dict[str, dict] = {}
    for r in results:
        for tag in (r.task.tags or ["未标注"]):
            d = dims.setdefault(tag, {"total": 0, "passed": 0})
            d["total"] += 1
            d["passed"] += int(r.passed)

    def _avg(key: str) -> float:
        vals = [r.stats.get(key, 0) for r in results if r.stats]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    return {
        "model": model,
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "total_duration_s": round(sum(r.duration_s for r in results), 1),
        "efficiency": {
            "avg_turns": _avg("turns"),
            "avg_tool_calls": _avg("tool_calls"),
            "avg_total_tokens": int(_avg("total_tokens")),
        },
        "dimensions": dims,
        "results": [
            {"id": r.task.id, "passed": r.passed, "duration_s": r.duration_s,
             "tags": r.task.tags, "stats": r.stats,
             "check_output": r.check_output[-200:], "error": r.error}
            for r in results
        ],
    }


def write_report(results: list[TaskResult], model: str, out_dir: Path) -> tuple[Path, Path]:
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = summarize(results, model)
    jpath = out_dir / f"report-{ts}.json"
    mpath = out_dir / f"report-{ts}.md"
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    dim_rows = "\n".join(
        f"| {tag} | {d['passed']}/{d['total']} |" for tag, d in data["dimensions"].items()
    )
    rows = "\n".join(
        f'| {r.task.id} | {"✅ 通过" if r.passed else "❌ 失败"} | {r.duration_s}s '
        f'| {"、".join(r.task.tags) or "-"} '
        f'| {r.stats.get("turns", "-")}轮/{r.stats.get("tool_calls", "-")}工具/{r.stats.get("total_tokens", "-")}tok |'
        f'| {((r.check_output or r.error).strip().splitlines() or [""])[-1][:70]} |'
        for r in results
    )
    eff = data["efficiency"]
    mpath.write_text(
        f"# you-agent 评测报告\n\n- 模型：`{model}`　任务：{data['total']}　"
        f"通过：{data['passed']}（**{data['pass_rate']:.0%}**）　总耗时：{data['total_duration_s']}s\n"
        f"- 效率均值：{eff['avg_turns']} 轮 / {eff['avg_tool_calls']} 次工具调用 / "
        f"{eff['avg_total_tokens']} tokens 每任务\n\n"
        f"## 能力维度\n\n| 维度 | 通过 |\n|---|---|\n{dim_rows}\n\n"
        f"## 明细\n\n| 任务 | 结果 | 耗时 | 维度 | 效率 | 判定输出（末行） |\n|---|---|---|---|---|---|\n{rows}\n",
        encoding="utf-8",
    )
    return jpath, mpath


def compare(prev: dict, cur: dict) -> list[str]:
    """对照上次报告，输出逐任务变化（新增/修复/退化）与通过率变化。"""
    p = {r["id"]: r["passed"] for r in prev.get("results", [])}
    c = {r["id"]: r["passed"] for r in cur.get("results", [])}
    lines = []
    for tid, ok in c.items():
        if tid not in p:
            lines.append(f"[新增] {tid}：{'通过' if ok else '失败'}")
        elif p[tid] != ok:
            lines.append(f"[{'修复 ✅' if ok else '退化 ❌'}] {tid}：{'失败→通过' if ok else '通过→失败'}")
    for tid in p:
        if tid not in c:
            lines.append(f"[移除] {tid}")
    lines.append(
        f"通过率：{prev.get('pass_rate', 0):.0%} -> {cur.get('pass_rate', 0):.0%}"
    )
    return lines
