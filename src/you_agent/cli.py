"""you CLI —— 交互 REPL（含 /plan /act /rewind /cost 命令）与 print 模式。

对照 Pi 的交互 / print 两种模式，并落地方案书三件核心能力：
- 计划模式：/plan 只读调研出 PLAN.md，/act 批准后恢复全量工具执行；
- 检查点：每轮执行前 git 快照（refs/you-cp/*），/rewind 文件与会话双回滚；
- 成本护栏：message_end 累计 usage，提醒 → 暂停（abort，上下文保留）→ 停止。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path

import typer
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from . import __version__
from .checkpoint import CheckpointManager
from .compaction import Compactor
from .govern import CostGuard
from .loop import Agent
from .mcp_client import attach_mcp_tools, find_mcp_config, load_mcp_config
from .permissions import PermissionGate
from .providers import FauxStream, litellm_stream, model_context_window
from .resource_loader import load_project_context
from .session import SessionManager, list_sessions
from .skills import load_skills, skills_section
from .tools.builtin import default_tools, read_only_tools
from .types import AssistantMessage, SystemMessage, TextContent, ToolCall, UserMessage

app = typer.Typer(add_completion=False, help="you-agent — 你的编码智能体（Python 内核，参照 Pi 设计）")
console = Console()

DEFAULT_MODEL = "deepseek/deepseek-chat"

DEFAULT_PROMPT = """You are you-agent, an expert coding assistant operating in the user's terminal.
You help by reading files, running commands, editing code, and writing files.

Rules:
- Explore with read/grep/find/ls first; use bash (or powershell) for builds, tests, and git.
- For edits use edit with a unique old_text snippet; use write only for new files or full rewrites.
- Prefer the smallest change that solves the task, then verify it (run/test) when possible.
- Be concise. Always show file paths clearly."""

HELP_TEXT = r"""[bold]命令[/]
  /plan <任务>   进入计划模式：只读工具调研，生成 PLAN.md 草案（不修改任何文件）
  /act           批准计划：恢复全量工具，要求严格按 PLAN.md 执行
  /rewind        回滚：选择历史检查点，同时还原文件状态与对话分支
  /cost          查看会话用量（token / 估算成本 / 护栏档位）
  /compact       手动压缩上下文（把早期对话替换为模型生成的摘要）
  /model <名称>  运行中切换模型（如 deepseek/deepseek-reasoner）
  /tools [集合]  查看/切换活动工具（readonly | default | 逗号清单）
  /sessions      列出并打开历史会话（也可启动时 --continue / --resume）
  /help          本帮助
  exit / quit    退出

[bold]运行中输入[/]
  直接输入文字 = 插话（下一轮生效）；输入 stop = 当前工具批次后停止
  行尾两个反斜杠可续行（多行输入）
  权限：--permission ask|allow-all|readonly，规则见 .you/permissions.json，审计 .you-audit.jsonl"""

READONLY_NAMES = [t.name for t in read_only_tools()]


def _load_dotenv() -> None:
    """极简 .env 加载（当前目录与 ~/.you/.env），已设置的环境变量优先。

    容错：utf-8-sig 吃掉记事本的 BOM；剥离值的引号与 export 前缀。
    """
    for env_path in (Path.cwd() / ".env", Path.home() / ".you" / ".env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                line = line.removeprefix("export ")
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)


def _faux_demo_script() -> list[AssistantMessage]:
    """无 Key 演示脚本：完整走一遍 模型 -> 工具调用 -> 结果回填 -> 最终回答。"""
    call = ToolCall(id="demo_1", name="ls", arguments={"path": "."})
    return [
        AssistantMessage(
            content=[TextContent(text="好的，我先看看当前目录。"), call],
            stop_reason="tool_calls",
        ),
        AssistantMessage(
            content=[TextContent(
                text="我已列出当前目录内容。这是 you-agent 的无 Key 演示：模型决策 -> 工具调用 -> 结果回填 -> 最终回答的完整 agent 循环。"
            )],
        ),
    ]


def _make_printer(quiet: bool):
    def on_event(ev) -> None:
        if quiet:
            return
        if ev.type == "message_update" and ev.delta:
            if ev.kind == "text":
                console.print(ev.delta, end="", markup=False, highlight=False)
            elif ev.kind == "thinking":
                console.print(ev.delta, end="", style="dim", markup=False, highlight=False)
        elif ev.type == "tool_execution_start" and ev.tool_call:
            args = ", ".join(f"{k}={v!r}" for k, v in list(ev.tool_call.arguments.items())[:3])
            console.print(f"\n[bold yellow]⚙ {ev.tool_call.name}[/]({args})", highlight=False)
        elif ev.type == "tool_execution_end" and ev.tool_call:
            mark = "✗" if ev.is_error else "✓"
            style = "red" if ev.is_error else "green"
            console.print(f"[{style}]{mark} {ev.tool_call.name} 完成[/{style}]")
    return on_event


class Runtime:
    """把 Agent / 会话 / 检查点 / 护栏组装到一起（对照 Pi 的 agent-session 中枢）。"""

    def __init__(self, agent: Agent, session: SessionManager | None,
                 cpman: CheckpointManager, guard: CostGuard, printer,
                 gate: PermissionGate | None = None):
        self.agent = agent
        self.session = session
        self.cpman = cpman
        self.guard = guard
        self.printer = printer
        self.gate = gate
        self._system_saved = False
        self.turns = 0        # 效率指标：LLM 轮数
        self.tool_calls = 0   # 效率指标：工具调用次数
        if gate is not None:
            agent.hooks.before_tool_call = gate
        agent.hooks.on_event = self._on_event

    def _on_event(self, ev) -> None:
        if ev.type == "turn_start":
            self.turns += 1
        elif ev.type == "tool_execution_start":
            self.tool_calls += 1
        elif ev.type == "message_end" and ev.message is not None and ev.message.type == "assistant":
            self.guard.on_message_end(ev.message)
            if self.session is not None:
                self.session.append(ev.message)
            if self.guard.should_abort():
                self.agent.abort()
                console.print("\n[bold red][护栏] 用量达到暂停阈值，本轮将在工具批次完成后停下（上下文保留）。[/]")
        elif ev.type == "tool_result" and ev.message is not None and self.session is not None:
            self.session.append(ev.message)
        elif ev.type == "user_message" and ev.message is not None:
            if self.session is not None:
                self.session.append(ev.message)
            console.print(f"\n[dim]（已插话：{ev.message.content[:60]}…）[/]", highlight=False)
        elif ev.type == "compaction":
            p = ev.payload
            if self.session is not None:
                self.session.append_raw({"compact": p})
            console.print(
                f"\n[yellow][压缩] 移除 {p.get('removed', '?')} 条早期消息，"
                f"估算 {p.get('before_tokens', '?')} -> {p.get('after_tokens', '?')} tokens[/]",
                highlight=False,
            )
        self.printer(ev)

    async def compact_now(self) -> None:
        if self.agent.compactor is None:
            console.print("[yellow]压缩未启用（--no-compact）[/]")
            return
        ok = await self.agent.compactor.compact(self.agent)
        console.print("[green]已压缩。[/]" if ok else "[yellow]当前无需压缩（上下文未达阈值）[/]")

    async def run(self, text: str, checkpoint: bool = True) -> AssistantMessage:
        """一次执行：可选打文件检查点 -> 记会话 -> 跑循环。"""
        if checkpoint and self.session is not None and self.cpman.available:
            ref = self.cpman.create(label=text[:40])
            if ref:
                self.session.append_checkpoint(ref, text[:40])
        if self.session is not None:
            if not self._system_saved:
                self.session.append(self.agent.messages[0])
                self._system_saved = True
            self.session.append(UserMessage(content=text))
        return await self.agent.run(text)

    # ---------- 计划模式 ----------

    async def plan(self, task: str) -> AssistantMessage:
        self.agent.set_active_tools(READONLY_NAMES)
        instr = SystemMessage(content=(
            "进入计划模式：你只能使用只读工具（read/grep/find/ls）做调研，禁止修改任何文件。"
            "请基于调研输出一份可执行的 Markdown 计划：目标、步骤（含要动的文件）、验证方式。"
            f"任务：{task}"
        ))
        self.agent.messages.append(instr)
        if self.session is not None:
            self.session.append(instr)
        final = await self.run(task, checkpoint=False)
        plan_path = Path.cwd() / "PLAN.md"
        plan_path.write_text(final.text, encoding="utf-8")
        console.print(f"\n[bold green]计划已保存到 {plan_path}[/]，检查后输入 [bold]/act[/] 批准执行")
        return final

    def act(self) -> None:
        self.agent.set_active_tools(None)  # 恢复全量工具
        msg = SystemMessage(content=(
            "计划已获用户批准（见 PLAN.md）。请严格按计划执行：逐项完成并在回答中标注进度"
            "（如 [1/3] 完成）；写操作使用 edit/write；完成后运行可用的验证手段。"
        ))
        self.agent.messages.append(msg)
        if self.session is not None:
            self.session.append(msg)
        console.print("[bold green]工具已恢复全量。[/]直接说「按计划执行」开始。")

    # ---------- 检查点回滚 ----------

    def rewind(self) -> None:
        if self.session is None:
            console.print("[yellow]当前未开启会话记录，无检查点[/]")
            return
        cps = self.session.checkpoints()
        if not cps:
            console.print("[yellow]没有检查点（不在 git 仓库，或还没有执行过写操作轮）[/]")
            return
        console.print("[bold]检查点：[/]")
        for i, e in enumerate(cps):
            cp = e["checkpoint"]
            console.print(f"  [{i}] {cp['label'] or '(无标签)'}  [dim]{cp['ref']}[/]", highlight=False)
        choice = Prompt.ask("回滚到哪个序号（回车取消）", default="")
        if not choice.strip():
            return
        try:
            cp = cps[int(choice)]
        except (ValueError, IndexError):
            console.print("[red]序号无效[/]")
            return
        if not self.cpman.available:
            console.print("[red]git 不可用，无法还原文件[/]")
            return
        ok, detail = self.cpman.restore(cp["checkpoint"]["ref"])
        if not ok:
            console.print(f"[red]文件还原失败：{detail}[/]")
            return
        self.session.checkout(cp["id"])
        self.agent.messages = self.session.messages_up_to(cp["id"])
        console.print(f"[bold green]已回滚：[/]{detail}\n对话已切到该分支，之后的对话从这里分叉。")

    def resume(self, path: Path) -> None:
        """切换到既有会话文件：重建上下文，后续追加写同一文件（延续树）。"""
        s = SessionManager.open(path)
        self.session = s
        if s.leaf:
            msgs = s.messages_up_to(s.leaf)
            self.agent.messages = msgs if msgs else [self.agent.messages[0]]
        self._system_saved = True


def _resume_flow(rt: Runtime, pick: bool, allow_cancel: bool = True) -> None:
    """会话恢复流程：--continue 直接取最近，--resume / /sessions 列出选择。"""
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]当前目录没有历史会话[/]")
        return
    if not pick:
        path = sessions[0]
    else:
        console.print("[bold]历史会话（新→旧）：[/]")
        for i, p in enumerate(sessions[:10]):
            console.print(f"  [{i}] {p.name}", highlight=False)
        choice = Prompt.ask("打开哪个序号（回车取消）", default="")
        if not choice.strip():
            if allow_cancel:
                return
            choice = "0"
        try:
            path = sessions[int(choice)]
        except (ValueError, IndexError):
            console.print("[red]序号无效[/]")
            return
    rt.resume(path)
    console.print(
        f"[green]已恢复会话：{path.name}（{len(rt.agent.messages)} 条消息，"
        "继续对话将追加到该会话的树上）[/]"
    )


def _banner(rt: Runtime, model: str, faux: bool, toolset) -> None:
    """启动横幅：品牌 + 环境/能力状态一览。"""

    def status(label: str, ok: bool, off_note: str = "") -> tuple[str, str]:
        if ok:
            return f"✓ {label}", "green"
        return f"✗ {label}{off_note}", "yellow"

    head = Text.assemble(("you", "bold deep_sky_blue1"), ("-agent", "bold"),
                         (f"  v{__version__}", "dim"))
    tagline = Text("你的编码智能体 · Python 内核 · 参照开源框架 Pi 设计", style="dim italic")
    model_row = Text.assemble(("模型   ", "dim"), (model, "bold cyan"),
                              ("  · 无 Key 演示模式" if faux else "", "yellow"))
    tools_row = Text.assemble(("工具   ", "dim"),
                              (f"{len(toolset)} 个 · " + " ".join(t.name for t in toolset), "white"))
    c1 = status("检查点", rt.cpman.available, "(非 git 仓库)")
    c2 = status("压缩", rt.agent.compactor is not None, "(--no-compact)")
    c3 = status("成本护栏", True)
    perm = rt.gate.mode if rt.gate is not None else "allow-all"
    c4 = (f"✓ 权限:{perm}", "green") if perm != "allow-all" else ("✗ 权限:allow-all", "yellow")
    caps_row = Text.assemble(("能力   ", "dim"), c1, ("   ", ""), c2, ("   ", ""), c3, ("   ", ""), c4)
    hint = Text("/model /tools /sessions · /plan /act /rewind /cost /compact · /help", style="dim")
    console.print(
        Panel(
            Group(head, tagline, Text(""), model_row, tools_row, caps_row, Text(""), hint),
            box=box.ROUNDED, border_style="deep_sky_blue1", padding=(0, 2),
        )
    )


@app.command()
def main(
    prompt: str = typer.Option(None, "-p", "--prompt", help="print 模式：执行单个任务后退出"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LiteLLM 模型名（deepseek/deepseek-chat 等）"),
    faux: bool = typer.Option(False, "--faux", help="无 Key 演示模式（脚本化假模型）"),
    tools: str = typer.Option("default", "--tools", help="default | readonly | 逗号分隔工具名"),
    plan_only: bool = typer.Option(False, "--plan", help="只生成计划 PLAN.md，不执行（print 模式）"),
    no_session: bool = typer.Option(False, "--no-session", help="不落盘会话"),
    cont: bool = typer.Option(False, "--continue", help="恢复当前目录最近的会话"),
    resume: bool = typer.Option(False, "--resume", help="列出会话并选择恢复"),
    no_compact: bool = typer.Option(False, "--no-compact", help="关闭自动上下文压缩"),
    mcp: Path = typer.Option(None, "--mcp", help="MCP 配置文件（默认自动查找 .you/mcp.json 或 mcp.json）"),
    permission: str = typer.Option(None, "--permission", help="权限模式：ask | allow-all | readonly（默认 REPL=ask，print/评测=allow-all）"),
    stream: bool = typer.Option(False, "--stream", help="print 模式流式输出增量"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="不加载 MCP 服务器"),
    stats_file: Path = typer.Option(None, "--stats-file", help="运行结束后把效率指标（轮数/工具调用/token）写入该 JSON 文件"),
    context_window: int = typer.Option(0, "--context-window", help="模型上下文窗口（token）；0=按模型自动探测，压缩阈值为窗口的 80%"),
    warn_tokens: int = typer.Option(200_000, help="护栏：提醒阈值（累计 token）"),
    pause_tokens: int = typer.Option(500_000, help="护栏：暂停阈值"),
    stop_tokens: int = typer.Option(1_000_000, help="护栏：停止阈值"),
):
    _load_dotenv()
    if tools == "default":
        toolset = default_tools()
    elif tools == "readonly":
        toolset = read_only_tools()
    else:
        wanted = {t.strip() for t in tools.split(",") if t.strip()}
        toolset = [t for t in default_tools() if t.name in wanted]

    stream = FauxStream(_faux_demo_script()) if faux else litellm_stream
    # 项目上下文（AGENTS.md/CLAUDE.md 规范，对照 Pi resource-loader）：拼进系统提示词
    project_ctx = load_project_context()
    # 技能索引（agentskills.io 规范）：只注入 name+description，正文按需 read（渐进式上下文）
    skills = load_skills()
    system_prompt = DEFAULT_PROMPT
    if project_ctx:
        system_prompt += f"\n\n# 项目上下文\n{project_ctx}"
    if skills:
        system_prompt += "\n\n" + skills_section(skills)
    if context_window <= 0:  # 自动从 litellm 模型目录探测，失败回退 64k
        context_window = model_context_window(model) or 64_000
    agent = Agent(
        stream_fn=stream,
        tools=toolset,
        system_prompt=system_prompt,
        model=model,
        compactor=None if no_compact else Compactor(
            stream_fn=stream, model=model, context_window=context_window
        ),
    )
    session = None if no_session else SessionManager()
    interactive = prompt is None
    perm = permission or ("ask" if (interactive and not faux) else "allow-all")
    gate = PermissionGate(mode=perm, interactive=interactive and not faux)
    if perm == "ask":
        console.print("[dim]权限模式 ask：写操作会逐项确认（.you/permissions.json 可配规则，审计写入 .you-audit.jsonl）[/dim]")
    rt = Runtime(
        agent, session,
        cpman=CheckpointManager(),
        guard=CostGuard(warn=warn_tokens, pause=pause_tokens, stop=stop_tokens),
        gate=gate,
        printer=_make_printer(quiet=not (interactive or stream)),  # print 模式默认不流式（管道友好），--stream 打开
    )
    if project_ctx:
        console.print("[dim]已加载项目上下文（AGENTS.md）[/dim]")
    if skills:
        console.print(f"[dim]已加载技能 {len(skills)} 个：{', '.join(s.name for s in skills)}[/dim]")
    mcp_cfg = None if no_mcp else find_mcp_config(mcp)

    async def _app() -> None:
        # 整个应用主体跑在同一个事件循环里：MCP 会话（stdio 子进程）无法跨 asyncio.run 存活
        async with AsyncExitStack() as stack:
            if mcp_cfg is not None:
                try:
                    servers = load_mcp_config(mcp_cfg)
                    attached, mcp_errors = await attach_mcp_tools(agent, stack, servers)
                    for a in attached:
                        console.print(f"[green]MCP 已接入：{a}[/]")
                    for err in mcp_errors:
                        console.print(f"[yellow]MCP 接入失败：{err}[/]")
                except Exception as e:  # noqa: BLE001
                    console.print(f"[yellow]MCP 配置读取失败：{e}[/]")
            if (cont or resume) and not no_session:
                _resume_flow(rt, pick=resume)

            # ---------- print 模式 ----------
            if prompt is not None:
                if rt.guard.level == "stop":
                    console.print("[red]已达停止阈值，拒绝执行（可用 --stop-tokens 调整）[/]")
                    raise typer.Exit(1)

                def _friendly(e: Exception) -> str:
                    first = str(e).strip().splitlines() or [""]
                    return f"{type(e).__name__}: {first[0][:300]}"

                try:
                    final = await (rt.plan(prompt) if plan_only else rt.run(prompt))
                except Exception as e:
                    console.print(f"[red]执行失败：{_friendly(e)}[/]")
                    raise typer.Exit(1) from e
                if stats_file is not None:
                    tokens_in = sum(v[0] for v in rt.guard.per_model.values())
                    tokens_out = sum(v[1] for v in rt.guard.per_model.values())
                    stats = {
                        "model": model,
                        "turns": rt.turns,
                        "tool_calls": rt.tool_calls,
                        "messages": len(agent.messages),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "total_tokens": rt.guard.session_tokens,
                    }
                    Path(stats_file).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
                if console.is_terminal:
                    console.print()
                    console.print(Markdown(final.text or "(无输出)"))
                else:
                    print(final.text)
                if session:
                    console.print(f"[dim]会话已保存：{session.path}[/dim]", highlight=False)
                return

            # ---------- 交互 REPL（常驻读入线程：支持运行中插话/stop 与多行输入） ----------
            _banner(rt, model, faux, toolset)
            q: asyncio.Queue[str] = asyncio.Queue()
            prompt_box = {"p": "\x1b[36myou ›\x1b[0m "}
            aloop = asyncio.get_running_loop()

            def _reader() -> None:
                while True:
                    try:
                        line = input(prompt_box["p"])
                    except (EOFError, KeyboardInterrupt):
                        line = "\x00EOF"
                    aloop.call_soon_threadsafe(q.put_nowait, line)
                    if line == "\x00EOF":
                        return

            threading.Thread(target=_reader, daemon=True, name="you-reader").start()

            async def _get_line() -> str:
                line = await q.get()
                if line == "\x00EOF":
                    return line
                while line.endswith("\\"):  # 行尾反斜杠续行 = 多行输入
                    prompt_box["p"] = "\x1b[2m…›\x1b[0m "
                    nxt = await q.get()
                    if nxt == "\x00EOF":
                        return nxt
                    line = line[:-1] + "\n" + nxt
                prompt_box["p"] = "\x1b[36myou ›\x1b[0m "
                return line

            async def _run_with_steering(coro) -> bool:
                """执行长任务，期间把输入行变成 插话/stop；返回是否收到 EOF。"""
                task = asyncio.create_task(coro)
                eof = False
                while not task.done():
                    try:
                        extra = await asyncio.wait_for(q.get(), timeout=0.2)
                    except TimeoutError:
                        continue
                    if extra == "\x00EOF":
                        rt.agent.abort()
                        eof = True
                        break
                    s_extra = extra.strip()
                    if not s_extra:
                        continue
                    if s_extra in ("stop", "中止"):
                        rt.agent.abort()
                        console.print("[yellow]（已请求停止：当前工具批次完成后停下）[/]")
                    else:
                        rt.agent.steer(s_extra)
                await task
                return eof

            eof_seen = False
            while not eof_seen:
                text = (await _get_line()).strip()
                if not text:
                    continue
                if text in ("exit", "quit"):
                    break
                try:
                    if text == "/help":
                        console.print(HELP_TEXT)
                    elif text == "/cost":
                        console.print(rt.guard.summary(), highlight=False)
                    elif text.startswith("/model"):
                        new_model = text[6:].strip()
                        if new_model:
                            rt.agent.model = new_model
                            if rt.agent.compactor is not None:
                                rt.agent.compactor.model = new_model
                            console.print(f"[green]模型已切换：{new_model}[/]")
                        else:
                            console.print(f"当前模型：{rt.agent.model}。用法：/model <模型名>")
                    elif text.startswith("/tools"):
                        arg = text[6:].strip()
                        if arg == "readonly":
                            rt.agent.set_active_tools(READONLY_NAMES)
                        elif arg in ("default", "all", "*"):
                            rt.agent.set_active_tools(None)
                        elif arg:
                            wanted = [t.strip() for t in arg.split(",") if t.strip()]
                            rt.agent.set_active_tools(wanted)
                        console.print(f"活动工具：{', '.join(rt.agent.tools)}")
                    elif text == "/sessions":
                        _resume_flow(rt, pick=True, allow_cancel=True)
                    elif text.startswith("/plan"):
                        task = text[5:].strip()
                        if not task:
                            console.print("用法：/plan <任务描述>")
                        else:
                            eof_seen = await _run_with_steering(rt.plan(task)) or eof_seen
                    elif text == "/act":
                        rt.act()
                    elif text == "/rewind":
                        rt.rewind()
                    elif text == "/compact":
                        await rt.compact_now()
                    else:
                        lvl = rt.guard.level
                        if lvl == "stop":
                            if Prompt.ask("[red]已达停止阈值，继续执行将产生费用。仍要继续？[/]",
                                          choices=["y", "n"], default="n") != "y":
                                continue
                        elif lvl == "pause":
                            console.print("[yellow][护栏] 当前处于暂停档位，继续执行请注意用量[/]")
                        elif lvl == "warn":
                            console.print(f"[yellow][护栏] {rt.guard.session_tokens:,} tokens，已过提醒阈值[/]")
                        eof_seen = await _run_with_steering(rt.run(text)) or eof_seen
                except KeyboardInterrupt:
                    console.print("\n[yellow]（已中断本轮）[/]")
                except Exception as e:  # noqa: BLE001 REPL 内单轮失败不退出会话
                    first = str(e).strip().splitlines() or [""]
                    console.print(f"\n[red]本轮执行失败：{type(e).__name__}: {first[0][:300]}[/]")
                console.print()

    asyncio.run(_app())


def _ensure_utf8_stdio() -> None:
    # Windows 控制台/管道默认 GBK，⚙ 等符号会 UnicodeEncodeError；errors=replace 兜底不崩
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def cli() -> None:
    _ensure_utf8_stdio()
    app()  # 必须经 typer 应用入口调用，参数才会被解析（直接调 main() 会拿到 OptionInfo 哨兵）


if __name__ == "__main__":
    cli()
