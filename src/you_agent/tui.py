"""you-tui —— Textual 全屏界面：流式输出 + 常驻输入（插话/stop 是天然能力）。

与 REPL 共用同一 Runtime/Agent/权限/护栏；区别只在展示层：
- RichLog 流式渲染（文本增量按行缓冲，工具调用高亮）；
- 输入框常驻：任务运行中输入 = steering 插话 / stop = 停止（无需额外线程）；
- 底部状态栏：模型 / 累计 token / 轮数 / 工具调用 / 权限模式。
注意：TUI 下权限默认 allow-all（规则与审计仍生效）——交互确认弹窗留给后续版本。
"""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from . import __version__
from .checkpoint import CheckpointManager
from .cli import DEFAULT_PROMPT, HELP_TEXT, READONLY_NAMES, Runtime, _faux_demo_script, _load_dotenv
from .compaction import Compactor
from .govern import CostGuard
from .loop import Agent
from .permissions import PermissionGate
from .providers import FauxStream, litellm_stream, model_context_window
from .resource_loader import load_project_context
from .session import SessionManager
from .skills import load_skills, skills_section
from .tools.builtin import default_tools


class AskScreen(ModalScreen[str]):
    """权限确认弹窗：dismiss 返回 "y" | "always" | "n"。"""

    DEFAULT_CSS = """
    AskScreen { align: center middle; }
    #ask-box { border: round $primary; padding: 1 2; width: auto; max-width: 80; }
    Horizontal { align: center middle; height: auto; margin-top: 1; }
    Button { margin: 0 1; }
    """

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-box"):
            yield Static(self.question)
            with Horizontal():
                yield Button("允许 (y)", id="y", variant="success")
                yield Button("本会话总是 (a)", id="always", variant="warning")
                yield Button("拒绝 (n)", id="n", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(str(event.button.id))


class TuiPrinter:
    """把 Runtime 事件流渲染进 RichLog；文本增量按行缓冲避免碎片。"""

    def __init__(self, app: YouTUI):
        self.app = app
        self.buf = ""

    def __call__(self, ev) -> None:
        log = self.app.logw
        if log is None:
            return
        if ev.type == "message_update" and ev.delta:
            if ev.kind == "text":
                self.buf += ev.delta
                while "\n" in self.buf:
                    line, self.buf = self.buf.split("\n", 1)
                    log.write(Text(line))
        elif ev.type == "tool_execution_start" and ev.tool_call:
            self.flush()
            args = ", ".join(f"{k}={v!r}" for k, v in list(ev.tool_call.arguments.items())[:3])
            log.write(Text(f"⚙ {ev.tool_call.name}({args})", style="bold yellow"))
        elif ev.type == "tool_execution_end" and ev.tool_call:
            mark = "✗" if ev.is_error else "✓"
            log.write(Text(f"{mark} {ev.tool_call.name} 完成", style="red" if ev.is_error else "green"))
        elif ev.type == "message_end":
            self.flush()
        elif ev.type == "compaction":
            self.flush()
            log.write(Text(
                f"[压缩] 移除 {ev.payload.get('removed', '?')} 条，"
                f"{ev.payload.get('before_tokens', '?')} -> {ev.payload.get('after_tokens', '?')} tokens",
                style="yellow"))

    def flush(self) -> None:
        if self.buf and self.app.logw is not None:
            self.app.logw.write(Text(self.buf))
        self.buf = ""


class YouTUI(App):
    TITLE = f"you-agent {__version__}"
    BINDINGS = [("ctrl+q", "quit", "退出"), ("ctrl+c", "stop_run", "停止本轮")]  # noqa: RUF012 Textual 框架的标准类属性模式

    DEFAULT_CSS = """
    RichLog { height: 1fr; }
    #status { color: $text-muted; padding: 0 1; }
    """

    def __init__(self, rt: Runtime, model: str, faux: bool):
        super().__init__()
        self.rt = rt
        self.model = model
        self.faux = faux
        self.logw: RichLog | None = None
        self.writes: list[str] = []  # 测试断言用：每条渲染的纯文本
        self._busy = False
        self._task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", wrap=True)
        yield Input(placeholder="you › 任务或命令（/help · 运行中输入=插话 · stop=停止 · Ctrl+Q 退出）")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.logw = self.query_one("#log", RichLog)
        real_write = self.logw.write

        def traced(*args, **kwargs):  # 透传任意签名；记录首参纯文本供测试断言
            if args:
                self.writes.append(str(args[0]))
            return real_write(*args, **kwargs)

        self.logw.write = traced  # type: ignore[method-assign]
        self.say(f"you-agent v{__version__} · {self.model}" + (" · 无 Key 演示" if self.faux else ""))
        self.say("输入任务开始；/help 查看命令；运行中输入会作为插话生效。")
        self._refresh_status()

    def say(self, text: str, style: str | None = None) -> None:
        if self.logw is not None:
            self.logw.write(Text(text, style=style) if style else Text(text))

    def _refresh_status(self) -> None:
        bar = self.query_one("#status", Static)
        g = self.rt.guard
        perm = self.rt.gate.mode if self.rt.gate else "allow-all"
        bar.update(
            f" 模型 {self.rt.agent.model} · tokens {g.session_tokens:,}（{g.level}） · "
            f"轮数 {self.rt.turns} · 工具 {self.rt.tool_calls} · 权限 {perm}"
        )

    # ---------- 输入 ----------

    def submit_text(self, text: str) -> None:
        """供事件处理与测试共用：异步派发一行输入。"""
        self._task = asyncio.get_running_loop().create_task(self._handle(text))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self.submit_text(text)

    async def _handle(self, text: str) -> None:
        if not text:
            return
        if self._busy:  # 运行中：输入即插话 / stop
            if text in ("stop", "中止"):
                self.rt.agent.abort()
                self.say("（已请求停止：当前工具批次完成后停下）", style="yellow")
            else:
                self.rt.agent.steer(text)
                self.say(f"（已插话：{text[:60]}）", style="dim")
            return
        if text in ("exit", "quit"):
            self.exit()
            return
        self.say(f"you › {text}", style="bold cyan")
        try:
            if text == "/help":
                self.say(HELP_TEXT)
            elif text == "/cost":
                self.say(self.rt.guard.summary())
            elif text.startswith("/model"):
                new_model = text[6:].strip()
                if new_model:
                    self.rt.agent.model = new_model
                    if self.rt.agent.compactor is not None:
                        self.rt.agent.compactor.model = new_model
                    self.say(f"模型已切换：{new_model}", style="green")
                else:
                    self.say(f"当前模型：{self.rt.agent.model}")
            elif text.startswith("/tools"):
                arg = text[6:].strip()
                if arg == "readonly":
                    self.rt.agent.set_active_tools(READONLY_NAMES)
                elif arg in ("default", "all", "*"):
                    self.rt.agent.set_active_tools(None)
                elif arg:
                    self.rt.agent.set_active_tools([t.strip() for t in arg.split(",") if t.strip()])
                self.say(f"活动工具：{', '.join(self.rt.agent.tools)}")
            elif text == "/sessions":
                self.say("（TUI 暂不支持会话切换，请用 you --continue / --resume）", style="yellow")
            else:
                self._busy = True
                try:
                    await self.rt.run(text)
                finally:
                    self._busy = False
        except Exception as e:  # noqa: BLE001 单轮失败不退出应用
            self.say(f"本轮失败：{type(e).__name__}: {str(e)[:200]}", style="red")
        (self.rt.printer.flush() if isinstance(self.rt.printer, TuiPrinter) else None)
        self._refresh_status()

    def action_stop_run(self) -> None:
        self.rt.agent.abort()
        self.say("（已请求停止）", style="yellow")


def build_tui_app(model: str = "deepseek/deepseek-chat", faux: bool = False,
                  no_session: bool = False, no_compact: bool = False,
                  permission: str | None = None) -> YouTUI:
    """构建 TUI 应用（与 CLI 同一套内核件；展示层换成 TuiPrinter）。"""
    stream = FauxStream(_faux_demo_script()) if faux else litellm_stream
    project_ctx = load_project_context()
    skills = load_skills()
    system_prompt = DEFAULT_PROMPT
    if project_ctx:
        system_prompt += f"\n\n# 项目上下文\n{project_ctx}"
    if skills:
        system_prompt += "\n\n" + skills_section(skills)
    agent = Agent(
        stream_fn=stream,
        tools=default_tools(),
        system_prompt=system_prompt,
        model=model,
        compactor=None if no_compact else Compactor(
            stream_fn=stream, model=model,
            context_window=model_context_window(model) or 64_000,
        ),
    )
    session = None if no_session else SessionManager()
    app = YouTUI(rt=None, model=model, faux=faux)  # type: ignore[arg-type]

    async def _ask(question: str) -> str:
        return await app.push_screen_wait(AskScreen(question))

    gate = PermissionGate(mode=permission or ("allow-all" if faux else "ask"),
                          interactive=True, ask_fn=_ask)
    rt = Runtime(
        agent, session,
        cpman=CheckpointManager(),
        guard=CostGuard(),
        printer=TuiPrinter(app),
        gate=gate,
    )
    app.rt = rt
    return app


def main() -> None:
    import typer

    _load_dotenv()
    app_ = typer.Typer(add_completion=False)

    @app_.command()
    def run(
        model: str = typer.Option("deepseek/deepseek-chat", "--model"),
        faux: bool = typer.Option(False, "--faux"),
        no_session: bool = typer.Option(False, "--no-session"),
        no_compact: bool = typer.Option(False, "--no-compact"),
        permission: str = typer.Option(None, "--permission", help="ask（默认，弹窗确认） | allow-all | readonly"),
    ):
        build_tui_app(model=model, faux=faux, no_session=no_session,
                      no_compact=no_compact, permission=permission).run()

    app_()


if __name__ == "__main__":
    main()
