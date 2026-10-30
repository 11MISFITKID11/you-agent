"""you-eval —— 评测 CLI：跑任务集、出报告、对比上次结果（对照方案书方案⑤）。"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .runner import Task, compare, run_eval, write_report

app = typer.Typer(add_completion=False, help="you-agent 评测：任务集批量执行 / 判定 / 报告 / 对比")
console = Console()

DEFAULT_REPO = Path("evals") / "fixtures" / "demo_repo"
DEFAULT_TASKS = Path("evals") / "tasks.jsonl"


@app.command()
def main(
    tasks_file: Path = typer.Argument(None, help="任务集 JSONL（默认 evals/tasks.jsonl）"),
    repo: Path = typer.Option(None, "--repo", help="任务针对的仓库（默认 evals/fixtures/demo_repo）"),
    model: str = typer.Option("deepseek/deepseek-chat", "--model"),
    jobs: int = typer.Option(1, "--jobs", min=1, max=8, help="并行任务数（各自独立隔离目录）"),
    compare_to: Path = typer.Option(None, "--compare", help="上次报告 JSON，输出逐任务变化"),
    keep: bool = typer.Option(False, "--keep", help="保留隔离目录便于排查"),
    task_timeout: int = typer.Option(300, "--task-timeout", help="单任务超时秒数"),
    out_dir: Path = typer.Option(Path("evals") / "reports", "--out", help="报告输出目录"),
):
    from .cli import _load_dotenv

    _load_dotenv()  # 子进程继承 DEEPSEEK_API_KEY 等

    tasks_file = tasks_file or DEFAULT_TASKS
    repo = repo or DEFAULT_REPO
    if not tasks_file.exists():
        console.print(f"[red]任务集不存在：{tasks_file}[/]")
        raise typer.Exit(1)
    if not repo.exists():
        # fixture 仓库以 git bundle 形式入库（嵌套 .git 无法直接跟踪），缺失时自动克隆还原
        bundle = repo.with_name(repo.name + ".bundle")
        if bundle.exists():
            subprocess.run(["git", "clone", "--quiet", str(bundle), str(repo)], check=True)
            console.print(f"[dim]已从 {bundle} 还原 fixture 仓库[/]")
        else:
            console.print(f"[red]仓库目录不存在：{repo}[/]")
            raise typer.Exit(1)

    tasks = Task.load_jsonl(tasks_file)
    console.print(f"任务 {len(tasks)} 个 | 模型 [cyan]{model}[/] | 仓库 {repo} | 并行 {jobs}")
    results = asyncio.run(
        run_eval(tasks, repo, jobs=jobs, model=model, keep=keep, task_timeout=task_timeout)
    )

    jpath, mpath = write_report(results, model, out_dir)
    table = Table(title="评测结果")
    table.add_column("任务")
    table.add_column("结果")
    table.add_column("耗时", justify="right")
    for r in results:
        table.add_row(r.task.id, "[green]通过[/]" if r.passed else "[red]失败[/]", f"{r.duration_s}s")
    console.print(table)

    passed = sum(1 for r in results if r.passed)
    console.print(f"通过率：[bold]{passed}/{len(results)}[/]")

    if compare_to and compare_to.exists():
        prev = json.loads(compare_to.read_text(encoding="utf-8"))
        cur = json.loads(jpath.read_text(encoding="utf-8"))
        console.print("[bold]与上次对比：[/]")
        for line in compare(prev, cur):
            console.print(f"  {line}", highlight=False)

    console.print(f"报告：[link=file://{mpath}]{mpath}[/link]")
    console.print(f"      [link=file://{jpath}]{jpath}[/link]")


def cli() -> None:
    from .cli import _ensure_utf8_stdio

    _ensure_utf8_stdio()
    app()


if __name__ == "__main__":
    cli()
