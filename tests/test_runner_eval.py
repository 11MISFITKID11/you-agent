"""评测 harness 测试：注入假执行器，离线验证 隔离执行 / 判定 / 报告 / 对比 全链路。"""

from __future__ import annotations

import asyncio
import json

from you_agent.runner import Task, TaskRunner, compare, run_eval, summarize, write_report


def _src_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "check_ok.py").write_text("print('OK')\n", encoding="utf-8")
    (repo / "check_bad.py").write_text("print('BAD')\nraise SystemExit(1)\n", encoding="utf-8")
    return repo


def _fake_executor(write_file: str | None = None, exit_code: int = 0):
    """假 agent：可选在隔离目录写一个文件，然后返回给定退出码。"""

    async def exec(workdir, prompt):
        if write_file:
            from pathlib import Path

            (Path(workdir) / write_file).write_text("done\n", encoding="utf-8")
        return exit_code, "fake agent output"

    return exec


def test_task_jsonl_roundtrip(tmp_path):
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        '{"id": "a", "prompt": "p", "check": "python check_ok.py", "tags": ["迭代纠错", "自我修复"]}\n'
        '# 注释行应被忽略\n'
        '{"id": "b", "prompt": "q", "check": "echo hi", "setup": "echo prep"}\n',
        encoding="utf-8",
    )
    tasks = Task.load_jsonl(f)
    assert [t.id for t in tasks] == ["a", "b"]
    assert tasks[0].tags == ["迭代纠错", "自我修复"]
    assert tasks[1].setup == "echo prep" and tasks[1].tags == []


def test_runner_pass_and_fail(tmp_path):
    repo = _src_repo(tmp_path)
    # 通过：假 agent 正常退出 + check_ok 判定通过
    ok_task = Task(id="t-ok", prompt="do", check="python check_ok.py")
    r1 = asyncio.run(TaskRunner(executor=_fake_executor()).run_task(ok_task, repo))
    assert r1.passed and r1.agent_exit == 0

    # 失败：判定命令退出码非 0
    bad_task = Task(id="t-bad", prompt="do", check="python check_bad.py")
    r2 = asyncio.run(TaskRunner(executor=_fake_executor()).run_task(bad_task, repo))
    assert not r2.passed and "BAD" in r2.check_output

    # 失败：agent 自身退出码非 0（判定哪怕通过也算失败）
    crash_task = Task(id="t-crash", prompt="do", check="python check_ok.py")
    r3 = asyncio.run(TaskRunner(executor=_fake_executor(exit_code=3)).run_task(crash_task, repo))
    assert not r3.passed and r3.agent_exit == 3


def test_isolation_between_tasks(tmp_path):
    repo = _src_repo(tmp_path)
    task = Task(id="t-iso", prompt="写文件", check="python check_ok.py")
    asyncio.run(
        TaskRunner(executor=_fake_executor(write_file="NEW.md")).run_task(task, repo)
    )
    assert not (repo / "NEW.md").exists()  # 隔离副本里的改动不污染源仓库


def test_stats_pickup_and_dimensions(tmp_path):
    """假执行器落 you-stats.json -> TaskResult.stats -> summarize 的维度分组与效率均值。"""
    import json as _json
    from pathlib import Path

    repo = _src_repo(tmp_path)

    async def exec_with_stats(workdir, prompt):
        (Path(workdir) / "you-stats.json").write_text(
            _json.dumps({"turns": 4, "tool_calls": 7, "total_tokens": 1234}), encoding="utf-8"
        )
        return 0, "ok"

    tasks = [
        Task(id="s1", prompt="a", check="python check_ok.py", tags=["迭代纠错"]),
        Task(id="s2", prompt="b", check="python check_ok.py", tags=["迭代纠错", "探索"]),
        Task(id="s3", prompt="c", check="python check_bad.py", tags=["探索"]),
    ]
    results = asyncio.run(run_eval(tasks, repo, jobs=2, executor=exec_with_stats))
    assert all(r.stats.get("tool_calls") == 7 for r in results)

    data = summarize(results, "fake")
    assert data["dimensions"] == {
        "迭代纠错": {"total": 2, "passed": 2},
        "探索": {"total": 2, "passed": 1},
    }
    assert data["efficiency"]["avg_turns"] == 4.0
    assert data["efficiency"]["avg_total_tokens"] == 1234
    assert any(r["id"] == "s3" and not r["passed"] for r in data["results"])


def test_report_and_compare(tmp_path):
    repo = _src_repo(tmp_path)
    tasks = [
        Task(id="t1", prompt="a", check="python check_ok.py"),
        Task(id="t2", prompt="b", check="python check_ok.py"),
    ]
    results = asyncio.run(
        run_eval(tasks, repo, jobs=2, executor=_fake_executor(write_file="out.md"))
    )
    assert len(results) == 2 and all(r.passed for r in results)

    jpath, mpath = write_report(results, "faux-model", tmp_path / "reports")
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["pass_rate"] == 1.0 and data["passed"] == 2
    assert "评测报告" in mpath.read_text(encoding="utf-8")

    # 对比：t2 从通过退化为失败
    cur = summarize(
        [results[0],
         type(results[1])(task=tasks[1], passed=False, agent_exit=0)],
        "faux-model",
    )
    prev = json.loads(jpath.read_text(encoding="utf-8"))
    lines = compare(prev, cur)
    assert any("退化" in l and "t2" in l for l in lines)
    assert any("通过率：100% -> 50%" in l for l in lines)
