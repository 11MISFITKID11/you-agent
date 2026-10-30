"""后台任务测试：JobManager 生命周期（启动/轮询/输出/终止）与 bash 后台分支。"""

from __future__ import annotations

import asyncio
import sys
import time

from you_agent.jobs import JOB_MANAGER, JobManager, JobsTool
from you_agent.tools.builtin import BashTool


def _wait_done(m: JobManager, jid: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = m.status(jid)[0]
        if not st["running"]:
            return
        time.sleep(0.05)
    raise AssertionError("后台任务超时未结束")


def test_job_start_poll_tail(tmp_path):
    m = JobManager()
    jid = m.start(f'"{sys.executable}" -c "print(\'start\'); import time; time.sleep(0.2); print(\'done\')"')
    assert m.status(jid)[0]["running"] is True
    _wait_done(m, jid)
    st = m.status(jid)[0]
    assert st["running"] is False and st["exit"] == 0
    assert "start" in m.tail(jid) and "done" in m.tail(jid)


def test_job_kill(tmp_path):
    m = JobManager()
    jid = m.start(f'"{sys.executable}" -c "import time; time.sleep(30)"')
    assert m.kill(jid) is True
    _wait_done(m, jid, timeout=5)
    assert m.status(jid)[0]["exit"] != 0
    assert m.kill(jid) is False  # 已结束


def test_jobs_tool_list_and_check():
    m = JOB_MANAGER  # JobsTool 查询的是全局单例，任务要注册到同一实例
    jid = m.start(f'"{sys.executable}" -c "print(\'job-out\')"')
    _wait_done(m, jid)
    tool = JobsTool()
    res = asyncio.run(tool.execute("t", {"action": "list"}))
    assert jid in res.content
    res2 = asyncio.run(tool.execute("t", {"action": "check", "id": jid}))
    assert "job-out" in res2.content and "exit=0" in res2.content
    res3 = asyncio.run(tool.execute("t", {"action": "check", "id": "nope"}))
    assert res3.is_error


def test_bash_run_in_background_branch():
    res = asyncio.run(BashTool().execute("t", {
        "command": f'"{sys.executable}" -c "print(\'bg-out\')"',
        "timeout": 30, "run_in_background": True,
    }))
    assert not res.is_error and "已后台启动" in res.content
    jid = res.content.split("[")[1].split("]")[0]
    deadline = time.time() + 10
    while time.time() < deadline and JOB_MANAGER.status(jid)[0]["running"]:
        time.sleep(0.05)
    assert "bg-out" in JOB_MANAGER.tail(jid)
