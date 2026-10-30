"""TUI 冒烟测试：用 textual 的 run_test 驱动——命令、对话轮、运行中插话。"""

from __future__ import annotations

import asyncio

from you_agent.tui import build_tui_app


def _drain_tasks(app) -> None:
    pass


def test_tui_command_and_turn():
    async def go():
        app = build_tui_app(faux=True, no_session=True)
        async with app.run_test() as pilot:
            app.submit_text("/cost")
            await pilot.pause()
            assert any("会话累计" in w for w in app.writes)

            app.submit_text("演示一下")  # faux：一轮 ls + 一轮回答
            assert app._task is not None
            await asyncio.wait_for(app._task, timeout=10)
            await pilot.pause()
            assert any("无 Key 演示" in w for w in app.writes)
            assert app._busy is False

    asyncio.run(go())


def test_tui_steering_while_busy():
    """运行中输入：busy 状态下新输入变成插话（不触发第二个 run）。"""

    async def go():
        app = build_tui_app(faux=True, no_session=True)
        async with app.run_test() as pilot:
            # 人为制造 busy：跑一个永不结束的任务会卡 faux 脚本，改为直接置位验证分流
            app._busy = True
            app.submit_text("中途改需求")
            await pilot.pause()
            assert app._task is not None and app._task.done()
            assert any("已插话" in w for w in app.writes)
            # 内核队列里确实有一条
            assert app.rt.agent._steering and app.rt.agent._steering[0].content == "中途改需求"

            app.submit_text("stop")
            await pilot.pause()
            assert any("已请求停止" in w for w in app.writes)

    asyncio.run(go())
