"""基于 git 的文件状态检查点 —— 对照 Cursor checkpoints / WidelCode 运行前快照。

设计：
- 快照存私有命名空间 refs/you-cp/<name>（stash create 产生的提交或 HEAD），不动用户 stash；
- 恢复用 read-tree -u --reset 把已跟踪文件还原到快照状态；
- 已知限制：快照之后新产生的"未跟踪"文件不会被删除（git 只管已跟踪内容）。
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


class CheckpointManager:
    def __init__(self, repo_dir: Path | None = None):
        self.repo_dir = repo_dir or Path.cwd()
        self.available = self._git("rev-parse", "--show-toplevel") is not None

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess | None:
        try:
            r = subprocess.run(
                ["git", *args], cwd=self.repo_dir, capture_output=True, text=True, encoding="utf-8"
            )
        except FileNotFoundError:
            return None
        if check and r.returncode != 0:
            return None
        return r

    def create(self, label: str = "") -> str | None:
        """创建快照，返回 ref 全名；不在 git 仓库 / 空仓库时返回 None。"""
        if not self.available:
            return None
        commit: str | None = None
        r = self._git("stash", "create", check=False)  # 工作区干净时输出为空
        if r is not None and r.returncode == 0 and r.stdout.strip():
            commit = r.stdout.strip()
        if commit is None:
            head = self._git("rev-parse", "HEAD", check=False)
            if head is not None and head.returncode == 0 and head.stdout.strip():
                commit = head.stdout.strip()
        if not commit:
            return None
        name = f"cp-{int(time.time())}-{uuid4short()}"
        ref = f"refs/you-cp/{name}"
        if self._git("update-ref", ref, commit) is None:
            return None
        _ = label  # label 由会话条目记录，ref 名保持确定性
        return ref

    def restore(self, ref: str) -> tuple[bool, str]:
        r = self._git("read-tree", "-u", "--reset", ref, check=False)
        if r is None:
            return False, "git 不可用"
        if r.returncode != 0:
            return False, (r.stderr.strip() or f"git 退出码 {r.returncode}")
        return True, "已还原已跟踪文件到快照状态（未跟踪文件保留）"


def uuid4short() -> str:
    import uuid

    return uuid.uuid4().hex[:6]
