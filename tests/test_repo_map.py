"""repo_map 测试：扫描回退路径（不依赖 ctags）、排序加成、工具输出。"""

from __future__ import annotations

import asyncio

from you_agent.repo_map import RepoMapTool, _scan


def _mini_repo(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\nclass Calc:\n    def run(self):\n        return add(1, 2)\n",
        encoding="utf-8",
    )
    (tmp_path / "strings_util.py").write_text(
        "def shout(s):\n    return s.upper()\n", encoding="utf-8",
    )
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "core.js").write_text(
        "function launch() {}\nconst fire = (x) => x;\n", encoding="utf-8",
    )


def test_scan_extracts_py_js_symbols(tmp_path):
    _mini_repo(tmp_path)
    syms = _scan(tmp_path)
    by_name = {s["name"]: s for s in syms}
    assert set(by_name) >= {"add", "Calc", "run", "shout", "launch", "fire"}
    assert by_name["Calc"]["kind"] == "class"
    assert by_name["add"]["file"] == "calc.py"
    assert by_name["fire"]["kind"] == "arrow"
    assert by_name["launch"]["file"].replace("\\", "/") == "pkg/core.js"
    # 行号正确性：add 在 calc.py 第 1 行
    assert by_name["add"]["line"] == 1


def test_repo_map_tool_output_and_query_boost(tmp_path):
    _mini_repo(tmp_path)
    tool = RepoMapTool(root=tmp_path)
    res = asyncio.run(tool.execute("t1", {"query": None, "max_symbols": 20}))
    assert not res.is_error
    assert "add" in res.content and "Calc" in res.content
    assert "共" in res.content

    res2 = asyncio.run(tool.execute("t2", {"query": "Calc", "max_symbols": 5}))
    first_line = res2.content.splitlines()[0]
    assert "Calc" in first_line  # query 命中的符号排最前


def test_repo_map_empty_repo(tmp_path):
    tool = RepoMapTool(root=tmp_path)
    res = asyncio.run(tool.execute("t", {"query": None, "max_symbols": 10}))
    assert "未解析出" in res.content
