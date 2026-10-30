# you-agent — 你的编码智能体（Python 内核）

用 Python 自研的编码智能体（coding agent）内核，架构对标 Pi、Cursor、Aider 等业界方案，
并加入自建的能力评测基准：双层 Agent 循环、五钩子体系、截断防御、11 个内置工具、
树形 JSONL 会话、计划模式、检查点双轮回滚、成本护栏、上下文压缩自愈、MCP/Skills 生态接入，
模型走 LiteLLM（默认 DeepSeek）。

## 快速开始

```bash
cd ula
uv sync                 # 创建 .venv 并安装依赖（含本包可编辑安装）

# 无 Key 演示（脚本化假模型，完整走一遍 工具调用->回填->回答 循环）
uv run you --faux -p "演示"

# DeepSeek 真实模式：先配 Key（二选一）
#   1) cp .env.example .env  然后填入 DEEPSEEK_API_KEY
#   2) 环境变量：export DEEPSEEK_API_KEY=sk-xxx
uv run you -p "列出当前目录并总结这个项目是做什么的"
uv run you               # 交互 REPL
uv run you --tools readonly -p "只做分析，不改文件"
uv run you --plan -p "..."   # 只生成 PLAN.md 不执行
```

## 生态接入（v0.4）

**AGENTS.md**：项目根或任意祖先目录放 `AGENTS.md`（兼容 CLAUDE.md），启动自动注入系统提示词；
全局规范放 `~/.you/agent/AGENTS.md`。

**技能（agentskills.io 规范）**：`.you/skills/<技能名>/SKILL.md`（项目级）或
`~/.you/skills/`（全局）。frontmatter 写 name/description，只索引进提示词，正文由模型
按需 read（渐进式上下文），与 Claude/Pi 的技能生态同构。

**MCP 服务器**：`.you/mcp.json`（或 `--mcp 配置文件`）声明 stdio 服务器，工具自动桥接
为 `mcp_<服务器>_<工具>`：

```json
{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}
```

**内置工具 11 个**：read / bash / powershell / write / edit / grep / find / ls /
repo_map / **todo**（长任务清单）/ **web_fetch**（网页转纯文本）。只读工具同批并发执行，
写类工具强制整批串行（execution_mode）。

交互 REPL 命令：

| 命令 | 作用 |
|---|---|
| `/plan <任务>` | 计划模式：切只读工具调研，产出 `PLAN.md` 草案（不动任何文件） |
| `/act` | 批准计划：恢复全量工具，要求严格按 PLAN.md 逐项执行 |
| `/rewind` | 双轮回滚：选历史检查点，同时还原文件（git 快照）与对话分支（会话树） |
| `/cost` | 会话用量：token / 模型明细 / 估算成本 / 护栏档位 |
| `/help` / `exit` | 帮助 / 退出 |

成本护栏三档（`--warn-tokens/--pause-tokens/--stop-tokens` 可调）：提醒（提示）→ 暂停
（本轮工具批次后中止，上下文保留，可继续）→ 停止（拒绝新执行，需确认才可越过）。
检查点要求当前目录是 git 仓库（快照存私有命名空间 `refs/you-cp/*`，不动你的 stash）。

## 评测（you-eval）

SWE-bench 范式的基准，**按能力维度设计任务**（不只测编码正确性，重点测 agent 维度）：

| 维度 | 任务示例 | 为什么是"agent 能力" |
|---|---|---|
| 迭代纠错 / 自我修复 | `t6-iter-fix-build`：构建脚本埋了语法错/名字错/键名错三层陷阱，必须 反复 运行→看报错→修复 | 单次代码生成过不了，必须走 agent 循环 |
| 多源综合 / 探索 | `t7-synthesis`：两个 JSON 文件交叉分析写摘要 | 需要多工具调用 + 信息综合 |
| 规范遵循 | `t8-spec-driven`：按 SPEC.md 精确实现（含边界行为） | 测指令遵循的完整性 |
| 精确修改 | `t9-precision`：只许改一行，`git diff --name-only` 校验零误伤 | 测修改纪律，改多即挂 |
| 环境交互 / git | `t10-changelog`：从 git log 整理变更日志 | 测与真实环境的交互 |
| 编码-修bug/补功能/检索/批量重构/仓库地图 | t1~t5 | 传统编码与工具使用基线 |

评测维度不只通过率，还有**效率指标**（轮数 / 工具调用数 / token，评测运行时自动采集），
报告按能力维度分组，`--compare` 可对比两次运行的逐任务修复/退化。

```bash
uv run you-eval                                # 默认 evals/tasks.jsonl + demo_repo（10 任务）
uv run you-eval --jobs 3                       # 并行执行（worktree 隔离）
uv run you-eval --compare evals/reports/report-xxxx.json   # 对比上次
uv run you-eval my-tasks.jsonl --repo /path/to/project     # 对任意仓库出自己的题
```

fixture 仓库以 `evals/fixtures/demo_repo.bundle`（git bundle，含完整提交历史）入库，
`you-eval` 检测到 `demo_repo/` 缺失时自动从 bundle 克隆还原。

当前基线（DeepSeek，2026-10-30，v0.7）：**21/21（100%）**，效率均值 6.9 轮 / 7.8 次工具调用 /
24k tokens 每任务，报告见 `evals/reports/`。所有任务判定均为客观校验（单测通过 /
`git diff` 零误伤 / 产物内容精确匹配），无人工打分。

会话自动保存到 `~/.you/sessions/--工作目录转义--/<时间戳>_<id>.jsonl`（树形结构，
每条目带 id/parentId，分支不丢数据）。加 `--no-session` 关闭。

## 架构（对标业界方案）

| 文件 | 职责 | 业界对标 |
|---|---|---|
| `src/you_agent/types.py` | 消息模型 / 工具协议 / 事件 | packages/ai/src/types.ts、packages/agent/src/types.ts |
| `src/you_agent/providers.py` | LiteLLM 流式归一 + FauxStream 假模型 | packages/ai（api/*、streaming-json.ts、faux.ts） |
| `src/you_agent/loop.py` | Agent 循环：双层 while、五钩子、截断防御 | packages/agent/src/agent-loop.ts |
| `src/you_agent/tools/builtin.py` | read/bash/powershell/write/edit/grep/find/ls | packages/coding-agent/src/core/tools/ |
| `src/you_agent/session.py` | 树形 JSONL 会话（id/parentId） | packages/coding-agent/src/core/session-manager.ts |
| `src/you_agent/checkpoint.py` | git 文件快照检查点（refs/you-cp/*） | Cursor checkpoints / WidelCode 运行前快照 |
| `src/you_agent/compaction.py` | 上下文压缩：估算→切割→摘要替换 | packages/coding-agent/src/core/compaction/compaction.ts |
| `src/you_agent/govern.py` | 成本护栏（提醒/暂停/停止三档） | WidelCode 单次运行成本上限 |
| `src/you_agent/repo_map.py` | 仓库符号地图（ctags/扫描回退 + 排序） | Aider repo map / Cursor 索引 |
| `src/you_agent/runner.py` + `src/you_agent/evalcli.py` | 隔离执行 / 判定 / 报告 / 对比（you-eval） | SWE-bench 范式（方案书方案⑤） |
| `src/you_agent/cli.py` | REPL（/plan /act /rewind /cost /compact）/ print 模式 | packages/coding-agent/src/main.ts |

内核的关键语义保证（tests/test_loop.py 全覆盖）：

1. **截断防御**：模型输出因 max_tokens 截断（stop_reason=length）时，工具调用一律不执行，
   以错误结果回填——参数 JSON 可能不完整（对照 failToolCallsFromTruncatedMessage）；
2. **错误即反馈**：工具异常 / 参数校验失败 / 未知工具全部转为 is_error 结果回给模型，
   由模型自行补救，而不是进程崩溃；
3. **闸门钩子**：before_tool_call 可返回 `{"block": True, "reason": ...}` 阻断高危调用
   （安全闸门、成本护栏的挂点）；
4. **pydantic 宽松校验**：模型输出 `"5"` 自动纠正为 `5`（对照 pi validation.ts）。

## 安全与控制（v0.5）

- **权限三档**：`--permission ask`（REPL 默认，写操作逐项确认：y / always 本会话 / n 拒绝）、
  `allow-all`（print/评测默认；deny 规则仍生效）、`readonly`（只放行只读工具）。
  规则文件 `.you/permissions.json`：`{"allow": ["read", "bash:git *"], "deny": ["bash:rm *"]}`
  （deny 永远优先；headless 下无法确认的 ask 一律拒绝——fail-safe）。
  全量审计写入 `.you-audit.jsonl`（时间/工具/参数/结论）。
- **运行中插话（steering）**：REPL 里 agent 干活时直接输入文字 = 下一轮生效的插话；
  输入 `stop` = 当前工具批次后停止；模型已收尾时新到的插话会把 run 复活一轮。
- **多行输入**：行尾反斜杠续行；**print 流式**：`you -p ... --stream`。

## 全屏 TUI（v0.6）

```bash
you-tui                 # Textual 全屏界面（与 REPL 同一内核）
```

- 流式输出 + 工具调用高亮（⚙/✓/✗）+ 底部状态栏（模型/token/轮数/权限）；
- **输入框常驻：任务运行中输入自动变成插话（steering），stop 停止**——TUI 的天然优势；
- `Ctrl+C` 停止本轮，`Ctrl+Q` 退出；/help /cost /model /tools 命令同 REPL。

## 安装与发布（v0.6）

本仓库开发安装：`uv sync`。已具备 PyPI 发布条件（LICENSE/元数据/构建验证）：

```bash
uv build                                   # 产物在 dist/（wheel + sdist）
uv publish --token pypi-你的令牌            # 发布（名称 you-agent 目前未被占用）
```

## 测试

```bash
uv run pytest -q        # 58 个测试，无需 API Key
uv run ruff check .     # lint（CI 同款）
```

## Roadmap（对照《you-agent竞品差距分析与迭代方案》）

- [x] 内核：循环 + 钩子 + 消息模型 + 会话树 + 八工具（v0.1）
- [x] 计划模式 / 检查点双轮回滚 / 成本护栏 / repo_map / 评测基准 you-eval（v0.2）
- [x] 压缩管线 + 溢出检测自动压缩重试自愈（v0.3）
- [x] AGENTS.md / CLAUDE.md 项目上下文加载（cwd 向上遍历 + ~/.you/agent/ 全局）（v0.3）
- [x] 会话恢复 --continue / --resume / /sessions（恢复后追加写同一棵树）（v0.3）
- [x] edit 容错匹配（行尾空白归一 + 零命中返回最接近行号上下文）（v0.3）
- [x] REPL 运行中 /model /tools 切换 + ruff + GitHub Actions CI（v0.3）
- [x] P1：MCP 客户端 / Skills 加载器 / web_fetch / todo 工具 / 并行工具执行 / 自动上下文窗口（v0.4，评测 10/10 无回归）
- [ ] P1 余量：print 模式 --stream、REPL 多行输入、web_search
- [x] P2 第一批：权限系统（三档+规则+审计）/ steering 插话 / 多行输入 / --stream（v0.5）
- [x] P2 第二批：Textual TUI（you-tui）+ PyPI 发布就绪（构建/干净环境安装/名称可用性已验证）（v0.6）
- [x] P2 收官：后台任务（bash run_in_background + jobs 工具）/ TUI 权限确认弹窗 / 评测扩到 21 题（v0.7，21/21）
- [ ] 展望：双模型 A/B 基线（需第二家模型 Key）/ 评测难度 L3 层
- [x] ~~子智能体分发~~（已按决策移除，不做）
