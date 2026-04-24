# RD-Agent Loop Runner — 使用指南

> 一条命令启动 factor / quant 循环，**终端实时看高亮的关键节点**，
> 完整日志 tee 到 `logs/live_loop/`，Ctrl-C 可优雅中断。

---

## 1. 三个极简入口

三个入口都等价，按你打字习惯选一个即可：

### (a) PowerShell 包装（推荐 Windows 用户）

```powershell
# 默认：factor 模式 10 轮
.\scripts\lab\run_loop.ps1

# 指定参数
.\scripts\lab\run_loop.ps1 -Mode factor -LoopN 10
.\scripts\lab\run_loop.ps1 -Mode quant  -LoopN 5 -MaxNEpochs 20 -Timeout 7200

# dry-run：只打印将要执行的命令 + 日志路径，不真的运行
.\scripts\lab\run_loop.ps1 -DryRun
```

### (b) 直接 Python（跨平台：Windows 或 WSL 都能跑）

```powershell
# 默认
python scripts\lab\run_loop.py

# 全参数
python scripts\lab\run_loop.py --mode quant --loop_n 5 --max-n-epochs 20 --timeout 7200

# dry-run
python scripts\lab\run_loop.py --dry-run
```

### (c) 最原始的 bash 脚本（WSL 里直接用）

```bash
bash scripts/lab/_wsl_run_factor_loop10.sh        # 纯 factor 10 轮
bash scripts/lab/_wsl_run_quant_loop5_tuned.sh    # 混合 5 轮 + GRU 调优
```

> (a) (b) 都内置了 ANSI 高亮和 tee，**推荐优先使用**；(c) 是老式直裸 `>` 重定向，高亮 off。

---

## 2. 参数速查

| 参数 | 默认（factor）| 默认（quant）| 说明 |
|------|---------------|---------------|------|
| `--mode` | `factor` | — | `factor`：纯因子循环；`quant`：混合（因子 + 模型）|
| `--loop_n` | `10` | `5` | 外层循环轮数 |
| `--max-n-epochs` | *（不影响）* | `20` | 裁剪 LLM 提议的 `n_epochs`；`0` 表示完全透传 |
| `--timeout` | `1800` | `7200` | 单轮 qrun 硬超时（秒） |
| `--log-dir` | `logs/live_loop/` | 同 | tee 日志目录 |
| `--dry-run` | — | — | 只打印命令和路径，不启动子进程 |

耗时预估：

- **factor 模式**：每轮 ~7–10 分钟 → `loop_n=10` 约 1 小时 20 分钟
- **quant 模式**（带 `--max-n-epochs 20`）：每轮 ~15–25 分钟 → `loop_n=5` 约 1.5–2 小时
- **quant 模式**（不限 `n_epochs`）：随时可能单轮 1 小时触顶超时

---

## 3. 启动时会打印什么

```
==================================================================
  RD-Agent Loop Runner
==================================================================
  mode          : factor
  loop_n        : 10
  qrun timeout  : 1800s
  log file      : D:\...\logs\live_loop\run_loop_factor_10_<ts>.log
  estimated     : ~80 min (each factor loop ~7-10 min)
==================================================================
launching: wsl -d Ubuntu -- bash -lc "cd /mnt/d/... && ... python -u -m scripts.lab.run_rdagent_loop --mode=factor --loop_n=10"
Press Ctrl-C to abort; subprocess will receive SIGINT.
```

启动栏会告诉你：**跑什么、多少轮、超时多少、日志写在哪、大概多久**。

---

## 4. 运行中哪些节点会被高亮？

11 类事件会被识别并在原始日志行下面加一条醒目 banner：

| 事件 | banner 样式 | 触发正则 |
|------|-------------|----------|
| Loop / Step 切换 | `===== Loop N \| Step M \| PROPOSE/CODE/QRUN/FEEDBACK ===== [HH:MM:SS]`（青色）| `Start Loop N, Step M: <name>` |
| qrun 指标出炉 | `METRICS  IC: x vs SOTA y   Composite: a vs SOTA b`（黄色）| `IC of Current Result is ... composite_score...` |
| 本轮被选为 SOTA | `*** NEW SOTA ***`（绿色粗体）| `Decision (Whether this experiment is SOTA): True` |
| 本轮未达 SOTA | `(not SOTA this round)`（灰）| 同上 `False` |
| 单步耗时 | `(step took N.N min)`（灰）| `Running time: N seconds`（仅 ≥60s 打印）|
| qrun 超时 | `/!\ qrun timeout after Ns — results neutralised`（品红）| `running time exceeds N seconds` |
| 致命错误 | `!!! MergeError !!!` / `FactorEmptyError` / `Traceback` 等（红色）| 6 种 Python 异常名 |

非关键节点（`Workflow Progress: 50%`、`Using chat model ...` 等）**不**加 banner，避免喧宾夺主。

---

## 5. 日志 tee 说明

所有原始内容（ANSI 被剥干净）都写到：

```
logs/live_loop/run_loop_<mode>_<loop_n>_<YYYYMMDD_HHMMSS>.log
```

即使终端窗口关了、屏幕滚走了，也能事后用：

```powershell
Get-Content logs\live_loop\run_loop_factor_10_*.log -Tail 200
# 或 WSL：
less +F logs/live_loop/run_loop_factor_10_*.log
```

日志头部自动记录启动参数（mode、loop_n、timeout、max_n_epochs、完整子进程命令），尾部记录 `exit_code` + 结束时间戳，方便后续比对。

---

## 6. Ctrl-C 行为

- **第 1 次 Ctrl-C**：给子进程发 SIGINT；Windows 下额外 `wsl pkill -INT -f run_rdagent_loop` 兜底。RD-Agent 会把当前 step 跑完然后优雅退出，保住 session checkpoint。
- **第 2 次 Ctrl-C**：直接 `kill` 子进程。只在第一次没反应时用。

中途退出后，**下次不需要从 0 开始**——RD-Agent 的 session 机制会保留 trace。想从断点续跑直接加 `--path <session_dir>` 给 `run_rdagent_loop`，本 runner 暂未封装，但 session 目录在 `log/<timestamp>/__session__/` 里。

---

## 7. 典型使用场景剧本

**场景 A：快速迭代因子库（最推荐）**

```powershell
.\scripts\lab\run_loop.ps1 -Mode factor -LoopN 10
```

90 分钟内 10 轮都能产出真实 IC/Composite，LLM 有连续真实 feedback 可学。

**场景 B：因子稳定后上模型**

```powershell
# 先 factor 10 轮把因子池做扎实
.\scripts\lab\run_loop.ps1 -Mode factor -LoopN 10

# 再上混合 5 轮训练模型（GRU 限 20 epoch，不会再超时）
.\scripts\lab\run_loop.ps1 -Mode quant -LoopN 5 -MaxNEpochs 20
```

**场景 C：一次跑到天亮**

```powershell
.\scripts\lab\run_loop.ps1 -Mode factor -LoopN 20 -Timeout 3600
```

大约 3 小时；睡醒再来看 tee 日志。

**场景 D：只看命令不跑**

```powershell
.\scripts\lab\run_loop.ps1 -DryRun -Mode quant -LoopN 5
```

打印配置、预计耗时、完整 WSL 命令，什么也不执行。常用于确认 `.env` 加载正确、日志路径没写错。

---

## 8. 故障排查

| 症状 | 可能原因 | 排查 |
|------|----------|------|
| 启动就 `python: command not found` | Windows PATH 没 Python，或 PS 用的是 Store python | 换 `py scripts\lab\run_loop.py ...`；或把 conda/rdagent 的 python 加到 PATH |
| `wsl: 找不到 Linux 发行版` | WSL 没装或发行版名不是 "Ubuntu" | 改 `run_loop.py::_build_command` 里的 `"Ubuntu"` 为 `wsl -l -v` 列出的真实名字 |
| 启动立刻退出，log 里第一行是 `ImportError` | `.env` 没加载或 rdagent 环境出问题 | `wsl -d Ubuntu -- bash -lc "ls -la /home/administrator/.local/share/mamba/envs/rdagent/bin/qrun"` 验证环境存在 |
| 跑了很久屏幕一点输出没有 | qrun 在训练神经网络（loguru 不打 stdout） | 正常；看 log 文件里 `Start Loop N, Step 2: running` 后有没有新行。确认 `ps` 里有 qrun 进程 |
| 出现 `MergeError` banner | 见 `docs/STAGE_I_REPORT.md §7`，应该已修好 | 若仍复现：检查 `rdagent_overrides/factor_template/conf_combined_factors.yaml` 里 `StaticDataLoader.config` 是否被改回绝对路径 |
| banner 乱码 / 看到 `[1m[36m` 字面量 | 终端 ANSI 未启用 | 用 Windows Terminal / PowerShell 7 / VS Code 终端；旧 `cmd.exe` 不建议 |

---

## 9. 相关文件一览

```
scripts/lab/
├─ run_loop.py                        # 核心 Python 启动器（跨平台）
├─ run_loop.ps1                       # PowerShell 包装
├─ run_rdagent_loop.py                # 薄壳调用 factor_lab.runners.rdagent_loop
├─ _wsl_run_factor_loop10.sh          # 老式 bash 入口：纯 factor 10 轮
├─ _wsl_run_quant_loop5_tuned.sh      # 老式 bash 入口：混合 5 轮 + 调优
├─ _verify_run_loop_highlight.py      # banner 正则的离线单元测试
└─ _verify_loop_config.py             # mode / cap / launcher 配置的离线验证

factor_lab/
├─ runners/rdagent_loop.py            # 带 mode 参数的对外入口
└─ adapters/patch_qlib_conda.py       # 所有 RD-Agent monkey-patch（含 n_epochs cap）

docs/
├─ LOOP_RUNNER_GUIDE.md               # 本文
└─ STAGE_I_REPORT.md §7               # 三件事的架构/验收记录
```
