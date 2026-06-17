# 阶段 D 测试与运维指南（真实回测 + 边际贡献 + 自动闭环）

> 本指南覆盖阶段 D 交付的运维级产物：
>
> 1. 4 个新 L2 check（轻量回测 / 真实回测 / 中性化残差 / 双 LGB A/B）
> 2. 三套 profile 的 D 版重构（default / strict / exploratory）
> 3. L1 RD-Agent 导出器（`factor_lab/exporters/rdagent_log_exporter.py`）
> 4. L1→L2→L3 自动闭环（`scripts/lab/run_lab_cycle.py` + `.ps1`）
> 5. Baseline 预测抽取工具（`scripts/validate/prepare_baseline_prediction.py`）
>
> 配套阅读：
>
> - `docs/STAGE_C_TEST_GUIDE.md` —— 前序 L2 基础 4 check + Promote 流水线
> - `docs/STAGE_D_TEST_PLAYBOOK.md` —— 阶段 D 的复制即跑测试
> - `docs/STAGE_D_REPORT.md` —— 阶段 D 交付清单与结论

---

## 1. 测试分层（pytest 快速索引）

| 层                          | 目录                                                                     | 说明                                                       | 典型命令                                                          |
| -------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| 4 个新 L2 check              | `tests/factor_validation/test_{backtest,rqalpha_backtest,marginal,marginal_training}_check.py` | 每个 check 独立用例；RQAlpha 用 FakeRunner 注入                    | `pytest tests/factor_validation -q -k "backtest or marginal"` |
| L2 编排 / profile schema    | `tests/factor_validation/test_orchestrator.py`                        | profile 加载 / context 注入 / aggregate / decide             | `pytest tests/factor_validation/test_orchestrator.py -v`      |
| L1 RD-Agent 导出器            | `tests/factor_lab/exporters/test_rdagent_log_exporter.py`             | log 扫描 / 反射 pkl / result.h5→parquet / factor_id 幂等      | `pytest tests/factor_lab/exporters -v`                        |
| 自动闭环                       | `tests/scripts/lab/test_run_lab_cycle.py`                             | export + validate + promote 的组合 + cycle 报告              | `pytest tests/scripts/lab -v`                                 |
| Baseline prediction 工具    | `tests/scripts/validate/test_prepare_baseline_prediction.py`          | flat/multi-index/synthetic-zero 三路径 + CLI                | `pytest tests/scripts/validate -v`                            |
| Promote / retire（阶段 C 延续） | `tests/promote/`                                                       | validate / promote / retire CLI                          | `pytest tests/promote -q`                                     |
| L3 存储（阶段 C 延续）            | `tests/factor_registry/` · `tests/feature/`                           | ParquetStore / Registry / ProductionFactorLoader + 桥接   | `pytest tests/factor_registry tests/feature -q`               |

### 一键全量

```powershell
conda activate qlib_zhengshi
python -m pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote tests/scripts -q
```

阶段 D 交付基准：**275 项全 PASS，整体 <10 s**（纯离线单测，不需要真实 qlib / rqalpha / lightgbm 数据）。

---

## 2. 4 个新 L2 check 的使用

### 2.1 `backtest`（D.1α，轻量 long-short）

| 字段                         | 含义                                             | 默认（default / exploratory） |
| -------------------------- | ---------------------------------------------- | ------------------------- |
| `quantile`                 | 顶 / 底分位宽度                                      | 0.2                       |
| `min_long_short_sharpe`    | 年化 Sharpe 下限                                   | 1.0 / 0.5                 |
| `max_drawdown`             | 最大回撤上限                                         | 0.40 / 0.60               |
| `min_annual_return`        | 年化收益下限                                         | 0.0 / 0.0                 |
| `min_win_ratio`            | 日胜率下限                                          | 0.50 / 0.45               |
| `allow_negative`           | 是否允许翻号后再评（候选因子取 `-x`）                          | true                      |

**语义约束**：纯 long-short，不吃费率 / 不吃涨跌停 / 等权。用来在秒级确认"候选方向是否
能赚钱"；不是真实回测，**不能**代替 strict 的 `backtest_rqalpha`。

### 2.2 `backtest_rqalpha`（D.1β，真实回测）

| 字段                    | 含义                                          | 默认（strict）                                          |
| --------------------- | ------------------------------------------- | -------------------------------------------------- |
| `min_sharpe`          | Sharpe 下限                                   | 1.2                                                |
| `max_drawdown`        | 最大回撤上限                                      | 0.30                                               |
| `min_annual_return`   | 年化收益下限                                      | 0.05                                               |
| `min_total_return`    | 总收益下限                                       | 0.0                                                |
| `allow_negative`      | 是否允许翻号                                      | false                                              |
| `full_invested`       | 是否强制满仓                                      | false                                              |
| `score_col`           | 预测 CSV 里的分数列名                               | `final`                                            |
| `rqalpha_config_path` | RQAlpha 账户 / 基准 / 费率配置                      | `config/rqalpha_config.yaml`                       |
| `output_root`         | `report.json` 落盘根目录                         | `factor_validation/reports/rqalpha`                |

**工程细节**：

- 运行前把候选因子 rank → prediction CSV（`date,code,score`）；
- Runner 通过 `_load_rqalpha_runner()` 延迟导入，tests 通过 monkeypatch 注入
  `FakeRunner` 做纯离线单测；
- 结果写 `output_root/<factor_id>_<ts>/report.json`；check 从里面读 `sharpe`、
  `max_drawdown`、`annual_return`、`total_return` 四项。

### 2.3 `marginal`（D.2A，中性化残差 IC）

| 字段                     | 含义                                 | 默认（default / strict） |
| ---------------------- | ---------------------------------- | -------------------- |
| `min_residual_rank_ic` | 残差 rank IC 均值下限                    | 0.008 / 0.012        |
| `min_residual_ic_ir`   | 残差 IC 信息率下限                        | 0.15 / 0.25          |
| `allow_negative`       | 允许翻号                               | true / false         |

**数据依赖**：`CheckContext.baseline_prediction_parquet`
（= production ensemble OOS 预测）。用 `scripts/validate/prepare_baseline_prediction.py`
离线产出；没有它 → check 会失败并给出清晰错误消息。

### 2.4 `marginal_training`（D.2B，双 LGB A/B）

| 字段                     | 含义                          | 默认（strict） |
| ---------------------- | --------------------------- | ---------- |
| `min_ic_uplift`        | `rank_ic_B − rank_ic_A` 下限 | 0.003      |
| `train_ratio`          | 时间序列 train 占比                | 0.6        |
| `num_boost_round`      | LGB 迭代上限                     | 200        |
| `learning_rate`        | 学习率                         | 0.05       |
| `num_leaves`           | 叶子上限                        | 31         |
| `min_data_in_leaf`     | 每叶样本下限                      | 50         |
| `feature_fraction`     | 每树列采样率                      | 0.9        |
| `bagging_fraction`     | 每树行采样率                      | 0.9        |
| `bagging_freq`         | 采样间隔                        | 5          |
| `random_seed`          | 随机种                         | 42         |
| `min_test_days`        | OOS 最少天数                    | 30         |
| `allow_negative`       | 允许翻号                        | false      |

**超参是"稳定可复现"取向而非"追 SOTA"**：固定种、保守迭代轮数，跑 10~30 秒 / 候选，
测 B 模型相对 A 的 rank IC 净增益。

---

## 3. 三套 profile 的 D 版形态

详细阈值见 `STAGE_D_REPORT.md §6 附录 A`。每条 profile 顶部注释说明其用途、
启用的 check、数据源依赖、典型运行时长 —— **修改 profile 前务必先读那段注释**。

关键约定：

| profile       | 适用场景                                  | 典型运行时长     | baseline pred 依赖 |
| ------------- | ------------------------------------- | ---------- | ---------------- |
| `exploratory` | RD-Agent 自动闭环的预筛                     | ~秒         | 不需要              |
| `default`     | 新因子提交默认档；`run_lab_cycle` 精验           | ~秒（含轻量回测） | 需要（`marginal`）   |
| `strict`     | Production 准入；手动触发                     | 5~20 min   | 需要（`marginal` + `marginal_training`）|

---

## 4. L1→L2→L3 自动闭环（`scripts.lab.run_lab_cycle`）

### 4.1 输入 / 输出

**输入**：

- `--log-root`（默认 `log/`）或 `--log-run-dir`（指定单个 run）
- `--rdagent-workspace-root`（默认 `git_ignore_folder/RD-Agent_workspace/`）
- `--exploratory-profile` / `--default-profile`（默认读 `config/factor_lab.yaml`）

**产物**（按 `<cycle_id>` 组织）：

```
factor_lab/workspace/candidates/<factor_id>/{factor.py,values.parquet,c1.json}
factor_validation/certificates/<cycle_id>/<factor_id>.exploratory.json
factor_validation/certificates/<cycle_id>/<factor_id>.default.json        # 仅 exploratory PASS 才生成
factor_registry/data/manifest.json                                         # 仅 default PASS 才更新
factor_validation/reports/lab_cycle_<cycle_id>.{json,md}                   # 汇总报告
```

返回码：

- `0`：执行完毕（不一定每个候选都 PASS，看报告）。
- `1`：运行过程中发生未捕获异常（导出失败、磁盘满、等）。

### 4.2 典型调用

```powershell
# 周期用法：扫最近一次 RD-Agent run，默认不覆盖已存在 staging
python -m scripts.lab.run_lab_cycle `
    --run-filter 2026-04-20 `
    --cycle-id   labcycle-20260420-weekly

# 指定单个 run；同时覆盖 staging 与已入库同名因子（谨慎）
python -m scripts.lab.run_lab_cycle `
    --log-run-dir log/2026-04-20_01-19-42-374811 `
    --overwrite-export `
    --allow-promote-overwrite

# 改默认 tag；默认是 "production,rdagent,lab_cycle"
python -m scripts.lab.run_lab_cycle `
    --promote-tags "production,rdagent,lab_cycle,q2_experiment"
```

### 4.3 与 `scripts.lab.export_rdagent_candidates` 的差别

| 维度        | `export_rdagent_candidates` | `run_lab_cycle`                    |
| --------- | --------------------------- | ---------------------------------- |
| 跑 L2 验证？  | ✗                           | ✓（exploratory + default 两档）        |
| 自动入 L3？   | ✗                           | ✓（default PASS 即 promote）          |
| 出报告？      | 仅 `--summary-json`（可选）      | `lab_cycle_<id>.{json,md}` 强制      |
| 幂等？       | 内容级幂等（content-hash skip）   | 同上，另加 `allow_promote_overwrite` 开关 |

**典型用法分流**：日常人工调试用 `export_rdagent_candidates`（只跑第一步）；定时自动跑用
`run_lab_cycle`。

---

## 5. Windows Task Scheduler（任务计划程序）配置

### 5.1 为什么选 Task Scheduler 而不是 cron

项目主开发环境是 Windows，Task Scheduler 是系统原生调度器，无需额外部署；
PS1 包装器 `scripts/lab/run_lab_cycle.ps1` 已经处理了：

- conda 环境的延迟 activate 逻辑（由 `--PythonExe` 指定使用的解释器；默认 `python`）；
- 日志落盘（`factor_validation/reports/scheduler-logs/run_lab_cycle-<ts>.log`）；
- exit code 透传（`exit $LASTEXITCODE`）。

### 5.2 配置步骤（GUI）

1. 打开「任务计划程序」→ 右侧「创建任务…」。
2. **常规**：
   - 名称：`qlib_lab_cycle_weekly`
   - 描述：`RD-Agent → L2 → L3 自动闭环，每周跑一次`
   - 选「不管用户是否登录都要运行」，勾选「使用最高权限运行」。
3. **触发器** → 新建：
   - 开始任务：`按计划`、`每周`、`星期六 02:30`、`重复间隔 1 周`。
4. **操作** → 新建：
   - 操作：`启动程序`
   - 程序/脚本：`powershell.exe`
   - 添加参数：
     ```
     -NoProfile -ExecutionPolicy Bypass -File "D:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\scripts\lab\run_lab_cycle.ps1"
     ```
   - 如果需要传自定义参数（例如只跑某个 run filter）：
     ```
     -NoProfile -ExecutionPolicy Bypass -File "…\run_lab_cycle.ps1" -RunFilter "2026-04-20" -CycleId "labcycle-20260420-weekly"
     ```
   - 如果 conda 环境的 `python` 不在全局 PATH，额外加 `-PythonExe "C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe"`。
5. **条件**：按需取消「仅在计算机使用交流电源时启动任务」（服务器常开则无所谓）。
6. **设置**：勾选「如果任务运行时间超过下列时间则停止任务」→ 建议 `2 小时`
   （weekly cycle 吃不到这个阈值；给 RQAlpha 误触留的保险）。

### 5.3 配置步骤（命令行，推荐在服务器上用）

```powershell
# 以管理员身份开一个 PowerShell，粘贴整块：
$Action   = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File D:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project\scripts\lab\run_lab_cycle.ps1"

$Trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 2:30AM

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest -LogonType S4U

Register-ScheduledTask `
    -TaskName "qlib_lab_cycle_weekly" `
    -Description "RD-Agent → L2 → L3 auto-closed-loop, weekly" `
    -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal
```

查看 / 删除：

```powershell
Get-ScheduledTask -TaskName "qlib_lab_cycle_weekly"
Unregister-ScheduledTask -TaskName "qlib_lab_cycle_weekly" -Confirm:$false
```

### 5.4 手动演练（强烈建议首次配置后跑一次）

```powershell
Start-ScheduledTask -TaskName "qlib_lab_cycle_weekly"
# 看日志
Get-Content (Get-ChildItem factor_validation/reports/scheduler-logs -File |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

### 5.5 运维 checklist

- ✅ 任务跑完后 `factor_validation/reports/lab_cycle_<id>.md` 里 `promoted >= 1` → 关注新入 L3 的 factor_id；
- ✅ `failed > 0` 且 `error` 字段非空 → 读 `lab_cycle_<id>.json` 定位异常类型；
- ✅ 每月巡检 `factor_validation/reports/scheduler-logs/` 的体积，按需清理 90 天以上；
- ✅ RD-Agent log/ 目录按月归档；`run_lab_cycle` 默认幂等，旧 run 重复扫不会重复 promote。

---

## 6. 手工排障清单（常见问题 → 定位）

| 症状                                                                    | 多半原因                                                           | 排查                                                             |
| --------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| `marginal` FAIL 且 detail 里写 `baseline_prediction_parquet 缺失`          | 没有先跑 `prepare_baseline_prediction.py`，或 profile 里路径写错          | 对照 profile `data_sources.baseline_prediction_parquet`          |
| `backtest_rqalpha` FAIL 且 `RQAlpha runner not importable`              | rqalpha 没装；或 Python 环境切错                                       | `python -c "import rqalpha"` 确认；strict 跑不了就暂时降到 default        |
| Cycle 结束 `promoted=0`，每个 candidate 都卡在 exploratory                    | 候选质量真的差；或 exploratory 阈值没过（读 cert 看 check_results）              | 开 `--log-level DEBUG` 重跑，盯 `ic_check` / `coverage_check` 细节   |
| `factor_id` 冲突：promote 报 `already registered`                         | 别的 cycle 已入过；或改了代码 id 却没改；`allow_promote_overwrite` 默认关        | 检查 manifest 对应 `factor_id`，确认要不要 overwrite；或改 RD-Agent 输出名   |
| 某个 candidate staging 已存在却被跳过                                          | **正常幂等**：内容哈希一致                                                | 加 `--overwrite-export` 强制覆盖                                   |
| Windows 控制台 `UnicodeEncodeError`                                      | GBK stdout 遇到中文；落盘 UTF-8 的报告文件不受影响                              | 直接读 `lab_cycle_<id>.md`                                         |
| `cycle_exception: RuntimeError: workspace_hash mismatch`              | RD-Agent run 的 pickle 指向的 workspace 已被 RD-Agent 清理             | 用 `--rdagent-workspace-root` 指向正确目录；或等 RD-Agent 下次出新 run     |

---

## 7. 回归测试清单（新增因子 / 改 check 时必跑）

| 动了什么                                                    | 必跑                                                                                                                   |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 改任一 `factor_validation/checks/*.py`                      | `pytest tests/factor_validation -q`                                                                                  |
| 改任一 `factor_validation/profiles/*.yaml`                  | `pytest tests/factor_validation/test_orchestrator.py -v`，再抽样跑一次 `run_lab_cycle` dry-run（不加 `--allow-promote-overwrite`）|
| 改 `factor_lab/exporters/rdagent_log_exporter.py`         | `pytest tests/factor_lab/exporters -v`                                                                               |
| 改 `scripts/lab/run_lab_cycle.py`                         | `pytest tests/scripts/lab -v`                                                                                        |
| 改 `scripts/validate/prepare_baseline_prediction.py`      | `pytest tests/scripts/validate -v`                                                                                   |
| 改 `scripts/promote/*.py`                                 | `pytest tests/promote -q`（阶段 C 覆盖，阶段 D 未变动）                                                                          |
| 改 `factor_registry/*.py`                                 | `pytest tests/factor_registry tests/feature -q`                                                                      |
| **任何跨模块的改动**                                            | `pytest tests/factor_lab tests/factor_validation tests/factor_registry tests/feature tests/promote tests/scripts -q` |

---

## 8. 已知限制与下一步

| 限制                                                                                 | 原因                                                | 计划                                                                          |
| ---------------------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------- |
| `run_lab_cycle` 默认不跑 strict                                                        | strict 含 RQAlpha 真实回测，时长不适合周级自动                    | 人工触发；未来接 CI/队列                                                              |
| `marginal_training` 的 baseline 特征集简化为"baseline prediction 自己"                     | 真·多特征 A/B 需要额外数据管道                                | 阶段 E：接 `feature/qlib_feature_pipeline.py`                                   |
| RD-Agent log 的 workspace 路径在 Windows 被清理后不可复现                                      | RD-Agent 清理策略不在本项目控制                              | 提示用户在自动闭环前不要手动删 `git_ignore_folder/RD-Agent_workspace/`                    |
| Cycle 报告是单 run 视角；跨 cycle 对比还要查询多文件                                                | 不在 D 阶段范围                                         | 阶段 F：dashboard                                                              |
| `backtest_check` 不模拟交易成本                                                           | 有意：作为 default/exploratory 秒级打分；strict 用 RQAlpha 补 | 保持现状                                                                        |
