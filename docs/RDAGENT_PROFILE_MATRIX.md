# RD-Agent 优化矩阵使用说明

本项目已内置一套可执行的 RD-Agent 优化矩阵，用于对比不同策略配置下的 `fin_quant` 表现。

## 文件位置

- 配置矩阵：`config/rdagent_profile_matrix.json`
- 执行脚本：`scripts/run_rdagent_profile_matrix.ps1`
- 启动入口：`scripts/run_fin_quant.py`

## 预置 Profile

- `A_short_cycle_baseline`
  - 目标：快速基线，最小演化成本
  - 参数：`evolving_n=1`, `action_selection=bandit`, CoSTEER loop=2
- `B_with_fundamental_bias`
  - 目标：增加因子尝试次数，加入基本面偏置
  - 参数：`evolving_n=2`, factor loop=3
- `C_imported_factor_explore`
  - 目标：引入 `rdagent_imported` 思路并用 LLM 选动作
  - 参数：`action_selection=llm`, factor/model loop=3

## 运行方式

### 仅演示命令（不执行）

```powershell
powershell -ExecutionPolicy Bypass -File "scripts/run_rdagent_profile_matrix.ps1" -DryRun
```

### 运行单个 Profile

```powershell
powershell -ExecutionPolicy Bypass -File "scripts/run_rdagent_profile_matrix.ps1" -ProfileName A_short_cycle_baseline -LoopN 1
```

### 依次运行全部 Profile

```powershell
powershell -ExecutionPolicy Bypass -File "scripts/run_rdagent_profile_matrix.ps1" -LoopN 1
```

## 输出与评估

- 每轮日志在 `log/<timestamp>/`
- 核心检查：
  - 是否完整执行 `direct_exp_gen -> coding -> running -> feedback`
  - `feedback` 中是否出现通过候选
  - 是否出现 `All tasks are failed`

## 注意事项

- 脚本会从用户级环境变量读取 `DEEPSEEK_API_KEY` 并生成 `.env/.env.wsl`。
- `feature_sets` 当前作为“实验提示信息”写入 `RDAGENT_FEATURE_SET_HINT`，用于追踪 profile 语义；真正回测模板仍在 `rdagent_overrides/*`。
- 若要将 profile 的特征组合强制映射到 Qlib 执行模板，建议下一步把 `rdagent_overrides` 的 handler/feature 列表也做 profile 化拆分。
