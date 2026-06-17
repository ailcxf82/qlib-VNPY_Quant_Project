# 循环结束后产出什么 & 如何提取成果

> 适用入口：`run_loop.ps1` / `run_loop.py` / `python -m scripts.lab.run_rdagent_loop`  
> 更新日期：2026-04-25

---

## 一、文件产出总览

### 阶段 1：仅执行循环（`run_rdagent_loop`）

```
log/<run-timestamp>/
├─ Loop_0/
│   ├─ direct_exp_gen/          ← hypothesis + experiment proposal（pkl）
│   ├─ coding/evo_loop_N/       ← 代码进化过程（pkl）
│   └─ feedback/feedback/       ← 本轮 feedback（pkl，含 decision: SOTA/not）
├─ Loop_1/ … Loop_N/
└─ __session__/                 ← 断点；续跑用 --path 指向这里

git_ignore_folder/RD-Agent_workspace/<hash>/
├─ factor.py                    ← 本轮生成的因子代码
└─ result.h5                    ← 因子值（原始 HDF5）

logs/live_loop/
└─ run_loop_factor_10_<时间戳>.log   ← 剥净 ANSI 的完整运行日志
```

> `log/`、`git_ignore_folder/`、`logs/live_loop/` 均在 `.gitignore` 中，不进版本库。

---

### 阶段 2：导出候选因子（`export_rdagent_candidates`）

```
factor_lab/workspace/candidates/<factor_id>/
├─ factor.py                    ← 同 RD-Agent workspace，但带元信息命名
├─ values.parquet               ← 因子截面值（可直接读取）
└─ c1.json                      ← C1 契约包：IC/Composite + 溯源 _meta
```

---

### 阶段 3：L2 验证 + Promote（`run_lab_cycle`）

```
factor_validation/
├─ reports/
│   ├─ lab_cycle_<cycle_id>.json    ← 每轮验证结果结构化数据
│   └─ lab_cycle_<cycle_id>.md      ← 同上，可读 Markdown 版
└─ certificates/<cycle_id>/
    ├─ <factor_id>.exploratory.json ← 宽松验证结果
    └─ <factor_id>.default.json     ← 标准验证结果（PASS 才可 promote）

factor_registry/
├─ data/
│   ├─ manifest.json            ← active/retired 因子索引（主权威）
│   └─ certified/<factor_id>.json   ← 证书副本（promote 时写入）
└─ parquet/
    └─ factors_v<N>.parquet     ← 生产因子宽表（模型可直接消费）

factor_lab/workspace/feedback/
├─ latest.json                  ← 最新 FeedbackBundle（下轮 RAG 用）
├─ latest_<universe>.json       ← 按股票池归档
└─ history/<stamp>.json         ← 历史记录
```

---

## 二、快速提取成果

### 2.1 导出这次循环的所有候选因子

```powershell
# 导出最新一次 run（自动找 log/ 下最新时间戳目录）
python -m scripts.lab.export_rdagent_candidates

# 指定某次 run
python -m scripts.lab.export_rdagent_candidates --log-run-dir log/<run-timestamp>

# 同时输出摘要 JSON
python -m scripts.lab.export_rdagent_candidates --summary-json exports/summary.json
```

产出：`factor_lab/workspace/candidates/<factor_id>/`（每个因子一个目录）

---

### 2.2 一键跑完整闭环（导出 → L2 验证 → 自动 Promote）

```powershell
python -m scripts.lab.run_lab_cycle
```

**结束后可读的汇总文件**：

| 文件 | 用途 |
|------|------|
| `factor_validation/reports/lab_cycle_<id>.md` | **人可读的验证报告**（推荐先看这个）|
| `factor_validation/reports/lab_cycle_<id>.json` | 同上，结构化数据（程序处理用）|
| `factor_registry/data/manifest.json` | 当前所有 active 因子列表 |
| `factor_lab/workspace/feedback/latest.json` | 完整 FeedbackBundle（含历史对比）|

---

### 2.3 刷新 FeedbackBundle（不重新跑循环）

```powershell
python -m scripts.lab.build_feedback_bundle
```

产出：`factor_lab/workspace/feedback/latest.md`（Markdown，方便直接阅读）

---

### 2.4 查看 active 因子一览

```powershell
# 读 manifest，列出所有 active 因子
python -c "
import json
m = json.load(open('factor_registry/data/manifest.json'))
for f in m['factors']:
    if f['status'] == 'active':
        print(f['name'], f.get('ic'), f.get('composite_score'))
"
```

---

### 2.5 读取生产因子宽表

```python
import pandas as pd
df = pd.read_parquet("factor_registry/parquet/factors_v1.parquet")
print(df.head())
```

---

## 三、完整流程图（文件视角）

```
run_loop.ps1 / run_loop.py
        │
        ▼
log/<ts>/Loop_N/          ← RD-Agent 原生产出（pkl、因子代码、result.h5）
git_ignore_folder/...
logs/live_loop/*.log      ← tee 日志
        │
        ▼  python -m scripts.lab.export_rdagent_candidates
        │
factor_lab/workspace/candidates/<factor_id>/
    ├─ factor.py
    ├─ values.parquet     ← 直接可用的因子值
    └─ c1.json            ← IC / Composite / 溯源信息
        │
        ▼  python -m scripts.lab.run_lab_cycle
        │
factor_validation/reports/lab_cycle_<id>.md    ← ✅ 人可读汇总
factor_validation/certificates/<id>/*.json      ← PASS/FAIL 证书
        │
        ▼  （PASS 自动 promote）
        │
factor_registry/data/manifest.json             ← ✅ 最终 active 因子权威
factor_registry/parquet/factors_v<N>.parquet   ← ✅ 生产宽表
factor_lab/workspace/feedback/latest.md        ← ✅ RAG/人工审阅摘要
```

---

## 四、关键文件速查

| 想了解 | 看这个文件 |
|--------|-----------|
| 每轮 IC / Composite 指标 | `factor_validation/reports/lab_cycle_<id>.md` |
| 当前 active 因子列表 | `factor_registry/data/manifest.json` |
| 某因子完整代码 | `factor_lab/workspace/candidates/<id>/factor.py` |
| 某因子截面值 | `factor_lab/workspace/candidates/<id>/values.parquet` |
| 历史 feedback 摘要 | `factor_lab/workspace/feedback/latest.md` |
| 生产因子宽表 | `factor_registry/parquet/factors_v<N>.parquet` |
| 完整运行日志 | `logs/live_loop/run_loop_*.log` |

---

## 五、相关文档

- `docs/LOOP_RUNNER_GUIDE.md` — 启动参数、高亮事件、故障排查
- `docs/ARCHITECTURE_FACTOR_LAB.md` — L1/L2/L3 三层架构全貌
- `docs/STAGE_H_LIVE_LOOP_LOG.md` — 真实 loop 日志解读示例
- `docs/STAGE_I_REPORT.md` — mode=factor/quant 验收记录
