# GRU 强化路线图 2026 H2

> **文档性质**：本路线图为 GRU/时序模型在本项目的中长期演进规划。本文档包含 3 个 Phase：Phase 1 工程修复（短期，1 周内）、Phase 2 PatchTST 引入（中期，2-3 周）、**Phase 3 高频数据驱动 GRU（核心，1-2 月）**。Phase 3 是本文档的重点。

**制定日期**：2026-04-29  
**关联文档**：`docs/OPTIMIZATION_ROADMAP_2026.md`、`docs/PREDICTION_LOGIC.md`、`docs/CHANGELOG_2026-01-14.md`  
**问题背景**：GRU 在最终融合层权重几乎为 0（见第 0 节诊断），`base_models=["lgb","gru"]` 但 `final` 由 LGB 主导。

---

## 0. 诊断回顾（为什么需要这份路线图）


| 编号  | 问题                                      | 影响                                                | 文件位置                              |
| --- | --------------------------------------- | ------------------------------------------------- | --------------------------------- |
| D1  | `_load_ic_histories` 未包含 `ic_gru`       | 最终 `RankICDynamicWeighter.blend` 时 GRU 权重 = 0     | `run_predict_chan.py:126-142`     |
| D2  | `meta_oof_builder` 整行 dropna            | GRU 因 NaN 行被剔除，Ridge 系数压低                         | `utils/meta_oof_builder.py:79-84` |
| D3  | `GRURegressor.load` dropout 默认 1.2（bug） | 加载时网络可能崩坏                                         | `models/gru_model.py`             |
| D4  | GRU 网络结构是 Qlib baseline 简化版             | 单向 1 层 hidden=64，能力不足                             | `config/model_gru.yaml`           |
| D5  | GRU 与 LGB 信息源同质                         | `gru_short_cycle` 和 `rdagent_exported` 都是日频价量+基本面 | `config/pipeline.yaml:19-26`      |
| D6  | 全项目仅有日频数据                               | GRU 没有"独有信息"可学                                    | 全项目无 1min/tick 数据                 |


**核心结论**：D1-D4 是工程瑕疵，Phase 1 处理；D5-D6 是结构性问题——**只有让 GRU 拿到 LGB 看不到的信息（即高频数据），它才能在融合中真正"独立加权"而不是"陪跑"**。

---

## 1. Phase 1 — 工程修复（1 周）

### 1.1 范围

让 GRU 真正参与最终加权，而不是被乘 0。**完成 Phase 1 之前，引入更强模型没有意义**——再强的模型也会被同样的瑕疵卡掉。

### 1.2 行动清单


| ID   | 动作                                                                                  | 文件                                                                  | 验收                                                                   |
| ---- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| P1-1 | `_load_ic_histories` 加上 `ic_gru` 读取，缺失时用 `ic_lgb` 兜底                                | `run_predict_chan.py`、`docs/PREDICTION_LOGIC.md` 同步示例               | `data/predictions/.../weights.json` 中 `gru` 权重 > 0                   |
| P1-2 | `GRURegressor.load` 的 `dropout` 默认值改为 `0.2`（与 fit/save 一致）                          | `models/gru_model.py`                                               | 加载已保存模型再预测，输出与训练日预测一致                                                |
| P1-3 | `meta_oof_builder` 不再整行 dropna；GRU 缺失值改为按当日截面均值填补 + 增加 `gru_coverage` 列供 Ridge 加权时用 | `utils/meta_oof_builder.py`                                         | `data/oof/.../meta_oof.parquet` 行数 ≈ LGB OOF 行数（不再因 GRU 缺失被砍 30-50%） |
| P1-4 | `model_gru.yaml`：`num_layers: 1→2`、新增 `bidirectional: true`、`seq_len: 20→40`        | `config/model_gru.yaml`、`models/gru_model.py`（支持 bidirectional）     | 在相同窗口下重训，valid IC 不下降；GPU 显存翻倍可接受                                    |
| P1-5 | 新增 RankIC loss / ListMLE loss，作为可选项；early-stop 已经是 rankic                           | `models/gru_model.py` 增加 `loss="rankic"` 分支                         | 切换 loss 后 valid Rank IC 提升 ≥ 0.005                                   |
| P1-6 | GRU 输入归一化改为**逐股票时序 z-score**（而非现在的 per_model 截面 z-score）                            | `models/ensemble_manager.py` 的 `_apply_norm` 增加 `per_instrument` 模式 | 不同尺度股票的 GRU 输入分布对齐；valid IC 稳定不下降                                    |


### 1.3 Phase 1 验收门槛（DoD）

- `training_metrics.csv` 中 `ic_gru` 中位数 ≥ 0.020
- 最终 `final_score` 与 `pred_gru` 的 Pearson 相关系数 ≥ 0.15（当前可能 < 0.05）
- 日志可见 GRU 权重在 `[min_weight, max_weight]` 区间内合理分布

---

## 2. Phase 2 — PatchTST 引入（2-3 周）

### 2.1 选择 PatchTST 的理由

- 在 [arxiv 2603.16886](https://arxiv.org/abs/2603.16886) 的 9 模型 financial forecasting 评测中，PatchTST 排第二（mean rank 2.0），仅次于 ModernTCN
- Channel-independent 设计天然适合"每只股票独立建模"，与本项目的 panel 数据结构一致
- 输入仍是日频特征序列，**不依赖高频数据**——可以与 Phase 3 解耦推进
- 实现量约 300 行 PyTorch，调试压力可控

### 2.2 行动清单


| ID   | 动作                                                                                                                                           | 文件                                        |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| P2-1 | 新建 `models/patch_tst_model.py`（与 `GRURegressor` 同接口：`fit/predict/save/load/predict`）                                                         | `models/patch_tst_model.py`（新增）           |
| P2-2 | 在 `model_registry.py` 注册 `"patchtst"`                                                                                                        | `models/model_registry.py`                |
| P2-3 | 新增 `config/model_patchtst.yaml`：`patch_len=10`、`stride=5`、`d_model=128`、`n_heads=8`、`e_layers=3`、`seq_len=60`                                | `config/model_patchtst.yaml`（新增）          |
| P2-4 | `pipeline.yaml`：`base_models: ["lgb","gru","patchtst"]`；`model_features.patchtst` 用与 GRU 不同的特征子集（如 `gru_long_cycle` 或新建 `patchtst_features`） | `config/pipeline.yaml`、`config/data.yaml` |
| P2-5 | `meta_oof_builder` 升级：从二元 `(lgb,gru)` 扩展为 N 元 `(lgb,gru,patchtst,...)` 通用版                                                                   | `utils/meta_oof_builder.py`               |
| P2-6 | `_load_ic_histories` 同步增加 `ic_patchtst` 列读取                                                                                                  | `run_predict_chan.py`                     |


### 2.3 Phase 2 验收门槛

- PatchTST 单模型 valid IC ≥ GRU valid IC + 0.005
- 加入 PatchTST 后组合 ICIR ≥ 当前 + 5%

### 2.4 与 Phase 3 的关系

PatchTST 在**日频**就能给出价值；如果将来 Phase 3 完成，PatchTST 也是 HF 数据消费者的最佳候选（本就是 patch-based）。

---

## 3. Phase 3 — 高频数据驱动 GRU（核心，1-2 月）

> **本节是文档重点**。Phase 1/2 让 GRU "活过来"，Phase 3 让 GRU 真正**强过 LGB 的那一类信息**——日内动力学。

### 3.1 为什么"高频数据 → GRU"是最大杠杆

#### 3.1.1 GRU 的天生优势在哪里

GRU/LSTM 这类 RNN 在量化里的本质优势是**捕捉同一标的在时间维度上的动力学结构**——动量、反转、波动率聚集、量价配合的相位关系。这些信号在日频数据上**已经被传统因子（ROC、RSI、MACD、ATR）和 LGB 充分挖掘**，GRU 很难拿到独有信息。

但在**日内分钟级**，存在大量 LGB 永远看不到的结构：


| 信号类型                                  | 日频可见 | 1min 可见 | GRU 可学到 |
| ------------------------------------- | ---- | ------- | ------- |
| 开盘 30min 收益 vs 全天收益                   | ❌    | ✅       | ✅       |
| 尾盘 30min 集合竞价行为                       | ❌    | ✅       | ✅       |
| 日内 VWAP 偏离曲线                          | ❌    | ✅       | ✅       |
| 大单买卖压（结合分钟成交量分布）                      | 粗糙   | ✅       | ✅       |
| Realized Variance / Bipower Variation | ❌    | ✅       | ✅       |
| 量价相关性的相位（领先/滞后）                       | ❌    | ✅       | ✅       |
| 日内 U 型成交量曲线偏离                         | ❌    | ✅       | ✅       |
| 跳空缺口的日内修复速度                           | ❌    | ✅       | ✅       |


**核心论点**：GRU 在日频上和 LGB 抢饭碗永远抢不过，但**在分钟级序列上 LGB 完全失效**（树模型无法吃 240×D 的展开向量），这是 GRU 唯一无法被取代的领地。

#### 3.1.2 学术证据

- [arxiv 2603.16886](https://arxiv.org/abs/2603.16886)：在 cryptocurrency / forex / equity 上，时序模型的优势在**短 horizon（4h/24h）最显著；横向比较时 LSTM 排末位，但在加入分钟级 patch 后**，PatchTST/ModernTCN 跃升到前两名
- [arxiv 2603.01820](https://www.arxiv.org/pdf/2603.01820)：**LPatchTST（LSTM + PatchTST 双流）Sharpe 2.31 vs vanilla LSTM 1.6**——双流的关键就是 LSTM 吃高频段、PatchTST 吃低频段
- 经验值：A 股截面上日内统计因子（VWAP 偏离、上午/下午 IC）在多家私募的 alpha pool 里 IC 0.04-0.07，**接近 LGB 全因子集合的水平**

#### 3.1.3 本项目的"独占价值"

- LGB 用 `rdagent_exported` 因子 → 都是日频聚合后的横截面因子
- daily GRU 用 `gru_short_cycle` → 也是日频
- **HF-GRU 用分钟数据 → 第三个完全独立的信息源**

三路融合时 Ridge/Meta-Stacker 才有"分工"可学，否则就是同一信号的噪声平均。

---

### 3.2 数据源选型

#### 3.2.1 候选数据源对比（A 股）


| 数据源                                  | 频率              | 历史长度           | 价格                 | 接入难度        | 推荐        |
| ------------------------------------ | --------------- | -------------- | ------------------ | ----------- | --------- |
| **Tushare Pro `stk_mins`**           | 1/5/15/30/60min | 5 年（2019-至今）   | 5000 积分（约 ¥2000/年） | 低（HTTP API） | ⭐⭐⭐⭐⭐     |
| **Akshare `stock_zh_a_hist_min_em`** | 1/5/15/30/60min | 仅最近 5 个交易日（实时） | 免费                 | 低           | ⭐⭐（仅适合实盘） |
| **JoinQuant**                        | 1min            | 全历史            | 免费（学术）/ 付费         | 中           | ⭐⭐⭐⭐      |
| **Wind / Choice**                    | 1min/tick       | 全历史            | 商业贵                | 高           | ⭐⭐⭐       |
| **VNPY + RQData/咏星**                 | 1min/tick       | 全历史            | 付费                 | 高           | ⭐⭐        |
| **CTP 实时 tick 落库**                   | tick            | 仅前向            | 免费                 | 高（要部署 CTP）  | ⭐（用于实盘）   |


**推荐组合**：

- **历史训练数据**：Tushare Pro `stk_mins` + 5min 起步（成本/数据量平衡），后期加 1min
- **实盘推理数据**：VNPY/CTP 接实时 1min（项目本就规划了 VNPY 集成）
- **冗余备份**：Akshare 用于 sanity check 与节假日修补

#### 3.2.2 数据量估算


| 频率               | 单股票每日条数 | csi500 × 252 天 × 2 年 | 字段（OHLCV+amount=6 列） | 原始大小         | parquet 压缩  |
| ---------------- | ------- | -------------------- | -------------------- | ------------ | ----------- |
| 60min            | 4       | 1,008,000            | 6 列 × 4 字节           | ~24 MB       | ~5 MB       |
| 30min            | 8       | 2,016,000            |                      | ~48 MB       | ~10 MB      |
| 15min            | 16      | 4,032,000            |                      | ~96 MB       | ~20 MB      |
| **5min**         | **48**  | **12,096,000**       |                      | **~290 MB**  | **~60 MB**  |
| **1min**         | **240** | **60,480,000**       |                      | **~1.45 GB** | **~300 MB** |
| tick（约 4000/日/股） | 4,000   | 1,008,000,000        |                      | ~24 GB       | ~5 GB       |


**结论**：1min 在硬件/存储/训练时间上完全可控。本路线图以 **1min** 为目标，5min 作为 Phase 3 第一阶段的过渡。

---

### 3.3 数据存储与 Schema 设计

#### 3.3.1 目录结构（与现有 Qlib 日频共存）

```text
/mnt/d/qlib_data/
├── qlib_data/                    # 现有日频（保持不动）
│   ├── calendars/day.txt
│   ├── instruments/csi500.txt
│   └── features/sh600000/
│       └── close_qfq.day.bin
└── qlib_data_1min/               # 新增分钟频
    ├── calendars/1min.txt        # 240 个时间点 × 252 天
    ├── instruments/csi500.txt    # 复用日频 instrument list
    └── features/sh600000/
        ├── open.1min.bin
        ├── high.1min.bin
        ├── low.1min.bin
        ├── close.1min.bin
        ├── volume.1min.bin
        └── amount.1min.bin
```

Qlib 原生支持多频率：

```python
qlib.init(
    provider_uri={
        "day":  "/mnt/d/qlib_data/qlib_data",
        "1min": "/mnt/d/qlib_data/qlib_data_1min",
    },
    region="cn",
)
```

#### 3.3.2 时间对齐规约

- **交易日历**：一日 240 根（09:30-11:30、13:00-15:00），不含集合竞价 09:25 与 14:57-15:00 的特殊处理
- **timestamp 约定**：bar 时间戳为**该 bar 的结束时刻**（即 09:31 的 bar = 09:30:00 ~ 09:30:59 的成交）
- **复权处理**：训练时使用前复权 OHLC（与日频 `_qfq` 一致），分红/拆股事件由数据层统一处理
- **缺数处理**：
  - 涨跌停未成交时段：volume=0、price=last close（不丢弃，用 mask 标记）
  - 临时停牌：整段 NaN，由 `instruments` 文件控制可交易区间（不进训练 batch）

---

### 3.4 数据接入工程

#### 3.4.1 拉取脚本

新建 `scripts/fetch_minute_bars.py`：

- 输入：`--start 2022-01-01 --end 2026-04-27 --instruments csi500 --freq 1min --source tushare`
- 流程：
  1. 读 `instruments/csi500.txt` 拿股票列表
  2. 分批调用 `pro.stk_mins(ts_code=..., start_date=..., end_date=..., freq='1min')`
  3. 限速：Tushare Pro 每分钟 500 次，自动 retry + sleep
  4. 落盘：`git_ignore_folder/minute_bars_raw/{code}.parquet`（按股票分文件）
  5. 增量：next-day 模式只拉最新交易日的差分
- 风险控制：
  - 拉取过程产生 detailed log（每 100 只股票一行）
  - 失败列表写入 `failed_stocks.txt` 供重试
  - md5 校验防止重复下载

#### 3.4.2 Qlib bin 转换

新建 `scripts/dump_qlib_minute.py`（参照 Qlib 官方 `dump_bin.py`）：

- 输入：`minute_bars_raw/*.parquet`
- 流程：
  1. 合并为统一 panel：`(datetime[1min], instrument) × OHLCV+amount`
  2. 复权：根据 `data/factors/adj_factor.parquet` 计算 `*_qfq`
  3. 写出 Qlib bin 格式 + `calendars/1min.txt` + `instruments/csi500.txt`
  4. sanity check：随机抽 10 只股票 × 10 个交易日，对比原始 parquet 与 bin 读取结果
- DoD：`D.features(["sh600000"], ["$close"], freq="1min", start_time="2024-01-01")` 能正常读出 240×N 行

#### 3.4.3 多频特征 Loader

新建 `feature/qlib_hf_pipeline.py`：

提供两类输出：

**输出 A：日内统计因子**（每日一行，喂给 daily GRU 或 LGB）

```text
feat_intraday[date, instrument] = [
    am_ret,          # 上午收益（09:30-11:30）
    pm_ret,          # 下午收益（13:00-15:00）
    open30_ret,      # 开盘 30min 收益
    close30_ret,     # 尾盘 30min 收益
    vwap_dev,        # 全天 VWAP / close - 1
    rv,              # realized variance（5min 收益平方和）
    bv,              # bipower variation（抗跳跃）
    rv_signed,       # 上行/下行 RV 比值
    vol_skew,        # 上午/下午成交量比
    vol_kurt,        # 日内成交量峰度
    pv_corr,         # 1min return × volume 的截面相关
    pv_lag1,         # return_t × volume_{t-1}
    big_buy_ratio,   # 单笔金额 > 100 万的成交占比（如有逐笔）
    ...              # 总计 ~25-50 维
]
```

**输出 B：原始分钟序列**（喂给 HF-GRU）

```text
feat_minute[(date, instrument)] = ndarray(240, 6)
   # 240 根 1min × [open, high, low, close, volume, amount] 标准化后
```

实现要点：

- 与现有 `feature/qlib_feature_pipeline.py` 共用 instrument filter / time window 逻辑
- 增加 `freq="1min"` 分支，调用 `D.features(...freq="1min")`
- 缓存：日内统计因子写到 `data/cache/intraday_stats.parquet`，避免每次训练重新算

---

### 3.5 RD-Agent 挖日内因子（HF-RDAgent）

> **核心思路**：现有 RD-Agent 已经在挖日频因子并产出 `combined_factors_df.parquet` 给 LGB 消费；只需做几处定向改造，就能让 RD-Agent **自动挖日内因子**，产出 `combined_intraday_factors.parquet`，供方案 A 的 daily GRU、HiGRU 的日频拼接分支、以及 LGB 同时消费。这是把 LLM 的"语义化因子生成能力"延伸到日内的最便宜路径。

#### 3.5.1 为什么 RD-Agent 适合挖日内因子

- **LLM 善于生成"语义化"因子表达式**：日内因子（开盘 30min 收益、VWAP 偏离、上下午波动率比、量价相位等）有明确语义，LLM 比人工更容易系统化遍历
- **日内 alpha 池开发不充分**：日频因子（动量、反转、价值、质量）已被广泛挖掘，剩余空间小；日内因子相对未充分开发
- **现有闭环可复用**：RD-Agent 的 hypothesis → factor_runner → qrun → IC 评估 → 入库 流程已稳定运行（见 `docs/P0-1_RUN_GUIDE.md`），改造成本低
- **Qlib 表达式语言原生支持多频率**：相同表达式 `Mean($close, 30)` 在 `freq="day"` 时是 30 日均价，在 `freq="1min"` 时是 30 分钟均价——只需切换 `qlib_init.freq` 即可

#### 3.5.2 改造点（与现有 daily RD-Agent 的差异）

| 维度 | 现有 daily RD-Agent | HF-RDAgent（新增） |
|------|---------------------|---------------------|
| `qlib_init.provider_uri` | `/mnt/d/qlib_data/qlib_data` | `{"day": "...", "1min": "/mnt/d/qlib_data/qlib_data_1min"}` |
| 因子计算频率 | `freq="day"` | **`freq="1min"`**（计算）+ 收盘聚合 → `freq="day"`（评估） |
| 因子模板目录 | `rdagent_overrides/factor_template/` | **新增 `rdagent_overrides/intraday_factor_template/`** |
| LLM prompt 算子字典 | `Mean/Std/Ref/Slope/...` 常规算子 | **扩充：`OpenAuction/CloseAuction/AmReturn/PmReturn/VWAPDev/RealizedVar/...`** |
| 评估 horizon | T+1 收益 | T+1 收益（与日频对齐，确保可融合） |
| 评估指标 | daily Rank IC + composite_score | daily Rank IC + composite_score（**完全一致**） |
| 落地 parquet | `git_ignore_folder/combined_factors_df.parquet` | **`git_ignore_folder/combined_intraday_factors.parquet`** |
| 消费方 | LGB | **daily GRU（方案 A）+ HiGRU 拼接分支 + LGB（可选）** |

**关键设计决策**：保持评估指标 100% 一致（daily Rank IC），这样：
1. 现有 promotion 门槛、quality gate、registry 流程**完全不需要改动**
2. 日内因子和日频因子可以**同台 PK**，无需为日内单独维护一套阈值
3. 因子产出后直接进入与日频因子相同的 L3 registry（version 体系不变）

#### 3.5.3 日内因子表达式样例

Qlib 表达式在 `freq="1min"` 下的日内因子可以分为 5 类：

**类 1：时段切片（最直接）**

```yaml
# 上午收益（09:30 收盘价 / 09:30 开盘价 - 1）
am_ret: "Slc($close, '11:30') / Slc($open, '09:30') - 1"

# 下午收益
pm_ret: "Slc($close, '15:00') / Slc($open, '13:00') - 1"

# 开盘 30min 收益
open30_ret: "Slc($close, '10:00') / Slc($open, '09:30') - 1"

# 尾盘 30min 收益
close30_ret: "Slc($close, '15:00') / Slc($close, '14:30') - 1"
```

> 注：`Slc(expr, "HH:MM")` 是项目需扩展的算子，等价于"取该日内时刻 bar 的值"。Qlib 原生没有，需要在 `feature/qlib_hf_pipeline.py` 注册自定义算子（约 30 行）。

**类 2：日内统计（VWAP / 波动率结构）**

```yaml
# 全日 VWAP 偏离收盘
vwap_dev: "(Sum($close * $volume, 240) / Sum($volume, 240)) / Slc($close, '15:00') - 1"

# 已实现波动率（5min 收益平方和）
realized_var: "Sum(Power($close / Ref($close, 1) - 1, 2), 240)"

# Bipower variation（抗跳跃波动）
bipower_var: "Sum(Abs($close/Ref($close,1)-1) * Abs(Ref($close,1)/Ref($close,2)-1), 240)"

# 上行/下行波动率比
up_down_var_ratio: "Sum(If($close>Ref($close,1), Power($close/Ref($close,1)-1,2), 0), 240) / Sum(If($close<Ref($close,1), Power($close/Ref($close,1)-1,2), 0), 240)"
```

**类 3：量价配合**

```yaml
# 量价相关性
pv_corr: "Corr($close/Ref($close,1)-1, $volume/Ref($volume,1)-1, 240)"

# 量在前价在后（领先指标）
pv_lead: "Corr($close/Ref($close,1)-1, Ref($volume/Ref($volume,1)-1, 5), 240)"

# 上午 vs 下午成交量比
am_pm_vol_ratio: "Sum(SlcRange($volume, '09:30', '11:30'), 1) / Sum(SlcRange($volume, '13:00', '15:00'), 1)"
```

**类 4：日内动量/反转**

```yaml
# 开盘动量延续（开盘 30min 收益 vs 全天收益的比值）
open_continuation: "(Slc($close, '10:00') / Slc($open, '09:30') - 1) / (Slc($close, '15:00') / Slc($open, '09:30') - 1)"

# 尾盘反转强度
close_reversal: "(Slc($close, '15:00') / Slc($close, '14:30') - 1) - (Slc($close, '14:30') / Slc($open, '09:30') - 1)"

# 日内最高价时刻（早高 vs 晚高的差异）
high_time_offset: "ArgMax($high, 240) / 240"  # 0~1，0.5 表示中午
```

**类 5：跳空与缺口**

```yaml
# 隔夜跳空
overnight_gap: "Slc($open, '09:30') / Ref(Slc($close, '15:00'), 1) - 1"

# 跳空修复速度（开盘 30min 内修复了多少）
gap_repair_30min: "(Slc($close, '10:00') - Slc($open, '09:30')) / (Slc($open, '09:30') - Ref(Slc($close, '15:00'), 1))"
```

LLM 在 prompt 引导下可以**生成数百个候选**，由 RD-Agent 的 IC 闸门自动筛选。

#### 3.5.4 工作流

```text
┌─────────────────────────────────────────────────────────────────────┐
│  HF-RDAgent Loop（与现有 daily RD-Agent 并行运行）                       │
└─────────────────────────────────────────────────────────────────────┘

Step 1  LLM (DeepSeek/GPT) 提议 N 个日内因子 hypothesis
        prompt 含：日内算子字典、已有因子、市场 regime 描述

Step 2  factor_runner 在 freq="1min" qlib 上计算因子值
        每个因子在 240 个 1min bar 上各有一个值

Step 3  日内 → 日频聚合：取每天最后一根 bar (15:00) 的因子值
        输出 panel: (date, instrument) → 因子值

Step 4  与日频 label (T+1 3 日收益) join，跑 SignalRecord/SigAnaRecord
        计算 daily IC / Rank IC / ICIR

Step 5  composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+turnover)
        过门槛 (IC>0.02, ICIR>0.5) 的因子写入：
        git_ignore_folder/combined_intraday_factors.parquet

Step 6  scripts/lab/prepare_rdagent_data.py 升级支持双 parquet 同步
        promote 到 L3 registry，新增字段 source="intraday"
```

#### 3.5.5 双 RD-Agent 并行（推荐拓扑）

```text
┌──────────────────────┐         ┌──────────────────────┐
│  Daily RD-Agent      │         │  HF-RDAgent (新增)    │
│  (现有)              │         │                      │
│  freq=day            │         │  freq=1min            │
│  factor_template/    │         │  intraday_factor_     │
│                      │         │  template/            │
└─────────┬────────────┘         └─────────┬────────────┘
          │                                │
          ▼                                ▼
combined_factors_df.parquet      combined_intraday_factors.parquet
          │                                │
          │                                │
          └────────┬───────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  L3 Factor Registry  │
        │  (manifest.json)     │
        │  source: daily | intraday
        └─────────┬───────────┘
                  │
       ┌──────────┼──────────────────┐
       │          │                  │
       ▼          ▼                  ▼
     LGB    daily GRU (方案 A)    HiGRU (方案 B 拼接分支)
```

**资源隔离**：
- 两个 RD-Agent 共用 LLM 配额（按 token 计费），但分别使用独立 conda env：
  - 现有 `rdagent` env → daily
  - 新增 `rdagent_hf` env（克隆 + 装多频 qlib）→ HF
- 共用 `git_ignore_folder/RD-Agent_workspace/` 但 workspace 名称用 `hf_<uuid>` 前缀区分

#### 3.5.6 与方案 A/B 的关系（重点）

| 对接方 | 作用 | HF-RDAgent 的价值 |
|--------|------|---------------------|
| **方案 A** (daily GRU + 日内统计) | HF-RDAgent **直接产出** 25-50 维日内统计因子 | ⭐⭐⭐⭐⭐ 完全替代手工日内特征 |
| **方案 B** (HiGRU) | HiGRU 的"日频拼接分支"消费 HF-RDAgent 因子 | ⭐⭐⭐⭐ 给 HiGRU 提供额外语义化日内 alpha |
| **LGB** | LGB 直接消费日内因子（横截面） | ⭐⭐⭐⭐ LGB 在日内上的能力被解锁 |

**核心论点**：HF-RDAgent 实际上让"高频数据 → GRU"变成了"高频数据 → 三家共享"，**ROI 从 +10% IC 提升到 +20%~+30%**。

#### 3.5.7 行动清单（HF-RDAgent 子阶段）

| ID | 动作 | 文件 |
|----|------|------|
| P3-RD-1 | 复制 `rdagent_overrides/factor_template/` → `intraday_factor_template/` | 新增目录 |
| P3-RD-2 | 改 `intraday_factor_template/conf_combined_factors.yaml`：`provider_uri` 加 1min 路径、`freq="1min"`、加日内时段过滤 | 改造 |
| P3-RD-3 | 注册 Qlib 自定义算子：`Slc/SlcRange/AmRet/PmRet/RealizedVar/BipowerVar/...` | `feature/qlib_hf_operators.py`（新增） |
| P3-RD-4 | 编写 HF prompt 模板：日内算子字典 + 因子语义示例 | `rdagent_overrides/intraday_factor_template/prompt_intraday.md`（新增） |
| P3-RD-5 | `scripts/lab/run_rdagent_loop.py` 增加 `--mode intraday` 选项 | 改造 |
| P3-RD-6 | `scripts/lab/prepare_rdagent_data.py` 升级支持双 parquet 同步 | 改造 |
| P3-RD-7 | `feature/qlib_hf_pipeline.py` 增加 `intraday_rdagent` 特征源分支（消费 `combined_intraday_factors.parquet`） | 改造 |
| P3-RD-8 | L3 registry schema 增加 `source: daily \| intraday` 字段 | `factor_registry/schema.py` 改造 |
| P3-RD-9 | `factor_dashboard.py` 增加 "日内因子" 子页 | 改造 |
| P3-RD-10 | 跑 5 轮 HF-RDAgent loop，验证至少 5 个日内因子通过门槛 | （执行） |

**预计时间**：2-3 周（与 Phase 3 主线并行可压到 1 周）

#### 3.5.8 风险与缓解

| 风险 | 缓解 |
|------|------|
| 1min Qlib 数据加载慢，单次 qrun 耗时翻 10x | 给 HF-RDAgent 单独的轻量评估窗口（仅 1 年训练，对比 daily 的 4 年） |
| LLM 生成的日内表达式语法错误率高 | 在 prompt 中提供 5-10 个 verified 表达式样例 + 严格的语法校验环节 |
| 日内因子换手率高 → composite_score 被 turnover 惩罚 | 复用现有 composite_score 公式，让 RDAgent 自动避开高换手因子 |
| 日内因子与日频因子高度相关，融合无增益 | 在 promotion 门槛中增加 `\|corr_with_daily_factors\| < 0.7` 检查 |
| HF-RDAgent 与 daily RD-Agent 抢 GPU/LLM 配额 | 错峰运行（HF 跑夜班，daily 跑日班） |

---

### 3.6 GRU 架构升级：三种方案

针对"高频数据 → GRU"，按从低成本到高雄心给三种方案，**推荐先做 A 再做 B**。

#### 方案 A — 日内统计 + 现有 daily GRU（**Phase 3.1**）

```text
分钟数据 → [aggregator] → 25-50 维日内因子 ─┐
                                           ├─→ 拼到现有 daily GRU 的输入 (D=70~80)
现有 daily 价量因子（30 维） ──────────────┘
                                                    │
                                                    ▼
                                            daily GRU (T=40, D=80)
                                                    │
                                                    ▼
                                                  pred
```

- **改造量**：小（仅在 `feature/qlib_feature_pipeline.py` 增加日内统计因子注入点）
- **训练成本**：与现有 daily GRU 相同
- **预期 IC 增益**：+0.003 ~ +0.008
- **风险**：低（日内统计是可解释的因子，便于调试）

#### 方案 B — Hierarchical GRU（HiGRU，**Phase 3.2，推荐主线**）

```text
分钟数据 (240, D_min=6) ─→ Intraday GRU (T_in=240, hidden=32) ─→ 日内 latent h_day (32 维)
                                                                       │
                                                                       │ 拼接日频特征
                                                                       │ + 30 维 daily 价量
                                                                       ▼
                                                       (T_out=20, D_total=62)
                                                                       │
                                                                       ▼
                                                              Cross-day GRU (hidden=64)
                                                                       │
                                                                       ▼
                                                                      pred
```

- **改造量**：中（新建 `models/hf_gru_model.py`、`datasets/hf_sequence_builder.py`）
- **训练成本**：单 epoch 约 daily GRU 的 8-12x（240×20 = 4800 步序列，比 daily 20 步增加 240x，但 batch_size 可压缩）
- **预期 IC 增益**：+0.008 ~ +0.018
- **可选增强**：
  - Intraday GRU 可换为 PatchTST（patch_len=20min，更高效）
  - Cross-day 用 Mamba2 处理超长序列
  - Intraday encoder 共享参数 vs 每股票独立（建议共享，减少过拟合）

#### 方案 C — 双流 + 路由器（**Phase 3.3，雄心方案**）

```text
分钟流：分钟数据 → Intraday Encoder (PatchTST) → h_intraday (64 维)
                                                       │
日频流：日频特征 → Daily Encoder (GRU/Transformer) → h_daily (64 维)
                                                       │
                                                       ▼
                                            Gating Router (TRA-style)
                                                  │   │   │
                                                  ▼   ▼   ▼
                                                head1 head2 head3
                                                       │
                                                       ▼
                                                      pred (Sinkhorn-routed)
```

- 灵感来自 Qlib `TRA` + LPatchTST
- 训练复杂，但**对市场状态切换（牛熊、波动率突变）最稳**
- 列为 Phase 3.3 选择性目标，不强制

---

### 3.7 训练范式（HiGRU 方案 B 的细节）

#### 3.7.1 Sequence Builder

新建 `datasets/hf_sequence_builder.py`：

- 输入：`feat_minute` panel (`(date, instrument)` 索引 → `(240, 6)` ndarray)
- 输出：
  ```text
  X_intraday: (N, T_out=20, T_in=240, D_min=6)  float32
  X_daily   : (N, T_out=20, D_daily=30)          float32
  endpoint  : (N,) MultiIndex
  ```
- 约束：
  - `T_out` 个交易日严格连续（与现有 `sequence_builder` 一致）
  - 每天必须有完整 240 根分钟（否则丢弃；或允许 ≥ 230 根并 mask）
  - 跨日不接（中间放 mask token）

#### 3.7.2 网络实现要点

```python
class HFGRUNet(nn.Module):
    def __init__(self, d_min, d_daily, intraday_hidden=32, cross_day_hidden=64, T_in=240, T_out=20):
        self.intraday_gru = nn.GRU(d_min, intraday_hidden, num_layers=1, batch_first=True)
        self.cross_day_gru = nn.GRU(intraday_hidden + d_daily, cross_day_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(cross_day_hidden * 2, 1)

    def forward(self, x_intra, x_daily):
        # x_intra: (B, T_out, T_in, D_min)
        # x_daily: (B, T_out, D_daily)
        B, T_out, T_in, D_min = x_intra.shape
        flat = x_intra.reshape(B * T_out, T_in, D_min)
        _, h_in = self.intraday_gru(flat)  # h_in: (1, B*T_out, intraday_hidden)
        h_day_token = h_in.squeeze(0).reshape(B, T_out, -1)  # (B, T_out, intraday_hidden)
        x_combined = torch.cat([h_day_token, x_daily], dim=-1)
        h_seq, _ = self.cross_day_gru(x_combined)
        last = h_seq[:, -1, :]
        return self.head(last).squeeze(-1)
```

- Intraday encoder 输出**只取 hidden state**（不返回每分钟），避免输出爆炸
- Cross-day 用双向 GRU + 2 层，相当于 Phase 1 升级后的 daily GRU 架构

#### 3.7.3 训练超参（建议初始值）

```yaml
# config/model_hf_gru.yaml
model:
  T_in: 240             # 日内分钟数
  T_out: 20             # 跨日窗口
  d_min: 6              # 分钟特征维度（OHLCV+amount）
  d_daily: 30           # 日频特征维度
  intraday_hidden: 32
  cross_day_hidden: 64
  num_layers_cross: 2
  bidirectional: true
  dropout: 0.3
  lr: 1.0e-4            # 比 daily GRU 小（参数更多）
  weight_decay: 5.0e-4
  batch_size: 64        # 比 daily GRU 小（显存约束）
  max_epochs: 30
  patience: 5
  loss: rankic          # Phase 1 的 loss 改造延续
  amp: true             # 混合精度必开
  device: cuda
```

#### 3.7.4 训练流程

1. **预处理阶段**（一次性）：
  - 拉取分钟数据 → 转 Qlib bin（约 4 小时）
  - 计算分钟标准化统计（按股票时序 z-score 的 mean/std），写入 `data/cache/minute_norm_stats.parquet`
2. **滚动训练**（每次 train_days 推进）：
  - `RollingTrainer.fit_one_window` 增加 `hf_gru` 模型分支
  - HF-GRU 单独使用 `gru: 240` 训练天数（不影响 LGB 的 480）
  - DataLoader 用 `num_workers=4`、`pin_memory=true`
3. **OOF 生成**：
  - HF-GRU 输出加入 `meta_oof.parquet`，新增列 `pred_hf_gru`
  - Ridge 系数自动学习（如 HF-GRU IC 强会被分配更高权重）

---

### 3.8 推理与回测对齐

#### 3.8.1 推理流程（关键约束）

- **特征延迟**：T 日预测使用截至 **T-1 日 15:00** 的分钟数据（不可包含 T 日盘中数据，避免泄露）
- **滚动推理**：与 daily GRU 的推理对齐，输出仍是 daily 频次的截面预测分数
- **缺数处理**：T-1 日某股票分钟数据缺失（停牌等）→ 跳过该股票，不出预测

#### 3.8.2 回测对齐

- 回测仍按日频 T+1 执行（与现有 RQAlpha 对齐），HF-GRU 不改变交易频率
- 回测数据加载端：先确保 T 日预测时 T-1 日分钟数据完整，否则该 T 日不出仓
- 实盘部署：VNPY/CTP 推送实时 1min bar → 当日 14:57 触发推理 → 14:59 提交订单

#### 3.8.3 与 LGB 的融合

- `pipeline.yaml.base_models = ["lgb", "gru", "patchtst", "hf_gru"]`
- meta_oof 现已支持 N 元（Phase 2 已升级）
- 验证 `pred_hf_gru` 与 `pred_lgb` 的相关系数 < 0.6（信息独立性指标）

---

### 3.9 资源估算

#### 3.9.1 存储


| 组件                          | 大小          |
| --------------------------- | ----------- |
| 1min Qlib bin（csi500 × 4 年） | ~600 MB     |
| Tushare 拉取的原始 parquet       | ~1 GB       |
| 日内统计因子缓存                    | ~50 MB      |
| 训练中间产物（pickle、checkpoint）   | ~2 GB       |
| **合计**                      | **~3.5 GB** |


#### 3.9.2 计算

- **Tushare 拉取**：5000 积分/分钟限速下，csi500 × 4 年 1min 数据约 4-6 小时
- **Qlib bin 构建**：单次 ~30 分钟
- **HF-GRU 单 epoch（RTX 3090）**：
  - 序列长度 240×20=4800，但 hidden 较小
  - 估算 8-15 分钟/epoch（vs daily GRU ~1 分钟）
  - 30 epochs × 5 分钟（早停）≈ 2.5 小时/滚动窗口
- **完整滚动训练**（按 step_days=10 推进 6 个月数据）：约 18-25 小时

#### 3.9.3 推理

- 单日推理：HF-GRU 处理 csi500 × 240×20 序列约 30 秒（GPU），完全在 14:57 决策窗口内

---

### 3.10 Phase 3 行动清单


| ID                         | 动作                                                        | 文件                                   | 预计时间  |
| -------------------------- | --------------------------------------------------------- | ------------------------------------ | ----- |
| **3.1 数据基础**               |                                                           |                                      |       |
| P3-1                       | 申请 Tushare Pro 5000 积分                                    | （非代码）                                | 1 天   |
| P3-2                       | `scripts/fetch_minute_bars.py` 拉取脚本                       | 新增                                   | 2 天   |
| P3-3                       | `scripts/dump_qlib_minute.py` Qlib bin 转换                 | 新增                                   | 3 天   |
| P3-4                       | `feature/qlib_hf_pipeline.py` 多频 Loader                   | 新增                                   | 3 天   |
| P3-5                       | 1min 数据完整性 sanity check 报告                                | `data/hf/data_quality_report.md`（新增） | 1 天   |
| **3.2 HF-RDAgent（推荐与 3.3 并行）** |                                                       |                                      |       |
| P3-RD-1                    | 复制并改造 `rdagent_overrides/intraday_factor_template/`     | 新增目录                                 | 1 天   |
| P3-RD-2                    | 注册 Qlib 自定义算子（`Slc/AmRet/RealizedVar/...`）             | `feature/qlib_hf_operators.py`（新增）   | 3 天   |
| P3-RD-3                    | 编写 HF prompt 模板（日内算子字典 + 5-10 个 verified 表达式样例）          | `intraday_factor_template/prompt_intraday.md`（新增） | 2 天   |
| P3-RD-4                    | `run_rdagent_loop.py` 增加 `--mode intraday` 选项            | 改造                                   | 1 天   |
| P3-RD-5                    | `prepare_rdagent_data.py` 升级支持双 parquet 同步               | 改造                                   | 1 天   |
| P3-RD-6                    | L3 registry schema 增加 `source: daily \| intraday` 字段     | `factor_registry/schema.py` 改造        | 0.5 天 |
| P3-RD-7                    | 跑 5 轮 HF-RDAgent loop，验证至少 5 个日内因子通过门槛                  | （执行）                                 | 3 天   |
| P3-RD-8                    | HF-RDAgent 验收：`combined_intraday_factors.parquet` 含 ≥ 20 个因子 | （报告）                                | 0.5 天 |
| **3.3 方案 A：日内统计**          |                                                           |                                      |       |
| P3-6                       | 在 `feature/qlib_feature_pipeline.py` 增加 `intraday_rdagent` 特征源分支（消费 HF-RDAgent parquet） | 改造                                   | 2 天   |
| P3-7                       | 重训现有 daily GRU（输入维度扩展 30→50~70），验证 IC 提升                  | （配置）                                 | 1 天   |
| P3-8                       | A 验收：valid IC ≥ 当前 + 0.005（注：因有 HF-RDAgent 增益，门槛比纯手工高）    | （报告）                                 | 0.5 天 |
| **3.4 方案 B：HiGRU**         |                                                           |                                      |       |
| P3-9                       | `datasets/hf_sequence_builder.py` 多层级序列构建器                | 新增                                   | 3 天   |
| P3-10                      | `models/hf_gru_model.py` HiGRU 实现                         | 新增                                   | 5 天   |
| P3-11                      | `config/model_hf_gru.yaml` 超参配置                           | 新增                                   | 0.5 天 |
| P3-12                      | `model_registry.py` 注册 `"hf_gru"`                         | 改造                                   | 0.5 天 |
| P3-13                      | `pipeline.yaml.base_models` 扩到包含 `hf_gru`                 | 改造                                   | 0.5 天 |
| P3-14                      | `meta_oof_builder` 增加 `pred_hf_gru` 列                     | 改造（已 Phase 2 通用化）                    | 0.5 天 |
| P3-15                      | `_load_ic_histories` 增加 `ic_hf_gru` 读取                    | 改造                                   | 0.5 天 |
| P3-16                      | 滚动训练 6 个月数据，输出 OOF + valid 指标                             | （执行）                                 | 2-3 天 |
| P3-17                      | B 验收：HF-GRU 单模型 IC ≥ daily GRU + 0.005，组合 ICIR ≥ 当前 + 10% | （报告）                                 | 1 天   |
| **3.5 监控与可视化**             |                                                           |                                      |       |
| P3-18                      | `scripts/factor_dashboard.py` 增加 HF-GRU 状态卡 + 日内因子子页      | 改造                                   | 1.5 天 |
| P3-19                      | 高频数据健康监控（每日凌晨自动跑 sanity check）                            | `scripts/hf_data_health.py`（新增）      | 1 天   |
| **3.6 实盘对接（可选 Phase 3.4）** |                                                           |                                      |       |
| P3-20                      | VNPY 接 CTP 实时 1min → 写入 Qlib bin（增量）                      | `scripts/vnpy_minute_writer.py`（新增）  | 5 天   |
| P3-21                      | 实盘 14:57 推理触发 → 订单生成                                      | `live/hf_inference_runner.py`（新增）    | 5 天   |


**总时间估算**：
- 数据基础（P3-1 ~ P3-5）约 1.5 周
- HF-RDAgent（P3-RD-1 ~ P3-RD-8）约 2 周（**可与方案 A/B 并行**）
- 方案 A（P3-6 ~ P3-8）约 0.5 周
- 方案 B（P3-9 ~ P3-17）约 3 周
- **P3 主线 5-6 周**完成（HF-RDAgent 与方案 A/B 重叠时压缩到 5 周）
- 可选实盘对接（P3-20 ~ P3-21）再加 2 周

### 3.11 验收门槛（DoD）

- **数据层**：1min Qlib bin 可被 `D.features(...freq="1min")` 正常读取，csi500 × 2 年覆盖率 ≥ 99%
- **HF-RDAgent**：
  - `combined_intraday_factors.parquet` 含 ≥ 20 个通过门槛的日内因子
  - 至少 5 个日内因子的 daily Rank IC ≥ 0.025
  - 日内因子与现有日频因子的相关系数中位数 ≤ 0.5（信息独立）
- **方案 A**（融合 HF-RDAgent 因子）：daily GRU 加入日内因子后 valid IC 提升 ≥ 0.005，无回归
- **方案 B**（HiGRU）：
  - HF-GRU 单模型 valid IC ≥ daily GRU + 0.005
  - HF-GRU 与 LGB 预测 Pearson 相关系数 ≤ 0.6（信息独立）
  - 加入 HF-GRU 后组合 ICIR ≥ 当前 + 10%
  - 在双倍交易成本下回测 Sharpe 提升 ≥ 0.15
- **稳定性**：滚动 6 个月期间 HF-GRU 权重不出现单点 → 0 或 → max_weight 的极端值

### 3.12 风险与回退


| 风险                        | 概率  | 影响            | 回退方案                                   |
| ------------------------- | --- | ------------- | -------------------------------------- |
| Tushare 限速导致拉取失败          | 中   | 数据缺失          | 落到 5min 起步，或换 JoinQuant                |
| 1min 数据存在大量噪声             | 中   | HF-GRU IC 不达标 | 降级到方案 A（日内统计）                          |
| HF-GRU 训练显存爆              | 中   | 训练失败          | 减小 batch_size，或 gradient checkpointing |
| HF-GRU 推理延迟过高             | 低   | 实盘失败          | 改为 5min 输入（48 步而非 240 步）               |
| HF-GRU 与 LGB 信号实际相关 > 0.7 | 低   | 融合无增益         | 引入路由器（方案 C），或换 PatchTST                |


---

## 4. 总体时间线


| 周           | Phase           | 主要交付                                       | 累计成果                                       |
| ----------- | --------------- | ------------------------------------------ | ------------------------------------------ |
| W1          | Phase 1         | P1-1 ~ P1-6 完成；GRU 真正参与最终加权                | GRU 在融合中权重稳定在 [10%, 50%]                   |
| W2-W4       | Phase 2         | PatchTST 引入；3 模型组合上线                       | base_models = ["lgb","gru","patchtst"]     |
| W5-W6       | Phase 3.1       | 1min 数据接入（拉取 + Qlib bin + Loader）          | 1min 数据可被 `D.features(...,freq="1min")` 读取  |
| W6-W7       | **Phase 3.2**   | **HF-RDAgent 启动 → 自动产出日内因子**（与 3.3 并行）    | `combined_intraday_factors.parquet` 含 ≥ 20 因子 |
| W7          | Phase 3.3       | 方案 A：daily GRU 消费日内因子（含 HF-RDAgent 产出）    | daily GRU IC ↑                             |
| W8-W10      | Phase 3.4       | HiGRU 完整训练 + 上线                            | HF-GRU 进入 base_models                      |
| W11         | 收尾              | Dashboard 集成 + 报告                          | 4 模型组合稳定运行                                 |
| W12-W13（可选） | Phase 3.5       | 双流 + 路由器（方案 C）                             | 雄心方案                                       |
| W14-W15（可选） | Phase 3.6       | VNPY 实盘对接                                  | HF-GRU 上实盘                                 |


**核心节点**：

- **W1 末**：GRU 在最终预测里权重 ≥ 10%
- **W4 末**：3 模型组合 ICIR ≥ 当前 + 5%
- **W7 末**：HF-RDAgent 已产出 ≥ 20 个日内因子并入 L3 registry；daily GRU + 日内因子 IC ↑
- **W10 末**：4 模型（LGB + daily GRU + PatchTST + HF-GRU）组合 ICIR ≥ 当前 + 15%

---

## 5. 与现有路线图的衔接

- 本文档的 Phase 1/2/3 与 `docs/OPTIMIZATION_ROADMAP_2026.md` 的 P0/P1/P2 **互不冲突**
- **优先级建议**：先完成 OPTIMIZATION_ROADMAP_2026 的 P0（闭环打通），再启动本文档 Phase 1
- HF-GRU 上线后，OPTIMIZATION_ROADMAP_2026 的 P0-3（4 基模型扩容）实际产物为：`["lgb_rd", "gru", "patchtst", "hf_gru"]`，与原计划的 `["lgb", "lgb_rd", "gru", "mlp_resid"]` 相比更注重时序模型差异化
- **HF-RDAgent 是现有 RD-Agent 闭环的"日内分支"**：复用 hypothesis loop / IC 闸门 / promotion 流程；产出与 daily 因子统一进入 L3 registry，仅 `source` 字段区分

---

## 6. 待用户确认的决策点

1. **数据源选型**：是否同意以 Tushare Pro `stk_mins` 为主、Akshare 为备份？需要先确认 Tushare 积分预算（约 ¥2000/年）
2. **频率起点**：第一阶段从 5min 还是直接 1min？1min 数据量 5x 但信息更丰富
3. **方案选择**：Phase 3 是按 A→B 顺序推进，还是直接跳到 B？A 改造小、风险低；B 增益更大但工程量大 3 倍
4. **HF-RDAgent 启动时机**：与方案 A/B 并行（推荐，节省 1-2 周）还是顺序（先 A/B 完成再启动 HF-RDAgent）？
5. **HF-RDAgent LLM 配额**：HF-RDAgent 与 daily RD-Agent 共用 LLM API 时 token 消耗预估翻倍，是否扩容 LLM 预算？
6. **实盘对接（Phase 3.6）**：是否在本路线图内完成？还是仅做训练 + 离线回测，实盘对接放到 2026 H2 末
7. **GPU 资源**：HF-GRU 训练对 GPU 要求显著（推荐 RTX 3090/4090 24GB），是否需要扩容

确认后我将按照路线图开始 Phase 1 的代码改动。