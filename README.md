# Qlib 因子选股与回测系统

面向生产级的多模型量化因子工程，基于 Qlib 框架构建，覆盖特征抽取、LightGBM/MLP/Stack 模型训练、IC 动态加权预测以及带风控的组合构建与回测。

## 1. 目录结构

```
project/
├── config/                # 全局配置文件（数据、模型、pipeline）
├── data/                  # 模型、日志、预测、回测等持久化目录
├── feature/               # 特征工程模块
├── models/                # LightGBM、MLP、Stack 模块
├── trainer/               # 滚动训练器
├── predictor/             # 多模型预测与动态加权
├── portfolio/             # 风控约束下的组合构建
├── utils/                 # 工具方法（配置、数据集、持久化）
├── run_train.py           # 训练脚本
├── run_predict.py         # 预测脚本
└── run_backtest.py        # 简易回测脚本
```

## 2. 功能概述

- **特征工程**：`QlibFeaturePipeline` 基于 `D.features` 提取行情与因子数据，自动对齐标签 `Ref($close,-5)/$close - 1`，并进行标准化。
- **模型体系**：
  - `LightGBMModelWrapper` 封装 qlib 原生 LightGBM，输出预测值与叶子索引；
  - `MLPRegressor` 由 PyTorch 实现的一层或多层感知机；
  - `LeafStackModel` 支持 OneHot 或哈希压缩方式处理 LGB 叶子索引，用 MLP 学习 residual 并与原 LGB 预测融合，避免稀疏矩阵爆内存。
- **多模型协同**：`EnsembleModelManager` 基于 qlib `AverageEnsemble` 统一训练/推理 LGB、MLP 等模型，可通过 `pipeline.yaml -> ensemble` 增减模型并获取融合预测。
- **IC 动态加权**：`RankICDynamicWeighter` 基于 rank-IC 半衰期均值与波动计算权重，支持 min/max/负值裁剪。
- **组合构建**：`PortfolioBuilder` 设定最大仓位、单股权重与行业敞口，生成最终权重。
- **滚动训练**：`RollingTrainer` 按 `pipeline.yaml` 的窗口设置执行训练、评估、模型保存以及训练日志记录。
- **预测 & 回测**：`PredictorEngine` 加载多模型并执行动态加权，`run_backtest.py` 使用预测结果与标签收益计算简单净值曲线。

## 3. 快速开始

1. **准备数据**：确保已经通过 `qlib` 下载对应市场的数据（默认为 `~/.qlib/qlib_data/cn_data`）。如需调整行情范围、特征或标签，请编辑 `config/data.yaml`。
2. **训练模型**：
   ```bash
   cd project
   python run_train.py --config config/pipeline.yaml
   ```
   训练过程中会按月度窗口滚动，结果分别写入 `data/models` 与 `data/logs/training_metrics.csv`。
3. **生成预测**：
   ```bash
   python run_predict.py --config config/pipeline.yaml --start 2024-01-01 --end 2024-01-31 --tag auto
   ```
   其中 `--tag auto` 会自动选择最新训练窗口；预测文件输出到 `data/predictions`。
4. **组合回测**：
   - **简化回测**（快速验证，无交易成本）：
     ```bash
     python run_backtest.py --config config/pipeline.yaml --prediction data/predictions/pred_<tag>.csv
     ```
   - **RQAlpha 真实回测**（T+1 交易，包含手续费、滑点）：
     ```bash
     python run_backtest.py --config config/pipeline.yaml --use-rqalpha --rqalpha-config config/rqalpha_config.yaml
     ```
   行业文件需包含 `instrument,industry` 列，可选；回测结果写入 `data/backtest/` 目录。

## 4. 配置说明

- `config/data.yaml`：数据提供者地址、交易品种、时间区间、因子列表、标签与标准化方式。
- `config/model_lgb.yaml`：LightGBM 超参（叶子数、学习率、bagging 等）。
- `config/model_mlp.yaml`：MLP 网络结构、训练批量、学习率等。
- `config/model_stack.yaml`：Stack MLP 结构、`alpha`（融合权重）、`encoding/hash_dim` 等叶子编码策略。
- `config/pipeline.yaml`：滚动窗口长度、IC 动态加权窗口、模型/日志/预测/回测路径，以及组合约束参数；新增 `ensemble` 区块用于声明多模型名单、配置引用与 qlib 融合策略。

## 4.1 启用 GRU（只改配置即可）

本工程支持将 GRU 作为第三个基模型接入，并自动参与：
- **序列数据集构建**：同一股票过去 T=60 个交易日的扁平特征堆叠为 `[T, D]`
- **walk-forward folds 的 OOF 生成**（与其它基模型共享同一套 split）
- **Meta-Stacking**：`X_meta = [pred_lgb, pred_mlp, pred_gru, ...]` 自动包含 GRU 列

只需要在 `config/pipeline.yaml` 做两处配置：

1) 在 `model_gru:` 中填写超参（T、hidden、layers、dropout、lr、batch_size、epochs、patience 等）

2) 在 `base_models:` 中加入 `"gru"`：

```yaml
base_models: ["lgb", "mlp", "gru"]
oof_stacking:
  enabled: "auto"   # 推荐保持 auto：当 base_models 含 gru 时自动开启 OOF+MetaStacking
```

## 4.2 训练 / 推理 / 缓存（README 风格使用说明）

### 训练（跑全流程）

```bash
python run_train.py --config config/pipeline.yaml
```

说明：
- 会按 `rolling` 滚动窗口训练 LGB/MLP/（可选 GRU）
- 若 `base_models` 包含 `gru` 且 `oof_stacking.enabled=auto/true`：
  - 会自动按 time-series/walk-forward folds 训练各 fold，生成 OOF
  - 会训练并保存二层 `MetaStacker`（Ridge/Linear）

### 推理（加载已训练模型，在最新日期输出信号）

```bash
python run_predict.py --config config/pipeline.yaml --start 2025-01-01 --end 2025-01-31 --tag auto
```

说明：
- `--tag auto` 会选择最新窗口的 tag（`YYYYMMDD`）
- 若该 tag 下存在 `*_meta.pkl`（并启用了 oof_stacking），最终 `final` 会走同一个元模型输出；否则回退到原有动态加权 `final`

### 缓存与复用（避免重复训练）

- **OOF 缓存目录**：`data/oof/{tag}/`
  - `{model_name}_{fold}.npy`
  - `y_{fold}.npy`
  - `index_{fold}.pkl`
  - 由 `paths.oof_dir` 控制，且 `oof_stacking.use_cache=true` 时自动复用已有缓存

- **模型权重目录**：`data/models/`
  - LGB：`{tag}_lgb.txt` / `{tag}_lgb_meta.json`
  - MLP：`{tag}_mlp.pt` / `{tag}_mlp_meta.json`
  - GRU：`{tag}_gru.pt` / `{tag}_gru_meta.json`
  - MetaStacker：`{tag}_meta.pkl` / `{tag}_meta_meta.json`

- `config/rqalpha_config.yaml`：RQAlpha 回测配置，包含回测周期、初始资金、手续费（万分之3）、滑点（万分之1）、交易限制等参数。

## 5. 扩展建议

- **自定义特征**：可在 `data.yaml` 中追加表达式或扩展 `feature/qlib_feature_pipeline.py` 以对接外部特征。
- **更丰富的模型**：可在 `models/` 中新增模型并在训练/预测器中注册。
- **真实回测**：目前 `run_backtest.py` 基于标签收益，如需交易级回测，可结合 qlib 内置 `backtest` 模块或接入实盘模拟。
- **线上部署**：可将 `run_predict.py` 嵌入任务调度，定期生成预测与组合信号，再结合 OMS/交易执行。
- **工程细节**：详见 `docs/ENGINEERING.md`，包含信号解释、回测流程等代码级文档。
- **流程图**：详见 `docs/WORKFLOW.md`，包含完整的系统架构、数据流、模块依赖等可视化流程图。

## 6. 依赖环境

- Python 3.10+
- `qlib` 最新版本（需提前 `pip install pyqlib` 并运行 `qlib.init` 所需的数据准备）
- 其他依赖：`lightgbm`, `pytorch`, `pandas`, `numpy`, `scikit-learn`, `pyyaml`
- **RQAlpha 回测**（可选）：`pip install rqalpha`，用于真实 T+1 交易回测

在生产部署时，建议使用虚拟环境并按需启用 GPU（MLP 支持 `cuda` 自动切换）。


