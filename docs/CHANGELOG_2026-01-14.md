## 说明文档：2026-01-14 本次迭代改动全记录（训练/推理/OOF/Stacking/GRU）

> 这份文档用“故事线”把今天做的所有改动串起来：从 **按日滚动训练** → **新增 GRU 基模型** → **OOF 缓存** → **方案A：LGB+GRU+Ridge Stacking** → **推理端自动复现** → **若干线上报错修复**。
>
> 目标是：你明天回头看也能一眼明白“为什么这么改、改了哪里、怎么用、出错怎么排查”。

---

## 一、最终我们做到了什么（给老板/同事的 30 秒版本）

- **训练频率从按月变为按日滚动**（窗口/步长均支持 day-based）。
- **新增 GRU 基模型**（序列输入 `[T=60, D]`，不足窗口样本丢弃，强断言不泄露未来）。
- **引入统一 walk-forward folds 的 OOF 生成与缓存**，并支持复用缓存跳过重训。
- **落地方案A：LGBM + GRU + Ridge Stacking（OOF）**
  - 训练：严格使用 OOF 构建 `meta_oof` → 按日横截面标准化 → Ridge 拟合 → 保存 `meta_ridge.json`
  - 推理：LGB/GRU test 预测按 `(date, code)` join → 同样标准化 → Ridge 输出 `final_score`
- **把“只改配置即可启用/停用模型”做成可用机制**：`base_models: ["lgb","gru"]` 即可关闭 MLP 训练、启用 GRU。

---

## 二、改动涉及的核心模块与数据流（从入口讲起）

### 2.1 训练入口：`run_train.py` → `trainer/RollingTrainer`

训练流程主干还是：

1. `run_train.py` 读取 `config/pipeline.yaml`
2. 解析股票池，给每个池生成临时 data/pipeline config（保持原结构）
3. 对每个池调用 `trainer/RollingTrainer(...).train()`

今天额外做了两点“工程加固”：

- 临时文件清理改为“安全删除”，避免 Windows 下 `FileNotFoundError/IndentationError`。
- 修复了 `config/model_stack.yaml` 的 YAML 缩进问题（之前会导致训练在初始化 stack 阶段直接崩溃）。

### 2.2 “按日滚动训练”

在 `trainer/trainer.py` 的窗口生成逻辑里：如果 `pipeline.yaml` 中存在 `train_days/valid_days/step_days`，则按 `pd.Timedelta(days=...)` 生成滚动窗口；否则兼容旧的按月模式（`pd.DateOffset(months=...)`）。

这让同一套训练器同时支持：
- 老配置（按月）
- 新配置（按日）

### 2.3 基模型声明：`base_models`（只改配置即可启用/停用）

新增了“配置驱动”的模型选择方式：

- 在 `config/pipeline.yaml` 里声明：
  - `base_models: ["lgb", "gru"]`（今天实际用这个来停用 MLP）
  - `model_gru:`（GRU 超参段，直接写在 pipeline 里）
- `models/ensemble_manager.py` 若发现 `ensemble.models: []`，则自动从 `base_models` 派生 specs：
  - lgb → 用 `lightgbm_config`（路径）
  - gru → 优先用 `model_gru`（dict 配置）

这样你只改配置，就能从 **LGB+MLP** 切到 **LGB+GRU**，无需动代码。

---

## 三、GRU 基模型：输入是什么、怎么保证不泄露未来、怎么训练与保存

### 3.1 GRU 模型位置与最小接口

- 文件：`models/gru_model.py`
- 类：`GRURegressor`
- 接口：`fit/predict/save/load`

该接口与工程现有模型（例如 `LightGBMModelWrapper`）的调用方式一致：能被 `EnsembleModelManager` 与 `OOFManager` 统一调度。

### 3.2 序列构建器（可复用组件）

- 文件：`datasets/sequence_builder.py`
- 函数：`build_panel_sequences(feat, seq_len=60, ...)`

关键约束：

- **同一股票**按 **交易日排序**，取过去 `T=60` 个交易日的扁平特征 `[D]` 堆叠成 `[T, D]`
- 不足窗口的样本 **直接丢弃**（先不做 padding/mask，避免引入复杂度）
- 强断言：
  - 序列最后一天特征 == 样本当天特征
  - 序列日期严格递增且不包含未来日期

并提供了 `unittest`：

- `tests/test_sequence_builder.py`

### 3.3 训练工程能力补齐（可复现、AMP、梯度裁剪、早停、best checkpoint）

GRU 训练增强点：

- 固定随机种子（python/numpy/torch/cudnn）：`utils/torch_utils.py`
- AMP（可配置开关）
- 梯度裁剪：`clip_grad_norm_`（默认 1.0）
- EarlyStopping：可监控 `loss` 或 `rankic`
- checkpoint：保存 best 权重与训练 history，`load()` 后可直接 `predict()`

配置文件：

- `config/model_gru.yaml`（默认 `seq_len: 60`）
- `config/pipeline.yaml` 中也有一份 `model_gru:` dict（优先使用）

---

## 四、OOF（walk-forward folds）是怎么生成、缓存、复用的

### 4.1 Fold 切分逻辑（与 LGB 完全一致）

- 文件：`trainer/oof_manager.py`
- 类：`TimeSeriesFoldSplitter`

切分是“日期级别”的 walk-forward：

- `split(dt_index)` 返回 `List[Fold]`
- `Fold` 里包含：
  - `train_dates: pd.Index`
  - `valid_dates: pd.Index`

之后在 `OOFManager` 里用日期 mask 把 `MultiIndex(datetime,instrument)` 的全局样本切出训练/验证样本，确保：

- 所有模型共享同一套 folds
- OOF 对齐严格基于 index，不依赖数组顺序

### 4.2 OOF 缓存目录与命名（per-fold）

默认目录：

- `paths.oof_dir` → `data/oof/{tag}/`

缓存文件：

- `index_{fold}.pkl`：验证集 MultiIndex（全局样本索引体系）
- `y_{fold}.npy`：验证集标签数组（顺序与 index 对齐）
- `{model_name}_{fold}.npy`：该模型 OOF 预测数组（顺序与 index 对齐）

### 4.3 统一 meta_oof.parquet（供方案A训练 Ridge）

训练阶段（主流程内）额外落盘：

- `paths.meta_dir/{tag}_meta_oof.parquet`（默认 `data/meta/`）

包含列：

- `[date, code, fold, y, lgb, gru]`

这样后续 Ridge 训练不需要再“猜数组顺序”，全部按 `(date, code)` join。

---

## 五、方案A：LGB + GRU + Ridge Stacking（OOF）

### 5.1 按日横截面标准化（训练与推理一致）

- 文件：`utils/normalize.py`
- 函数：`normalize_by_date(df, cols, date_col="date", mode="demean|zscore", eps=1e-6)`

特性：
- 只按当日截面计算 mean/std（不跨天、不用未来）
- 提供测试：`tests/test_normalize_by_date.py`

### 5.2 meta_oof 构建（严格 join key + 强断言）

- 文件：`utils/meta_oof_builder.py`
- 函数：`build_meta_oof(lgb_oof_path, gru_oof_path, y_path, out_path, norm_mode, norm_eps)`

强约束：
- 只能用 `(date, code)` join（禁止假设顺序一致）
- 断言：
  - 无重复 key
  - fold 一致
  - `pred_lgb/pred_gru` 非 NaN（NaN 则统一过滤）
- 输出：`meta_oof.parquet`（列：`[date, code, fold, y, pred_lgb, pred_gru]`）

### 5.3 Ridge 训练与推理（参数 JSON 化，推理复现列顺序与 normalize）

- 文件：`utils/meta_ridge.py`
- 训练：`train_meta_ridge(meta_oof_path, out_json, alpha=1.0, grid=None, norm_mode, norm_eps)`
  - 打印/保存：`coef_ / intercept_ / OOF RankIC / MSE`
  - 保存到：`{meta_dir}/{tag}_meta_ridge.json`
- 推理：`predict_meta_ridge(lgb_pred_path, gru_pred_path, ridge_json, out_path)`
  - `(date, code)` join
  - 同样 `normalize_by_date`
  - 输出：`date, code, final_score`

### 5.4 已集成到主流程（训练与推理）

训练侧：
- `trainer/trainer.py`：当 `oof_stacking.enabled=auto` 且 base_models 含 gru
  - 自动生成 OOF
  - 自动落 meta_oof.parquet
  - 自动训练 ridge 并保存 meta_ridge.json

推理侧：
- `predictor/predictor.py`：优先尝试加载 ridge 参数并做融合；失败会清晰日志并回退到原逻辑。

---

## 六、配置怎么写（最小可跑）

```yaml
base_models: ["lgb", "gru"]
ensemble:
  models: []   # 让系统从 base_models 自动派生

paths:
  model_dir: "data/models"
  oof_dir: "data/oof"
  meta_dir: "data/meta"

model_gru:
  seq_len: 60
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  lr: 0.0005
  batch_size: 1024
  max_epochs: 20
  patience: 5

oof_stacking:
  enabled: "auto"
  use_cache: true
  n_splits: 5
  valid_days_per_fold: 20
  min_train_days: 120
  meta_model:
    model_type: "ridge"
    alpha: 1.0
  normalize_mode: "zscore"
  normalize_eps: 1e-6
```

---

## 七、怎么运行（训练/推理）

### 训练（全流程）

```bash
python run_train.py --config config/pipeline.yaml
```

关键输出位置：

- 模型：`data/models/{tag}_lgb.txt`、`data/models/{tag}_gru.pt`、（以及 stack/归一化参数等）
- OOF 缓存：`data/oof/{tag}/`
- meta 文件：
  - `data/meta/{tag}_meta_oof.parquet`
  - `data/meta/{tag}_meta_ridge.json`

### 推理（最新 tag）

```bash
python run_predict.py --config config/pipeline.yaml --start 2025-01-01 --end 2025-01-31 --tag auto
```

若 ridge 文件存在且配置开启，将自动做：
- join（date, code）
- normalize_by_date（与训练一致）
- ridge 输出 final_score

---

## 八、今天修过的两个“真·拦路虎”报错（排查/复盘）

### 8.1 `IndentationError`（run_train.py 临时文件删除）

现象：
- `IndentationError: expected an indented block after 'if'`

处理：
- 修复 `os.unlink(...)` 的缩进，统一用安全删除（先 exists 再 unlink）。

### 8.2 `ParserError`（model_stack.yaml）

现象：
- YAML 缩进错误导致 stack 初始化失败。

处理：
- 统一缩进为 `stack:` 下的键值对结构（见本文件上文）。

---

## 九、附录：今天新增/改动文件清单（便于 code review）

新增：
- `models/gru_model.py`
- `datasets/sequence_builder.py`
- `trainer/oof_manager.py`
- `models/meta_stacker.py`
- `utils/meta_oof_builder.py`
- `utils/meta_ridge.py`
- `utils/normalize.py`
- `utils/torch_utils.py`
- `tests/test_sequence_builder.py`
- `tests/test_normalize_by_date.py`

修改：
- `config/pipeline.yaml`
- `config/model_stack.yaml`
- `run_train.py`
- `trainer/trainer.py`
- `predictor/predictor.py`
- `models/ensemble_manager.py`
- `models/model_registry.py`
- `README.md`

> 注：`data/oof/`、`data/meta/` 与 `__pycache__/` 属于运行产物，建议后续加入 `.gitignore`。



