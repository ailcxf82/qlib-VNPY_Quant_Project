# pip 依赖安装说明（Windows / PowerShell）

本文档汇总本仓库 **Python 代码中实际引用** 的第三方库，并提供 **一次性安装** 方式。更权威的机器可读清单见项目根目录 **`requirements.txt`**。

## 1. Python 版本建议

- 推荐：`Python 3.10` 或 `3.11`
- 谨慎：`Python 3.12+`（部分轮子/量化生态可能滞后）
- 不建议：`Python 3.13`（`pyqlib` / `torch` 等可能暂无稳定 wheel，需自行验证）

## 2. 创建并激活虚拟环境

```powershell
cd d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 3. 一键安装（推荐）

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

说明：

- **`requirements.txt`** 与下表一致，便于版本管理与 CI。
- **PyTorch**：未锁定 CUDA 变体；若需 GPU，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 `pip` 命令替换其中的 `torch` 安装步骤（可先装其余包，再单独装 `torch`）。

## 4. 依赖总表（与代码模块对应）

| pip 包名 | 用途 | 典型引用位置 |
| --- | --- | --- |
| `numpy` | 数值计算 | 全项目 |
| `pandas` | 表格数据 | 全项目 |
| `scipy` | 统计等 | `models/dynamic_ensemble.py`、`tests/test_icir.py` 等 |
| `PyYAML` | 读写 YAML 配置 | `utils/config.py`、`monitor/*.py` 等（`import yaml`） |
| `scikit-learn` | 回归、预处理、编码 | `models/weighted_ensemble.py`、`models/stack_model.py`、`models/meta_stacker.py`、`utils/meta_ridge.py` |
| `joblib` | 模型持久化 | `models/meta_stacker.py` |
| `lightgbm` | LightGBM 及 qlib GBDT | `models/lightgbm_model.py` |
| `pyarrow` | Parquet 引擎（OOF、缓存） | `trainer/trainer.py`、`utils/meta_oof_builder.py`、`backtest/msa/tushare_client.py` 等 |
| `requests` | HTTP（如企业微信通知） | `monitor/notifier/wechat.py` |
| `torch` | GRU/MLP、损失函数 | `models/gru_model.py`、`models/multi_task_gru.py`、`utils/loss_functions.py` |
| `pyqlib` | **Microsoft Qlib** 数据与模型 | `utils/dataset.py`、`models/lightgbm_model.py`、`models/ensemble_manager.py` 等 |
| `flask` | 预测 API | `api/predict_api.py` |
| `akshare` | 监控/回测行情（可选数据源） | `monitor/data_source.py`、`monitor/data_hub.py`、`monitor/strategy_library/fusion/real_data_provider.py` |
| `tushare` | 财务/行情 Pro 接口 | `backtest/msa/tushare_client.py`、`monitor/ts_client.py` 等 |
| `rqalpha` | RQAlpha 回测引擎 | `backtest/rqalpha_backtest.py`、`backtest/rqalpha_chan_strategy.py`、`backtest/msa/rqalpha_msa_strategy.py` |
| `h5py` | 读取 RQAlpha bundle `stocks.h5`（波动率等） | `backtest/msa/run_msa_signal.py` |
| `matplotlib` | 回测报告出图（Agg 后端） | `monitor/strategy_library/backtest/report_generator.py`（未安装则相关图表功能不可用） |
| `openai` | LLM 分析（可选） | `monitor/llm_analyzer.py`（未安装会记录警告并降级） |
| `pytest` | 单元测试 | `monitor/strategy_library/fusion/tests/` 等 |

### 4.1 Qlib 安装注意

项目中大量 `from qlib...`。**请安装 Microsoft 发行的包名 `pyqlib`**，不要安装 PyPI 上同名但非官方实现的 `qlib`，否则会出现 `No module named 'qlib.data'` 等错误。

优先：

```powershell
pip install pyqlib
```

若 PyPI 不可用，可从源码安装：

```powershell
pip install git+https://github.com/microsoft/qlib.git
```

验证：

```powershell
python -c "import qlib; from qlib.data import D; print('qlib ok')"
```

### 4.2 RD-Agent（可选）

若使用 `scripts/rdagent/` 相关自动化流程，可额外安装：

```powershell
pip install rdagent
```

RD-Agent 执行生成代码时**通常仍依赖 Docker**；详见 [RD-Agent 安装说明](https://rdagent.readthedocs.io/en/stable/installation_and_configuration.html)。

### 4.3 工程中未在 `requirements.txt` 中列出的名称

- **`vnpy`**：仓库名含 VNPY，但当前 **Python 源码中未发现 `import vnpy`**；若你本地要对接 VNPY，请自行追加安装。
- **`feature/qlib_feature_pipeline.py`**：在 `.gitignore` 中，本地若存在该文件，同样依赖上述 **`pyqlib`** 栈。

## 5. 分场景最小集（可选）

仅当磁盘或网络受限、需要裁剪时参考：

| 场景 | 建议包 |
| --- | --- |
| 仅跑单元测试（不触 qlib） | `numpy` `pandas` `pytest` 等测试文件直接依赖 |
| 训练 / 预测（Qlib + LGB + 可选 Torch） | 表第 4 节中除 `rqalpha`、`akshare`、`tushare`、`h5py`、`matplotlib`、`openai` 外的主干 |
| RQAlpha 回测 | 在上述基础上加 `rqalpha`；读 bundle 时加 `h5py` |
| 监控大屏与行情 | 加 `akshare`、`tushare`、`flask`（按功能开） |

完整开发环境直接使用 **`pip install -r requirements.txt`** 即可。

## 6. 安装完成后自检

```powershell
python -c "import numpy,pandas,scipy,yaml,sklearn,joblib,lightgbm,pyarrow,requests,torch,flask; print('core third-party ok')"
python -c "import qlib; from qlib.data import D; print('qlib ok')"
python -m pytest tests/test_icir.py tests/test_normalize_by_date.py tests/test_sequence_builder.py -q
```

按需验证可选组件：

```powershell
python -c "import akshare, tushare, rqalpha, h5py, matplotlib, openai; print('optional ok')"
```

## 7. 常见问题

- **`ModuleNotFoundError: No module named 'numpy'`**  
  未激活虚拟环境或未执行安装；先激活 `.venv`，再 `pip install -r requirements.txt`。

- **`No module named 'qlib.data'`**  
  多半误装了错误的 `qlib` 包：

  ```powershell
  pip uninstall -y qlib
  pip install pyqlib
  ```

- **RQAlpha 导入失败**  
  执行 `pip install rqalpha`；若与当前 `pandas`/`numpy` 版本冲突，需按 RQAlpha 文档调整版本或使用独立虚拟环境。

- **训练 GPU 相关报错**  
  检查 `config/model_lgb*.yaml` 的 `device`；无 GPU 时改为 CPU。
