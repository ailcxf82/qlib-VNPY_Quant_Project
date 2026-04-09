# pip 依赖安装说明（Windows / PowerShell）

本文档用于快速搭建本项目运行环境，覆盖训练、预测、测试与 API 依赖。

## 1. Python 版本建议

- 推荐：`Python 3.10` 或 `3.11`
- 不建议：`Python 3.13`（部分量化/深度学习依赖，尤其 qlib 相关，可能没有可用 wheel）

## 2. 创建并激活虚拟环境

```powershell
cd d:\quant_project\Qlib_Quant\qlib-VNPY_Quant_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 3. 安装基础依赖（必装）

```powershell
pip install numpy pandas scipy pyyaml scikit-learn lightgbm pyarrow requests
```

说明：
- `pyarrow` 用于 parquet 读写（如 OOF / meta 数据）
- `lightgbm` 是 `models/lightgbm_model.py` 的直接依赖

## 4. 安装深度学习依赖（GRU/MLP）

```powershell
pip install torch
```

如需指定 CUDA 版本，请按 PyTorch 官方命令替换本行安装命令。

## 5. 安装 Qlib（核心依赖）

项目中大量模块使用 `from qlib...`（如 `feature/qlib_feature_pipeline.py`、`utils/dataset.py`）。

优先尝试：

```powershell
pip install pyqlib
```

若所在环境无法从 PyPI 获取 `pyqlib`，可尝试：

```powershell
pip install git+https://github.com/microsoft/qlib.git
```

注意：
- `pip install qlib` 可能安装到同名但非 Microsoft Qlib 的包，导致缺少 `qlib.data` 等模块。
- 安装后可用以下命令快速验证：

```powershell
python -c "import qlib; from qlib.data import D; print('qlib ok')"
```

## 6. 安装可选依赖

### 6.1 API 服务

```powershell
pip install flask
```

### 6.2 测试

```powershell
pip install pytest
```

### 6.3 Tushare 数据接口（若使用）

```powershell
pip install tushare
```

## 7. 一键安装命令（常用）

如你希望一次装齐常用依赖（训练 + 测试 + API）：

```powershell
pip install numpy pandas scipy pyyaml scikit-learn lightgbm pyarrow requests torch flask pytest tushare
```

然后再单独安装 Qlib（见第 5 节）。

## 8. 安装完成后自检

```powershell
python -c "import numpy,pandas,scipy,sklearn,lightgbm,torch,yaml,requests,flask; print('base deps ok')"
python -c "import qlib; from qlib.data import D; print('qlib ok')"
python -m pytest tests/test_training_metrics_icir.py -q
```

## 9. 常见问题

- **报错 `ModuleNotFoundError: No module named 'numpy'`**
  - 说明基础依赖未装，执行第 3 节命令。

- **报错 `ModuleNotFoundError: No module named 'qlib.data'`**
  - 大概率装错了 `qlib` 包；卸载后按第 5 节重装：
  ```powershell
  pip uninstall -y qlib
  pip install pyqlib
  ```

- **训练时 GPU 相关报错**
  - 检查 `config/model_lgb*.yaml` 的 `device: "gpu"`；
  - 无 GPU 环境可改为 `device: "cpu"`。

