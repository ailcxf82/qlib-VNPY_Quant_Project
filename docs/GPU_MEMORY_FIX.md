# GPU 内存溢出问题修复

## 问题描述

在预测阶段，MLP 模型（特别是 Stacking 模型中的 MLP）在预测大量数据时出现 CUDA 内存溢出错误：

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 22.11 GiB. 
GPU 0 has a total capacity of 15.93 GiB
```

**错误位置**：`models/mlp_model.py::predict_with_leaf_index()` 方法

**原因**：一次性将整个预测数据加载到 GPU，当数据量很大时（如 CSI300 全市场多日数据）会导致 GPU 内存不足。

## 修复方案

### 1. 添加批处理逻辑

修改 `models/mlp_model.py::predict_with_leaf_index()` 方法，将预测数据分批处理：

```python
def predict_with_leaf_index(self, leaf: np.ndarray, index: pd.Index) -> pd.Series:
    """使用叶子索引进行预测（批处理以避免 GPU 内存溢出）。"""
    # 获取批处理大小（默认 4096，可根据 GPU 内存调整）
    batch_size = self.config.get("predict_batch_size", 4096)
    n_samples = len(leaf)
    
    # 如果数据量较小，直接处理
    if n_samples <= batch_size:
        # ... 直接处理 ...
    
    # 批处理预测
    for i in range(0, n_samples, batch_size):
        # 处理每个批次
        # 每批后清理 GPU 缓存
        torch.cuda.empty_cache()
```

### 2. 添加配置项

在 `config/model_mlp.yaml` 中添加 `predict_batch_size` 配置项：

```yaml
predict_batch_size: 4096  # 预测时的批处理大小（避免 GPU 内存溢出）
```

### 3. 优化点

- **自动批处理**：当数据量超过 `predict_batch_size` 时，自动分批处理
- **GPU 缓存清理**：每批处理后清理 GPU 缓存，避免内存碎片
- **可配置**：通过配置文件调整批处理大小，适应不同 GPU 内存

## 使用方法

### 调整批处理大小

根据 GPU 内存大小，在 `config/model_mlp.yaml` 中调整 `predict_batch_size`：

- **GPU 内存 ≥ 24GB**：可以设置为 8192 或更大
- **GPU 内存 16GB**：建议设置为 4096（默认）
- **GPU 内存 8GB**：建议设置为 2048 或更小
- **GPU 内存 < 8GB**：建议设置为 1024 或使用 CPU

### 使用 CPU 预测

如果 GPU 内存不足，可以在 `models/mlp_model.py` 中修改设备设置：

```python
self.device = torch.device("cpu")  # 使用 CPU 而不是 GPU
```

## 验证

修复后，重新运行预测：

```bash
python run_predict.py --start 2023-10-01 --end 2025-10-01
```

应该不再出现 GPU 内存溢出错误。

## 注意事项

1. **批处理大小**：批处理大小过小会影响预测速度，过大可能导致内存溢出
2. **GPU 内存监控**：可以使用 `nvidia-smi` 监控 GPU 内存使用情况
3. **CPU 备选**：如果 GPU 内存始终不足，可以考虑使用 CPU 预测（速度较慢但更稳定）

## 相关文件

- `models/mlp_model.py`：MLP 模型实现（已修复）
- `config/model_mlp.yaml`：MLP 配置文件（已添加 `predict_batch_size`）
- `models/stack_model.py`：Stacking 模型（调用 MLP 的 `predict_with_leaf_index`）


