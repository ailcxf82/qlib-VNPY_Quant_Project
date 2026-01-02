# 训练功能说明

## 梯度裁剪 (Gradient Clipping)

### 功能说明
梯度裁剪用于防止训练过程中的梯度爆炸问题，通过限制梯度的最大范数来稳定训练。

### 配置方法
在 `config_industry_rotation.yaml` 中添加：

```yaml
industry_gru_config:
  grad_clip: 1.0  # 梯度裁剪最大范数
```

### 参数说明
- **grad_clip**: 
  - `None` 或未设置：不进行梯度裁剪
  - 数值（如 `1.0`）：梯度向量的最大 L2 范数，超过此值会被缩放

### 使用建议
- **默认值**：`1.0` 是一个常用的安全值
- **较小值**（如 `0.5`）：更严格的梯度控制，适合训练不稳定时使用
- **较大值**（如 `5.0`）：较宽松的控制，适合梯度较小的情况
- **不裁剪**：如果训练稳定，可以设置为 `None`

### 实现原理
使用 PyTorch 的 `torch.nn.utils.clip_grad_norm_()` 函数：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
```

---

## 学习率调度器 (Learning Rate Scheduler)

### 功能说明
学习率调度器用于在训练过程中动态调整学习率，提高模型收敛速度和最终性能。

### 配置方法
在 `config_industry_rotation.yaml` 中添加：

```yaml
industry_gru_config:
  lr_scheduler: "plateau"  # 学习率调度器类型
```

### 支持的调度器类型

#### 1. ReduceLROnPlateau（平台衰减）
- **类型**：`"plateau"`
- **工作原理**：监控验证损失，当损失在指定轮数内不再下降时，自动降低学习率
- **适用场景**：验证集损失陷入平台期时使用

**调度器参数**（当前实现）：
- `mode="min"`：监控指标越小越好（验证损失）
- `factor=0.5`：学习率衰减因子（每次降低 50%）
- `patience=5`：等待 5 个 epoch 没有改善后降低学习率
- `verbose=True`：打印学习率变化信息

**示例**：
```
Epoch 10: lr=0.0007
Epoch 15: 验证损失未改善，降低学习率: 0.0007 -> 0.00035
Epoch 20: 验证损失未改善，降低学习率: 0.00035 -> 0.000175
```

#### 2. 其他调度器（未来扩展）
- `"step"`：按固定步长降低学习率
- `"cosine"`：余弦退火调度器
- `None`：不使用调度器（固定学习率）

### 参数说明
- **lr_scheduler**: 
  - `None` 或未设置：不使用学习率调度器，保持固定学习率
  - `"plateau"`：使用 ReduceLROnPlateau 调度器
  - 其他值：当前版本不支持，会输出警告并忽略

### 使用建议

#### 何时使用 ReduceLROnPlateau
- ✅ 训练过程中验证损失陷入平台期
- ✅ 希望自动调整学习率，无需手动干预
- ✅ 训练时间较长，需要精细的学习率控制

#### 何时不使用调度器
- ✅ 训练时间较短，固定学习率已足够
- ✅ 验证损失持续下降，无需调整学习率
- ✅ 希望手动控制学习率变化

### 实现原理
```python
if lr_scheduler_type == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # 监控验证损失
        factor=0.5,      # 学习率衰减 50%
        patience=5,     # 等待 5 个 epoch
        verbose=True     # 打印信息
    )
    
# 每个 epoch 后更新
scheduler.step(val_loss)  # 传入验证损失
```

---

## 完整配置示例

### 示例 1：启用梯度裁剪和学习率调度器
```yaml
industry_gru_config:
  lr: 0.001
  grad_clip: 1.0          # 启用梯度裁剪
  lr_scheduler: "plateau" # 启用学习率调度器
  max_epochs: 50
  patience: 10
```

### 示例 2：仅启用梯度裁剪
```yaml
industry_gru_config:
  lr: 0.001
  grad_clip: 1.0          # 启用梯度裁剪
  lr_scheduler: null       # 不使用学习率调度器
  max_epochs: 50
  patience: 10
```

### 示例 3：仅启用学习率调度器
```yaml
industry_gru_config:
  lr: 0.001
  grad_clip: null          # 不使用梯度裁剪
  lr_scheduler: "plateau"  # 启用学习率调度器
  max_epochs: 50
  patience: 10
```

### 示例 4：都不使用（默认行为）
```yaml
industry_gru_config:
  lr: 0.001
  # grad_clip 和 lr_scheduler 都不设置，使用默认值（None）
  max_epochs: 50
  patience: 10
```

---

## 训练日志示例

### 启用梯度裁剪时的日志
```
2025-01-01 10:00:00 - INFO - 启用梯度裁剪，最大范数: 1.00
2025-01-01 10:00:00 - INFO - 未启用学习率调度器
```

### 启用学习率调度器时的日志
```
2025-01-01 10:00:00 - INFO - 未启用梯度裁剪
2025-01-01 10:00:00 - INFO - 使用 ReduceLROnPlateau 学习率调度器
...
2025-01-01 10:05:00 - INFO - IndustryGRU epoch 15/50: train_loss=0.123456, valid_loss=0.234567, lr=0.000700
2025-01-01 10:05:30 - INFO - Epoch    15: reducing learning rate of group 0 to 3.5000e-04.
2025-01-01 10:06:00 - INFO - IndustryGRU epoch 16/50: train_loss=0.120000, valid_loss=0.230000, lr=0.000350
```

---

## 注意事项

1. **梯度裁剪**：
   - 梯度裁剪在每个训练批次后执行
   - 如果梯度范数超过 `grad_clip`，会被缩放
   - 不会影响梯度方向，只影响梯度大小

2. **学习率调度器**：
   - ReduceLROnPlateau 需要验证集才能正常工作
   - 如果没有验证集，会使用训练损失作为监控指标
   - 学习率降低后不会自动恢复，需要重新训练

3. **组合使用**：
   - 梯度裁剪和学习率调度器可以同时使用
   - 两者都是训练稳定性的保障措施
   - 建议在训练不稳定时同时启用

---

## 当前配置

根据 `config_industry_rotation.yaml`，当前配置为：

```yaml
industry_gru_config:
  grad_clip: 1.0          # ✅ 已启用
  lr_scheduler: "plateau" # ✅ 已启用
```

这两个功能都已启用，可以有效提高训练稳定性和模型性能。

