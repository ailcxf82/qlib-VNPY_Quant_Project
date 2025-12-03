# 158 因子（Alpha158）使用指南

## 一、功能说明

本系统支持使用 qlib 提供的 158 个基础因子（Alpha158）作为特征。158 因子是 qlib 框架中预定义的一组技术指标和统计特征，涵盖了价格、成交量、技术指标等多个维度。

## 二、配置方法

### 1. 基本配置（使用完整 158 因子）

在 `config/data.yaml` 文件中，添加以下配置：

```yaml
data:
  # 是否使用 158 因子
  use_alpha158: true  # 设置为 true 启用，false 禁用
  
  # 其他配置...
  features:
    - "$open"
    - "$close"
    # ... 其他自定义特征
```

**注意**：使用完整 158 因子会消耗大量计算资源，建议使用下面的优化配置。

### 2. 优化配置（推荐）- 筛选因子

为了减少计算资源消耗，可以通过配置筛选因子：

```yaml
data:
  use_alpha158: true
  
  # 自定义 158 因子配置（筛选因子）
  alpha158_config:
    kbar: {}  # K线特征（9个因子）- 推荐保留
    price:
      windows: [0]  # 只使用当前价格（4个因子）
      feature: ["OPEN", "HIGH", "LOW", "VWAP"]  # 如果数据源没有VWAP，可以去掉
    rolling:
      windows: [5, 10, 20, 30, 60]  # 滚动窗口，可以只选 [5, 10, 20] 减少因子数量
      include: ["ROC", "MA", "STD", "BETA", "MAX", "MIN", "RANK", "RSV", "CORR", "CORD", "SUMP", "VMA"]  # 只使用这些操作符
  
  # 进一步筛选（可选）
  alpha158_filter:
    # include_windows: [5, 10, 20]  # 只使用这些窗口
    # max_factors: 50  # 最多使用50个因子
```

**推荐配置**（约 50-80 个因子）：
- KBar: 全部 9 个
- Price: 当前价格 4 个
- Rolling: 核心操作符（ROC, MA, STD, BETA, MAX, MIN, RANK, RSV, CORR, CORD, SUMP, VMA）× 5 个窗口 = 60 个
- **总计**: 约 73 个因子

详细因子清单请参考 `docs/ALPHA158_FACTOR_LIST.md`

### 2. 自动获取 158 因子（推荐）

系统会尝试自动从 qlib 获取 158 因子列表：

1. **优先使用配置文件中的手动列表**（如果提供）
2. **使用 `Alpha158DL.get_feature_config()` 方法**（qlib 官方标准方法）
3. **尝试从 `Alpha158` 处理器中获取**（备选方案）

如果自动获取成功，日志会显示：
```
通过 Alpha158DL.get_feature_config() 成功获取 158 因子列表，共 158 个因子
```

**注意**：`Alpha158DL.get_feature_config()` 返回 `(fields, names)` 元组，其中：
- `fields`：因子表达式列表（用于 `D.features()`）
- `names`：因子名称列表（用于列名）

### 3. 手动指定因子列表（备选方案）

如果自动获取失败，可以在配置文件中手动指定因子表达式列表：

```yaml
data:
  use_alpha158: true
  alpha158_factors:
    - "($close - Ref($close, 1)) / Ref($close, 1)"  # 示例因子表达式
    - "Mean($close, 5) / $close - 1"
    - "Std($close, 10)"
    # ... 添加更多因子表达式
```

**注意**：手动指定时，需要提供完整的因子表达式（qlib 表达式语法），而不是因子名称。

## 三、获取 158 因子列表的方法

### 方法1：从 qlib 源码获取

158 因子的定义在 qlib 的以下位置：
- **`qlib/contrib/data/loader.py`** 中的 `Alpha158DL.get_feature_config()` 方法（**主要方法**）
- `qlib/contrib/data/handler.py` 中的 `Alpha158` 类（内部使用 `Alpha158DL`）

**正确的获取方式**：
```python
from qlib.contrib.data.loader import Alpha158DL

# 获取 158 因子表达式列表
fields, names = Alpha158DL.get_feature_config()
# fields 是因子表达式列表（如 ["($close-$open)/$open", ...]）
# names 是因子名称列表（如 ["KMID", "KLEN", ...]）
print(f"共 {len(fields)} 个因子表达式")
```

### 方法2：使用 Python 代码获取

```python
import qlib
from qlib.data import D

# 初始化 qlib
qlib.init(provider_uri="D:/qlib_data/qlib_data", region="cn")

# 方法1: 使用 Alpha158DL.get_feature_config()（推荐方法）
try:
    from qlib.contrib.data.loader import Alpha158DL
    fields, names = Alpha158DL.get_feature_config()
    print(f"通过 Alpha158DL.get_feature_config() 获取到 {len(fields)} 个因子表达式")
    print("前5个因子表达式:", fields[:5])
    print("前5个因子名称:", names[:5])
except ImportError as e:
    print(f"无法导入 Alpha158DL: {e}")

# 方法2: 使用 Alpha158 处理器（备选方案）
try:
    from qlib.contrib.data.handler import Alpha158
    handler = Alpha158()
    if hasattr(handler, 'get_feature_config'):
        fields, names = handler.get_feature_config()
        print(f"通过 Alpha158.get_feature_config() 获取到 {len(fields)} 个因子表达式")
except Exception as e:
    print(f"获取失败: {e}")
```

### 方法3：查看 qlib 文档

参考 qlib 官方文档或 GitHub 仓库，查找 Alpha158 的因子定义。

## 四、使用示例

### 示例1：启用 158 因子

```yaml
# config/data.yaml
data:
  use_alpha158: true  # 启用 158 因子
  features:
    - "$close"
    - "$volume"
    # 158 因子会自动添加到特征列表中
```

运行训练时，系统会：
1. 检查 `use_alpha158` 配置
2. 自动获取 158 因子列表
3. 将 158 因子添加到特征列表
4. 使用所有特征进行训练

### 示例2：禁用 158 因子

```yaml
# config/data.yaml
data:
  use_alpha158: false  # 禁用 158 因子
  features:
    - "$close"
    - "$volume"
    # 仅使用自定义特征
```

### 示例3：手动指定因子列表

```yaml
# config/data.yaml
data:
  use_alpha158: true
  alpha158_factors:
    # 手动指定部分因子表达式
    - "($close - Ref($close, 1)) / Ref($close, 1)"
    - "Mean($close, 5) / $close - 1"
    - "Std($close, 10)"
    - "Max($high, 5) / $close - 1"
    # ... 更多因子
  features:
    - "$close"
```

## 五、验证配置

运行训练脚本后，查看日志输出：

```
INFO - 提取特征，共 167 个特征表达式
INFO - 通过 get_158_factor() 成功获取 158 因子列表，共 158 个因子
INFO - 已添加 158 因子，当前特征总数: 167 (原有: 9, 158因子: 158)
```

如果看到类似日志，说明 158 因子已成功加载。

## 六、常见问题

### Q1: 提示 "无法导入 Alpha158"

**原因**：qlib 版本可能不包含 Alpha158 模块，或导入路径不同。

**解决方案**：
1. 检查 qlib 版本：`pip show qlib`
2. 升级 qlib：`pip install --upgrade qlib`
3. 如果仍无法导入，使用手动指定因子列表的方式

### Q2: 自动获取失败

**原因**：qlib 版本或 API 变化导致自动获取方法失效。

**解决方案**：
1. 在配置文件中手动指定 `alpha158_factors` 列表
2. 参考 qlib 文档获取最新的因子列表
3. 查看日志中的详细错误信息

### Q3: 特征数量不对

**原因**：158 因子可能与自定义特征有重复，或某些因子表达式无效。

**解决方案**：
1. 检查日志中的特征总数
2. 验证因子表达式是否正确
3. 检查是否有重复的特征表达式

### Q4: 训练速度变慢

**原因**：158 因子增加了特征维度，计算量增大。

**解决方案**：
1. 这是正常现象，158 个因子会显著增加特征维度
2. 可以考虑特征选择，只使用部分重要因子
3. 调整模型参数以适应更多特征

## 七、性能建议

1. **首次使用**：建议先用 `use_alpha158: false` 测试自定义特征，确认流程正常
2. **逐步启用**：可以先手动指定部分因子，测试效果后再启用全部
3. **特征选择**：158 因子可能包含冗余特征，建议使用模型的特征重要性进行筛选
4. **内存管理**：158 因子会增加内存占用，注意监控系统资源

## 八、参考资源

- [qlib 官方文档](https://qlib.readthedocs.io/)
- [qlib GitHub 仓库](https://github.com/microsoft/qlib)
- [Alpha158 因子定义](https://github.com/microsoft/qlib/tree/main/qlib/contrib/data)

