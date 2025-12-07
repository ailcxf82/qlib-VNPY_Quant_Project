# 数据泄露问题分析报告

## 一、问题概述

经过检查，发现工程中存在**非时序归一化**问题，导致数据泄露。

## 二、发现的问题

### 2.1 ❌ 非时序归一化（严重问题）

**位置**：`feature/qlib_feature_pipeline.py::_fit_norm()`

**问题代码**：
```python
def _fit_norm(self, features: pd.DataFrame):
    """计算全局均值方差。"""
    self._feature_mean = features.mean()  # ❌ 使用整个数据集的均值
    std = features.std().replace(0, 1)   # ❌ 使用整个数据集的标准差
    self._feature_std = std
```

**问题描述**：
1. 在 `build()` 方法中，使用**整个数据集**（包括训练集和测试集）计算均值和标准差
2. 在滚动窗口训练时，每个窗口都使用这些**全局统计量**进行归一化
3. 这导致训练时使用了**未来数据的信息**（测试集的统计量），造成数据泄露

**影响**：
- 模型在回测时表现可能过于乐观
- 实际部署时性能会显著下降
- 违反了时间序列数据的基本假设

**示例**：
```
数据范围: 2017-10-01 到 2021-10-31
训练窗口1: 2017-10-01 到 2019-09-30
验证窗口1: 2019-10-01 到 2019-10-31

问题：归一化时使用了 2017-10-01 到 2021-10-31 的全局统计量
     训练窗口1 的归一化参数包含了 2020-2021 年的数据信息
```

### 2.2 ✅ 未来函数检查（无问题）

**检查结果**：
- 标签表达式：`Ref($close, -20)/$close - 1` ✅ 正确（标签使用未来数据是正常的）
- 特征表达式：未发现使用未来数据的特征 ✅
- Alpha158 因子：由 qlib 自动生成，通常不包含未来函数 ✅

### 2.3 ⚠️ 训练流程问题

**位置**：`trainer/trainer.py::train()`

**问题**：
```python
def train(self):
    self.pipeline.build()  # ❌ 先构建所有数据并归一化
    features, labels = self.pipeline.get_all()  # 获取已归一化的数据
    
    for idx, window in enumerate(self._generate_windows()):
        train_feat, train_lbl = self._slice(features, labels, ...)  # 使用全局归一化的数据
        # ...
```

**问题描述**：
- 在训练开始前，先调用 `build()` 对所有数据进行归一化
- 滚动窗口训练时，使用的是已经用全局统计量归一化的数据
- 每个窗口没有独立的归一化参数

## 三、修复方案

### 方案 1：滚动窗口归一化（推荐）

**原理**：每个训练窗口使用该窗口内的数据计算归一化参数

**实现步骤**：
1. 修改 `QlibFeaturePipeline`，不在 `build()` 时进行归一化
2. 在 `trainer/trainer.py` 中，对每个训练窗口单独计算归一化参数
3. 预测时，使用最近一个训练窗口的归一化参数

**优点**：
- 完全避免数据泄露
- 符合时间序列数据的特点
- 实现相对简单

**缺点**：
- 需要修改训练流程
- 预测时需要保存归一化参数

### 方案 2：在线滚动归一化

**原理**：使用滚动窗口（如60日）的均值和标准差进行归一化

**实现步骤**：
1. 修改 `_fit_norm()` 和 `_transform()`，支持滚动窗口归一化
2. 每个时间点的归一化参数只使用该时间点之前的数据

**优点**：
- 更符合实际交易场景
- 可以实时更新归一化参数

**缺点**：
- 实现较复杂
- 需要处理窗口边界问题

### 方案 3：保存归一化参数（临时方案）

**原理**：在训练时保存每个窗口的归一化参数，预测时使用

**实现步骤**：
1. 在训练时，为每个窗口计算并保存归一化参数
2. 预测时，使用最近一个窗口的归一化参数

**优点**：
- 修改量小
- 可以快速修复问题

**缺点**：
- 仍然存在一定的时间不一致性

## 四、推荐修复方案（方案1）

### 4.1 修改 `feature/qlib_feature_pipeline.py`

```python
def build(self):
    # ... 特征提取代码 ...
    
    # ❌ 删除这部分
    # self._fit_norm(features)
    # norm_feat = self._transform(features)
    
    # ✅ 改为：不归一化，保存原始特征
    self.features_df = features  # 保存原始特征
    self.label_series = label
```

### 4.2 修改 `trainer/trainer.py`

```python
def train(self):
    self.pipeline.build()
    features, labels = self.pipeline.get_all()  # 获取未归一化的数据
    
    for idx, window in enumerate(self._generate_windows()):
        train_feat, train_lbl = self._slice(features, labels, ...)
        valid_feat, valid_lbl = self._slice(features, labels, ...)
        
        # ✅ 对训练窗口单独归一化
        train_mean = train_feat.mean()
        train_std = train_feat.std().replace(0, 1)
        train_feat_norm = (train_feat - train_mean) / train_std
        train_feat_norm = train_feat_norm.clip(-5, 5)
        
        # ✅ 验证集使用训练集的归一化参数
        valid_feat_norm = (valid_feat - train_mean) / train_std
        valid_feat_norm = valid_feat_norm.clip(-5, 5)
        
        # 使用归一化后的特征训练
        self.ensemble.fit(train_feat_norm, train_lbl, valid_feat_norm, valid_lbl)
        # ...
```

### 4.3 修改 `predictor/predictor.py`

```python
def predict(self, features, ic_histories):
    # ✅ 使用最近一个训练窗口的归一化参数
    # 需要从模型元数据中加载归一化参数
    # ...
```

## 五、修复优先级

1. **【高优先级】** 修复归一化方法（方案1）
2. **【中优先级】** 验证修复后的效果
3. **【低优先级】** 优化归一化参数保存和加载

## 六、验证方法

修复后，可以通过以下方式验证：

1. **回测对比**：修复前后的回测结果应该有明显差异（修复后可能表现下降，但更真实）
2. **时间分割测试**：使用前70%数据训练，后30%数据测试，对比修复前后的测试集表现
3. **滚动窗口IC**：检查修复后各窗口的IC是否更稳定

## 七、参考

- [时间序列数据泄露问题](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [滚动窗口归一化](https://www.kaggle.com/code/ryanholbrook/leakage-and-data-snooping)

