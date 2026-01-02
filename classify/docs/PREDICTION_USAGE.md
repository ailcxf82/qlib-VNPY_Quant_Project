# 行业轮动预测使用指南

## 一、快速开始

### 1. 训练模型

首先需要训练模型：

```bash
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml
```

训练完成后，模型会保存在 `data/models/industry_rotation/` 目录下。

### 2. 进行预测

使用训练好的模型进行预测：

```bash
# 预测最新交易日
python classify/run_industry_predict.py --config classify/config_industry_rotation.yaml

# 预测指定日期
python classify/run_industry_predict.py --config classify/config_industry_rotation.yaml --date 2024-12-31

# 使用指定模型版本
python classify/run_industry_predict.py --config classify/config_industry_rotation.yaml --date 2024-12-31 --model_tag 20241231
```

---

## 二、预测输出

### 2.1 控制台输出

预测脚本会在控制台输出：

1. **排名前 K 的行业**（K 由配置文件中的 `portfolio.top_k` 设置，默认 10）
2. **完整排名**（所有行业的预测分数和排名）

示例输出：

```
================================================================================
行业预测排名（预测日期: 2024-12-31）
================================================================================
排名前 10 的行业:

   1. sw_l1_801010 (预测分数: 0.123456)
   2. sw_l1_801020 (预测分数: 0.098765)
   3. sw_l1_801030 (预测分数: 0.087654)
   ...
```

### 2.2 文件输出

预测结果会保存为 CSV 文件：

- **文件路径**：`data/predictions/industry_rotation/industry_pred_YYYYMMDD.csv`
- **文件格式**：
  ```csv
  排名,行业代码,预测分数
  1,sw_l1_801010,0.123456
  2,sw_l1_801020,0.098765
  3,sw_l1_801030,0.087654
  ...
  ```

---

## 三、配置说明

### 3.1 模型配置

在 `config_industry_rotation.yaml` 中：

```yaml
industry_gru_config:
  sequence_length: 60  # 时序长度：60天历史数据
  # ... 其他配置
```

### 3.2 组合配置

```yaml
portfolio:
  top_k: 10  # 推荐关注排名前 10 的行业
```

### 3.3 路径配置

```yaml
paths:
  model_dir: "data/models/industry_rotation"  # 模型保存目录
  prediction_dir: "data/predictions/industry_rotation"  # 预测结果保存目录
```

---

## 四、使用建议

### 4.1 预测日期选择

- **推荐**：使用最新交易日进行预测
- **注意**：预测日期需要有足够的历史数据（至少 60 天）

### 4.2 模型版本选择

- **默认**：使用 `latest` 标签（最新训练的模型）
- **指定版本**：使用 `--model_tag` 参数指定特定日期的模型

### 4.3 结果解读

1. **预测分数**：分数越高，表示该行业未来走势越好
2. **排名**：按预测分数降序排列，排名越靠前越好
3. **推荐行业**：关注排名前 `top_k` 的行业（默认前 10 名）

---

## 五、常见问题

### 5.1 模型加载失败

**问题**：`模型加载失败: FileNotFoundError`

**解决**：
1. 确认已运行训练脚本生成模型
2. 检查 `model_dir` 配置是否正确
3. 检查模型文件是否存在：`{model_dir}/{model_tag}_industry_gru.pt`

### 5.2 数据时间范围不足

**问题**：`无法获取历史特征数据`

**解决**：
1. 检查数据配置中的 `start_time` 和 `end_time`
2. 确保数据时间范围包含预测日期及之前至少 60 天的数据
3. 预测脚本会自动扩展时间范围，但需要 qlib 数据源中有相应数据

### 5.3 归一化参数缺失

**问题**：`未找到归一化参数文件`

**解决**：
1. 确认训练时已保存归一化参数
2. 检查归一化参数文件是否存在：`{model_dir}/{model_tag}_norm_meta.json`
3. 如果缺失，预测脚本会使用特征管道的归一化方法（可能不够准确）

---

## 六、预测流程说明

### 6.1 数据准备

1. 读取行业列表（从 `industry_index_path` 配置）
2. 计算需要的历史数据范围（预测日期前至少 60 天）
3. 提取特征数据（使用 QlibFeaturePipeline）

### 6.2 特征处理

1. 加载归一化参数（从模型元数据文件）
2. 对特征进行归一化（使用训练时的归一化参数）
3. 构建时序序列（60 天历史数据）

### 6.3 模型预测

1. 加载训练好的模型
2. 对每个行业进行预测
3. 输出预测分数

### 6.4 结果输出

1. 按预测分数排序
2. 显示排名前 K 的行业
3. 保存完整排名到 CSV 文件

---

## 七、示例

### 完整预测流程

```bash
# 1. 训练模型
python classify/run_industry_train.py --config classify/config_industry_rotation.yaml

# 2. 等待训练完成（可能需要几分钟到几十分钟）

# 3. 进行预测
python classify/run_industry_predict.py --config classify/config_industry_rotation.yaml --date 2024-12-31

# 4. 查看预测结果
cat data/predictions/industry_rotation/industry_pred_20241231.csv
```

### 预测结果示例

```csv
排名,行业代码,预测分数
1,sw_l1_801010,0.123456
2,sw_l1_801020,0.098765
3,sw_l1_801030,0.087654
4,sw_l1_801040,0.076543
5,sw_l1_801050,0.065432
...
```

---

## 八、注意事项

1. **数据时效性**：确保 qlib 数据源已更新到预测日期
2. **模型时效性**：使用最新训练的模型进行预测
3. **历史数据**：预测需要至少 60 天的历史数据
4. **归一化参数**：确保使用训练时的归一化参数，否则预测结果可能不准确

---

## 九、后续优化建议

1. **行业名称映射**：添加行业代码到行业名称的映射，使输出更易读
2. **预测置信度**：输出预测的置信度或不确定性
3. **历史表现**：显示该行业的历史预测准确度
4. **可视化**：生成预测结果的图表（如排名柱状图）

