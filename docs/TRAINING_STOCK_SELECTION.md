# 模型训练股票选择逻辑

## 一、股票池配置

### 1.1 配置位置
- **配置文件**: `config/data.yaml`
- **配置项**: `data.instruments`

### 1.2 当前配置
```yaml
data:
  instruments: "csi101"  # CSI101 指数成分股
  start_time: "2021-10-01"
  end_time: "2025-10-31"
  freq: "day"
```

### 1.3 股票池解析逻辑

股票池通过 `feature/qlib_feature_pipeline.py` 中的 `_parse_instruments()` 方法解析：

**支持的配置格式：**

1. **市场别名（字符串）**：如 `"csi101"`, `"csi300"`, `"csi500"`
   - 通过 `D.instruments()` 获取市场配置
   - 通过 `D.list_instruments()` 获取股票代码列表
   - 自动清理代码格式（去除 `.SH` 或 `.SZ` 后缀）

2. **字典配置**：如 `{"market": "csi101", "filter_pipe": []}`
   - 同样通过 `D.list_instruments()` 获取股票列表

3. **股票代码列表**：如 `["000001", "000002", ...]`
   - 直接使用，自动清理代码格式

**代码位置**：
```458:523:feature/qlib_feature_pipeline.py
    def _parse_instruments(inst_conf: Union[str, Dict[str, Any], Tuple[str, ...], list[str]]) -> list[str]:
        """
        将配置的股票池转换为股票代码列表。
        
        根据测试，D.features() 需要股票代码数组（如 ["000001", "000002", ...]），
        而不是字典配置。因此需要将市场别名转换为股票代码列表。

        - 字符串默认视作市场别名（如 "csi300"），通过 D.list_instruments() 获取股票列表
        - 字典配置（如 {"market": "csi300", "filter_pipe": []}），也转换为股票列表
        - list/tuple 视为具体股票代码集合，直接返回
        """
        if isinstance(inst_conf, str):
            # 如果是市场别名（如 "csi300"），先获取配置字典，再转换为股票列表
            try:
                market_config = D.instruments(inst_conf)
                # 使用 D.list_instruments() 获取股票代码列表
                stock_list = D.list_instruments(instruments=market_config, as_list=True)
                if isinstance(stock_list, list) and len(stock_list) > 0:
                    logger.info("从市场 '%s' 获取到 %d 只股票", inst_conf, len(stock_list))
                    # 确保返回的是纯数字股票代码（去掉 .SH 或 .SZ 后缀，如果存在）
                    cleaned_list = []
                    for code in stock_list:
                        # 如果代码包含点号，提取前面的数字部分
                        if '.' in str(code):
                            code = str(code).split('.')[0]
                        cleaned_list.append(str(code))
                    return cleaned_list
                else:
                    raise ValueError(f"无法从市场 '{inst_conf}' 获取股票列表，返回结果为空")
            except Exception as e:
                logger.error("无法从市场 '%s' 获取股票列表: %s", inst_conf, e)
                logger.error("请检查：1) qlib 数据源是否包含该市场定义；2) 市场名称是否正确")
                raise ValueError(f"无法解析股票池配置 '{inst_conf}': {e}")
        
        if isinstance(inst_conf, dict):
            # 如果是字典配置，也转换为股票列表
            try:
                stock_list = D.list_instruments(instruments=inst_conf, as_list=True)
                if isinstance(stock_list, list) and len(stock_list) > 0:
                    market_name = inst_conf.get("market", "未知市场")
                    logger.info("从市场配置 '%s' 获取到 %d 只股票", market_name, len(stock_list))
                    # 确保返回的是纯数字股票代码
                    cleaned_list = []
                    for code in stock_list:
                        if '.' in str(code):
                            code = str(code).split('.')[0]
                        cleaned_list.append(str(code))
                    return cleaned_list
                else:
                    raise ValueError(f"无法从市场配置获取股票列表，返回结果为空")
            except Exception as e:
                logger.error("无法从市场配置获取股票列表: %s", e)
                raise ValueError(f"无法从市场配置获取股票列表: {e}")
        
        if isinstance(inst_conf, (list, tuple)):
            # 如果是列表，确保格式正确（纯数字代码）
            result = []
            for code in inst_conf:
                code_str = str(code)
                # 如果包含点号，提取前面的数字部分
                if '.' in code_str:
                    code_str = code_str.split('.')[0]
                result.append(code_str)
            return result
        
        raise ValueError(f"不支持的股票池配置类型: {type(inst_conf)}")
```

## 二、数据提取流程

### 2.1 特征提取入口

**代码位置**: `feature/qlib_feature_pipeline.py` 的 `build()` 方法

```217:235:feature/qlib_feature_pipeline.py
        instruments = self._parse_instruments(self.feature_cfg["instruments"])
        start = self.feature_cfg["start_time"]
        end = self.feature_cfg["end_time"]
        freq = self.feature_cfg.get("freq", "day")
        label_expr = self.feature_cfg.get("label", "Ref($close, -5)/$close - 1")

        logger.info("提取特征，共 %d 个特征表达式", len(feats))
        
        
        
        # 提取特征和标签
        try:
            feature_panel = D.features(instruments=instruments, fields=feats, start_time=start, end_time=end, freq=freq)
            label_panel = D.features(instruments=instruments, fields=[label_expr], start_time=start, end_time=end, freq=freq)
        except Exception as e:
            logger.error("特征提取失败: %s", e)
            logger.error("提示：可能是 158 因子中的某些表达式在当前数据源中不可用")
            logger.error("建议：1) 检查数据源是否包含 VWAP 字段；2) 使用 alpha158_config 筛选因子")
            raise
```

### 2.2 数据范围

- **股票池**: CSI101 指数成分股（通过 `D.list_instruments()` 动态获取）
- **时间范围**: `2021-10-01` 至 `2025-10-31`
- **频率**: 日频（`freq: "day"`）

### 2.3 数据过滤

**涨停/跌停过滤**（如果配置）：
```yaml
filter:
  limit_up: true   # 过滤涨停股票
  limit_down: true # 过滤跌停股票
```

**代码位置**: `feature/qlib_feature_pipeline.py` 的 `build()` 方法中（如果配置了 filter）

## 三、滚动窗口训练

### 3.1 滚动窗口配置

**配置文件**: `config/pipeline.yaml`

```yaml
rolling:
  train_months: 24    # 训练窗口：24个月
  valid_months: 1     # 验证窗口：1个月
  test_months: 1      # 测试窗口：1个月（未使用）
  step_months: 1      # 步长：1个月
  min_samples: 2000   # 最小训练样本数
```

### 3.2 窗口生成逻辑

**代码位置**: `trainer/trainer.py` 的 `_generate_windows()` 方法

```48:70:trainer/trainer.py
    def _generate_windows(self) -> Iterable[Window]:
        rolling = self.cfg["rolling"]
        data_cfg = load_yaml_config(self.data_cfg_path)["data"]
        start = pd.Timestamp(data_cfg["start_time"])
        end = pd.Timestamp(data_cfg["end_time"])
        train_offset = pd.DateOffset(months=rolling["train_months"])
        valid_offset = pd.DateOffset(months=rolling["valid_months"])
        step = pd.DateOffset(months=rolling["step_months"])

        # cursor 指向验证起点，前推 train_offset 即训练区间
        cursor = start + train_offset
        while cursor + valid_offset <= end:
            train_start = cursor - train_offset
            train_end = cursor - pd.Timedelta(days=1)
            valid_start = cursor
            valid_end = cursor + valid_offset - pd.Timedelta(days=1)
            yield Window(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=valid_end.strftime("%Y-%m-%d"),
            )
            cursor += step
```

### 3.3 训练流程

**代码位置**: `trainer/trainer.py` 的 `train()` 方法

```105:174:trainer/trainer.py
    def train(self):
        self.pipeline.build()
        features, labels = self.pipeline.get_all()
        
        # 检查标签转换是否生效
        label_is_rank = getattr(self.pipeline, "_label_is_rank", False)
        if label_is_rank:
            logger.info("训练使用 Rank 转换后的标签（范围应在 [0, 1] 之间）")
            logger.info("标签值统计: min=%.6f, max=%.6f, mean=%.6f", 
                       labels.min(), labels.max(), labels.mean())
            if labels.min() < 0 or labels.max() > 1:
                logger.warning("标签值不在 [0, 1] 范围内！可能转换未生效")
        else:
            logger.info("训练使用原始标签（未进行 Rank 转换）")
            logger.info("标签值统计: min=%.6f, max=%.6f, mean=%.6f", 
                       labels.min(), labels.max(), labels.mean())
        
        os.makedirs(self.paths["model_dir"], exist_ok=True)
        os.makedirs(self.paths["log_dir"], exist_ok=True)
        metrics: List[Dict] = []

        # 记录数据时间范围，便于诊断
        if len(features) > 0:
            data_start = features.index.get_level_values("datetime").min()
            data_end = features.index.get_level_values("datetime").max()
            logger.info("特征数据时间范围: %s 至 %s，共 %d 条记录", data_start, data_end, len(features))
        
        for idx, window in enumerate(self._generate_windows()):
            logger.info("==== 滚动窗口 %d: 训练 [%s, %s] 验证 [%s, %s] ====", 
                       idx, window.train_start, window.train_end, window.valid_start, window.valid_end)
            train_feat, train_lbl = self._slice(features, labels, window.train_start, window.train_end)
            valid_feat, valid_lbl = self._slice(features, labels, window.valid_start, window.valid_end)
            
            if len(train_feat) < self.cfg["rolling"].get("min_samples", 1000):
                logger.warning("训练样本不足 (%d < %d)，跳过该窗口", 
                             len(train_feat), self.cfg["rolling"].get("min_samples", 1000))
                continue
            
            has_valid = valid_feat is not None and not valid_feat.empty and valid_lbl is not None and not valid_lbl.empty
            if not has_valid:
                logger.warning("窗口 %d 验证集为空 (特征: %d, 标签: %d)，退化为仅训练", 
                             idx, len(valid_feat) if valid_feat is not None else 0, 
                             len(valid_lbl) if valid_lbl is not None else 0)
                # 诊断：检查验证时间范围是否在数据范围内
                if len(features) > 0:
                    data_start = features.index.get_level_values("datetime").min()
                    data_end = features.index.get_level_values("datetime").max()
                    valid_start_ts = pd.Timestamp(window.valid_start)
                    valid_end_ts = pd.Timestamp(window.valid_end)
                    if valid_start_ts < data_start or valid_end_ts > data_end:
                        logger.warning("验证时间范围 [%s, %s] 超出数据范围 [%s, %s]", 
                                     window.valid_start, window.valid_end, data_start, data_end)
                valid_feat = None
                valid_lbl = None
            else:
                logger.info("窗口 %d: 训练样本 %d，验证样本 %d", idx, len(train_feat), len(valid_feat))

            # 修复：对每个训练窗口单独计算归一化参数，避免数据泄露
            logger.info("窗口 %d: 计算训练窗口归一化参数（仅使用训练集数据）", idx)
            train_feat_norm, norm_mean, norm_std = self.pipeline.normalize_features(train_feat)
            
            # 验证集使用训练集的归一化参数（不能使用验证集数据计算归一化参数）
            if has_valid:
                valid_feat_norm = (valid_feat - norm_mean) / norm_std
                valid_feat_norm = valid_feat_norm.clip(-5, 5)
            else:
                valid_feat_norm = None
            
            # 统一训练多模型（使用归一化后的特征）
            self.ensemble.fit(train_feat_norm, train_lbl, valid_feat_norm, valid_lbl)
```

### 3.4 数据切片逻辑

**代码位置**: `trainer/trainer.py` 的 `_slice()` 方法

```72:103:trainer/trainer.py
    def _slice(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        start: str,
        end: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """按时间范围切片特征和标签。"""
        idx = features.index
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(f"特征索引应为 MultiIndex，实际为 {type(idx)}")
        
        # 确保 datetime 层级存在
        if "datetime" not in idx.names:
            raise ValueError(f"索引层级中未找到 'datetime'，当前层级: {idx.names}")
        
        # 转换为 Timestamp 以确保正确比较
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        datetime_level = idx.get_level_values("datetime")
        mask = (datetime_level >= start_ts) & (datetime_level <= end_ts)
        
        feat = features.loc[mask]
        lbl = labels.loc[mask]
        
        logger.debug(
            "切片 [%s, %s]: 特征样本 %d，标签样本 %d",
            start, end, len(feat), len(lbl)
        )
        
        return feat, lbl
```

## 四、训练数据总结

### 4.1 股票范围
- **股票池**: CSI101 指数成分股（动态获取，具体数量取决于 qlib 数据源）
- **股票代码格式**: 纯数字代码（如 `"000001"`），自动去除 `.SH` 或 `.SZ` 后缀

### 4.2 时间范围
- **数据提取范围**: `2021-10-01` 至 `2025-10-31`
- **训练窗口**: 24个月（滚动）
- **验证窗口**: 1个月（滚动）
- **步长**: 1个月

### 4.3 数据过滤
- **涨停/跌停**: 根据配置决定是否过滤（当前配置：`limit_up: true`, `limit_down: true`）
- **缺失值处理**: 
  - 删除标签为 NaN 的行
  - 特征缺失值使用前向填充和后向填充，最后用 0 填充
  - 删除全为 NaN 的特征列

### 4.4 数据量要求
- **最小训练样本数**: 2000（如果不足，跳过该窗口）

## 五、如何修改股票池

### 5.1 使用市场别名
```yaml
data:
  instruments: "csi300"  # 改为 CSI300 指数
```

### 5.2 使用自定义股票列表
```yaml
data:
  instruments: ["000001", "000002", "600000", ...]  # 直接指定股票代码
```

### 5.3 使用字典配置
```yaml
data:
  instruments:
    market: "csi500"
    filter_pipe: []  # 可以添加过滤条件
```

## 六、注意事项

1. **股票池动态性**: CSI101 指数成分股会随时间变化，qlib 会根据时间范围自动获取对应时期的成分股
2. **数据完整性**: 确保 qlib 数据源包含所需时间范围内的所有股票数据
3. **标签计算**: 标签使用 `Ref($close, -20)/$close - 1`（未来20日收益率），因此最后20天的数据没有标签
4. **归一化**: 每个训练窗口单独计算归一化参数，避免数据泄露


