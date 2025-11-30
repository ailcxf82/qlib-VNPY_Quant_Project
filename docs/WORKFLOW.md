# 工程流程图

本文档以流程图形式展示系统的完整工作流程，从数据准备到最终回测的全链路。

## 1. 整体架构流程图

```mermaid
graph TB
    Start([开始]) --> Config[加载配置文件]
    Config --> DataPrep[数据准备阶段]
    DataPrep --> Train[训练阶段]
    Train --> Predict[预测阶段]
    Predict --> Backtest[回测阶段]
    Backtest --> End([结束])
    
    subgraph DataPrep["数据准备 (QlibFeaturePipeline)"]
        D1[初始化 Qlib] --> D2[提取特征 D.features]
        D2 --> D3[提取标签 Ref($close,-5)/$close-1]
        D3 --> D4[特征标签对齐]
        D4 --> D5[标准化处理]
        D5 --> D6[输出 DataFrame/Series]
    end
    
    subgraph Train["训练阶段 (RollingTrainer)"]
        T1[生成滚动窗口] --> T2[切片训练/验证集]
        T2 --> T3[训练多模型]
        T3 --> T4[计算验证 IC]
        T4 --> T5[保存模型与指标]
        T5 --> T6{是否还有窗口?}
        T6 -->|是| T1
        T6 -->|否| T7[生成 training_metrics.csv]
    end
    
    subgraph Predict["预测阶段 (PredictorEngine)"]
        P1[加载历史 IC] --> P2[加载多模型]
        P2 --> P3[提取预测期特征]
        P3 --> P4[多模型预测]
        P4 --> P5[IC 动态加权]
        P5 --> P6[生成 final 信号]
        P6 --> P7[保存预测 CSV]
    end
    
    subgraph Backtest["回测阶段 (run_backtest)"]
        B1[加载预测结果] --> B2[加载标签收益]
        B2 --> B3[按日期循环]
        B3 --> B4[组合构建]
        B4 --> B5[计算当期收益]
        B5 --> B6{是否还有日期?}
        B6 -->|是| B3
        B6 -->|否| B7[计算绩效指标]
        B7 --> B8[保存回测结果]
    end
```

## 2. 训练阶段详细流程

```mermaid
graph LR
    subgraph Window["滚动窗口循环"]
        W1[窗口 N] --> W2[训练集切片]
        W2 --> W3[验证集切片]
    end
    
    subgraph Ensemble["多模型训练 (EnsembleModelManager)"]
        E1[LightGBM 训练] --> E2[输出预测 + 叶子索引]
        E3[MLP 训练] --> E4[输出预测]
        E2 --> E5[模型预测结果]
        E4 --> E5
    end
    
    subgraph Stack["Stack 模型训练"]
        S1[计算 LGB Residual] --> S2[叶子编码<br/>OneHot/哈希]
        S2 --> S3[MLP 学习 Residual]
        S3 --> S4[融合 LGB + Residual]
    end
    
    subgraph Metrics["指标计算"]
        M1[验证集预测] --> M2[计算 Rank-IC]
        M2 --> M3[记录 IC 历史]
    end
    
    W3 --> Ensemble
    E5 --> Stack
    Stack --> Metrics
    Metrics --> Save[保存模型与指标]
    Save --> Next{下一个窗口?}
    Next -->|是| W1
    Next -->|否| End([训练完成])
```

## 3. 预测阶段详细流程

```mermaid
graph TB
    Start([开始预测]) --> LoadIC[加载 training_metrics.csv]
    LoadIC --> LoadModels[加载多模型]
    LoadModels --> ExtractFeat[提取预测期特征]
    
    subgraph MultiModel["多模型预测"]
        MM1[LightGBM 预测] --> MM2[输出 lgb_pred + leaf]
        MM3[MLP 预测] --> MM4[输出 mlp_pred]
        MM2 --> MM5[Stack 预测]
        MM5 --> MM6[输出 stack_pred]
        MM4 --> MM7[模型预测集合]
        MM6 --> MM7
    end
    
    subgraph QlibEnsemble["Qlib Ensemble (可选)"]
        QE1[标准化各模型预测] --> QE2[平均融合]
        QE2 --> QE3[输出 qlib_ensemble]
    end
    
    subgraph ICWeight["IC 动态加权"]
        IC1[读取历史 IC 序列] --> IC2[计算 IC-IR<br/>半衰期加权]
        IC2 --> IC3[权重归一化<br/>min/max 裁剪]
        IC3 --> IC4[生成权重字典]
    end
    
    ExtractFeat --> MultiModel
    MM7 --> QlibEnsemble
    QlibEnsemble --> ICWeight
    ICWeight --> Blend[加权融合<br/>final = Σ(weight × pred)]
    Blend --> SavePred[保存预测 CSV]
    SavePred --> End([预测完成])
```

## 4. 回测阶段详细流程

```mermaid
graph TB
    Start([开始回测]) --> LoadPred[加载预测 CSV]
    LoadPred --> LoadLabel[加载标签收益]
    LoadLabel --> LoadConfig[加载组合配置]
    
    subgraph DailyLoop["按日期循环"]
        DL1[获取当日信号] --> DL2[获取当日标签]
        DL2 --> DL3[组合构建]
    end
    
    subgraph Portfolio["组合构建 (PortfolioBuilder)"]
        P1[按 signal 排序] --> P2[取 Top-K]
        P2 --> P3[归一化权重]
        P3 --> P4[单股权重裁剪]
        P4 --> P5[行业权重裁剪]
        P5 --> P6[仓位限制调整]
        P6 --> P7[输出最终权重]
    end
    
    subgraph ReturnCalc["收益计算"]
        RC1[权重 × 标签收益] --> RC2[求和得到当期收益]
        RC2 --> RC3[累计收益计算]
    end
    
    LoadConfig --> DailyLoop
    DL3 --> Portfolio
    P7 --> ReturnCalc
    RC3 --> Next{下一个日期?}
    Next -->|是| DL1
    Next -->|否| Stats[计算统计指标]
    Stats --> SaveResult[保存回测结果]
    SaveResult --> End([回测完成])
```

## 5. 数据流图

```mermaid
graph LR
    subgraph Input["输入数据"]
        I1[Qlib 数据源<br/>~/.qlib/qlib_data/cn_data]
        I2[配置文件<br/>config/*.yaml]
    end
    
    subgraph Process["处理流程"]
        P1[特征工程] --> P2[模型训练]
        P2 --> P3[信号生成]
        P3 --> P4[组合构建]
    end
    
    subgraph Output["输出数据"]
        O1[训练模型<br/>data/models/]
        O2[训练指标<br/>data/logs/training_metrics.csv]
        O3[预测结果<br/>data/predictions/]
        O4[回测结果<br/>data/backtest/]
    end
    
    I1 --> P1
    I2 --> P1
    P1 --> P2
    P2 --> O1
    P2 --> O2
    O2 --> P3
    O1 --> P3
    P3 --> O3
    O3 --> P4
    P4 --> O4
```

## 6. 模块依赖关系

```mermaid
graph TD
    subgraph Core["核心模块"]
        C1[QlibFeaturePipeline] --> C2[EnsembleModelManager]
        C2 --> C3[RollingTrainer]
        C3 --> C4[PredictorEngine]
        C4 --> C5[PortfolioBuilder]
    end
    
    subgraph Models["模型模块"]
        M1[LightGBMModelWrapper]
        M2[MLPRegressor]
        M3[LeafStackModel]
    end
    
    subgraph Utils["工具模块"]
        U1[RankICDynamicWeighter]
        U2[ModelRegistry]
    end
    
    C2 --> M1
    C2 --> M2
    C3 --> M3
    C4 --> U1
    C2 --> U2
```

## 7. 关键文件与脚本

| 脚本/模块 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `run_train.py` | 训练入口 | `config/pipeline.yaml` | `data/models/`, `data/logs/training_metrics.csv` |
| `run_predict.py` | 预测入口 | 预测日期范围 | `data/predictions/pred_*.csv` |
| `run_backtest.py` | 回测入口 | 预测 CSV | `data/backtest/backtest_result.csv` |
| `QlibFeaturePipeline` | 特征工程 | `config/data.yaml` | 标准化特征与标签 |
| `RollingTrainer` | 滚动训练 | 特征/标签 | 模型文件 + IC 指标 |
| `PredictorEngine` | 信号生成 | 模型 + 特征 | 综合预测信号 |
| `PortfolioBuilder` | 组合构建 | 信号 + 约束 | 权重分配 |

## 8. 执行顺序

```mermaid
sequenceDiagram
    participant User as 用户
    participant Train as run_train.py
    participant Feature as QlibFeaturePipeline
    participant Trainer as RollingTrainer
    participant Models as 多模型
    participant Predict as run_predict.py
    participant Engine as PredictorEngine
    participant Backtest as run_backtest.py
    
    User->>Train: python run_train.py
    Train->>Feature: build()
    Feature->>Trainer: 返回特征/标签
    Trainer->>Models: fit()
    Models->>Trainer: 返回预测
    Trainer->>Trainer: 计算 IC
    Trainer->>Train: 保存模型与指标
    
    User->>Predict: python run_predict.py
    Predict->>Engine: load_models()
    Engine->>Engine: predict()
    Engine->>Predict: 返回 final 信号
    Predict->>User: 保存预测 CSV
    
    User->>Backtest: python run_backtest.py
    Backtest->>Backtest: 加载预测与标签
    Backtest->>Backtest: 构建组合
    Backtest->>Backtest: 计算收益
    Backtest->>User: 保存回测结果
```

## 9. 关键决策点

```mermaid
graph TD
    Start([开始]) --> CheckData{数据是否准备?}
    CheckData -->|否| InitQlib[初始化 Qlib 数据]
    CheckData -->|是| CheckTrain{是否训练?}
    InitQlib --> CheckTrain
    
    CheckTrain -->|是| TrainModels[训练模型]
    CheckTrain -->|否| CheckPredict{是否预测?}
    
    TrainModels --> CheckValid{验证集存在?}
    CheckValid -->|是| CalcIC[计算验证 IC]
    CheckValid -->|否| CalcTrainIC[计算训练 IC]
    CalcIC --> SaveMetrics[保存指标]
    CalcTrainIC --> SaveMetrics
    SaveMetrics --> CheckPredict
    
    CheckPredict -->|是| LoadModels[加载模型]
    CheckPredict -->|否| End([结束])
    LoadModels --> ICWeight[IC 动态加权]
    ICWeight --> GenSignal[生成 final 信号]
    GenSignal --> CheckBacktest{是否回测?}
    
    CheckBacktest -->|是| BuildPortfolio[构建组合]
    CheckBacktest -->|否| End
    BuildPortfolio --> CalcReturn[计算收益]
    CalcReturn --> End
```

---

**说明**：以上流程图使用 Mermaid 语法绘制，可在支持 Mermaid 的 Markdown 查看器中直接渲染。如需修改，请编辑对应的 Mermaid 代码块。

