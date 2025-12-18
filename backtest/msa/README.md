## MSA 多策略组合（RQAlpha）

该目录把 `backtest/投资策略` 里的想法工程化为一个可跑的 RQAlpha 策略：

- **资金分配**：总资产分成两份（可配置比例），分别运行两个子策略，最后合并成一个目标权重
- **策略1（CSI101 小市值）**：预测 Top20 → 过滤（科创/北交、ST、上市天数等）→ 选前 N 只等权，按周频/间隔调仓
- **策略2（CSI300 低估值）**：预测 Top20 → 过滤（近5日涨停、0<PB<1 等）→ 选前 N 只等权，按周频/间隔调仓
- **通用风控**：收盘前 N 分钟检查个股回撤 > 阈值则卖出，卖出后立即按策略补仓

### 依赖
- **必需**：RQAlpha（你已有）
- **可选**：Tushare（用于 ST/PB/涨停等过滤）
  - 安装：`pip install tushare`
  - 配置 token（二选一）：
    - 环境变量：`TUSHARE_TOKEN`
    - 本地密钥文件：复制 `config/secrets.yaml.example` 为 `config/secrets.yaml` 并填入 token（该文件已加入 `.gitignore`，不会提交到仓库）

### 运行方式

先确保你已经分别生成了 csi101/csi300 的预测文件（`data/predictions/pred_*.csv`）。

然后运行：

```bash
python backtest/msa/run_msa_backtest.py ^
  --rqalpha-config config/rqalpha_config.yaml ^
  --pred-csi101 data/predictions/pred_csi101_xxx.csv ^
  --pred-csi300 data/predictions/pred_csi300_xxx.csv
```

### 工程执行逻辑与修改指南
- 详见：`docs/MSA_STRATEGY_GUIDE.md`

可选参数：
- `--alloc1/--alloc2`：两份资金占比（默认 0.5/0.5）
- `--drawdown-stop`：个股回撤止损阈值（默认 0.08）

### 说明
- 如果未设置 `TUSHARE_TOKEN`，策略会输出 warning，并跳过 Tushare 相关过滤（仍可正常回测）。
- 你文档里策略2写了“Top2”，但又写“保持4只”，目前默认 `s2_target_holdings=4`，可通过 `extra.context_vars` 调整为 2。


python backtest/msa/run_msa_backtest.py --rqalpha-config config/rqalpha_config.yaml

