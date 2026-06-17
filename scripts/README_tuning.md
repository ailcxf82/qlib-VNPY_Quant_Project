# 微调对比组件使用说明

## 1. 准备实验规格
- 样例文件：`data/tuning/specs/sample_experiments.yaml`
- 每个实验至少包含：
  - `experiment_id`
  - `model_type` (`lgb` / `gru`)
  - `param_overrides`（点路径覆写）

## 2. 一键执行（推荐）

```bash
python scripts/tuning_pipeline.py --spec data/tuning/specs/sample_experiments.yaml
```

可选参数：

```bash
python scripts/tuning_pipeline.py --spec data/tuning/specs/sample_experiments.yaml --only gru --max-experiments 3
```

## 3. 分步执行

```bash
python scripts/run_tuning_experiments.py --spec data/tuning/specs/sample_experiments.yaml
python scripts/collect_tuning_results.py --runs-dir data/tuning/runs --out-dir data/tuning/summary
python scripts/render_tuning_report.py --summary-csv data/tuning/summary/experiment_summary.csv --window-csv data/tuning/summary/window_level_metrics.csv --out data/tuning/summary/tuning_report.html
```

## 4. 输出目录
- 单实验产物：`data/tuning/runs/<experiment_id>/`
  - `configs/`（临时实验配置）
  - `logs/training_metrics.csv`
  - `run.log`
  - `result.json`
- 汇总产物：`data/tuning/summary/`
  - `experiment_summary.csv`
  - `window_level_metrics.csv`
  - `tuning_report.html`

## 5. 注意
- 组件不会覆盖原始 `config/*.yaml`，只在实验目录生成临时配置。
- 仍调用原有 `run_train.py`，现有工程流程保持不变。
- `--dry-run` 仅生成配置，不会训练；默认汇总会自动跳过 dry-run/失败/非新鲜指标结果。
- 如需把历史残留结果也纳入汇总，可显式加：`--include-stale`。
