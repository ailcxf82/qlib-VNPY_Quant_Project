# Tushare MSA Backtest

This folder contains an independent Tushare-backed MSA backtest path. It does
not use RQAlpha bundle data and does not modify the existing RQAlpha scripts.

## Run

Set the Tushare token before running:

```powershell
$env:TUSHARE_API_KEY="your_token"
```

```powershell
conda run -n qlib_zhengshi python backtest/msa_tushare/tushare_msa_backtest.py --strategy-mode csi300_only --pred-csi300 data/predictions/pred_csi300.csv
```

By default the script runs the latest two signal dates. Use `--start-date` and
`--end-date` to select a wider signal-date range.

## Outputs

Files are written to `data/backtest/msa_tushare`:

- `msa_tushare_nav.csv`
- `msa_tushare_positions.csv`
- `msa_tushare_summary.json`

## Notes

- Market data, trading calendar, PB, limit-up lists, and close-to-close returns
  are pulled from Tushare and cached under `data/tushare_cache`.
- The script reads the token from `TUSHARE_API_KEY`, `TUSHARE_TOKEN`, or
  `config/secrets.yaml`; it does not accept tokens on the command line.
- This is a daily close-to-close backtest, not an RQAlpha order-matching
  simulation.
