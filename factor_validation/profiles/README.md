# ValidationProfile —— 验证标准配置

每个 YAML 文件即一个"验证标准"。完整字段说明见
[`docs/VALIDATION_PROFILE_GUIDE.md`](../../docs/VALIDATION_PROFILE_GUIDE.md)。

## 内置 profile（阶段 C 落地）

| 文件 | 用途 |
|---|---|
| `default.yaml` | 默认验证标准；中等严苛 |
| `strict.yaml` | Production 准入级；所有 check 必须通过 |
| `exploratory.yaml` | 早期探索；放宽 IC / 换手等阈值 |

## 自定义 profile

1. 复制 `default.yaml`，改名为 `my_profile.yaml`
2. 调阈值或 `enabled: false` 关掉某些 check
3. 命令行用 `--profile my_profile` 引用

每次验证都会把 profile 的 sha256 写进 `CertifiedFactorRecord.profile_hash`，证书永久可复现。
