1) 输入与目标
输入：两份预测文件（csi101/csi300），每行至少包含 datetime + (instrument 或 rq_code) + final，可能还包含 _meta_shifted_next_day。
目标：用最后一条预测生成下一个交易日的交易建议，并把结果写入年度台账 data/trade_plans/MSA_YYYY.csv。
2) 找预测文件：_find_latest_prediction()
优先找固定文件名：data/predictions/pred_{pool}.csv（比如 pred_csi101.csv）
找不到再回退：pred_{pool}_*.csv、rqalpha_pred_{pool}.csv、rqalpha_pred_{pool}_*.csv 等（按修改时间选最新）
3) 判断预测日期语义：pred_dates_are
在 main()：
你可以传 --pred-dates-are：auto/signal_date/trade_date
auto 的判断方式是 _infer_pred_dates_are()：如果预测文件里 _meta_shifted_next_day=1 → 认为文件 datetime 是 trade_date，否则认为是 signal_date。
> 关键点：脚本必须知道预测文件 datetime 到底代表“信号生成日”还是“交易建议日”，否则会偏一天。
4) 读取预测文件：load_prediction_csv(..., dates_are=pred_dates_are)
这里是最关键的对齐点：
读取 csv 后，会把数据变成 PredictionBook(by_date={date -> {rq_code -> score}})
如果 dates_are 是 auto/signal_date：并且文件 _meta_shifted_next_day=1，会在 loader 内把 trade_date 反向还原到 signal_date（避免隐性 T+2）
如果 dates_are=trade_date：则不做反向还原，按 trade_date 保存
5) 选“最后一条预测日期”作为 asof
在 main()：
先取 book101.by_date.keys() 的最大日期
如果有 book300，取两者共同日期的最大值（避免一份缺日期）
--asof 不传时，默认用这个最大日期（也就是“最后一条预测”）
6) 计算 signal_date / trade_date / pred_dt
如果 pred_dates_are == trade_date：
trade_date = asof
signal_date = 前一交易日(trade_date)
pred_dt = trade_date（取信号用 trade_date 键）
否则（signal_date）：
signal_date = asof
trade_date = 下一交易日(signal_date)
pred_dt = signal_date（取信号用 signal_date 键）
> 这样保证：你永远是在用“最后一条预测”去生成“下一交易日”的交易建议，且不会出现 T+2。
7) 构造两个子策略配置（S1/S2）
SubStrategyConfig 里核心字段：
topk_pred：候选池大小（你现在的逻辑里会被强制 >=100）
target_holdings：目标输出股票数（你现在两边默认都是 10）
filter_cfg：过滤条件（ST、上市天数、PB、近N日涨停等）
selection_mode：equal_weight 或 scheme_c（TopM→低波动→风险预算权重）
8) 子策略选股核心：_select_for_substrategy(...)
这是脚本里真正“怎么从预测变成持仓”的地方：
(1) 取当天信号
signals = book.get(pred_dt) 得到 {rq_code -> score}
(2) 构建候选池（硬编码保证足够顺延）
topn = max(sub.topk_pred, 100)
cand_raw = topk(signals, topn) 得到候选列表（按 score 降序）
(3) 严格过滤
cand = apply_basic_filters(cand_raw, signal_date, sub.filter_cfg, ts_client)
signal_date 用于取 PB/涨停等“当日可得信息”
(4) 计算波动率（如需要）
当 scheme_c 或启用 vol 过滤时，会用 tushare 或 bundle 计算 vol20/60/120
(5) 形成排序列表
scored = [(code, score)]，按 score 降序
(6) 初选 picks
scheme_c：按波动率在 TopM 内挑 K，再按风险预算给权重
否则：直接 Top target_holdings
(7) 同行业限制（每行业最多2只）
通过 index_member_all 建 ts_code -> 行业 映射
从 scored 里顺序挑选：行业满 2 就跳过，继续找下一只
(8) 回填补足到 10 只（非常关键，解决你说的 csi300 只出3只）
若过滤/行业cap 后 picks < target_holdings：
放松 PB 和涨停过滤（仍保留基础过滤 + 行业cap）
从 cand_raw 的排名继续往后补，直到尽量补满 10
回填发生后，sub_weights 会回退为空，后面统一走等权
(9) 权重
若没有可用的风险预算权重：回退等权，保证权重和为 sub.allocation
9) 合并两子策略 → 总权重
把两子策略的目标权重叠加到一起
再整体归一化到 1.0（满仓）
10) 写年度台账（MSA_YYYY.csv）
读取历史台账，找上一期调仓快照
如果本次目标权重与上一期一样 → 不更新
否则：
记录 SUMMARY 行（收益、换手、成本、净值等）
记录 POSITION 行（每只股票一行，包含 score、权重、波动率等）
写入：data/trade_plans/MSA_{year}.csv
