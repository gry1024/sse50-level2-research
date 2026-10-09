from pathlib import Path
import json,html
import polars as pl
import numpy as np
R=Path('results');a=json.loads((R/'summary.json').read_text());e=json.loads((R/'execution_summary.json').read_text());f=pl.read_parquet(R/'features.parquet');grid=pl.read_csv(R/'training_grid.csv');et=pl.read_csv(R/'execution_trades.csv').filter(pl.col('split')=='afternoon')
spread=f['spread_bps'].to_numpy();bystock=f.group_by('symbol').agg(pl.col('spread_bps').median())['spread_bps'].to_numpy()
raw=pl.read_parquet('sse50_20260923/snp_20260923.parquet');symbols=raw['Symbol'].unique().sort().to_list()
trend=[]
for sym in symbols:
 g=raw.filter((pl.col('Symbol')==sym)&(pl.col('Time')>=93000000)&(pl.col('Time')<145700000)&(pl.col('BidPrice1')>0)).sort('Time')
 trend.append(float((g['BidPrice1'][-1]+g['AskPrice1'][-1])/(g['BidPrice1'][0]+g['AskPrice1'][0])-1)*1e4)
obs={'spread_p50_bps':float(np.median(spread)),'spread_p90_bps':float(np.quantile(spread,.9)),'stock_median_spread_min_max_bps':[float(bystock.min()),float(bystock.max())],'stocks_down':sum(x<0 for x in trend),'equal_stock_open_to_1457_mid_bps':float(np.mean(trend)),'eligible_roundtrip_configs':grid.filter(pl.col('eligible')).height,'positive_eligible_roundtrip_configs':grid.filter(pl.col('eligible')&(pl.col('train_net_bps')>0)).height,'execution_savings_after_extra_5bps':float(et['saving_cny'].sum()-et['notional'].sum()*.0005)}
(R/'observations.json').write_text(json.dumps(obs,indent=2))
text=f'''# SSE50 Level2：数据规律与成本约束下的交易策略

完成日期：2026-10-09。研究样本：2026-09-23，50 只股票、单个交易日。

**结论：可落地为模拟盘的候选是“已有买入计划的延迟执行”，不是已证明盈利的选股策略。下午节省 30.10 元（2.90 bp），留出股票节省为负，统计证据不足。短线卖出买回策略在保守交易成本下不成立。**

## 数据处理与审计

| 数据 | 原始记录数 | 用途 |
|---|---:|---|
| 逐笔委托 ord | 6,276,360 | 字段/事件码分布审计 |
| 逐笔成交 exe | 3,792,934 | 分钟成交额、量、上一完整分钟买卖方向 |
| 十档快照 snp | 252,631 | 清洗、盘口失衡、价差、模拟成交 |

清洗后保留 236,581 条连续竞价快照，剔除 16,050 条非目标交易时段/无效记录。过滤 09:30–11:30、13:00–14:57，避免午休与收盘集合竞价跨越；Symbol+Time 重复记录为 0，连续竞价零报价/锁定或交叉报价为 0。HHMMSSmmm 正确转为秒。原始文件 SHA256、完整 schema 在 `results/summary.json`。

逐笔委托 OrderKind=65（ASCII A）4,533,409 条、68（D）1,742,701 条、83（S）250 条；FunctionCode 的 B/S 分布和状态码单独记录。没有供应商字段规范，故不将这些代码擅自解释为完整订单簿生命周期，也没有声称逐笔重建订单簿。成交 BSFlag=B/S 按其方向字段汇总，0 为未知，未知不计入方向分母。

![委托与成交事件活动](results/order_activity.png)

## 观察到的规律

1. 开盘成交活动明显高于盘中，午后开盘和收盘前再次活跃；图中午休不连线，避免伪造休市成交。
2. 快照价差中位数 **{obs['spread_p50_bps']:.2f} bp**，90 分位 **{obs['spread_p90_bps']:.2f} bp**；各股中位价差范围 **{bystock.min():.2f}–{bystock.max():.2f} bp**。同一信号在不同股票上的执行成本差别很大。
3. 一档盘口失衡 OBI=(BidVolume1−AskVolume1)/(BidVolume1+AskVolume1) 的极端值与随后 30/60 秒中间价变动同向，但图中的量级通常仅几个 bp。长周期曲线混入当日趋势，不应直接称为可套利规律。
4. 当日 {obs['stocks_down']}/50 只股票从开盘至 14:57 前最后快照下跌，等权中间价变动 {obs['equal_stock_open_to_1457_mid_bps']:.2f} bp。等待买入的收益可能只是当天整体下跌带来的结果，不能据此断言买盘占优必然引发回落。

![市场微观结构](results/microstructure.png)

分桶图为下午所有有效样本的描述性均值，未来标签重叠、股票间存在共同市场冲击；不使用这些快照数作为独立统计样本数。

## 可实施候选：计划买入的延迟执行

使用场景：已经决定买入某股并持有，比较立刻买与稍后买相同股数的成本。**信号只控制执行时机，不产生买入投资决策。**

- 五档失衡 `OBI5=(sum BidVolume1:5−sum AskVolume1:5)/(sum BidVolume1:5+sum AskVolume1:5)` ≥ 0.6 时触发。
- 每只股票/实验时段只触发一次，信号后至少 1 秒取得基准报价，延迟 900 秒买入；不跨午休或 14:57，不在当天卖出。
- 股数按 100 股向下取整，最多约 10,000 元，且不超过信号时一档卖量的 1%；基准执行时不超过一档深度的 10%。无法满足则等待下一次信号，不强行成交。
- 买入按十档卖盘逐档计算 VWAP，每档最多使用显示量的 10%，再加 0.01 元/股不利滑点。下午 21 笔十档容量检查全部通过；显示深度仍不等于真实成交保证。
- 两种执行方式都计一次买入佣金：万二且最低 5 元、过户费万分之 0.1；买入无卖出印花税。比较同股数完整成本，避免把两笔交易成本错误地加到单次买入择时上。
- 策略投入模拟盘；实盘启用条件是新增多日冻结规则验证、加大滑点后收益仍为正、库存/成交失败有实测结果。当前条件未满足。

| 样本 | 笔数 | 节省金额 CNY | 按成交金额加权节省 bp | 正节省比例 |
|---|---:|---:|---:|---:|
| 上午训练股票 | 25 | 198.10 | 15.35 | 48.00% |
| 下午全部股票 | 21 | 30.10 | 2.90 | 47.62% |
| 下午留出股票 | 4 | −8.00 | −4.38 | 25.00% |

下午按股票/交易重采样的**等权每笔**节省 95% 区间为 [{e['metrics']['afternoon_all']['mean_trade_bps_ci95'][0]:.2f}, {e['metrics']['afternoon_all']['mean_trade_bps_ci95'][1]:.2f}] bp，跨零；这与表中**金额加权**平均为不同估计量。额外增加 5 bp 执行成本后，下午节省变为 {obs['execution_savings_after_extra_5bps']:.2f} 元。没有证明成本优势稳定存在。

![买入执行比较](results/execution.png)

## 被否定的候选：底仓先卖、15 分钟后买回

按上午训练选择五档 OBI≤−0.6、持有间隔 900 秒；需要开盘前已结算的可卖底仓。上午/下午是相互独立的反事实实验，不能将上午买回的股票当作下午可卖库存。

36 组固定网格：3 种信号（一档失衡、五档失衡、前一完整分钟成交失衡）×2 方向×2 阈值（0.6/0.8）×3 间隔（300/900/1800 秒）。仅上午训练股票选参数；股票按 Symbol%5=0 留出。要求至少 10 笔训练交易且没有买回容量失败。符合条件的 {obs['eligible_roundtrip_configs']} 组中，净收益为正的为 **{obs['positive_eligible_roundtrip_configs']} 组**。

双边主动成交使用卖一/买一、每腿一跳不利滑点、每腿万二且最低 5 元佣金、每腿 0.1 bp 过户费、卖出 5 bp 印花税。上午训练 12 对交易净损失 **163.00 元（−36.85 bp）**。下午 9 对报价标记损失 **128.80 元（−28.14 bp）**，其中 1 对买回深度不足；**因此下午总计是失败记录未剔除的理论 markout，不是可执行的已实现收益**。保留失败情况，不使用未来容量筛选美化回测。

![底仓策略与成本敏感度](results/strategy.png)

## 验证与限制

本研究快速完成，曾观察初版下午结果并修正仓位/容量和执行策略类别，故本次“上午训练、下午测试”仅为**探索性时间切分**，不是完全未接触的样本外证据。没有按下午收益挑选股票，但过程存在研究者自由度。

单日样本不足以年化收益、计算可靠年化 Sharpe 或宣称长期盈利。等待期间价格上涨会使买入更贵；信号下单时刻的比较也不能代替随机时间/无条件等待基准。未来应收集至少 20–60 个独立交易日，预注册规则、加入无条件延迟/随机时点/去市场收益基准，并用按日聚类区间验证。

已通过 6 个行为测试：毫秒时间解析、休市/集合竞价边界、最低佣金与印花税、下单延迟/整手约束、禁止跨午休、十档 VWAP 和容量失败。随机种子固定为 42。

## 复现

原始教学数据来自 [课程 issue #6](https://github.com/aslan9/pku_quantllm/issues/6)。将三个 parquet 放入 `sse50_20260923/`。

```bash
uv sync --extra dev
uv run python research.py
uv run python execution.py
uv run python order_flow.py
uv run python make_report.py
uv run python -m pytest
```

代码、图表、训练网格、逐笔实验记录、统计 JSON 可公开复核；原始 parquet 和派生完整快照特征不上传。`results/dashboard.html` 可直接打开。

交易规则参考：[上交所 2026 年交易规则 3.1.4](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)、[上交所股票交易税费说明](https://one.sse.com.cn/onething/gptz/)。佣金和过户费是本实验假设，实际以账户费率为准。
'''
Path('README.md').write_text(text,encoding='utf8')
# Static self-contained report: embedded figures, no network dependencies.
import base64
cards=''.join('<section><h2>'+title+'</h2><img src="data:image/png;base64,'+base64.b64encode((R/name).read_bytes()).decode()+'"></section>' for title,name in [('委托与成交事件','order_activity.png'),('微观结构','microstructure.png'),('计划买入的延迟执行','execution.png'),('被否定的底仓策略','strategy.png')])
body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SSE50 Level2 研究</title><style>body{font:16px/1.7 system-ui;background:#f4f6f8;color:#182333;max-width:1180px;margin:40px auto;padding:24px}section{background:white;border-radius:14px;padding:24px;margin:24px 0}img{width:100%;height:auto}h1{font-size:32px}.metric{font-size:24px;color:#236d86}pre{white-space:pre-wrap;background:#fff;padding:24px;border-radius:14px}</style><h1>SSE50 Level2 · 单日探索性研究</h1><p>2026-09-23 · 50 只股票 · 10,321,925 条原始记录</p><p class="metric">计划买入延迟执行：下午 21 笔，成本节省 30.10 元 / 2.90 bp</p><p>留出股票结果为负，95% 区间跨零；新增 5 bp 成本后优势消失。当前仅适合模拟验证，没有证明长期盈利。</p>'''+cards+'<h2>完整研究说明</h2><pre>'+html.escape(text)+'</pre></html>'
(R/'dashboard.html').write_text(body,encoding='utf8')
policy=dict(mode='paper_only',signal='obi5',condition='>=0.6',delay_seconds=900,max_order_cny=10000,lot_size=100,max_signal_ask1_participation=.01,max_fill_depth_participation=.1,max_trades_per_stock_per_day=1,allow_same_day_sell=False,enable_live=False,reason='Single-day exploratory results, confidence interval crosses zero, held-out stocks negative.')
Path('strategy_config.json').write_text(json.dumps(policy,indent=2))
print(json.dumps(obs,indent=2))

