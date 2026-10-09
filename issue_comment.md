[**作业仓库：SSE50 Level2 数据分析与策略验证**](https://github.com/gry1024/sse50-level2-research)

- 完成 **50 只股票、1,032 万条 Level2 记录**的清洗、微观结构可视化及含税费、滑点、十档容量约束的策略实验。
- **核心发现：预测性不等于可交易收益。**价差中位数 3.90 bp；满足训练成交约束的 6 组短线回转参数均亏损。
- 提出“已有买入计划延迟 15 分钟执行”候选：下午 21 笔模拟节省 **30.10 元／2.90 bp**，但留出股票结果为负、置信区间跨零，尚未证明稳定有效。

[完整分析与复现说明](https://github.com/gry1024/sse50-level2-research#readme) · [逐笔实验记录](https://github.com/gry1024/sse50-level2-research/blob/main/results/execution_trades.csv)

**盘口信号与执行成本：**
![微观结构与盘口失衡响应](https://raw.githubusercontent.com/gry1024/sse50-level2-research/main/results/microstructure.png)

**候选策略的逐股结果与验证差异：**
![延迟买入的成本节省与验证结果](https://raw.githubusercontent.com/gry1024/sse50-level2-research/main/results/execution.png)
