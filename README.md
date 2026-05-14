# AutoSolver: AI Agent for Delivery Assignment

美团 AI Hackathon 命题四参赛作品：让 AI Agent 自主求解配送分配问题。

## 作品简介

本项目实现了一个面向配送分配问题的 AutoSolver Agent。系统读取候选的订单-骑手组合、预估得分与接单意愿，自动进行多策略搜索、概率覆盖评估和局部替换优化，在限时内输出当前最优分配方案。相比示例贪心只给每个订单匹配单一低分候选，本方案会综合考虑多骑手指派、接单概率、合单组合与分数成本之间的权衡。

## 核心思路

- 解析 `task_id_list / courier_id / total_score / willingness` 候选集合。
- 使用接单概率估计每个订单被至少一名骑手接起的概率。
- 通过多组参数贪心探索“接单覆盖”和“分数成本”的不同权重。
- 使用随机扰动跳出单一排序策略带来的局部最优。
- 使用局部替换在时间预算内持续改进方案。
- 始终保留历史最优结果，接近时间上限时直接输出。

## 使用方式

参赛平台只需要导入 `solver.py`，其中定义了官方要求的接口：

```python
def solve(input_text: str) -> list:
    ...
```

本地测试：

```bash
python solver.py < large_seed301.txt
```

## 文件说明

- `solver.py`: 主求解器，包含解析、评分、搜索和输出逻辑。
- `README.md`: 作品说明。
- `.gitignore`: 忽略比赛数据和本地生成结果。

## 本地验证

在 `large_seed301.txt` 上，本地运行时间约 9 秒。基于内部概率覆盖指标，相比示例 baseline，期望接单覆盖有明显提升。
