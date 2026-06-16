# 模型校准误差报告

已评估比赛数: 16

胜平负命中率: 43.8%
精确比分命中率: 12.5%
Brier 分数: 0.2452
对数损失: 1.1558
比分平均绝对误差: 0.8438
预测总进球/实际总进球: 54.48 / 46.00
总进球校准比率: 0.844

## 本轮参数微调

- 球队1基础进球倍率: 0.9520
- 球队2基础进球倍率: 0.9400
- Elo 进球影响倍率: 0.9900
- 主场进球加成调整: -0.0028
- 冷门/平局保护: diff_shrink=0.639, draw_boost=1.177
- 强弱悬殊大比分尾部: tail_boost=1.011
- 说明: 样本较少时使用强收缩，避免两三场比赛导致参数剧烈摆动。

## 逐场误差

| 场次 | 比赛 | 预测比分 | 实际比分 | 胜平负是否命中 | 精确比分是否命中 | 期望总进球 | 实际总进球 |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Mexico vs South Africa | 2-0 | 2-0 | 是 | 是 | 3.89 | 2 |
| 2 | South Korea vs Czech Republic | 2-1 | 2-1 | 是 | 是 | 3.04 | 3 |
| 7 | Canada vs Bosnia & Herzegovina | 2-0 | 1-1 | 否 | 否 | 3.52 | 2 |
| 8 | Qatar vs Switzerland | 0-2 | 1-1 | 否 | 否 | 3.16 | 2 |
| 13 | Brazil vs Morocco | 2-1 | 1-1 | 否 | 否 | 3.10 | 2 |
| 14 | Haiti vs Scotland | 1-2 | 0-1 | 是 | 否 | 2.72 | 1 |
| 19 | USA vs Paraguay | 2-1 | 4-1 | 是 | 否 | 2.73 | 5 |
| 20 | Australia vs Turkey | 2-1 | 2-0 | 是 | 否 | 2.79 | 2 |
| 25 | Germany vs Curacao | 4-0 | 7-1 | 是 | 否 | 5.46 | 8 |
| 26 | Ivory Coast vs Ecuador | 1-2 | 1-0 | 否 | 否 | 2.82 | 1 |
| 31 | Netherlands vs Japan | 2-1 | 2-2 | 否 | 否 | 3.13 | 4 |
| 32 | Sweden vs Tunisia | 2-1 | 5-1 | 是 | 否 | 2.77 | 6 |
| 37 | Belgium vs Egypt | 2-1 | 1-1 | 否 | 否 | 3.73 | 2 |
| 38 | Iran vs New Zealand | 2-1 | 2-2 | 否 | 否 | 3.43 | 4 |
| 43 | Spain vs Cape Verde | 4-0 | 0-0 | 否 | 否 | 5.39 | 0 |
| 44 | Saudi Arabia vs Uruguay | 1-2 | 1-1 | 否 | 否 | 2.79 | 2 |