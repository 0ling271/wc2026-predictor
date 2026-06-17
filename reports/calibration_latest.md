# 模型校准误差报告

已评估比赛数: 20

胜平负命中率: 50.0%
精确比分命中率: 10.0%
Brier 分数: 0.2085
对数损失: 1.0137
比分平均绝对误差: 0.9250
预测总进球/实际总进球: 62.97 / 62.00
总进球校准比率: 0.985

## 本轮参数微调

- 球队1基础进球倍率: 1.0333
- 球队2基础进球倍率: 0.9400
- Elo 进球影响倍率: 1.0000
- 主场进球加成调整: +0.0029
- 冷门/平局保护: diff_shrink=0.669, draw_boost=1.142
- 强弱悬殊大比分尾部: tail_boost=1.037
- 说明: 样本较少时使用强收缩，避免两三场比赛导致参数剧烈摆动。

## 逐场误差

| 场次 | 比赛 | 预测比分 | 实际比分 | 胜平负是否命中 | 精确比分是否命中 | 期望总进球 | 实际总进球 |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Mexico vs South Africa | 2-0 | 2-0 | 是 | 是 | 3.41 | 2 |
| 2 | South Korea vs Czech Republic | 2-1 | 2-1 | 是 | 是 | 2.76 | 3 |
| 7 | Canada vs Bosnia & Herzegovina | 2-0 | 1-1 | 否 | 否 | 3.12 | 2 |
| 8 | Qatar vs Switzerland | 0-2 | 1-1 | 否 | 否 | 3.05 | 2 |
| 13 | Brazil vs Morocco | 2-1 | 1-1 | 否 | 否 | 2.84 | 2 |
| 14 | Haiti vs Scotland | 1-2 | 0-1 | 是 | 否 | 2.60 | 1 |
| 19 | USA vs Paraguay | 2-1 | 4-1 | 是 | 否 | 2.53 | 5 |
| 20 | Australia vs Turkey | 1-2 | 2-0 | 否 | 否 | 2.61 | 2 |
| 25 | Germany vs Curacao | 2-0 | 7-1 | 是 | 否 | 4.73 | 8 |
| 26 | Ivory Coast vs Ecuador | 1-2 | 1-0 | 否 | 否 | 2.69 | 1 |
| 31 | Netherlands vs Japan | 2-1 | 2-2 | 否 | 否 | 2.89 | 4 |
| 32 | Sweden vs Tunisia | 2-1 | 5-1 | 是 | 否 | 2.56 | 6 |
| 37 | Belgium vs Egypt | 2-1 | 1-1 | 否 | 否 | 3.29 | 2 |
| 38 | Iran vs New Zealand | 2-1 | 2-2 | 否 | 否 | 3.07 | 4 |
| 43 | Spain vs Cape Verde | 4-0 | 0-0 | 否 | 否 | 5.44 | 0 |
| 44 | Saudi Arabia vs Uruguay | 1-2 | 1-1 | 否 | 否 | 2.68 | 2 |
| 49 | France vs Senegal | 2-1 | 3-1 | 是 | 否 | 3.47 | 4 |
| 50 | Iraq vs Norway | 1-2 | 1-4 | 是 | 否 | 2.85 | 5 |
| 55 | Argentina vs Algeria | 2-1 | 3-0 | 是 | 否 | 3.62 | 3 |
| 56 | Austria vs Jordan | 2-1 | 3-1 | 是 | 否 | 2.77 | 4 |