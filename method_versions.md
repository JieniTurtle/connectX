# ConnectX 三版方法核心说明

本文档只保留三种版本的核心结构和主要局限。三版都从强化学习网络出发，区别主要在推理阶段是否加入前瞻和 MCTS。

## 版本一：残差策略价值网络

### 核心结构

- 输入：两个 6x7 棋盘平面，分别表示当前玩家棋子和对手棋子。
- 主干网络：卷积层 + 残差块，用于提取横向、纵向、斜向连线特征。
- 策略头：输出每个合法落子位置的动作概率。
- 价值头：输出当前局面对当前玩家的胜负价值。
- 推理方式：只做一次网络前向传播，屏蔽非法动作后选择策略概率最高的列。

### 主要局限

- 只依赖一次网络预测，关键局面缺少额外验证。
- 策略概率高的动作不一定长期最优，可能忽略对手下一步反击。
- 价值头虽然存在，但在决策时没有展开候选动作后再比较。

对应脚本：`version1_policy_value.py`

## 版本二：策略价值网络 + 轻量前瞻

### 核心结构

- 网络结构沿用版本一，不重新训练或更换模型。
- 先用策略头得到合法动作概率。
- 对若干候选动作分别模拟一步落子，得到下一局面。
- 用价值头评估每个下一局面。
- 综合当前策略概率和下一局面价值选择动作。

简化评分形式：

```text
score(a) = alpha * log P(s, a) - beta * V(s')
```

其中 `V(s')` 是落子后对手视角下的局面价值，所以对当前玩家取负号。

### 主要局限

- 只做一层前瞻，仍然难以处理多步陷阱。
- 每个候选动作只评估一次，没有访问次数统计。
- 没有 PUCT 探索项，可能漏掉初始概率较低但实际更强的动作。

对应脚本：`version2_light_lookahead.py`

## 版本三：AlphaZero + MCTS

### 核心结构

- 网络结构仍是策略价值网络。
- 策略头提供 MCTS 分支先验概率 `P(s, a)`。
- 价值头评估搜索叶节点局面价值 `V(s)`。
- MCTS 使用 PUCT 公式选择分支，反复执行选择、扩展、评估、回传。
- 最终选择访问次数最多的动作。

当前最终版在 `submission.py` 中实现，每步使用 `_SIMULATION_NUM = 200` 次模拟：

```text
score = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

### 主要局限

- 推理开销高于前两版。
- 搜索强度受 Kaggle 单步时间限制影响。
- 训练轮数不足时，价值网络在少见局面上仍可能不稳定。

对应脚本：`submission.py`

## 本地运行和测试

安装依赖：

```powershell
pip install kaggle-environments torch numpy
```

测试版本一：

```powershell
python version1_policy_value.py --episodes 40 --opponent random
python version1_policy_value.py --episodes 40 --opponent negamax
```

测试版本二：

```powershell
python version2_light_lookahead.py --episodes 40 --opponent random
python version2_light_lookahead.py --episodes 40 --opponent negamax
```

测试最终版：

```powershell
python submission.py
```

前两个脚本默认会分别测试先手和后手，并输出胜场数、负场数、平局数和胜率。为了报告更稳定，建议每组至少运行 40 局。

