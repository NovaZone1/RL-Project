# Q-Learning 强化学习实验

本项目是强化学习实践的第三个阶段。

前两个阶段分别完成了：

- Bandit：学习 `Q(a)`、探索与利用、ε-greedy
- GridWorld：学习 State、Episode、Return、Value、Bellman、Policy Iteration

本阶段进一步学习：

- `Q(s, a)`
- TD Target
- TD Error
- Q-Learning 更新
- ε-greedy 探索
- ε 衰减
- Train / Evaluation 分离
- 最优策略提取

最终目标是：

> 不提前知道完整环境模型，只通过与环境交互，学习出最优 Q-table 和最优策略。

---

## 1. 项目结构

```text
q_learning/
├── environment.py
├── 01_q_table.py
├── 02_q_learning.py
├── 03_epsilon_decay.py
├── 04_train_evaluate.py
├── README.md
└── plots/
    ├── epsilon_compare.png
    └── epsilon_decay.png
```

各文件作用：

| 文件 | 主要内容 |
|---|---|
| `environment.py` | 4×4 GridWorld 环境 |
| `01_q_table.py` | Q-table 与一次 Q-Learning 更新 |
| `02_q_learning.py` | 完整 Q-Learning 训练与策略输出 |
| `03_epsilon_decay.py` | Fixed ε 与 Decaying ε 对比 |
| `04_train_evaluate.py` | 训练与评估分离 |
| `plots/` | 保存实验曲线 |

---

## 2. 环境设置

使用 4×4 GridWorld：

```text
S . . .
. . . .
. . . .
. . . G
```

其中：

- `S`：起点 `(0, 0)`
- `G`：终点 `(3, 3)`

动作：

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

Reward 设计：

```text
普通移动：-1
进入 Goal：0
```

因此 Agent 想最大化长期 Return，就必须尽量减少到达 Goal 的步数。

理论最短路径长度：

```text
6 steps
```

---

## 3. 从 Q(a) 到 Q(s, a)

Bandit 阶段使用：

```text
Q(a)
```

它表示：

> 动作 a 长期来看有多好？

GridWorld 有多个状态，因此升级为：

```text
Q(s, a)
```

它表示：

> 在状态 s 下执行动作 a，长期来看有多好？

例如：

```python
Q[2, 1, 3]
```

表示：

```text
state = (2, 1)
action = RIGHT
```

对应：

```text
Q((2,1), RIGHT)
```

---

## 4. Q-table

GridWorld 一共有：

```text
4 × 4 = 16 states
```

每个状态有：

```text
4 actions
```

因此 Q-table：

```python
Q = np.zeros(
    (env.rows, env.cols, 4)
)
```

形状：

```text
(4, 4, 4)
```

可以理解为：

```text
每一个 state
都有一组：

[Q(UP), Q(DOWN), Q(LEFT), Q(RIGHT)]
```

例如：

```text
state = (2,2)

Q[2,2] =
[-2.02, -1.00, -2.13, -1.00]
```

说明：

```text
UP     = -2.02
DOWN   = -1.00
LEFT   = -2.13
RIGHT  = -1.00
```

因此 DOWN 和 RIGHT 并列最优。

---

## 5. Q-Learning 更新公式

Q-Learning 的核心更新：

```text
Q(s,a) ← Q(s,a)
         + α [r + γ max Q(s',a') - Q(s,a)]
```

可以拆成三个部分：

```text
TD Target
=
r + γ max Q(s',a')
```

```text
TD Error
=
TD Target - Q(s,a)
```

```text
Q_new
=
Q_old + α × TD Error
```

整体结构仍然和 Bandit 一样：

```text
新估计
=
旧估计
+
学习率 × 预测误差
```

区别只是 Bandit 的目标是：

```text
reward
```

而 Q-Learning 的目标加入了下一状态的未来价值：

```text
reward + future value
```

---

## 6. 实验 1：单次 Q-Learning 更新

文件：

```text
01_q_table.py
```

人工指定：

```text
state = (2,2)
action = RIGHT
```

环境返回：

```text
reward = -1
next_state = (2,3)
done = False
```

初始 Q-table 全为 0。

因此：

```text
old Q = 0
best next Q = 0
```

TD Target：

```text
target = -1 + 1 × 0
       = -1
```

TD Error：

```text
td_error = -1 - 0
         = -1
```

学习率：

```text
alpha = 0.1
```

更新后：

```text
new Q = 0 + 0.1 × (-1)
      = -0.1
```

实验输出：

```text
Q values at state (2,2):
[ 0.   0.   0.  -0.1]
```

说明只更新了真正经历过的：

```text
Q((2,2), RIGHT)
```

其他动作没有被执行，因此不更新。

---

## 7. Terminal State 的处理

普通情况下：

```text
TD Target
=
r + γ max Q(s',a')
```

但如果：

```text
done = True
```

说明已经进入 Goal，Episode 结束。

这时没有未来动作，因此：

```text
TD Target = reward
```

代码：

```python
if done:
    target = reward
else:
    target = reward + gamma * best_next_q
```

这个逻辑和前面 Bellman Value 中：

```python
if done:
    next_value = 0
```

本质一致。

---

## 8. 实验 2：完整 Q-Learning

文件：

```text
02_q_learning.py
```

训练过程：

```text
初始化 Q-table
    ↓
reset 环境
    ↓
根据 ε-greedy 选择 action
    ↓
执行 action
    ↓
得到 reward 和 next_state
    ↓
计算 TD Target
    ↓
计算 TD Error
    ↓
更新 Q(s,a)
    ↓
state = next_state
    ↓
直到 done
```

训练参数：

```python
n_episodes = 1000
max_steps = 500
alpha = 0.1
gamma = 1.0
epsilon = 0.1
```

---

## 9. ε-greedy

训练过程中不能始终只选当前 Q 最大的动作，否则可能过早陷入错误策略。

所以使用 ε-greedy：

```text
概率 ε：
    随机选择动作
    → Exploration

概率 1 - ε：
    选择当前 Q 最大的动作
    → Exploitation
```

代码：

```python
if rng.random() < epsilon:
    action = rng.integers(4)
else:
    q_values = Q[row, col]

    best_actions = np.flatnonzero(
        np.isclose(
            q_values,
            q_values.max()
        )
    )

    action = rng.choice(
        best_actions
    )
```

并列最大时随机选择，避免固定偏向某个动作。

---

## 10. 完整 Q-Learning 实验结果

固定：

```text
ε = 0.1
```

训练 1000 个 Episode 后：

```text
Average steps in first 100 episodes: 13.97
Average steps in last 100 episodes: 6.63
```

理论最短路径为：

```text
6 steps
```

因此：

```text
13.97 → 6.63
```

说明 Agent 通过 trial-and-error 逐渐学会了接近最短路径的策略。

后期没有严格等于 6，是因为训练时仍然保留 10% 的随机探索。

---

## 11. Q-table 结果分析

训练得到的部分 Q 值：

```text
state = (0,0)

[-5.81, -5.00, -5.83, -5.00]
```

对应：

```text
UP     = -5.81
DOWN   = -5.00
LEFT   = -5.83
RIGHT  = -5.00
```

最大的是：

```text
DOWN
RIGHT
```

因此从起点向下或向右都属于最优动作。

再看：

```text
state = (2,3)

[-1.50, 0.00, -1.66, -0.52]
```

DOWN 的 Q 值最大：

```text
Q((2,3), DOWN) = 0
```

因为 `(2,3)` 向下直接进入 Goal。

---

## 12. Learned Policy

训练结束后，根据：

```text
argmax_a Q(s,a)
```

可以直接得到策略。

实验输出：

```text
Learned Policy:

↓→  ↓→  ↓→  ↓
↓→  ↓→  ↓→  ↓
↓→  ↓→  ↓→  ↓
→   →   →   G
```

其中：

```text
↓→
```

表示 DOWN 和 RIGHT 并列最优。

最终策略和前面 Policy Iteration 得到的最优策略一致。

这说明：

> Q-Learning 在不知道完整环境模型的情况下，也能够通过交互学习出最优策略。

---

## 13. Policy Iteration 与 Q-Learning 的区别

Policy Iteration：

```text
知道完整环境模型
↓
可以遍历所有 state / action
↓
使用 Bellman 方程计算
↓
得到最优策略
```

Q-Learning：

```text
不知道完整环境模型
↓
只能实际执行 action
↓
观察 reward 和 next_state
↓
通过样本更新 Q(s,a)
↓
最终得到最优策略
```

因此：

```text
Policy Iteration
→ Dynamic Programming
→ 依赖环境模型

Q-Learning
→ Model-Free RL
→ 不需要完整转移模型
```

---

## 14. 实验 3：Fixed ε vs Decaying ε

文件：

```text
03_epsilon_decay.py
```

比较两种探索策略。

### Fixed ε

```text
ε = 0.1
```

整个训练过程中保持不变。

实验结果：

```text
First 100 episodes: 13.95
Last 100 episodes: 6.55
```

### Decaying ε

设置：

```text
ε_start = 1.0
ε_end = 0.01
decay_episodes = 800
```

训练前期高探索，后期逐渐转向利用。

实验结果：

```text
First 100 episodes: 45.43
Last 100 episodes: 6.06
```

---

## 15. 为什么 Decaying ε 前期更差？

Decay 一开始：

```text
ε = 1.0
```

意味着几乎完全随机行动。

因此：

```text
前 100 Episode
Fixed ε: 13.95
Decay ε: 45.43
```

Decay 明显更差。

但这是主动设计的：

```text
前期牺牲短期表现
↓
换取更多 Exploration
```

---

## 16. 为什么 Decaying ε 后期更好？

Fixed ε：

```text
ε = 0.1
```

即使已经学会最优策略，仍然有 10% 概率随机行动。

因此后期：

```text
6.55 steps
```

Decaying ε 后期：

```text
ε = 0.01
```

只有约 1% 随机探索。

因此：

```text
6.06 steps
```

已经非常接近理论最短路径：

```text
6 steps
```

实验结果：

| 策略 | 前 100 Episode | 后 100 Episode |
|---|---:|---:|
| Fixed ε = 0.1 | 13.95 | 6.55 |
| Decaying ε | 45.43 | 6.06 |

核心结论：

```text
Fixed ε
→ 前期表现较好
→ 后期仍持续探索

Decaying ε
→ 前期探索更多
→ 后期利用更充分
```

---

## 17. ε、α、γ 的区别

三个参数不能混淆。

| 参数 | 含义 | 控制什么 |
|---|---|---|
| ε | Exploration Rate | 怎么选择动作 |
| α | Learning Rate | Q 更新多快 |
| γ | Discount Factor | 未来 Reward 有多重要 |

可以记成：

```text
ε = 怎么选
α = 怎么学
γ = 看多远
```

---

## 18. 实验 4：Training 与 Evaluation

文件：

```text
04_train_evaluate.py
```

训练阶段：

```text
ε-greedy
↓
探索环境
↓
持续更新 Q-table
```

评估阶段：

```text
冻结 Q-table
↓
ε = 0
↓
纯 Greedy
↓
只测试，不更新
```

这是强化学习实验中非常重要的区别：

```text
Training Performance
≠
Learned Policy Performance
```

训练阶段需要 Exploration。

评估阶段不需要 Exploration。

---

## 19. Evaluation 结果

训练结束后，使用 100 个 Evaluation Episode。

结果：

```text
Average steps: 6.0
Minimum steps: 6
Maximum steps: 6
Success rate: 1.0
```

即：

```text
100% success rate
```

并且每次都是：

```text
6 steps
```

说明学到的 greedy policy 已经达到理论最优。

---

## 20. Greedy Trajectory

Evaluation 中的一条实际轨迹：

```text
[(0, 0),
 (1, 0),
 (2, 0),
 (2, 1),
 (2, 2),
 (3, 2),
 (3, 3)]
```

对应：

```text
(0,0)
  ↓
(1,0)
  ↓
(2,0)
  →
(2,1)
  →
(2,2)
  ↓
(3,2)
  →
(3,3)
```

共：

```text
6 steps
```

虽然和 Policy Iteration 打印出的路径不完全一样，但两者都是最短路径。

因此：

```text
最优策略不一定只有一条具体路径
```

---

## 21. 为什么 Q-Learning 能学出最优策略？

Q-Learning 的 TD Target 使用：

```text
r + γ max Q(s',a')
```

其中：

```text
max Q(s',a')
```

表示：

> 到达下一状态以后，假设未来选择当前认为最好的动作。

所以 Q-Learning 的学习目标不断向最优动作价值靠近。

最终：

```text
Q(s,a) → Q*(s,a)
```

然后：

```text
π*(s) = argmax_a Q*(s,a)
```

就得到最优策略。

---

## 22. Bandit、GridWorld、Q-Learning 的联系

### Bandit

学习：

```text
Q(a)
```

更新：

```text
Q(a)
←
Q(a) + α [R - Q(a)]
```

### GridWorld

加入：

```text
State
Episode
Return
Value
Bellman
```

学习：

```text
Vπ(s)
```

### Q-Learning

进一步学习：

```text
Q(s,a)
```

更新：

```text
Q(s,a)
←
Q(s,a)
+
α [r + γ max Q(s',a') - Q(s,a)]
```

三阶段主线：

```text
Bandit
Q(a)
   ↓
GridWorld
Vπ(s)
   ↓
Q-Learning
Q(s,a)
```

---

## 23. Q-Learning 的完整训练闭环

整个项目可以总结为：

```text
初始化 Q-table
      ↓
观察 state
      ↓
ε-greedy 选择 action
      ↓
Environment 执行动作
      ↓
得到 reward 和 next_state
      ↓
计算 TD Target
      ↓
计算 TD Error
      ↓
更新 Q(s,a)
      ↓
state = next_state
      ↓
重复
```

训练完成后：

```text
Q-table
   ↓
argmax Q(s,a)
   ↓
Greedy Policy
   ↓
Evaluation
```

最终得到：

```text
Success Rate = 100%
Average Steps = 6
```

---

## 24. 本阶段最重要的结论

### 24.1 Q(s,a) 同时包含状态和动作信息

```text
V(s)
→ 当前状态有多好

Q(s,a)
→ 当前状态下做这个动作有多好
```

### 24.2 Q-Learning 是 TD 方法

它不需要等整个 Episode 完成才更新。

每一步交互以后就可以：

```text
立即更新 Q(s,a)
```

### 24.3 Q-Learning 不需要完整环境模型

Agent 不需要提前知道：

```text
P(s'|s,a)
```

只需要通过交互观察：

```text
(s, a, r, s')
```

即可学习。

### 24.4 Exploration 影响经验数据

ε 不直接修改 Q 更新公式。

它决定：

```text
Agent 会收集到哪些经验
```

### 24.5 Training 和 Evaluation 必须分开

训练：

```text
需要探索
需要更新
```

评估：

```text
不探索
不更新
只测试最终策略
```

### 24.6 Q-table 本身就包含策略

训练完成后：

```text
π(s) = argmax_a Q(s,a)
```

就可以直接得到 greedy policy。

---

## 25. 与 DQN 的联系

Q-Learning 当前把所有动作价值直接存入：

```text
Q-table
```

当状态空间很小时，这种方法很好用。

但是如果状态变成：

```text
机器人关节角
相机图像
激光雷达
连续位置
高维传感器
```

就不可能为每一个状态单独存一行 Q-table。

于是下一阶段需要把：

```text
Q-table
```

替换成：

```text
Neural Network
```

即：

```text
state
  ↓
Q Network
  ↓
Q(s,a1)
Q(s,a2)
Q(s,a3)
...
```

也就是：

```text
Q(s,a)
→
Qθ(s,a)
```

这就是 DQN。

---

## 26. 下一阶段

下一阶段进入：

```text
DQN
```

重点学习：

```text
Q-table
→ Neural Network

TD Target
→ DQN Target

单步在线样本
→ Replay Buffer

同一个网络估计目标
→ Target Network
```

学习路线：

```text
Bandit
   ↓
GridWorld
   ↓
Q-Learning
   ↓
DQN
   ↓
Policy Gradient
   ↓
Actor-Critic
   ↓
PPO
```
