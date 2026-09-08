# DQN 强化学习实验

本项目是强化学习实践的第四个阶段。

前面已经完成：

- Bandit：`Q(a)`、ε-greedy、探索与利用
- GridWorld：State、Return、Value、Bellman、Policy Iteration
- Q-Learning：`Q(s,a)`、TD Target、TD Error、Model-Free Learning

本阶段进一步把 Q-Learning 从：

```text
Q-table
```

升级为：

```text
Neural Network
```

即：

```text
Q(s,a)
→
Qθ(s,a)
```

最终实现完整的 Deep Q-Network（DQN），并在 4×4 GridWorld 中训练出 100% 成功率、6 步最短路径的最优策略。

---

## 1. 项目结构

```text
dqn/
├── environment.py
├── 01_q_network.py
├── 02_online_dqn.py
├── 03_replay_buffer.py
├── 04_minibatch_update.py
├── 05_target_network.py
├── 06_dqn_train.py
├── README.md
└── plots/
    ├── training_steps.png
    └── training_loss.png
```

各文件作用：

| 文件 | 主要内容 |
|---|---|
| `environment.py` | 4×4 GridWorld 环境 |
| `01_q_network.py` | 用神经网络替代 Q-table |
| `02_online_dqn.py` | 单条 transition 的神经网络 TD 更新 |
| `03_replay_buffer.py` | Replay Buffer 与随机经验采样 |
| `04_minibatch_update.py` | Mini-Batch DQN 更新 |
| `05_target_network.py` | Online Network 与 Target Network |
| `06_dqn_train.py` | 完整 DQN 训练与评估 |
| `plots/training_steps.png` | 训练步数曲线 |
| `plots/training_loss.png` | Loss 曲线 |

---

## 2. 为什么从 Q-Learning 升级到 DQN？

Q-Learning 使用：

```text
Q-table
```

直接保存：

```text
Q(s,a)
```

对于 4×4 GridWorld：

```text
16 states × 4 actions
```

Q-table 很小，因此可以直接存储。

但是现实任务中状态可能是：

```text
机器人位置
关节角
速度
相机图像
激光雷达
触觉信号
```

状态空间可能巨大甚至连续，因此无法为每一个状态单独建立一行 Q-table。

于是使用神经网络：

```text
state
  ↓
Q Network
  ↓
Qθ(s,UP)
Qθ(s,DOWN)
Qθ(s,LEFT)
Qθ(s,RIGHT)
```

也就是：

```text
Q-table
→
Function Approximation
```

DQN 的核心思想可以概括为：

```text
使用神经网络近似 Q(s,a)
```

---

## 3. 环境设置

继续使用前面的 4×4 GridWorld：

```text
S . . .
. . . .
. . . .
. . . G
```

其中：

```text
S = Start = (0,0)
G = Goal  = (3,3)
```

动作：

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

Reward：

```text
普通移动：-1
进入 Goal：0
```

理论最短路径：

```text
6 steps
```

因此最终目标仍然是：

```text
尽量减少到达 Goal 的步数
```

---

## 4. 实验 1：Q Network

文件：

```text
01_q_network.py
```

### 4.1 State 表示

GridWorld 中 state 原本是：

```python
state = (2, 2)
```

为了输入神经网络，使用 one-hot 编码。

4×4 GridWorld 一共有：

```text
16 states
```

状态 `(row, col)` 转为：

```text
state_index = row * cols + col
```

例如：

```text
(2,2)
→
2 × 4 + 2
→
index 10
```

one-hot：

```text
[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
```

因此：

```text
state_dim = 16
```

---

## 5. Q Network 结构

网络：

```text
16
 ↓
64
 ↓
4
```

代码：

```python
self.network = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, action_dim)
)
```

含义：

```text
16
= 输入 state 的维度

64
= hidden units
= 人工选择的网络超参数

4
= action 数量
```

其中 64 并不是 DQN 的固定要求。

它只是一个较常见的小型隐藏层宽度。

对于当前 GridWorld：

```text
16 → 32 → 4
16 → 64 → 4
16 → 128 → 4
```

都可以使用。

---

## 6. Q-table 与 Q Network 的对应关系

Q-Learning：

```python
Q[row, col]
```

输出：

```text
[Q(UP), Q(DOWN), Q(LEFT), Q(RIGHT)]
```

DQN：

```python
q_network(state_tensor)
```

同样输出：

```text
[Qθ(UP), Qθ(DOWN), Qθ(LEFT), Qθ(RIGHT)]
```

因此：

```text
Q[row,col]
      ↓
q_network(state)
```

本质上只是把：

```text
直接存储
```

换成：

```text
神经网络预测
```

---

## 7. Q Network 初始输出

实验中：

```text
State: (2,2)
```

网络随机初始化后输出：

```text
[-0.0820, -0.0233, -0.0421, -0.0881]
```

对应：

```text
UP     = -0.0820
DOWN   = -0.0233
LEFT   = -0.0421
RIGHT  = -0.0881
```

这些值一开始没有学习意义。

因为：

```text
Network Parameters
```

只是随机初始化。

训练的目标就是让这些预测逐渐接近真正的动作价值。

---

## 8. 实验 2：Online Neural Q-Learning

文件：

```text
02_online_dqn.py
```

这一阶段只处理一条 transition。

实验：

```text
state      = (2,2)
action     = RIGHT
reward     = -1
next_state = (2,3)
done       = False
```

网络当前预测：

```text
Predicted Q:
0.0849
```

TD Target：

```text
-0.8874
```

Loss：

```text
0.9454
```

一次网络更新后：

```text
New predicted Q:
-0.0232
```

说明：

```text
0.0849
→
-0.0232
```

正在向：

```text
-0.8874
```

靠近。

---

## 9. DQN 的 TD Target

Q-Learning 中：

```text
Target
=
r + γ max Q(s',a')
```

DQN 中仍然保留同一个思想：

```text
Target
=
r + γ max Qθ(s',a')
```

区别只是：

```text
Q(s',a')
```

不再从表格读取，而由神经网络预测。

---

## 10. 从 TD Error 到 Loss

Tabular Q-Learning：

```text
TD Error
=
Target - Q(s,a)
```

然后：

```text
Q_new
=
Q_old + α × TD Error
```

DQN 无法直接修改某一个 Q 值。

因此改为：

```text
Predicted Q
    ↓
Target
    ↓
Loss
    ↓
Backward
    ↓
Gradient
    ↓
Update θ
```

本实验使用 MSE Loss：

```text
Loss
=
(Predicted Q - Target)^2
```

因此：

```text
TD Error
→
Loss
→
Gradient
→
Network Parameters
```

---

## 11. PyTorch 三步更新

核心代码：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

分别表示：

```text
optimizer.zero_grad()
→ 清空上一轮梯度

loss.backward()
→ 反向传播并计算梯度

optimizer.step()
→ 根据梯度修改网络参数
```

整体：

```text
Loss
 ↓
Backpropagation
 ↓
Gradient
 ↓
Optimizer
 ↓
θ Update
```

---

## 12. 神经网络更新与 Q-table 更新的区别

Q-table：

```text
更新 Q(s,RIGHT)
```

只修改一个表格元素。

神经网络：

```text
更新 Qθ(s,RIGHT)
```

实际上修改的是共享网络参数。

因此一次更新可能同时影响：

```text
UP
DOWN
LEFT
RIGHT
```

甚至可能影响其他 state 的 Q 预测。

这就是神经网络的：

```text
Parameter Sharing
```

和：

```text
Function Approximation
```

特性。

这既是神经网络可以处理大状态空间的原因，也是深度强化学习训练更容易不稳定的原因之一。

---

## 13. 实验 3：Replay Buffer

文件：

```text
03_replay_buffer.py
```

Replay Buffer 存储：

```text
(state,
 action,
 reward,
 next_state,
 done)
```

即：

```text
(s,a,r,s',done)
```

例如：

```text
((2,0), RIGHT, -1, (2,1), False)
```

---

## 14. 为什么需要 Replay Buffer？

强化学习数据按时间连续产生：

```text
transition 1
transition 2
transition 3
transition 4
...
```

相邻样本高度相关。

如果神经网络始终按照这种顺序训练：

```text
最新 transition
↓
立刻训练
↓
下一条 transition
↓
立刻训练
```

容易产生不稳定。

Replay Buffer 改为：

```text
Environment
    ↓
不断产生 transition
    ↓
存进 Buffer
    ↓
随机抽取旧经验和新经验
    ↓
训练
```

核心作用：

```text
Sequential Experience
→
Random Sampling
```

从而减弱样本之间的时间相关性。

---

## 15. Replay Buffer 的容量

实验中：

```python
buffer = deque(
    maxlen=capacity
)
```

如果：

```text
capacity = 5000
```

最多保存：

```text
5000 transitions
```

超过容量后：

```text
最旧经验自动删除
```

因此可以理解为：

```text
First In
First Out
```

---

## 16. Random Sampling

随机 Batch：

```python
indices = rng.choice(
    len(self.buffer),
    size=batch_size,
    replace=False
)
```

其中：

```text
replace=False
```

表示：

```text
同一个 buffer index
不会在同一个 batch 中重复抽取
```

但两个不同 index 可能保存内容完全相同的 transition。

这是正常现象。

---

## 17. 实验 4：Mini-Batch Update

文件：

```text
04_minibatch_update.py
```

之前：

```text
1 transition
→
1 update
```

现在：

```text
8 transitions
→
1 batch
→
1 update
```

实验：

```text
batch_size = 8
```

网络输出：

```text
Q values shape:
torch.Size([8,4])
```

表示：

```text
8 states
×
4 actions
```

---

## 18. gather()

网络一次输出：

```text
[batch_size, action_dim]
```

例如：

```text
sample 0:
[Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]

sample 1:
[Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]

...
```

但每条 transition 只真正执行了一个 action。

所以需要：

```python
predicted_q = q_values.gather(
    1,
    actions_tensor
).squeeze(1)
```

例如：

```text
sample 0
action = RIGHT

sample 1
action = DOWN
```

最终取：

```text
Q(s0,RIGHT)
Q(s1,DOWN)
```

所以：

```text
[8,4]
→
gather()
→
[8]
```

---

## 19. Mini-Batch TD Target

对于整个 batch：

```text
Target_i
=
reward_i
+
γ × best_next_q_i × (1 - done_i)
```

代码：

```python
targets = (
    rewards_tensor
    +
    gamma
    *
    best_next_q
    *
    (1.0 - dones_tensor)
)
```

如果：

```text
done = False
```

则：

```text
1 - done = 1
```

保留未来 Q。

如果：

```text
done = True
```

则：

```text
1 - done = 0
```

未来价值被清零。

因此 terminal transition：

```text
Target = reward
```

---

## 20. Mini-Batch Loss

对于 batch 中 N 条 transition：

```text
Loss
=
平均 [
    (Qθ(si,ai) - Target_i)^2
]
```

实验中：

```text
Loss:
0.6994
```

更新后，多条 Predicted Q 同时向各自 Target 靠近。

所以：

```text
Random Batch
    ↓
Batch Predicted Q
    ↓
Batch Target
    ↓
One Loss
    ↓
One Backward
```

这就是 Mini-Batch Gradient Descent。

---

## 21. 实验 5：Target Network

文件：

```text
05_target_network.py
```

如果只有一个 Q Network：

```text
Online Network
```

既负责：

```text
预测 Q(s,a)
```

又负责：

```text
计算 TD Target
```

那么：

```text
optimizer.step()
```

一执行，网络发生变化。

于是：

```text
Prediction 在变
Target 也在变
```

相当于：

```text
网络不断追逐一个自己不断移动的目标
```

这会降低训练稳定性。

---

## 22. Online Network 与 Target Network

DQN 使用两套网络：

```text
Online Network
Qθ
```

负责：

```text
当前 Q 预测
动作选择
Gradient Update
```

Target Network：

```text
Qθ-
```

负责：

```text
计算 TD Target
```

因此：

```text
Predicted Q
=
Qθ(s,a)
```

而：

```text
Target
=
r + γ max Qθ-(s',a')
```

---

## 23. Target Network 参数同步

初始化：

```python
target_network.load_state_dict(
    q_network.state_dict()
)
```

所以：

```text
Online = Target
```

实验结果：

```text
Before Online Update:
Networks equal: True
```

Online Network 更新一次以后：

```text
After Online Update:
Networks equal: False
```

因为：

```text
Online 更新
Target 不更新
```

再次执行：

```python
target_network.load_state_dict(
    q_network.state_dict()
)
```

得到：

```text
After Target Sync:
Networks equal: True
```

因此 Target Network 的工作方式：

```text
Online
每个 gradient step 都更新

Target
暂时冻结

每隔一段时间
从 Online 复制一次参数
```

---

## 24. DQN 的核心结构

到此为止，经典 DQN 的核心组件已经完整：

```text
Q-Learning
    +
Neural Network
    +
Experience Replay
    +
Target Network
```

可以写成：

```text
DQN
=
Q-Learning
+
Function Approximation
+
Replay Buffer
+
Target Network
```

---

## 25. 实验 6：完整 DQN

文件：

```text
06_dqn_train.py
```

完整训练过程：

```text
初始化 Online Network
        ↓
初始化 Target Network
        ↓
Target ← Online
        ↓
reset Environment
        ↓
ε-greedy 选择 action
        ↓
Environment Step
        ↓
得到 (s,a,r,s',done)
        ↓
存入 Replay Buffer
        ↓
随机抽 Mini-Batch
        ↓
Online Network
计算 Qθ(s,a)
        ↓
Target Network
计算 TD Target
        ↓
MSE Loss
        ↓
Backpropagation
        ↓
更新 Online Network
        ↓
定期同步 Target Network
        ↓
重复
```

---

## 26. 完整 DQN 训练参数

主要参数：

```python
n_episodes = 1200
max_steps = 200

gamma = 1.0

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_episodes = 800

buffer_capacity = 5000
batch_size = 32
min_buffer_size = 64

target_update_frequency = 100
```

Optimizer：

```python
optim.Adam(
    q_network.parameters(),
    lr=0.001
)
```

---

## 27. 参数含义

| 参数 | 含义 |
|---|---|
| `gamma` | 未来 Reward 的权重 |
| `epsilon` | Exploration 强度 |
| `lr` | 神经网络参数更新步长 |
| `batch_size` | 每次训练抽多少经验 |
| `buffer_capacity` | Replay Buffer 最大容量 |
| `min_buffer_size` | 至少积累多少经验后开始训练 |
| `target_update_frequency` | Target Network 同步频率 |
| `n_episodes` | 训练 Episode 数量 |

---

## 28. 为什么先积累经验再训练？

设置：

```text
min_buffer_size = 64
```

刚开始：

```text
Buffer size < 64
```

只进行：

```text
Exploration
+
Experience Collection
```

不立即训练。

达到 64 条经验以后：

```text
Random Mini-Batch Training
```

才开始。

原因是：

```text
Buffer 太小时
数据仍然高度相关
随机采样意义很弱
```

---

## 29. Target Network 更新频率

实验中：

```text
target_update_frequency = 100
```

不是：

```text
100 Episodes
```

而是：

```text
100 Gradient Steps
```

每执行一次：

```python
optimizer.step()
```

就：

```text
gradient_steps += 1
```

当：

```text
gradient_steps % 100 == 0
```

执行：

```text
Target ← Online
```

因此：

```text
Online Network
连续学习 100 次

Target Network
同步一次
```

---

## 30. 完整训练结果

实际实验结果：

```text
Episode  100 | epsilon = 0.877 | avg steps = 48.39
Episode  200 | epsilon = 0.754 | avg steps = 25.33
Episode  300 | epsilon = 0.630 | avg steps = 17.66
Episode  400 | epsilon = 0.506 | avg steps = 12.35
Episode  500 | epsilon = 0.382 | avg steps = 10.23
Episode  600 | epsilon = 0.259 | avg steps = 8.80
Episode  700 | epsilon = 0.135 | avg steps = 7.32
Episode  800 | epsilon = 0.011 | avg steps = 6.38
Episode  900 | epsilon = 0.010 | avg steps = 6.10
Episode 1000 | epsilon = 0.010 | avg steps = 6.08
Episode 1100 | epsilon = 0.010 | avg steps = 6.03
Episode 1200 | epsilon = 0.010 | avg steps = 6.08
```

整体趋势：

```text
48.39
→
25.33
→
17.66
→
12.35
→
8.80
→
7.32
→
6.38
→
6.08
```

理论最优：

```text
6 steps
```

因此网络成功从随机行为逐渐收敛到接近最短路径。

---

## 31. 为什么训练初期步数很高？

训练开始：

```text
epsilon ≈ 1.0
```

因此 Agent 大量随机探索。

例如 Episode 100：

```text
epsilon = 0.877
```

意味着仍然有约 87.7% 概率随机探索。

所以：

```text
avg steps = 48.39
```

完全正常。

随着：

```text
epsilon
↓
```

Agent 越来越多地使用：

```text
argmax Qθ(s,a)
```

平均步数就逐渐下降。

---

## 32. 训练后期为什么是 6.08？

训练结束时：

```text
epsilon = 0.01
```

仍然有 1% 概率随机探索。

因此即使已经学到最优策略，偶尔仍然会：

```text
选错动作
→
多走一步
```

所以：

```text
Average steps in last 100 episodes:
6.08
```

略高于理论最优：

```text
6
```

---

## 33. Evaluation

训练结束后停止探索：

```text
epsilon = 0
```

使用纯 greedy policy：

```text
action
=
argmax Qθ(s,a)
```

Evaluation 结果：

```text
Average steps: 6.0
Minimum steps: 6
Maximum steps: 6
Success rate: 1.0
```

即：

```text
100% Success Rate
```

并且：

```text
100 次 Evaluation
全部恰好 6 步
```

说明网络已经学到了最优策略。

---

## 34. Greedy Trajectory

实际 Evaluation 轨迹：

```text
[(0,0),
 (0,1),
 (0,2),
 (1,2),
 (1,3),
 (2,3),
 (3,3)]
```

对应：

```text
(0,0)
  →
(0,1)
  →
(0,2)
  ↓
(1,2)
  →
(1,3)
  ↓
(2,3)
  ↓
(3,3)
```

动作：

```text
RIGHT
RIGHT
DOWN
RIGHT
DOWN
DOWN
```

一共：

```text
6 steps
```

因此是最短路径。

---

## 35. Training 与 Evaluation

训练：

```text
ε-greedy
+
Network Update
+
Replay Buffer
```

评估：

```text
epsilon = 0
+
No Gradient Update
+
Pure Greedy
```

因此：

```text
Training Performance
≠
Learned Policy Performance
```

训练最后：

```text
6.08 steps
```

评估：

```text
6.0 steps
```

原因正是训练阶段仍然存在 Exploration。

---

## 36. Training Steps Curve

训练曲线保存：

```text
plots/training_steps.png
```

使用 Moving Average：

```text
window = 50
```

因此不是直接画每一个 Episode 的原始步数，而是：

```text
最近 50 个 Episode
平均需要多少步
```

主要观察：

```text
Steps to Goal
从很高
逐渐下降
最终接近 6
```

理论最优线：

```text
6 steps
```

---

## 37. Loss Curve

Loss 曲线保存：

```text
plots/training_loss.png
```

横坐标：

```text
Gradient Step
```

而不是：

```text
Episode
```

因为一次 Episode 内可能执行多次：

```python
optimizer.step()
```

所以：

```text
Episode Number
≠
Gradient Step
```

---

## 38. 为什么 DQN Loss 不一定持续平滑下降？

监督学习中常见：

```text
Loss
持续下降
```

但 DQN 的训练目标不断变化：

```text
Policy 在变
↓
采样数据在变

Replay Buffer 在变
↓
Batch 在变

Online Network 在变
↓
Q Prediction 在变

Target Network 定期同步
↓
TD Target 在变
```

因此 Loss 可能明显波动。

所以判断 DQN 是否成功，不能只看 Loss。

更重要的是：

```text
Episode Return
Steps to Goal
Success Rate
Evaluation Performance
```

本实验中：

```text
Success Rate = 100%
Average Evaluation Steps = 6
```

因此可以确认训练成功。

---

## 39. Q-Learning 与 DQN 对比

| Q-Learning | DQN |
|---|---|
| Q-table | Q Network |
| `Q[s,a]` | `Qθ(s,a)` |
| 直接更新一个表格元素 | 通过梯度更新网络参数 |
| 单条 transition 更新 | Replay Buffer Mini-Batch |
| 当前 Q-table 算 Target | Target Network 算 Target |
| 适合小离散状态空间 | 可以扩展到大规模状态空间 |

---

## 40. 两者不变的核心

虽然实现方式变化很大，但核心学习目标没有变。

Q-Learning：

```text
Q(s,a)
≈
r + γ max Q(s',a')
```

DQN：

```text
Qθ(s,a)
≈
r + γ max Qθ-(s',a')
```

因此 DQN 可以理解为：

```text
Q-Learning
+
Neural Network Function Approximation
```

再加：

```text
Replay Buffer
+
Target Network
```

提高训练稳定性。

---

## 41. 从 Bandit 到 DQN 的知识主线

### Bandit

```text
Q(a)
```

解决：

```text
哪个动作长期平均 Reward 更高？
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

并通过：

```text
TD Target
TD Error
```

直接更新 Q-table。

### DQN

进一步：

```text
Q(s,a)
↓
Qθ(s,a)
```

使用：

```text
Neural Network
Replay Buffer
Mini-Batch
Target Network
Backpropagation
```

学习最优 Q function。

---

## 42. 完整知识链

```text
Bandit
Q(a)
   ↓
GridWorld
Vπ(s)
   ↓
Q-Learning
Q(s,a)
   ↓
DQN
Qθ(s,a)
```

对应能力升级：

```text
简单动作选择
   ↓
序列决策
   ↓
Model-Free Value Learning
   ↓
Deep Value Function Approximation
```

---

## 43. DQN 阶段最重要的结论

### 43.1 DQN 没有抛弃 Q-Learning

核心仍然是：

```text
TD Learning
+
Bootstrapping
+
Greedy Target
```

DQN 只是把：

```text
Q-table
```

换成：

```text
Neural Network
```

---

### 43.2 Neural Network 是函数近似器

网络学习的是：

```text
state
→
Q values
```

即：

```text
Qθ(s,a)
```

参数 θ 决定整个 Q function。

---

### 43.3 Replay Buffer 解耦数据采集和训练

不再：

```text
刚产生一条经验
→
只训练这一条
```

而是：

```text
收集大量经验
→
随机采样
→
Mini-Batch Training
```

---

### 43.4 Target Network 提供更稳定的学习目标

Online Network：

```text
持续学习
```

Target Network：

```text
暂时固定
```

因此：

```text
TD Target
```

短时间内更稳定。

---

### 43.5 DQN 是 Value-Based 方法

DQN 并没有单独训练一个 Policy Network。

它仍然通过：

```text
Qθ(s,a)
```

选择：

```text
argmax_a Qθ(s,a)
```

得到策略。

因此：

```text
Q function
→
Policy
```

---

## 44. 下一阶段：Policy Gradient

到 DQN 为止，我们一直走的是：

```text
Value-Based RL
```

核心思想：

```text
先学习 Value / Q
↓
再根据 Value 选动作
```

下一阶段开始进入：

```text
Policy-Based RL
```

即：

```text
直接学习 Policy
```

从：

```text
Qθ(s,a)
→
argmax
→
action
```

升级为：

```text
πθ(a|s)
→
直接产生动作概率
```

这就是：

```text
Policy Gradient
```

后续路线：

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
