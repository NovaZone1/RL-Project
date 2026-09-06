# GridWorld 强化学习实验

本项目是强化学习实践的第二个阶段。

前一阶段 Bandit 主要学习：

- Action
- Reward
- Q(a)
- Exploration vs Exploitation
- ε-greedy

GridWorld 在此基础上第一次正式引入：

- State
- State Transition
- Episode
- Return
- Discount Factor
- State Value
- Bellman Equation
- Monte Carlo Value Estimation
- Policy Evaluation
- Policy Improvement
- Policy Iteration

本阶段的核心目标是从：

```text
Action
  ↓
Reward
```

升级为：

```text
State
  ↓
Action
  ↓
Reward + Next State
```

也就是正式进入序列决策问题。

---

## 1. 项目结构

```text
gridworld/
├── environment.py
├── 01_environment_test.py
├── 02_random_agent.py
├── 03_episode_return.py
├── 04_state_value.py
├── 05_bellman_value.py
├── 06_policy_improvement.py
├── 07_policy_iteration.py
├── README.md
└── plots/
```

各文件作用：

| 文件 | 主要内容 |
|---|---|
| `environment.py` | 定义 4×4 GridWorld 环境 |
| `01_environment_test.py` | 测试状态转移、Reward、done |
| `02_random_agent.py` | Random Policy 与 Episode |
| `03_episode_return.py` | Return 与 Discount Factor |
| `04_state_value.py` | Monte Carlo 估计 V(s) |
| `05_bellman_value.py` | Bellman Policy Evaluation |
| `06_policy_improvement.py` | 根据 V(s) 改进策略 |
| `07_policy_iteration.py` | Policy Evaluation + Improvement 反复迭代 |

---

## 2. GridWorld 环境

环境使用 4×4 网格：

```text
S . . .
. . . .
. . . .
. . . G
```

其中：

```text
S = Start
G = Goal
```

坐标：

```text
(0,0)  (0,1)  (0,2)  (0,3)
(1,0)  (1,1)  (1,2)  (1,3)
(2,0)  (2,1)  (2,2)  (2,3)
(3,0)  (3,1)  (3,2)  (3,3)
```

起点：

```python
start_state = (0, 0)
```

终点：

```python
goal_state = (3, 3)
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

如果动作会走出边界，则 Agent 保持在原状态，但仍然获得：

```text
reward = -1
```

---

## 3. 环境接口

`environment.py` 中的核心接口：

```python
state = env.reset()

next_state, reward, done = env.step(action)
```

其中：

```text
state
= 当前状态

action
= 当前动作

next_state
= 执行动作后的状态

reward
= 当前动作得到的奖励

done
= Episode 是否结束
```

核心交互关系：

```text
(state, action)
        ↓
(reward, next_state)
```

---

## 4. 实验 1：Environment Test

文件：

```text
01_environment_test.py
```

第一阶段不使用学习算法，只人工规定动作：

```text
RIGHT
RIGHT
RIGHT
DOWN
DOWN
DOWN
```

轨迹：

```text
(0,0)
  →
(0,1)
  →
(0,2)
  →
(0,3)
  ↓
(1,3)
  ↓
(2,3)
  ↓
(3,3)
```

实验输出验证：

- 状态转移正确
- Reward 正确
- Goal 时 `done = True`
- 边界动作不会走出地图

这一阶段建立最基本的 MDP 交互：

```text
S_t + A_t
    ↓
R_t + S_(t+1)
```

---

## 5. 实验 2：Random Agent

文件：

```text
02_random_agent.py
```

使用 Random Policy：

```python
action = rng.integers(4)
```

四个动作概率相同：

```text
UP     25%
DOWN   25%
LEFT   25%
RIGHT  25%
```

这就是：

```text
π(a|s) = 1/4
```

Random Policy 不利用 state 信息。

实际一次实验中：

```text
到达终点，共走了 55 步
```

而理论最短路径只需要：

```text
6 步
```

说明随机策略虽然能够到达 Goal，但效率非常低。

---

## 6. Episode

GridWorld 第一次正式出现 Episode。

一个 Episode：

```text
reset()
   ↓
Start
   ↓
不断执行 action
   ↓
不断发生 state transition
   ↓
到达 Goal
   ↓
done = True
```

也就是：

```text
Episode Start
    ↓
S0
    ↓
A0
    ↓
S1
    ↓
A1
    ↓
...
    ↓
Terminal State
```

Bandit 中每一步基本独立。

GridWorld 中：

```text
上一轮 next_state
=
下一轮 state
```

因此不同时间步形成连续的状态链。

---

## 7. 实验 3：Episode Return

文件：

```text
03_episode_return.py
```

Reward 只表示当前一步的反馈。

Return 表示从当前时刻开始的累计未来 Reward。

公式：

```text
G_t
=
R_t
+ γ R_(t+1)
+ γ^2 R_(t+2)
+ ...
```

递推形式：

```text
G_t
=
R_t
+
γ G_(t+1)
```

实验中同一条 55 步轨迹分别使用：

```text
γ = 1.0
γ = 0.9
γ = 0.5
```

得到：

| γ | Episode Return |
|---:|---:|
| 1.0 | -54.0 |
| 0.9 | -9.9662 |
| 0.5 | -2.0 |

---

## 8. Discount Factor

γ 控制：

```text
未来 Reward 在当前看来有多重要
```

当：

```text
γ 接近 1
```

说明：

```text
更重视长期结果
```

当：

```text
γ 接近 0
```

说明：

```text
更重视眼前 Reward
```

在当前 GridWorld 中：

```text
每一步 = -1
Goal = 0
```

并且 Episode 最终会终止。

因此：

```text
γ = 1
```

非常适合表达：

```text
走得越久
→ Return 越低
→ 最短路径越好
```

---

## 9. Reward、Return、Value 的区别

可以直接区分为：

```text
Reward
= 这一步怎么样？

Return
= 从现在开始，整个未来怎么样？

Value
= 从这个状态出发，长期平均来看怎么样？
```

对应：

```text
R_t
G_t
Vπ(s)
```

---

## 10. 实验 4：Monte Carlo State Value

文件：

```text
04_state_value.py
```

状态价值定义：

```text
Vπ(s)
=
从状态 s 出发，
按照策略 π 行动时，
Return 的期望
```

Monte Carlo 估计方法：

```text
访问 state s
    ↓
得到一次 Return G
    ↓
再次访问 s
    ↓
再得到一个 G
    ↓
大量样本
    ↓
取平均
    ↓
Vπ(s)
```

代码中维护：

```python
returns_sum[row, col]
```

表示：

```text
状态 s 所有 Return 样本的总和
```

以及：

```python
returns_count[row, col]
```

表示：

```text
状态 s 被统计了多少次
```

最终：

```text
V(s)
=
returns_sum / returns_count
```

---

## 11. 从后往前计算 Return

Episode 结束后，从最后一步向前：

```python
G = 0.0

for t in reversed(range(len(rewards))):

    G = rewards[t] + gamma * G
```

对应：

```text
G_t
=
R_t
+
γ G_(t+1)
```

例如：

```text
Rewards:

-1  -1  -1   0
```

当：

```text
γ = 1
```

从后往前：

```text
0
↓
-1
↓
-2
↓
-3
```

于是 Episode 中不同状态分别获得不同的 Return 样本。

---

## 12. Monte Carlo 实验结果

最初设置：

```text
max_steps = 100
```

并且如果 Episode 没结束：

```python
if not done:
    continue
```

这样会把：

```text
100 步以内没有到 Goal 的长轨迹
```

全部丢掉。

这些轨迹通常 Return 很低。

因此 Monte Carlo 的 Value 会被明显高估。

原始结果大致为：

```text
[[-36.34 -32.93 -29.04 -26.12]
 [-32.93 -29.85 -25.34 -21.65]
 [-29.11 -25.44 -19.01 -12.31]
 [-26.59 -21.98 -12.50   0.00]]
```

后来将：

```text
max_steps = 10000
```

重新实验，得到：

```text
[[-59.70 -57.83 -54.08 -51.68]
 [-57.39 -54.71 -49.23 -44.70]
 [-54.11 -49.79 -40.37 -29.12]
 [-51.87 -45.10 -29.50   0.00]]
```

这一结果与 Bellman Evaluation 非常接近。

这个实验说明：

```text
采样方式
会直接影响 Value 估计
```

不合理地丢弃长 Episode 会产生明显偏差。

---

## 13. Every-Visit Monte Carlo

当前代码中：

```text
同一个 Episode
如果多次访问同一个 state
```

每一次都会统计对应 Return。

因此使用的是：

```text
Every-Visit Monte Carlo
```

即：

```text
每一次访问都作为一个样本
```

---

## 14. 实验 5：Bellman Policy Evaluation

文件：

```text
05_bellman_value.py
```

Monte Carlo 是：

```text
真的跑大量 Episode
↓
收集 Return
↓
求平均
```

Bellman 方法则利用已知环境模型：

```text
知道当前 state
知道所有 action
知道 reward
知道 next_state
↓
直接计算 Value
```

Random Policy 下：

```text
π(a|s) = 1/4
```

Bellman 更新：

```text
V(s)
=
Σ_a π(a|s)
[
    r + γ V(s')
]
```

在当前确定性 GridWorld 中，就是：

```text
四个动作各占 25%
↓
分别计算：

reward + γ × next_state_value

↓
加权求和
```

---

## 15. Bellman Iteration

一开始：

```python
V = np.zeros((4, 4))
```

然后不断：

```text
V0
↓
Bellman Update
↓
V1
↓
Bellman Update
↓
V2
↓
...
```

直到：

```text
V_(k+1)
≈
V_k
```

使用：

```python
difference = abs(
    new_V[row, col] - V[row, col]
)

delta = max(
    delta,
    difference
)
```

当：

```python
delta < theta
```

停止。

实验设置：

```text
theta = 1e-6
```

---

## 16. Bellman Evaluation 结果

实验最终：

```text
Converged after 700 iterations
```

Value：

```text
[[-58.43 -56.43 -53.29 -50.71]
 [-56.43 -53.57 -48.71 -44.14]
 [-53.29 -48.71 -39.86 -29.00]
 [-50.71 -44.14 -29.00   0.00]]
```

与修正后的 Monte Carlo 结果非常接近。

因此验证：

```text
Monte Carlo
≈
Bellman Policy Evaluation
```

前提是 Monte Carlo 采样不存在明显偏差。

---

## 17. Monte Carlo 与 Bellman 的区别

| Monte Carlo | Bellman / Dynamic Programming |
|---|---|
| 依赖实际采样 | 依赖环境模型 |
| 需要 Episode | 不需要真实跑完整 Episode |
| 从 Return 样本估计 Value | 直接用 Bellman 递推 |
| 不需要完整转移模型 | 需要知道动作后的结果 |

两者最终都在估计：

```text
Vπ(s)
```

---

## 18. 实验 6：Policy Improvement

文件：

```text
06_policy_improvement.py
```

前面已经得到：

```text
Vπ(s)
```

现在的问题是：

> 既然知道不同状态有多好，能否用这些 Value 改进策略？

对于某个状态 s 和动作 a：

```text
Qπ(s,a)
=
r + γ Vπ(s')
```

当前环境是确定性的，因此一个动作只对应一个 next_state。

对四个动作分别计算：

```text
UP
DOWN
LEFT
RIGHT
```

然后选择：

```text
argmax_a Qπ(s,a)
```

得到改进后的 greedy policy。

---

## 19. Policy Improvement 结果

得到：

```text
Improved Policy:

↓→  →   ↓   ↓
↓   ↓→  ↓   ↓
→   →   ↓→  ↓
→   →   →   G
```

其中：

```text
↓→
```

表示 DOWN 和 RIGHT 并列最优。

这个结果说明策略已经从：

```text
Random Policy
```

明显转向：

```text
朝 Goal 方向移动
```

---

## 20. V(s)、Q(s,a)、Policy 的关系

三个概念可以这样理解：

```text
Vπ(s)
= 当前状态整体有多好？

Qπ(s,a)
= 当前状态下先做动作 a 有多好？

π(s)
= 当前状态最终选择什么动作？
```

关系：

```text
Vπ(s)
    ↓
计算 Qπ(s,a)
    ↓
选择最大的 action
    ↓
得到 improved policy
```

也就是：

```text
π'(s)
=
argmax_a Qπ(s,a)
```

---

## 21. 实验 7：Policy Iteration

文件：

```text
07_policy_iteration.py
```

一次 Policy Improvement 并不能直接假设已经最优。

因此反复执行：

```text
Policy Evaluation
        ↓
Policy Improvement
        ↓
Policy Evaluation
        ↓
Policy Improvement
        ↓
...
```

直到：

```text
new policy
=
old policy
```

这就是：

```text
Policy Iteration
```

完整过程：

```text
π0
↓
Vπ0
↓
π1
↓
Vπ1
↓
π2
↓
...
↓
π*
```

---

## 22. Policy Iteration 实验结果

程序输出：

```text
Policy Iteration: 1
Policy Iteration: 2
Policy Iteration: 3
```

第 3 轮后策略稳定。

最终 State Value：

```text
[[-5. -4. -3. -2.]
 [-4. -3. -2. -1.]
 [-3. -2. -1.  0.]
 [-2. -1.  0.  0.]]
```

最终策略：

```text
↓  ↓  ↓  ↓
↓  ↓  ↓  ↓
↓  ↓  ↓  ↓
→  →  →  G
```

这是一个最优策略。

---

## 23. 为什么起点 Value 是 -5？

从：

```text
(0,0)
```

到：

```text
(3,3)
```

最短路径需要：

```text
6 actions
```

但 Reward 设计是：

```text
普通移动：-1
进入 Goal：0
```

所以最短路径的 Reward：

```text
-1
-1
-1
-1
-1
 0
```

总 Return：

```text
-5
```

因此：

```text
V*(0,0) = -5
```

---

## 24. 最优策略不一定唯一

从很多状态出发：

```text
DOWN
```

和：

```text
RIGHT
```

都能形成最短路径。

因此最优动作可能并列。

`np.argmax()` 遇到并列最大值时只返回第一个最大值，所以最终打印出的策略只是：

```text
某一个确定性的最优策略
```

并不意味着只有这一条最优路径。

---

## 25. Random Policy 与 Optimal Policy

Random Policy 的起点价值约为：

```text
V_random(0,0)
≈
-58.43
```

Policy Iteration 后：

```text
V*(0,0)
=
-5
```

说明：

```text
Random Policy
→ 平均会绕很多路

Optimal Policy
→ 走最短路径
```

策略改进带来了非常明显的长期 Return 提升。

---

## 26. GridWorld 阶段的完整知识链

整个项目可以概括为：

```text
Environment
    ↓
State / Action / Reward / Next State
    ↓
Random Policy
    ↓
Episode
    ↓
Return
    ↓
Discount Factor
    ↓
State Value
    ↓
Monte Carlo Evaluation
    ↓
Bellman Evaluation
    ↓
Policy Improvement
    ↓
Policy Iteration
    ↓
Optimal Policy
```

核心关系：

```text
S_t
 ↓
A_t
 ↓
R_t + S_(t+1)
 ↓
G_t
 ↓
Vπ(s)
 ↓
Qπ(s,a)
 ↓
π*
```

---

## 27. 与 Bandit 的联系

Bandit：

```text
Action
  ↓
Reward
```

学习：

```text
Q(a)
```

GridWorld：

```text
State
  ↓
Action
  ↓
Reward + Next State
```

进一步学习：

```text
Return
Vπ(s)
Qπ(s,a)
Policy
```

因此：

```text
Bandit
Q(a)
  ↓
GridWorld
Vπ(s), Qπ(s,a)
```

---

## 28. 与 Q-Learning 的联系

当前 Policy Iteration 能够直接计算最优策略，是因为：

```text
环境模型已知
```

也就是说可以主动查询：

```text
如果在 state s
执行 action a
会去哪里？
reward 是多少？
```

但真实任务中往往不知道完整：

```text
P(s'|s,a)
```

因此下一阶段进入：

```text
Q-Learning
```

不再提前知道完整环境模型，而是：

```text
真的执行 action
↓
观察 reward
↓
观察 next_state
↓
更新 Q(s,a)
```

核心更新：

```text
Q(s,a)
←
Q(s,a)
+
α [
    r
    +
    γ max Q(s',a')
    -
    Q(s,a)
]
```

也就是从：

```text
Model-Based Dynamic Programming
```

过渡到：

```text
Model-Free Reinforcement Learning
```

---

## 29. 本阶段最重要的结论

### State 让动作价值依赖当前处境

同一个 action 在不同 state 下效果不同。

因此强化学习从：

```text
Q(a)
```

逐渐发展到：

```text
Q(s,a)
```

### Reward 不等于 Return

```text
Reward
= 一步反馈

Return
= 从当前时刻开始的累计未来反馈
```

### Value 是 Expected Return

```text
Vπ(s)
=
按照策略 π，
从状态 s 出发的长期 Return 期望
```

### Bellman 来自 Return 的递推

Return：

```text
G_t
=
R_t
+
γ G_(t+1)
```

提升到 Value 层面：

```text
Vπ(s)
=
Expected[
    R_t
    +
    γ Vπ(S_(t+1))
]
```

### Value 依赖 Policy

同一个 state：

```text
Random Policy
```

和：

```text
Optimal Policy
```

可以具有完全不同的 Value。

### Policy Improvement 利用 Value 改进动作选择

```text
Vπ(s)
↓
Qπ(s,a)
↓
argmax
↓
better policy
```

### Policy Iteration 最终得到 Optimal Policy

```text
Evaluation
↔
Improvement
```

不断交替，直到策略稳定。

---

## 30. 下一阶段

下一阶段：

```text
Q-Learning
```

重点从：

```text
知道环境模型后直接规划
```

转向：

```text
不知道完整环境模型
通过交互自己学
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
