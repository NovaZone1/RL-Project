# HalfCheetah PPO

基于 **Gymnasium + MuJoCo + PyTorch** 实现的多维连续动作 PPO（Proximal Policy Optimization）项目。

本项目是在完成 Pendulum 连续控制 PPO 后，将同一套 PPO 框架迁移到更复杂的 `HalfCheetah-v5` 环境，并围绕多维连续动作、PPO 稳定性、KL、ClipFrac、Best Checkpoint 与 Actor 学习率进行了完整实验。

## 1. 项目目标

本项目主要完成以下内容：

- 将单维连续动作 PPO 迁移到多维连续动作环境
- 使用 Gaussian Policy 建模 6 维连续动作
- 使用 `tanh` 对动作进行压缩并映射到环境动作范围
- 正确处理 transformed action 的 log probability
- 使用 GAE 计算 Advantage
- 使用 PPO clipped objective 更新 Actor
- 使用 Value Loss 更新 Critic
- 记录 KL、ClipFrac、策略标准差等诊断指标
- 加入 Best Checkpoint
- 分析 KL Early Stopping 的作用
- 研究 Actor Learning Rate 对策略稳定性的影响
- 对最终最佳策略进行 deterministic evaluation

## 2. 环境

环境：

```text
HalfCheetah-v5
```

环境空间：

```text
Observation Space: (17,)
Action Space:      (6,)
Action Range:      [-1, 1]
```

因此：

```text
state_dim  = 17
action_dim = 6
```

HalfCheetah 的 Actor 一次输出 6 个动作均值，对应 6 个连续控制维度。

## 3. 项目结构

```text
halfcheetah_ppo/
├── models.py
├── train.py
├── evaluate.py
└── checkpoints/
    ├── actor.pth
    ├── critic.pth
    ├── best_actor.pth
    └── best_critic.pth
```

其中：

- `models.py`：Actor 与 Critic 网络
- `train.py`：PPO 训练流程
- `evaluate.py`：确定性策略评估
- `actor.pth`：最后一次训练得到的 Actor
- `critic.pth`：最后一次训练得到的 Critic
- `best_actor.pth`：训练过程中 evaluation return 最好的 Actor
- `best_critic.pth`：对应的 Critic

## 4. Actor 网络

Actor 输出 Gaussian Policy 的均值和标准差：

```python
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, x):
        mean = self.network(x)
        std = torch.exp(self.log_std)

        return mean, std
```

对于 HalfCheetah：

```text
mean.shape = (6,)
std.shape  = (6,)
```

即每一个动作维度都有自己的 Gaussian：

\[
u_i \sim \mathcal{N}(\mu_i,\sigma_i)
\]

整体动作：

\[
\mathbf{u}
=
[u_1,u_2,\dots,u_6]
\]

这里每个动作维度独立建模，但 PPO 使用的是整个 6 维联合动作的 log probability。

## 5. Critic 网络

Critic 输入状态：

\[
s\in\mathbb{R}^{17}
\]

输出：

\[
V(s)\in\mathbb{R}
\]

用于估计当前状态的状态价值。

## 6. 连续动作采样

Actor 首先输出：

\[
\mu,\sigma
\]

然后采样 raw action：

\[
u\sim\mathcal{N}(\mu,\sigma)
\]

Gaussian 本身是无界的，因此不能直接发送给动作范围为 `[-1,1]` 的 HalfCheetah。

使用：

\[
y=\tanh(u)
\]

得到：

\[
y\in(-1,1)
\]

再进行动作范围映射：

\[
a
=
action\_bias
+
action\_scale\cdot y
\]

HalfCheetah 中：

```text
action_scale = [1, 1, 1, 1, 1, 1]
action_bias  = [0, 0, 0, 0, 0, 0]
```

因此实际就是：

\[
a=\tanh(u)
\]

## 7. Tanh 后的 Log Probability

由于动作经过：

\[
u
\rightarrow
\tanh(u)
\rightarrow
a
\]

发生了变量变换，所以不能直接使用原始 Gaussian 的 log probability。

需要加入 Jacobian 修正：

\[
\log\pi(a|s)
=
\log\pi(u|s)
-
\log
\left(
action\_scale
\cdot
(1-\tanh^2(u))
\right)
\]

实现时对所有动作维度求和：

```python
log_prob = (
    dist.log_prob(raw_action)
    - torch.log(
        action_scale
        * (1 - squashed_action.pow(2))
        + 1e-6
    )
).sum(-1)
```

因此每一个时间步最终只有一个 joint log probability。

## 8. Advantage 与 GAE

Critic 估计：

\[
V(s_t)
\]

单步 TD error：

\[
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
\]

GAE：

\[
A_t
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+\cdots
\]

Critic 的训练目标：

\[
Return_t
=
V(s_t)
+
A_t
\]

Advantage 在进入 PPO 更新前进行 normalization：

\[
A
\leftarrow
\frac{A-\mu_A}
{\sigma_A+\epsilon}
\]

## 9. terminated 与 truncated

Gymnasium 中：

`terminated` 表示真正的 MDP terminal，因此 bootstrap value 设为 0。

`truncated` 通常表示时间上限等外部截断，不代表真实终止状态，因此仍允许 `V(next_state)` 参与 bootstrap。

本项目使用：

```text
terminated
```

控制 TD bootstrap，

使用：

```text
terminated or truncated
```

结束 episode 并阻止 GAE 跨 reset 传播。

## 10. PPO Ratio

PPO 的核心 ratio：

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{old}}(a_t|s_t)
}
\]

实现：

```python
ratio = torch.exp(
    new_log_probs
    - old_log_probs
)
```

`ratio = 1` 表示新旧策略对该动作的概率密度相同；`ratio > 1` 表示新策略提高了该动作的概率密度；`ratio < 1` 表示新策略降低了该动作的概率密度。

## 11. PPO Clipping

PPO 使用 clipped surrogate objective：

\[
L^{CLIP}
=
\min
\left(
r_tA_t,
\operatorname{clip}
(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\]

本项目：

```python
clip_epsilon = 0.2
```

因此 clipping 区间：

\[
[0.8,1.2]
\]

PPO clipping 是 loss 层面的软限制，并不会强制真实 ratio 永远位于该区间。

## 12. ClipFrac

ClipFrac 表示当前 PPO 更新中，有多少比例的样本 ratio 超出了 clipping 区间。

```python
clip_fraction = (
    (torch.abs(ratio - 1.0) > clip_epsilon)
    .float()
    .mean()
)
```

例如：

```text
ClipFrac = 0.20
```

表示约 20% 的样本已经进入 clipping 区域。

## 13. KL

KL 用于衡量新旧策略分布之间的变化程度。

本项目使用 rollout sample 做 approximate KL：

\[
D_{KL}
(
\pi_{old}
\|
\pi_{new}
)
\approx
\mathbb{E}
[
\log\pi_{old}
-
\log\pi_{new}
]
\]

实现：

```python
approx_kl = (
    old_log_probs
    - new_log_probs
).mean()
```

直觉上：

- KL 接近 0：新旧策略接近
- KL 增大：策略更新后移动得更远

ClipFrac 关注“多少样本越过 clipping 区域”，KL 关注“整个策略移动了多远”。

## 14. Baseline 实验

初始参数：

```python
rollout_steps = 2048
num_updates = 200

gamma = 0.99
gae_lambda = 0.95

ppo_epochs = 10
batch_size = 64
clip_epsilon = 0.2

actor_lr = 3e-4
critic_lr = 3e-4
```

训练过程中 Return 明显提高：

```text
Update 10   Return ≈ -182
Update 50   Return ≈ 275
Update 100  Return ≈ 818
Update 140  Return ≈ 1033
Update 170  Return ≈ 1240
```

说明 PPO 已经能够学习 HalfCheetah 控制策略。

但训练后期出现：

```text
KL        ≈ 0.10 ~ 0.15
ClipFrac  ≈ 0.50+
```

并伴随 Evaluation Return 明显波动。

## 15. Best Checkpoint

训练过程中加入 deterministic evaluation：

```text
每隔若干 update
↓
运行固定 seed evaluation
↓
如果 EvalReturn > best_eval_return
↓
保存 best_actor.pth
```

这样避免最后一个 checkpoint 覆盖训练过程中更好的模型。

即：

```text
last checkpoint != best checkpoint
```

## 16. KL Early Stopping 实验

实验加入过：

```python
target_kl = 0.03
```

以及：

```python
target_kl = 0.05
```

逻辑：

```text
完成一个 PPO epoch
↓
计算当前 epoch 的 mean KL
↓
如果 mean KL > target_kl
↓
停止继续使用当前 rollout
↓
重新采集下一批 rollout
```

实验表明 KL Early Stopping 能抑制策略过度更新，但阈值过小时会频繁出现：

```text
EpochsUsed = 1 ~ 3
```

降低 rollout 数据利用率。

## 17. Actor Learning Rate 实验

进一步将：

```python
actor_lr = 3e-4
```

降低到：

```python
actor_lr = 1e-4
```

同时取消 KL Early Stopping，并保持：

```python
ppo_epochs = 10
```

结果明显改善：

```text
Update 50   EvalReturn:  426.34
Update 60   EvalReturn:  843.37
Update 70   EvalReturn: 1237.35
Update 80   EvalReturn: 1442.91
Update 90   EvalReturn: 1540.91
Update 100  EvalReturn: 1677.11
Update 110  EvalReturn: 1885.64
Update 120  EvalReturn: 2086.65
Update 130  EvalReturn: 2365.18
Update 140  EvalReturn: 2501.78
Update 160  EvalReturn: 2567.98
Update 170  EvalReturn: 2671.57
Update 180  EvalReturn: 2822.88
Update 200  EvalReturn: 2836.18
```

同时：

```text
KL         ≈ 0.01 ~ 0.02
ClipFrac   ≈ 0.15 ~ 0.24
EpochsUsed = 10
```

这次实验表明，在当前配置中，降低 Actor Learning Rate 比严格依赖 KL Early Stopping 更适合控制策略更新幅度。

直观理解：

```text
降低 actor_lr
=
每次 optimizer.step() 走得更小
```

而：

```text
KL Early Stopping
=
发现策略已经走远以后停止继续走
```

## 18. 当前最终参数

当前最终配置：

```python
rollout_steps = 2048
num_updates = 200

gamma = 0.99
gae_lambda = 0.95

ppo_epochs = 10
batch_size = 64
clip_epsilon = 0.2

actor_lr = 1e-4
critic_lr = 3e-4
```

KL Early Stopping 默认关闭。

KL 与 ClipFrac 保留作为训练诊断指标。

Best Checkpoint 机制保留。

## 19. 最终 Evaluation

最终使用：

```text
best_actor.pth
```

进行 20 个 deterministic evaluation episodes。

评估时使用 Actor 的均值动作：

\[
a
=
action\_scale
\cdot
\tanh(\mu)
+
action\_bias
\]

最终结果：

```text
Evaluation Average: 2870.40
Evaluation Std:        92.00
Evaluation Min:      2731.97
Evaluation Max:      3036.09
```

平均回报：

\[
2870.40
\]

标准差：

\[
92.00
\]

相对于平均值约为：

\[
\frac{92}{2870.40}\approx 3.2\%
\]

20 个 evaluation episodes 的回报集中在：

```text
2731.97 ~ 3036.09
```

说明当前 best policy 在这些固定 evaluation seeds 下具有较高平均回报，并且 episode 之间波动较小。

## 20. 实验总结

本项目从 Pendulum 的一维连续动作 PPO 出发，成功迁移到了：

```text
17 维 observation
+
6 维 continuous action
```

的 HalfCheetah 环境。

主要认识：

- PPO Clipping 是 soft constraint，不是硬性限制 ratio。
- ClipFrac 用于观察多少样本进入 clipping 区域。
- KL 用于观察新旧策略整体移动程度。
- 最后一个 checkpoint 不一定是最好的 checkpoint，因此需要 Best Checkpoint。
- KL Early Stopping 可以限制策略过度更新，但阈值过严会降低数据利用率。
- Actor Learning Rate 会直接影响每次参数更新的步长。
- 当前实验中，将 `actor_lr` 从 `3e-4` 降低到 `1e-4` 后，KL 和 ClipFrac 明显下降，同时完整保留了 10 个 PPO epochs，并获得了更高、更稳定的 evaluation return。

## 21. 运行训练

```bash
cd ~/reinforcement-learning-practice/halfcheetah_ppo
python train.py
```

训练完成后的模型保存在：

```text
checkpoints/
```

## 22. Evaluation

```bash
python evaluate.py
```

默认评估：

```text
checkpoints/best_actor.pth
```

并使用 deterministic policy。

## 23. 学习路线

目前已经完成：

```text
Bandit
↓
GridWorld
↓
Q-Learning
↓
DQN
↓
REINFORCE
↓
Actor-Critic
↓
Discrete PPO
↓
Pendulum Continuous PPO
↓
HalfCheetah Multi-Dimensional Continuous PPO
```

下一阶段：

```text
PPO Expert
↓
Demonstration Dataset
↓
Behavior Cloning
↓
DAgger
↓
Imitation Learning
```
