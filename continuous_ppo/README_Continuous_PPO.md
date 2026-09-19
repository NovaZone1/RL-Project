# Continuous PPO on Pendulum-v1

本项目使用 PyTorch 从零实现 **连续动作空间 PPO（Proximal Policy Optimization）**，并在 Gymnasium 的 `Pendulum-v1` 环境中完成训练与评估。

项目重点不是调用现成 PPO 算法，而是手动实现并理解完整训练链路：

- Gaussian Actor
- `tanh` 动作压缩与动作范围映射
- 变换后策略的 log-probability 修正
- Critic / State Value
- TD Error
- GAE（Generalized Advantage Estimation）
- Advantage Normalization
- PPO Probability Ratio
- Clipped Surrogate Objective
- Mini-batch + Multiple Epoch Update
- `terminated` / `truncated` 的正确处理
- 模型保存与确定性评估

## 1. Project Structure

```text
continuous_ppo/
├── models.py
├── train.py
├── evaluate.py
└── checkpoints/
    ├── actor.pth
    └── critic.pth
```

`models.py` 定义 Actor 和 Critic；`train.py` 完成 rollout、GAE、PPO 更新与模型保存；`evaluate.py` 加载训练后的 Actor 并使用确定性策略评估。

## 2. Environment

使用：

```python
gym.make("Pendulum-v1")
```

状态：

```text
[cos(theta), sin(theta), theta_dot]
```

状态维度：

```text
state_dim = 3
```

动作空间：

```text
Box(-2.0, 2.0, (1,))
```

动作维度：

```text
action_dim = 1
```

## 3. Continuous Gaussian Policy

连续 Actor 输出：

\[
\mu(s)
\]

并维护可学习的：

\[
\log \sigma
\]

通过：

```python
std = torch.exp(log_std)
```

得到：

\[
\sigma > 0
\]

策略为：

\[
u_t \sim \mathcal N(\mu(s_t), \sigma)
\]

其中 `raw_action = u_t` 的范围是：

\[
(-\infty, +\infty)
\]

## 4. Tanh Action Squashing

先进行：

\[
y_t = \tanh(u_t)
\]

得到：

\[
y_t \in (-1, 1)
\]

再映射到环境动作范围：

\[
a_t = action\_scale \cdot y_t + action\_bias
\]

其中：

\[
action\_scale = \frac{high-low}{2}
\]

\[
action\_bias = \frac{high+low}{2}
\]

Pendulum 中：

\[
low=-2, \quad high=2
\]

所以：

\[
a_t = 2\tanh(u_t)
\]

## 5. Log Probability Correction

由于动作经过了 `tanh + scale` 变换，不能直接使用 Gaussian 对 `raw_action` 的 log probability。

根据变量变换：

\[
\log \pi(a|s)
=
\log p_U(u)
-
\log
\left[
action\_scale
\left(
1-\tanh^2(u)
\right)
\right]
\]

实现：

```python
base_log_prob = distribution.log_prob(raw_action)

log_prob = (
    base_log_prob
    - torch.log(
        action_scale
        * (1 - squashed_action.pow(2))
        + 1e-6
    )
)

log_prob = log_prob.sum(dim=-1)
```

## 6. Critic

Critic 学习：

\[
V(s_t)
\]

表示当前状态下按照当前策略继续执行时的期望累计回报。

网络结构：

```text
3
↓
64
↓
ReLU
↓
64
↓
ReLU
↓
1
```

## 7. TD Error

一步 TD Error：

\[
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
\]

代码：

```python
delta = (
    rewards[t]
    + gamma
    * (1.0 - float(terminateds[t]))
    * next_values[t]
    - values[t]
)
```

如果 `terminated=True`，则：

\[
V(s_{t+1})=0
\]

## 8. GAE

GAE 将多个未来 TD Error 衰减累积：

\[
A_t^{GAE}
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\cdots
\]

递推形式：

\[
A_t^{GAE}
=
\delta_t
+
\gamma\lambda A_{t+1}^{GAE}
\]

代码：

```python
gae = (
    delta
    + gamma
    * gae_lambda
    * (1.0 - float(episode_ends[t]))
    * gae
)
```

GAE 用于估计：

\[
A(s,a)=Q(s,a)-V(s)
\]

即当前动作相对于当前状态平均水平的优势。

## 9. terminated vs truncated

Gymnasium 中区分：

```python
terminated
truncated
```

`terminated=True` 表示真正的 MDP 终止，因此：

\[
V(s_{t+1})=0
\]

`truncated=True` 表示时间限制等外部原因结束 episode，因此仍允许：

\[
V(s_{t+1})
\]

进行 bootstrap。

本项目使用：

```python
episode_end = terminated or truncated
```

TD bootstrap 使用：

\[
1-	ext{terminated}
\]

GAE 跨步传播使用：

\[
1-	ext{episode\_end}
\]

从而避免 GAE 跨越 reset 后的新 episode。

## 10. Critic Target

构造：

\[
\hat R_t^{\lambda}
=
V_{old}(s_t)
+
\hat A_t^{GAE}
\]

代码：

```python
returns_tensor = (
    advantages_tensor
    + values_tensor
)
```

Critic loss：

\[
L_{critic}
=
MSE
\left(
V_{\phi}(s_t),
\hat R_t^{\lambda}
\right)
\]

## 11. Advantage Normalization

在构造 Critic target 后，对 Advantage 标准化：

```python
advantages_tensor = (
    advantages_tensor
    - advantages_tensor.mean()
) / (
    advantages_tensor.std(unbiased=False)
    + 1e-8
)
```

目标是：

\[
mean(A) \approx 0
\]

\[
std(A) \approx 1
\]

标准化后的 Advantage 主要用于 Actor。

## 12. PPO Ratio

Rollout 时保存：

\[
\log\pi_{old}(a_t|s_t)
\]

PPO update 时，使用当前 Actor 对同一个旧 `raw_action` 重新计算：

\[
\log\pi_{new}(a_t|s_t)
\]

然后：

\[
r_t(\theta)
=
\frac{
\pi_{new}(a_t|s_t)
}{
\pi_{old}(a_t|s_t)
}
\]

代码：

```python
ratio = torch.exp(
    new_log_probs
    - batch_old_log_probs
)
```

第一次 PPO update 前应满足：

\[
ratio \approx 1
\]

## 13. PPO Clipped Objective

\[
surr_1 = r_tA_t
\]

\[
surr_2 =
clip(r_t, 1-\epsilon, 1+\epsilon)A_t
\]

Actor objective：

\[
L^{CLIP}
=
E
\left[
\min(surr_1,surr_2)
\right]
\]

由于 PyTorch optimizer 做最小化：

```python
actor_loss = -torch.min(
    surr1,
    surr2
).mean()
```

## 14. Mini-batch PPO

当前设置：

```python
rollout_steps = 2048
batch_size = 64
ppo_epochs = 10
```

因此一个 epoch：

\[
2048 / 64 = 32
\]

个 mini-batch。

10 个 epoch：

\[
32 \times 10 = 320
\]

次 mini-batch optimization。

## 15. Main Hyperparameters

```python
rollout_steps = 2048
num_updates = 100

gamma = 0.99
gae_lambda = 0.95

ppo_epochs = 10
batch_size = 64
clip_epsilon = 0.2

actor_lr = 3e-4
critic_lr = 3e-4
```

| Parameter | Effect |
| --- | --- |
| `rollout_steps` | 每次 PPO 更新前采多少环境经验 |
| `num_updates` | 总训练轮数 |
| `gamma` | 对长期 reward 的重视程度 |
| `gae_lambda` | GAE 的 bias-variance 权衡 |
| `ppo_epochs` | 同一批 rollout 重复训练多少遍 |
| `batch_size` | 每次梯度更新使用多少样本 |
| `clip_epsilon` | 限制新旧策略一次变化幅度 |
| `actor_lr` | Actor 参数更新速度 |
| `critic_lr` | Critic 参数更新速度 |
| `log_std` | 连续动作探索强度 |

## 16. Training Observation

早期使用：

```python
rollout_steps = 128
```

时，训练效果长期不稳定。

改为：

```python
rollout_steps = 2048
```

后，训练明显改善。

一次成功训练中的部分结果：

```text
Update 10, Return: -923.03
Update 20, Return: -539.49
Update 30, Return: -263.28
Update 40, Return: -224.45
Update 50, Return: -188.41
Update 60, Return: -147.41
Update 70, Return: -144.86
```

说明连续控制中，更充分的 rollout 对 GAE 和 PPO 更新稳定性有明显帮助。

## 17. Diagnostics

训练时记录：

### Policy Standard Deviation

```python
torch.exp(actor_network.log_std)
```

一次训练中：

```text
0.927
→ 0.822
→ 0.684
→ 0.531
→ 0.424
```

表示策略随着学习逐渐减少探索。

### Approximate KL

成功训练中约：

```text
0.005 ~ 0.008
```

### Clip Fraction

成功训练中约：

```text
0.04 ~ 0.07
```

说明策略更新幅度总体稳定。

## 18. Evaluation

训练阶段：

\[
u\sim\mathcal N(\mu,\sigma)
\]

评估阶段不再采样，而使用：

\[
u=\mu(s)
\]

最终动作：

\[
a
=
action\_scale
\cdot
\tanh(\mu(s))
+
action\_bias
\]

一次保存模型的评估结果：

```text
Evaluation Average: -196.99
Evaluation Min:     -431.91
Evaluation Max:       -0.45
```

不同随机初始化和训练 run 会产生不同结果。

## 19. Run

训练：

```bash
python train.py
```

训练结束后保存：

```text
checkpoints/actor.pth
checkpoints/critic.pth
```

评估：

```bash
python evaluate.py
```

## 20. Core Learning Path

从离散 PPO：

```text
state
↓
logits
↓
Categorical
↓
discrete action
```

迁移到连续 PPO：

```text
state
↓
mean / std
↓
Normal
↓
raw action
↓
tanh
↓
bounded continuous action
```

完整训练链：

```text
Environment
↓
Rollout
↓
Gaussian Actor
↓
tanh + Action Scaling
↓
Reward / Next State
↓
Critic Value
↓
TD Error
↓
GAE
↓
Advantage Normalization
↓
PPO Ratio
↓
Clipped Actor Loss
↓
Critic Value Loss
↓
Mini-batch Optimization
↓
Updated Policy
```

## 21. Current Status

```text
[✓] Continuous Gaussian Actor
[✓] Critic Network
[✓] Bounded Continuous Action
[✓] Tanh Log-Probability Correction
[✓] Rollout Buffer
[✓] GAE
[✓] Advantage Normalization
[✓] PPO Clipping
[✓] Mini-batch Update
[✓] Multiple PPO Epochs
[✓] terminated / truncated Handling
[✓] Training Diagnostics
[✓] Model Checkpoint
[✓] Deterministic Evaluation
[✓] Pendulum-v1 Training
```

该项目作为后续机器人连续控制、仿真强化学习与模仿学习项目的 PPO 基础实现。
