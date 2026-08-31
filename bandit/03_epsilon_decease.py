import numpy as np
import matplotlib.pyplot as plt

def epsilon_decay(step,epsilon_start,epsilon_end,decay_steps):

    ration = min(step/decay_steps,1.0)

    epsilon = epsilon_start + ration * (epsilon_end - epsilon_start)

    return epsilon


# =========================
# 1. Bandit 环境
# =========================

# 每个动作真实的平均奖励
# 这是环境真值，Agent 不知道
true_means = np.array([
    1.2,
    3.4,
    4.7,
    1.5,
    1.0,
    -0.5
])

n_actions = len(true_means)

# 真正的最优动作
optimal_action = np.argmax(true_means)


# =========================
# 2. 单次 Bandit 实验
# =========================

def run_bandit(steps, seed,step,epsilon_start,epsilon_end,decay_steps):

    rng = np.random.default_rng(seed)

    # Agent 对各动作价值的估计
    # 一开始什么都不知道
    q = np.zeros(n_actions)

    # 每个动作被选择的次数
    N = np.zeros(n_actions, dtype=int)

    epsilons_history = np.zeros(steps)

    # 保存每一步 Reward
    rewards = np.zeros(steps)

    # 记录每一步是否选择了真实最优动作
    optimal_actions = np.zeros(steps)

    for step in range(steps):

        epsilon = epsilon_decay(step,epsilon_start,epsilon_end,decay_steps)

        epsilons_history[step] = epsilon
        # =====================
        # epsilon-greedy
        # =====================

        if rng.random() < epsilon:

            # Exploration
            action = rng.integers(n_actions)

        else:

            # Exploitation
            # 注意：必须根据 Agent 自己的 q 来选
            best_actions = np.flatnonzero(
                q == q.max()
            )

            action = rng.choice(best_actions)

        # =====================
        # Environment
        # =====================

        # 只有环境生成 Reward 时，
        # 才允许使用 true_means
        reward = rng.normal(
            loc=true_means[action],
            scale=1.0
        )

        # =====================
        # 更新动作价值
        # =====================

        N[action] += 1

        alpha = 1.0 / N[action]

        q[action] = q[action] + alpha * (
            reward - q[action]
        )

        # =====================
        # 保存实验结果
        # =====================

        rewards[step] = reward

        if action == optimal_action:
            optimal_actions[step] = 1

    return rewards, optimal_actions, epsilons_history


# =========================
# 3. 多次实验设置
# =========================

n_runs = 1000
n_steps = 1000

all_rewards = np.zeros(
    (n_runs, n_steps)
)

all_optimal_actions = np.zeros(
    (n_runs, n_steps)
)

# reward_results = {}

# optimal_action_results = {}

# =========================
# 4. 比较 epsilon
# =========================

    # 注意这里是 n_runs
for run in range(n_runs):

    rewards, optimal_actions,_ = run_bandit(
        steps=n_steps,
        seed=run,
        step = n_steps,
        epsilon_start=0.5,
        epsilon_end=0.01,
        decay_steps=800
    )

    all_rewards[run] = rewards

    all_optimal_actions[run] = (
        optimal_actions
    )

    # 每一个时间步，在 1000 次实验上取平均
mean_rewards = np.mean(
    all_rewards,
    axis=0
)

optimal_action_rate = np.mean(
    all_optimal_actions,
    axis=0
)


# =========================
# 5. 平均 Reward 曲线
# =========================

plt.figure()

plt.plot(mean_rewards,label=f"epsilon 0.5->0.01")

plt.xlabel("Step")

plt.ylabel("Average Reward")

plt.title("Average Reward")

plt.legend()

plt.savefig("plots/epsilon_decay_average_reward.png",dpi=150,bbox_inches="tight"
)


# =========================
# 6. 最优动作选择率
# =========================

plt.figure()

plt.plot(optimal_action_rate * 100,label=f"epsilon 0.5->0.01")

plt.xlabel("Step")

plt.ylabel("Optimal Action (%)")

plt.title("Optimal Action Rate")

plt.legend()

plt.savefig("plots/epsilon_decay_optimal_action_rate.png",dpi=150,bbox_inches="tight"
)

plt.show()