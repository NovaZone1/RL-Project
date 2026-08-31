# #现在在单次bandit的基础上进行多次epsilon不同的实验，比较不同的epsilon对于学习的影响
# #1.确定bandit环境，即实际价值，动作个数,以及真实价值最高动作的序号
# #2.构建单次bandit实验，传入3个参数，训练轮数，epsilon，随机数种子  这样可以进一步实现实验的真实性
# #3.运行多次bandit实验，记录平均估计价值，以及选择真实价值最高动作的概率

# import numpy as np
# import matplotlib.pyplot as plt

# #实验环境准备
# Q = np.array([1.2,3.4,4.7,1.5,1.0,-0.5])

# N_actions = len(Q)

# maxQ_action = np.argmax(Q)

# #单次实验设置
# def run_bandit(steps,seed,epsilon):

#     rng = np.random.default_rng(seed)

#     q = np.zeros(N_actions)

#     n_actions = np.zeros_like(Q)

#     rewards = np.zeros(steps)

#     optional_actions = np.zeros(steps)

#     for step in range(steps):

#         if rng.random() < epsilon:

#             action = rng.integers(N_actions)

#         else:

#             best_action = np.flatnonzero(q == q.max())

#             action = rng.choice(best_action)

#         reward = rng.normal(loc=Q[action],scale = 1.0)

#         n_actions[action] += 1

#         alpha = 1.0 / n_actions[action]

#         q[action] = q[action] + alpha * (reward - q[action])

#         rewards[step] = reward

#         if action == maxQ_action:

#             optional_actions[step] = 1

#     return rewards,optional_actions

# #多次实验环境准备
# epsilons = [0.0,0.01,0.1,0.5]

# n_runs = 1000

# n_steps = 1000

# reward_results = {}

# optional_action_results = {}

# #进行多次实验

# for epsilon in epsilons:

#     all_rewards = np.zeros((n_runs,n_steps))

#     all_optional_actions = np.zeros((n_runs,n_steps))

#     for step in range(n_steps):

#         rewards,optional_actions = run_bandit(n_steps,step,epsilon)

#         all_rewards[step] = rewards

#         all_optional_actions[step] = optional_actions

#     mean_rewards = np.mean(all_rewards,axis = 0)

#     optional_action_rate = np.mean(all_optional_actions,axis = 0)

#     reward_results[epsilon] = mean_rewards

#     optional_action_results[epsilon] = optional_action_rate

# #平均Reward曲线
# plt.figure()

# for epsilon in epsilons:

#     plt.plot(reward_results[epsilon],label=f"epsilon={epsilon}")

# plt.xlabel("Step")

# plt.ylabel("Average Reward")

# plt.title("Average Reward")

# plt.legend()

# plt.savefig("plots/average_reward.png",dpi=150,bbox_inches="tight")

# #最优动作选择率曲线
# plt.figure()

# for epsilon in epsilons:

#     plt.plot(optional_action_results[epsilon] * 100,label=f"epsilon={epsilon}")

# plt.xlabel("Step")

# plt.ylabel("Optimal Action (%)")

# plt.title("Optimal Action Rate")

# plt.legend()

# plt.savefig("plots/optimal_action_rate.png",dpi=150,bbox_inches="tight")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt


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

def run_bandit(steps, seed, epsilon):

    rng = np.random.default_rng(seed)

    # Agent 对各动作价值的估计
    # 一开始什么都不知道
    q = np.zeros(n_actions)

    # 每个动作被选择的次数
    N = np.zeros(n_actions, dtype=int)

    # 保存每一步 Reward
    rewards = np.zeros(steps)

    # 记录每一步是否选择了真实最优动作
    optimal_actions = np.zeros(steps)

    for step in range(steps):

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

    return rewards, optimal_actions


# =========================
# 3. 多次实验设置
# =========================

epsilons = [
    0.0,
    0.01,
    0.1,
    0.5
]

n_runs = 1000
n_steps = 1000

reward_results = {}

optimal_action_results = {}


# =========================
# 4. 比较 epsilon
# =========================

for epsilon in epsilons:

    all_rewards = np.zeros(
        (n_runs, n_steps)
    )

    all_optimal_actions = np.zeros(
        (n_runs, n_steps)
    )

    # 注意这里是 n_runs
    for run in range(n_runs):

        rewards, optimal_actions = run_bandit(
            steps=n_steps,
            seed=run,
            epsilon=epsilon
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

    reward_results[epsilon] = mean_rewards

    optimal_action_results[epsilon] = (
        optimal_action_rate
    )


# =========================
# 5. 平均 Reward 曲线
# =========================

plt.figure()

for epsilon in epsilons:

    plt.plot(
        reward_results[epsilon],
        label=f"epsilon={epsilon}"
    )

plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.title("Average Reward")
plt.legend()

plt.savefig(
    "plots/fixed_average_reward.png",
    dpi=150,
    bbox_inches="tight"
)


# =========================
# 6. 最优动作选择率
# =========================

plt.figure()

for epsilon in epsilons:

    plt.plot(
        optimal_action_results[epsilon] * 100,
        label=f"epsilon={epsilon}"
    )

plt.xlabel("Step")
plt.ylabel("Optimal Action (%)")
plt.title("Optimal Action Rate")
plt.legend()

plt.savefig(
    "plots/fixed_optimal_action_rate.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()






        








