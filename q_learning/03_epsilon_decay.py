import numpy as np
import matplotlib.pyplot as plt

from environment import GridWorld


# =========================
# 1. 基本参数
# =========================

n_episodes = 1000
max_steps = 500

alpha = 0.1
gamma = 1.0


# =========================
# 2. Q-Learning 训练函数
# =========================

def train_q_learning(
    use_decay,
    seed=42
):

    env = GridWorld()

    rng = np.random.default_rng(seed)

    Q = np.zeros(
        (env.rows, env.cols, 4)
    )

    steps_per_episode = []

    epsilon_history = []


    for episode in range(n_episodes):

        # =========================
        # epsilon
        # =========================

        if use_decay:

            epsilon_start = 1.0
            epsilon_end = 0.01
            decay_episodes = 800

            ratio = min(
                episode / decay_episodes,
                1.0
            )

            epsilon = (
                epsilon_start
                +
                (
                    epsilon_end
                    -
                    epsilon_start
                )
                * ratio
            )

        else:

            epsilon = 0.1


        epsilon_history.append(
            epsilon
        )


        # =========================
        # 一个 Episode
        # =========================

        state = env.reset()


        for step in range(max_steps):

            row, col = state


            # =========================
            # epsilon-greedy
            # =========================

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


            # =========================
            # Environment
            # =========================

            next_state, reward, done = env.step(
                action
            )


            # =========================
            # 当前 Q
            # =========================

            old_q = Q[
                row,
                col,
                action
            ]


            # =========================
            # TD Target
            # =========================

            if done:

                target = reward

            else:

                next_row, next_col = next_state

                best_next_q = np.max(
                    Q[next_row, next_col]
                )

                target = (
                    reward
                    +
                    gamma * best_next_q
                )


            # =========================
            # TD Error
            # =========================

            td_error = (
                target
                -
                old_q
            )


            # =========================
            # Q Update
            # =========================

            Q[
                row,
                col,
                action
            ] = (
                old_q
                +
                alpha * td_error
            )


            # =========================
            # 状态推进
            # =========================

            state = next_state


            if done:

                steps_per_episode.append(
                    step + 1
                )

                break


        else:

            steps_per_episode.append(
                max_steps
            )


    return (
        Q,
        np.array(steps_per_episode),
        np.array(epsilon_history)
    )

# =========================
# 3. Fixed epsilon
# =========================

fixed_Q, fixed_steps, fixed_epsilon = (
    train_q_learning(
        use_decay=False,
        seed=42
    )
)


# =========================
# 4. Decaying epsilon
# =========================

decay_Q, decay_steps, decay_epsilon = (
    train_q_learning(
        use_decay=True,
        seed=42
    )
)

print("Fixed epsilon = 0.1")

print(
    "First 100 episodes:",
    np.mean(fixed_steps[:100])
)

print(
    "Last 100 episodes:",
    np.mean(fixed_steps[-100:])
)


print("\nDecaying epsilon")

print(
    "First 100 episodes:",
    np.mean(decay_steps[:100])
)

print(
    "Last 100 episodes:",
    np.mean(decay_steps[-100:])
)

# =========================
# 5. Moving Average
# =========================

def moving_average(data, window=50):

    return np.convolve(
        data,
        np.ones(window) / window,
        mode="valid"
    )


fixed_smooth = moving_average(
    fixed_steps,
    window=50
)

decay_smooth = moving_average(
    decay_steps,
    window=50
)


plt.figure(figsize=(8, 5))

plt.plot(
    fixed_smooth,
    label="Fixed epsilon = 0.1"
)

plt.plot(
    decay_smooth,
    label="Decaying epsilon"
)

plt.xlabel("Episode")
plt.ylabel("Steps to Goal")

plt.title(
    "Q-Learning: Fixed vs Decaying Epsilon"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/epsilon_compare.png",
    dpi=150
)

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(decay_epsilon)

plt.xlabel("Episode")
plt.ylabel("Epsilon")

plt.title(
    "Epsilon Decay"
)

plt.tight_layout()

plt.savefig(
    "plots/epsilon_decay.png",
    dpi=150
)

plt.close()