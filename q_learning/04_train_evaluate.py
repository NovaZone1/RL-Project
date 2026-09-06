import numpy as np

from environment import GridWorld


# =========================
# 基本参数
# =========================

n_episodes = 1000

max_steps = 500

alpha = 0.1

gamma = 1.0


# =========================
# Train
# =========================

def train_q_learning(seed=42):

    env = GridWorld()

    rng = np.random.default_rng(seed)

    Q = np.zeros(
        (env.rows, env.cols, 4)
    )


    for episode in range(n_episodes):

        # -------------------------
        # epsilon decay
        # -------------------------

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


        # -------------------------
        # 一个 Episode
        # -------------------------

        state = env.reset()


        for step in range(max_steps):

            row, col = state


            # epsilon-greedy
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


            # Environment
            next_state, reward, done = env.step(
                action
            )


            # 当前 Q
            old_q = Q[
                row,
                col,
                action
            ]


            # TD Target
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


            # TD Error
            td_error = (
                target
                -
                old_q
            )


            # Q Update
            Q[
                row,
                col,
                action
            ] = (
                old_q
                +
                alpha * td_error
            )


            state = next_state


            if done:
                break


    return Q

def evaluate_policy(
    Q,
    n_eval_episodes=100,
    seed=123
):

    env = GridWorld()

    rng = np.random.default_rng(seed)

    steps_per_episode = []


    for episode in range(n_eval_episodes):

        state = env.reset()


        for step in range(max_steps):

            row, col = state


            # =========================
            # Greedy Policy
            # epsilon = 0
            # =========================

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


    return np.array(
        steps_per_episode
    )

# =========================
# 1. Train
# =========================

Q = train_q_learning(
    seed=42
)


print("Training finished.")


# =========================
# 2. Evaluate
# =========================

eval_steps = evaluate_policy(
    Q,
    n_eval_episodes=100,
    seed=123
)


print("\nEvaluation Results:")

print(
    "Average steps:",
    np.mean(eval_steps)
)

print(
    "Minimum steps:",
    np.min(eval_steps)
)

print(
    "Maximum steps:",
    np.max(eval_steps)
)

print(
    "Success rate:",
    np.mean(
        eval_steps < max_steps
    )
)

# =========================
# 3. Show one greedy trajectory
# =========================

env = GridWorld()

state = env.reset()

trajectory = [state]


for step in range(max_steps):

    row, col = state

    q_values = Q[row, col]

    action = np.argmax(
        q_values
    )

    next_state, reward, done = env.step(
        action
    )

    trajectory.append(
        next_state
    )

    state = next_state


    if done:
        break


print("\nGreedy Trajectory:")

print(trajectory)

print(
    "Steps:",
    len(trajectory) - 1
)