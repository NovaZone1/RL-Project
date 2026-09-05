import numpy as np

from environment import GridWorld


env = GridWorld()

gamma = 1.0
theta = 1e-6

actions = [
    GridWorld.UP,
    GridWorld.DOWN,
    GridWorld.LEFT,
    GridWorld.RIGHT
]


action_names = {
    GridWorld.UP: "↑",
    GridWorld.DOWN: "↓",
    GridWorld.LEFT: "←",
    GridWorld.RIGHT: "→"
}


# =========================
# 1. 初始化 Random Policy
# =========================

policy = np.ones(
    (env.rows, env.cols, 4)
) / 4.0


# Goal 不需要动作
goal_row, goal_col = env.goal_state

policy[goal_row, goal_col] = 0.0


policy_iteration = 0


while True:

    # =========================
    # 2. Policy Evaluation
    # =========================

    V = np.zeros(
        (env.rows, env.cols)
    )


    while True:

        new_V = np.zeros_like(V)

        delta = 0.0


        for row in range(env.rows):

            for col in range(env.cols):

                state = (row, col)


                if state == env.goal_state:

                    new_V[row, col] = 0.0

                    continue


                state_value = 0.0


                for action in actions:

                    env.state = state

                    next_state, reward, done = env.step(
                        action
                    )


                    if done:

                        next_value = 0.0

                    else:

                        next_row, next_col = next_state

                        next_value = V[
                            next_row,
                            next_col
                        ]


                    action_prob = policy[
                        row,
                        col,
                        action
                    ]


                    state_value += (
                        action_prob
                        *
                        (
                            reward
                            +
                            gamma * next_value
                        )
                    )


                new_V[row, col] = state_value


                difference = abs(
                    new_V[row, col]
                    -
                    V[row, col]
                )


                delta = max(
                    delta,
                    difference
                )


        V = new_V


        if delta < theta:

            break


    # =========================
    # 3. Policy Improvement
    # =========================

    policy_stable = True


    for row in range(env.rows):

        for col in range(env.cols):

            state = (row, col)


            if state == env.goal_state:

                continue


            old_action = np.argmax(
                policy[row, col]
            )


            action_values = []


            for action in actions:

                env.state = state

                next_state, reward, done = env.step(
                    action
                )


                if done:

                    next_value = 0.0

                else:

                    next_row, next_col = next_state

                    next_value = V[
                        next_row,
                        next_col
                    ]


                value = (
                    reward
                    +
                    gamma * next_value
                )


                action_values.append(value)


            best_action = np.argmax(
                action_values
            )


            # 改成确定性 greedy policy
            policy[row, col] = 0.0

            policy[
                row,
                col,
                best_action
            ] = 1.0


            if best_action != old_action:

                policy_stable = False


    policy_iteration += 1


    print(
        f"Policy Iteration: "
        f"{policy_iteration}"
    )


    if policy_stable:

        break


# =========================
# 4. 输出最终 Value
# =========================

print("\nOptimal State Value:")

print(
    np.round(V, 2)
)


# =========================
# 5. 输出最终 Policy
# =========================

print("\nOptimal Policy:")


for row in range(env.rows):

    line = []

    for col in range(env.cols):

        state = (row, col)


        if state == env.goal_state:

            line.append("G")

        else:

            action = np.argmax(
                policy[row, col]
            )

            line.append(
                action_names[action]
            )


    print(
        "  ".join(line)
    )