# import numpy as np 
# from environment import GridWorld

# env = GridWorld()
# rng = np.random.default_rng(42)

# n_episodes = 1000
# max_steps = 500
# alpha = 0.1
# gamma = 1.0
# epsilon = 0.1

# Q = np.zeros((env.rows,env.cols,4))

# steps_per_episodes = []

# for episode in range(n_episodes):

#     state = env.reset()

#     for step in range(max_steps):

#         row,col = state

#         if rng.random() < epsilon:

#             action = rng.integers(4)

#         else:

#             q_values = Q[row,col]

#             best_actions = np.flatnonzero(q_values == q_values.max())

#             action = rng.choice(best_actions)

#         next_state,reward,done = env.step(action)

#         old_q = Q[row,col,action]

#         if done:

#             target = reward

#         else:

#             next_row,next_col = next_state

#             best_next_q = np.max(Q[next_row,next_col])

#             target = reward + gamma * best_next_q

#         td_error = target - old_q

#         Q[row,col,action] = old_q + alpha * td_error

#         state = next_state

#         if done:

#             steps_per_episodes.append(step + 1)

#             break

#     else:

#             steps_per_episodes.append(max_steps)

# print("Training finished.")

# print(
#     "Average steps in first 100 episodes:",
#     np.mean(steps_per_episodes[:100])
# )

# print(
#     "Average steps in last 100 episodes:",
#     np.mean(steps_per_episodes[-100:])
# )


# print("\nQ values:")

# print(
#     np.round(Q, 2)
# )

# action_names = {
#     GridWorld.UP: "↑",
#     GridWorld.DOWN: "↓",
#     GridWorld.LEFT: "←",
#     GridWorld.RIGHT: "→"
# }

# print("\nLearned Policy:")

# for row in range(env.rows):

#     line = []

#     for col in range(env.cols):

#         state = (row, col)

#         if state == env.goal_state:

#             line.append("G")

#             continue

#         q_values = Q[row, col]

#         best_value = np.max(q_values)

#         best_actions = np.flatnonzero(
#             np.isclose(q_values, best_value)
#         )

#         symbols = []

#         for action in best_actions:

#             symbols.append(
#                 action_names[action]
#             )

#         line.append(
#             "".join(symbols)
#         )

#     print(
#         "  ".join(line)
#     )

import numpy as np 
from environment import GridWorld

env = GridWorld()
rng = np.random.default_rng(42)

max_episodes = 1000
max_steps = 500
gamma =1.0
alpha = 0.1
n_actions = 4
epsilon =0.1
Q = np.zeros((env.rows,env.cols,n_actions))  # 需要维护这样一个Q table
steps_episodes = []

#train

for episode in range(max_episodes):

    state = env.reset()

    for step in range(max_steps):

        row,col = state

        if rng.random() < epsilon:

            action = rng.integers(n_actions)

        else:

            q_values = Q[row,col]

            best_actions = np.flatnonzero(q_values == q_values.max())

            action = rng.choice(best_actions)

        next_state,reward,done = env.step(action)

        if done:

            target = reward

        else:

            next_row,next_col = next_state

            target = reward + gamma * Q[next_row,next_col].max()

        state = next_state 

        q_old = Q[row,col,action]

        td_error = target - Q[row,col,action]

        Q[row,col,action] = q_old + alpha * td_error

        if done:

            steps_episodes.append(step + 1)
            break

    else:

        steps_episodes.append(max_steps)

print("Training Finished !")

print("前100轮平均步数:",np.mean(steps_episodes[:100]))

print("最后100轮平均步数:",np.mean(steps_episodes[-100:]))

#evaluation 

evaluation_episodes = 200
evaluation_steps = 500
steps_evaluation_episodes = []

for episode in range(evaluation_episodes):

    state = env.reset()

    for step in range(evaluation_steps):

        row,col = state

        q_values = Q[row,col]

        best_actions = np.flatnonzero(q_values == q_values.max())

        action = rng.choice(best_actions)

        next_state,reward,done = env.step(action)

        state = next_state

        if done:

            steps_evaluation_episodes.append(step + 1)

            break

    else:

        steps_evaluation_episodes.append(evaluation_steps)

steps_evaluation_episodes = np.array( steps_evaluation_episodes )

print("Evaluation Finished !")

print("Average Steps:",np.mean(steps_evaluation_episodes))

print("Success Rate:",np.mean(steps_evaluation_episodes < evaluation_steps))











