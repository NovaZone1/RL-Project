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

max_episodes = 1000 # 最大训练轮次
max_steps = 500 #每一轮次最大步数，尽量取大一点
gamma =1.0 #折扣率
alpha = 0.1 #学习率
n_actions = 4 #动作数
epsilon =0.1 #随机探索率
Q = np.zeros((env.rows,env.cols,n_actions))  # 需要维护这样一个Q table
steps_episodes = [] #统计每轮次的步数

#train

for episode in range(max_episodes):

    state = env.reset() #先初始化环境

    for step in range(max_steps):

        row,col = state #先存下此时的状态，便于后面取最大值

        if rng.random() < epsilon: #随机探索

            action = rng.integers(n_actions) 

        else:

            q_values = Q[row,col]

            best_actions = np.flatnonzero(q_values == q_values.max()) #取得此状态下价值最大的动作的序号

            action = rng.choice(best_actions) #随机取同价值的动作

        next_state,reward,done = env.step(action) 

        if done:

            target = reward #因为到达终点后不存在下一个Q

        else:

            next_row,next_col = next_state #同上

            target = reward + gamma * Q[next_row,next_col].max()

        state = next_state #更新环境

        q_old = Q[row,col,action]

        td_error = target - Q[row,col,action] #计算td_error

        Q[row,col,action] = q_old + alpha * td_error #更新Q值

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

    for step in range(evaluation_steps): #此时不要随机探索，因为前面那个Q table理论上已经存下最优路径

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











