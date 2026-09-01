#1.设置好实验参数，包括实验轮次，最大尝试步数
#2.初始化环境
#3.创建两个数组，分别记录每个状态下的总收益和访问这个状态的次数
#4.进行多次实验，记录下每次的reward和state，便于我们后续维护上面的两个主要数组

import numpy as np 
from environment import GridWorld 

env = GridWorld()

rng = np.random.default_rng(42)

n_episodes = 10000

max_steps = 10000

gamma = 1.0

returns_sum = np.zeros((env.rows,env.cols))

returns_count = np.zeros((env.rows,env.cols),dtype=int)

for episode in range(n_episodes):

    state = env.reset()

    states = []

    rewards = []

    for step in range(max_steps):

        states.append(state)

        action = rng.integers(4)

        next_state,reward,done = env.step(action)

        rewards.append(reward)

        state = next_state

        if done:

            break

    if not done:

        continue

    G = 0.0

    for t in reversed(range(len(rewards))):

        G = rewards[t] + gamma * G

        row,col = states[t]

        returns_sum[row,col] += G

        returns_count[row,col] += 1

V = np.zeros((env.rows,env.cols))

for row in range(env.rows):

    for col in range(env.cols):

        if returns_count[row,col] > 0:

            V[row,col] = returns_sum[row,col] / returns_count[row,col]

goal_row,goal_col = env.goal_state

V[goal_row,goal_col] = 0.0

print("State Value V(s):")

print(
    np.round(V, 2)
)


print("\nState Visit Count:")

print(
    returns_count
)




    

