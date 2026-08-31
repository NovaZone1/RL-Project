#1.确定各个动作的真实价值
#2.初始化各个动作的估计价值，各个动作被选择的次数
#3.初始化随机数种子
#4.进行带有随机性的贪婪算法
#5.得到最终的各个动作估计的平均价值

import numpy as np

Q = np.array([1.2,3.4,4.7,1.5,1.0,-0.5])

N_actions = len(Q)

q = np.zeros(N_actions)

n_actions = np.zeros_like(Q)

rng = np.random.default_rng(42)

epsilon = 0.5

learn_step = 1000

rewards = []

actions = []

for step in range(learn_step):

    if rng.random() < epsilon:

        action = rng.integers(N_actions)

    else:

        best_action = np.flatnonzero(Q==Q.max())

        action = rng.choice(best_action)

    reward = rng.normal(loc=Q[action],scale = 1.0)

    n_actions[action] += 1

    alpha = 1.0 / n_actions[action]

    q[action] = q[action] + alpha * (reward - q[action])

    rewards.append(reward) 

    actions.append(action)

print("真实的动作价值:",Q)

print("\n估计的动作价值:",q)

print("\n每个动作被选择的次数:",n_actions)

print("\n平均奖励:",np.mean(rewards))

print("\n最终认为的最优动作:",(np.argmax(q)+1))








