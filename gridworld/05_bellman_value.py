#1.初始化环境参数
#2.进行bellman迭代

import numpy as np 
from environment import GridWorld

env = GridWorld()

gamma = 1.0

theta = 1e-6

action_probability = 0.25

V = np.zeros((env.rows,env.cols))

iteration = 0

while True:

    new_V = np.zeros((env.rows,env.cols))

    delta = 0.0

    for row in range(env.rows):

        for col in range(env.cols):

            state = (row,col)

            if state == env.goal_state:

                new_V[row,col] = 0.0

                continue

            state_value = 0.0

            for action in [GridWorld.UP,
                           GridWorld.DOWN,
                           GridWorld.LEFT,
                           GridWorld.RIGHT]:

                env.state = state

                next_state,reward,done = env.step(action)

                if done:

                    next_value = 0.0

                else:

                    next_row,next_col = next_state

                    next_value = V[next_row,next_col]

                state_value += action_probability * (reward + gamma * next_value)

            new_V[row,col] = state_value

            difference = abs(new_V[row,col] - V[row,col])

            delta = max(delta,difference)

    V = new_V 

    iteration += 1

    if delta < theta:

        break

print(
    "Converged after",
    iteration,
    "iterations"
)

print("\nBellman State Value V(s):")

print(
    np.round(V, 2)
)
        

            