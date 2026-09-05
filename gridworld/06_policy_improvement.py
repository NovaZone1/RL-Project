import numpy as np 
from environment import GridWorld 

env = GridWorld()

gamma = 1.0

V = np.array([
    [-58.43, -56.43, -53.29, -50.71],
    [-56.43, -53.57, -48.71, -44.14],
    [-53.29, -48.71, -39.86, -29.00],
    [-50.71, -44.14, -29.00,   0.00]
])

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

policy = np.zeros((env.rows,env.cols),dtype = object)

for row in range(env.rows):

    for col in range(env.cols):

        state = (row,col)

        if state == env.goal_state:

            policy[row,col] = "G"

            continue

        action_value = []

        for action in actions:

            env.state = state

            next_state,reward,done = env.step(action)

            if done:

                next_value = 0.0

            else:

                next_row,next_col = next_state

                next_value = V[next_row,next_col]

            value = (reward + gamma * next_value)

            action_value.append(value)

        best_value = max(action_value)

        best_actions = []

        for i,value in enumerate(action_value):

            if np.isclose(value,best_value):

                best_actions.append(action_names[actions[i]])

        policy[row,col] = "".join(best_actions)


print("Improved Policy:")

for row in range(env.rows):

    print(
        "  ".join(policy[row])
    )

