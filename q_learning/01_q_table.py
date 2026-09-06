import numpy as np 
from environment import GridWorld 

env = GridWorld()

alpha = 0.1
gamma = 1.0

Q = np.zeros((env.rows,env.cols,4))

print("Q-table shape:")
print(Q.shape)

state = (2,2)
action = GridWorld.RIGHT

env.state = state
next_state,reward,done = env.step(action)

print("\nTransition:")

print("state:", state)
print("action:", action)
print("reward:", reward)
print("next_state:", next_state)
print("done:", done)

row,col = state

old_q = Q[row,col,action]

if done:

    target = reward

else:

    next_row,next_col = next_state

    best_next_q = np.max(Q[next_row,next_col])

    target = reward + gamma * best_next_q 

td_error = target - old_q

new_q = old_q + alpha * td_error

Q[row,col,action] = new_q


print("\nQ-Learning Update:")

print("old Q:", old_q)
print("target:", target)
print("TD error:", td_error)
print("new Q:", new_q)

print(
    "\nQ values at state (2,2):"
)

print(
    Q[2, 2]
)