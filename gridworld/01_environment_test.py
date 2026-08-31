from environment import GridWorld

env = GridWorld()

state = env.reset()

print("初始状态",state)

env.render()

actions = [
    GridWorld.RIGHT,
    GridWorld.RIGHT,
    GridWorld.RIGHT,
    GridWorld.DOWN,
    GridWorld.DOWN,
    GridWorld.DOWN,
]

for action in actions:

    next_state,reward,done = env.step(action)

    print("action:",action)
    print("next_state:",next_state)
    print("reward",reward)
    print("done",done)

    env.render()

    if done == True:
        break