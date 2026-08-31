#添加一个随机数生成器，作为最简单的policy

import numpy as np 
from environment import GridWorld

env = GridWorld()

rng = np.random.default_rng(1)

max_steps = 100

state = env.reset()

print("当前状态:",state)

for step in range(max_steps):

    action = rng.integers(4)

    next_state,reward,done = env.step(action)

    print(f"step={step + 1}, "
        f"state={state}, "
        f"action={action}, "
        f"reward={reward}, "
        f"next_state={next_state}, "
        f"done={done}"
        )

    state = next_state

    if done:
            print(
            f"\n到达终点,共走了 {step + 1} 步"
        )

            break

else:
    print(
        f"\n达到最大步数 {max_steps},"
        "仍未到达终点"
        )
                 