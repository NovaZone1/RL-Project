#这里要计算长期回报G
import numpy as np 
from environment import GridWorld

env = GridWorld()

rng = np.random.default_rng(42)

max_steps = 100

gamma = 0.5

rewards = []

G = 0.0

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

    rewards.append(reward)

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

for t,reward in enumerate(rewards):

     G += (gamma ** t) * reward

print("Reward 序列:")
print(rewards)

print("Episode Return:")
print(G)


