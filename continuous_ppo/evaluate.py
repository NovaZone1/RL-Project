import gymnasium as gym
import torch
import numpy as np

from models import ActorNetwork


env = gym.make("Pendulum-v1")

state_dim = 3
action_dim = 1


actor_network = ActorNetwork(
    state_dim,
    action_dim
)


# 加载已经训练好的参数
actor_network.load_state_dict(
    torch.load(
        "checkpoints/actor.pth",
        weights_only=True
    )
)

actor_network.eval()

action_high = torch.tensor(
    env.action_space.high,
    dtype=torch.float32
)

action_low = torch.tensor(
    env.action_space.low,
    dtype=torch.float32
)

action_scale = (
    action_high - action_low
) / 2.0

action_bias = (
    action_high + action_low
) / 2.0

eval_returns = []

for episode in range(10):

    state, info = env.reset(
        seed=1000 + episode
    )

    episode_return = 0.0

    while True:

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            mean, std = actor_network(
                state_tensor
            )

            action = (
                action_scale
                * torch.tanh(mean)
                + action_bias
            )

        next_state, reward, terminated, truncated, info = env.step(
            action.cpu().numpy()
        )

        episode_return += reward

        state = next_state

        if terminated or truncated:
            break

    eval_returns.append(
        episode_return
    )

print(
    "Evaluation Average:",
    np.mean(eval_returns)
)

print(
    "Evaluation Min:",
    np.min(eval_returns)
)

print(
    "Evaluation Max:",
    np.max(eval_returns)
)

env.close()