import gymnasium as gym
import torch
import numpy as np

from td3_bc_models import Actor


# ======================
# 创建环境
# ======================

env = gym.make(
    "HalfCheetah-v5"
)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


# ======================
# 创建 TD3 Actor
# ======================

actor = Actor(
    state_dim,
    action_dim
)


actor.load_state_dict(
    torch.load(
        "checkpoints/td3_bc_actor.pth",
        weights_only=True
    )
)


actor.eval()


# ======================
# Evaluation
# ======================

episode_rewards = []


episodes = 10


for episode in range(episodes):

    state, info = env.reset()

    episode_reward = 0.0


    while True:

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )


        with torch.no_grad():

            action = actor(
                state_tensor
            )


        next_state, reward, terminated, truncated, info = env.step(
            action.cpu().numpy()
        )


        episode_reward += reward


        state = next_state


        if terminated or truncated:
            break


    episode_rewards.append(
        episode_reward
    )


# ======================
# 输出结果
# ======================

episode_rewards = np.array(
    episode_rewards
)


print(
    "TD3+BC Average:",
    episode_rewards.mean()
)


print(
    "TD3+BC Std:",
    episode_rewards.std()
)


print(
    "TD3+BC Max:",
    episode_rewards.max()
)


print(
    "TD3+BC Min:",
    episode_rewards.min()
)