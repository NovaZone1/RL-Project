import gymnasium as gym
import torch
import numpy as np

from models import ActorNetwork


# 1. 创建环境
env=gym.make("HalfCheetah-v5")
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
# 2. 创建 Actor
actor_network=ActorNetwork(env.observation_space.shape[0],env.action_space.shape[0])
actor_network.eval()
# 3. 加载 best_actor.pth
actor_network.load_state_dict(torch.load("checkpoints/best_actor.pth",weights_only=True))

# 4. 创建保存列表
states=[]
actions=[]


# 5. rollout 专家策略

for episode in range(50):

    state,info=env.reset()

    while True:
        state_tensor=torch.tensor(state,dtype=torch.float32)
        with torch.no_grad():
            mean,std=actor_network(state_tensor)
            action=action_scale*torch.tanh(mean)+action_bias
        next_state,reward,terminated,truncated,info=env.step(action.numpy())
        states.append(state)
        actions.append(action.numpy())
        state=next_state
        if terminated or truncated:
            break


# 6. np.savez 保存
np.savez(
    "expert_dataset.npz",
    states=np.array(states),
    actions=np.array(actions)
)

print(
    "states:",
    np.array(states).shape
)

print(
    "actions:",
    np.array(actions).shape
)