import gymnasium as gym
import torch
import numpy as np

from ppo_models import PPOActorNetwork


env = gym.make(
    "HalfCheetah-v5"
)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]

action_high=torch.tensor(
    env.action_space.high,
    dtype=torch.float32
)

action_low=torch.tensor(
    env.action_space.low,
    dtype=torch.float32
)


action_scale=(
    action_high-action_low
)/2.0


action_bias=(
    action_high+action_low
)/2.0

# 创建 Expert Actor
expert_actor=PPOActorNetwork(state_dim,action_dim)

# 加载 PPO
expert_actor.load_state_dict(torch.load("checkpoints/expert_actor.pth",weights_only=True))

expert_actor.eval()

states=[]
actions=[]
rewards=[]
next_states=[]
dones=[]


for episode in range(50):

    state,info=env.reset()

    while True:

        state_tensor=torch.tensor(state,dtype=torch.float32)

        # expert action
        with torch.no_grad():

            expert_mean,expert_std=expert_actor(state_tensor)

            expert_action=action_scale*torch.tanh(expert_mean)+action_bias

        # env.step
        next_state,reward,terminated,truncated,info=env.step(expert_action.cpu().numpy())
        done=terminated or truncated
        # 保存 transition
        states.append(state)
        actions.append(expert_action.cpu().numpy())
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        state=next_state

        if done:
            break
        
np.savez(
    "offline_dataset.npz",

    states=np.array(states),

    actions=np.array(actions),

    rewards=np.array(rewards),

    next_states=np.array(next_states),

    dones=np.array(dones)
)

print("states:",np.array(states).shape)
print("actions:",np.array(actions).shape)
print("rewards:",np.array(rewards).shape)
print("next_states:",np.array(next_states).shape)
print("dones:",np.array(dones).shape)