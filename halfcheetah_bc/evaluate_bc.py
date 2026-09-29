import gymnasium as gym
import torch
import numpy as np 

from models import BCActorNetwork

env=gym.make("HalfCheetah-v5")
state_dims=env.observation_space.shape[0]
action_dims=env.action_space.shape[0]
bc_network=BCActorNetwork(state_dims,action_dims)
bc_network.load_state_dict(torch.load("checkpoints/bc_actor.pth",weights_only=True))
bc_network.eval()
rewards=[]
for episode in range(10):

    state,info=env.reset()
    episode_reward=0.0
    while True:

        state_tensor=torch.tensor(state,dtype=torch.float32)

        with torch.no_grad():

            action=bc_network(state_tensor)
            next_state,reward,terminated,truncated,info=env.step(action.numpy())
            

        episode_reward+=reward
        
        state=next_state

        if terminated or truncated:
            break
    rewards.append(episode_reward)

rewards=np.array(rewards)
print("平均奖励:",rewards.mean())
print("最大奖励:",rewards.max())
print("最小奖励:",rewards.min())
