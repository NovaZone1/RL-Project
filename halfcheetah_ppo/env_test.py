import gymnasium as gym
from models import ActorNetwork,CriticNetwork
import torch


env = gym.make("HalfCheetah-v5")

state_dims=env.observation_space.shape[0] #17
action_dims=env.action_space.shape[0] #6

state, info = env.reset(seed=42)

actor_network=ActorNetwork(state_dims,action_dims)
critic_network=CriticNetwork(state_dims)

state_tensor=torch.tensor(state,dtype=torch.float32)
with torch.no_grad():
    mean,std=actor_network(state_tensor)

    distribution=torch.distributions.Normal(mean,std)

    raw_action=distribution.sample()

    squashed_action=torch.tanh(raw_action)

    log_prob=distribution.log_prob(raw_action).sum(dim=-1)

    value=critic_network(state_tensor)

print("mean shape:", mean.shape)
print("std shape:", std.shape)
print("raw action shape:", raw_action.shape)
print("squashed action:", squashed_action)
print("log prob shape:", log_prob.shape)
print("value shape:", value.shape)




