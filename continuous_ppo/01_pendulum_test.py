import gymnasium as gym
import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim

env = gym.make("Pendulum-v1")

state_dims = 3
action_dims = 1

rollout_steps = 2048
num_updates=100

gamma=0.99
lambda_gae=0.95

ppo_epochs = 10
batch_size = 64
clip_epsilon = 0.2

class ActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        # 主网络：
        # state_dim
        # -> 64
        # -> ReLU
        # -> 64
        # -> ReLU
        # -> action_dim
        #
        # 输出 mean

        self.network = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )


        # 可学习的 log_std
        #
        # shape = [action_dim]
        #
        # 初始值先全部设为0

        self.log_std = nn.Parameter(torch.zeros(action_dim))


    def forward(self, x):

        # 1. 计算 mean
        mean = self.network(x)

        # 2. log_std -> std
        std = torch.exp(self.log_std)

        # 3. 返回 mean 和 std
        return mean,std

class CriticNetwork(nn.Module):

    def __init__(self,state_dim):
        super().__init__()

        self.network=nn.Sequential(

            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)

        )

    def forward(self,x):

        value = self.network(x)

        return value.squeeze(-1)
    

state, info = env.reset(seed=42)
actor_network=ActorNetwork(state_dims,action_dims)
critic_network=CriticNetwork(state_dims)
actor_optimizer=optim.Adam(actor_network.parameters(),lr=3e-4)
critic_optimizer=optim.Adam(critic_network.parameters(),lr=3e-4)

action_high=torch.tensor(env.action_space.high,dtype=torch.float32)
action_low=torch.tensor(env.action_space.low,dtype=torch.float32)
action_scale=(action_high-action_low)/2.0
action_bias=(action_high+action_low)/2.0

episode_return = 0.0
completed_returns = []


for update in range(num_updates):

    states = []
    raw_actions = []
    rewards = []
    terminateds = []
    episode_ends = []
    log_probs = []
    values = []
    next_values = []

    for step in range(rollout_steps):

        # state -> tensor

        state_tensor = torch.tensor(state,dtype=torch.float32)

        # Actor:
        with torch.no_grad():
            # mean, std
            mean,std=actor_network(state_tensor)
            # Normal
            distribution=torch.distributions.Normal(mean,std)
            # raw_action
            raw_action=distribution.sample()
            # tanh
            squashed_action=torch.tanh(raw_action)
            # final action

            final_action=action_scale*squashed_action+action_bias
            # corrected log_prob
            base_log_prob=distribution.log_prob(raw_action)
            log_prob=base_log_prob-torch.log(action_scale*(1-squashed_action.pow(2))+1e-6)
            log_prob=log_prob.sum(dim=-1)
            # Critic:
            
            # value
            value=critic_network(state_tensor)

            # env.step(final action)
        next_state,reward,terminated,truncated,info=env.step(final_action.cpu().numpy())
        episode_end=(terminated or truncated)

            # 计算 next_state 的 next_value
            # 注意这里还没 reset
        next_state_tensor = torch.tensor(next_state,dtype=torch.float32)

        with torch.no_grad():
            if terminated:

                next_value=torch.tensor(0.0)

            else:

                next_value=critic_network(next_state_tensor)

            # 保存:
            # state
            # raw_action
            # reward
            # terminated
            # episode_end
            # old_log_prob
            # value
            # next_value


        states.append(state)
        raw_actions.append(raw_action)
        rewards.append(reward)
        terminateds.append(terminated)
        episode_ends.append(episode_end)
        log_probs.append(log_prob)
        values.append(value)
        next_values.append(next_value)

        episode_return += reward

        if episode_end:

            completed_returns.append(episode_return)
            episode_return=0.0
            state, info = env.reset()
        else:
            state = next_state

    advantages=[]
    gae=0.0

    for t in reversed(range(len(rewards))):

        delta = rewards[t]+gamma*(1.0-float(terminateds[t]))*next_values[t]-values[t]

        gae = delta+gamma*lambda_gae*(1.0-float(episode_ends[t]))*gae

        advantages.append(gae)

    advantages.reverse()

    advantages_tensor=torch.stack(advantages)

    values_tensor=torch.stack(values)

    returns_tensor=advantages_tensor+values_tensor

    advantages_tensor=(advantages_tensor-advantages_tensor.mean())/(advantages_tensor.std(unbiased=False)+1e-8)

    states_tensor=torch.tensor(np.array(states),dtype=torch.float32)
    raw_actions_tensor=torch.stack(raw_actions)
    old_log_probs_tensor=torch.stack(log_probs)

    num_samples = len(states_tensor)

    kl_values = []
    clip_fractions = []

    for epoch in range(ppo_epochs):

        indices=torch.randperm(num_samples)

        for start in range(0,num_samples,batch_size):

            batch_indices=indices[start:start+batch_size]

            batch_states=states_tensor[batch_indices]

            batch_raw_actions=raw_actions_tensor[batch_indices]

            batch_old_log_probs=old_log_probs_tensor[batch_indices]

            batch_advantages=advantages_tensor[batch_indices]

            batch_returns=returns_tensor[batch_indices]

            mean,std=actor_network(batch_states)

            distribution=torch.distributions.Normal(mean,std)

            base_new_log_probs=distribution.log_prob(batch_raw_actions)

            batch_squashed_actions=torch.tanh(batch_raw_actions)

            new_log_probs=base_new_log_probs-torch.log(action_scale*(1-batch_squashed_actions.pow(2))+1e-6)

            new_log_probs=new_log_probs.sum(dim=-1)

            ratio=torch.exp(new_log_probs-batch_old_log_probs)

            approx_kl = (
                batch_old_log_probs
                - new_log_probs
            ).mean()

            clip_fraction = (
                (torch.abs(ratio - 1.0) > clip_epsilon)
                .float()
                .mean()
            )

            kl_values.append(
                approx_kl.item()
            )

            clip_fractions.append(
    clip_fraction.item()
)

            surr1=ratio*batch_advantages
            surr2=torch.clamp(ratio,1.0-clip_epsilon,1.0+clip_epsilon)*batch_advantages

            actor_loss=-torch.min(surr1,surr2).mean()

            new_values=critic_network(batch_states)
            critic_loss=torch.nn.functional.mse_loss(new_values,batch_returns)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

    # if (update + 1) % 10 == 0:

    #     if len(completed_returns) > 0:

    #         recent_returns = completed_returns[-10:]

    #         print(
    #             f"Update {update + 1}, "
    #             f"Return: {np.mean(recent_returns):.2f}, "
    #             f"std: {torch.exp(actor_network.log_std).item():.4f}, "
    #             f"KL: {np.mean(kl_values):.4f}, "
    #             f"ClipFrac: {np.mean(clip_fractions):.3f}"
    #          )

eval_env = gym.make("Pendulum-v1")

eval_returns = []

for episode in range(10):

    state, info = eval_env.reset(
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

        next_state, reward, terminated, truncated, info = eval_env.step(
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


            




















