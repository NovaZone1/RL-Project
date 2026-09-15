import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim 
from environment import GridWorld

env = GridWorld()
state_dims = env.rows * env.cols
action_dims = 4

max_episodes = 1000
max_steps = 200
ppo_epochs = 4
steps_episode = []

gamma = 1.0
learning_rate = 0.001
clip_epsilon = 0.2
gae_lambda = 0.95
entropy_coef = 0.01

class ActorNetwork(nn.Module):

    def __init__(self,state_dim,action_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )

    def forward(self,x):

        return self.network(x)

class CriticNetwork(nn.Module):

    def __init__(self,state_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,1)

        )

    def forward(self,x):

        value = self.network(x)

        return value.squeeze(-1)

def state_to_tensor(state):

    row,col = state

    index = env.cols * row + col

    one_hot = np.zeros(state_dims,dtype = np.float32)

    one_hot[index] = 1

    state_tensor = torch.tensor(one_hot)

    return state_tensor

actor_network = ActorNetwork(state_dims,action_dims)

actor_optimizer = optim.Adam(

    actor_network.parameters(),
    lr = learning_rate
)

critic_network = CriticNetwork(state_dims)

critic_optimizer = optim.Adam(

    critic_network.parameters(),
    lr = learning_rate
)

for episode in range(max_episodes):

    states = []
    actions = []
    rewards = []
    dones = []
    old_log_probs = []
    values = []

    state = env.reset()

    for step in range(max_steps):

        state_tensor = state_to_tensor(state)

        logits = actor_network(state_tensor)

        distribution = torch.distributions.Categorical(

            logits = logits
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        value = critic_network(state_tensor)

        next_state, reward, done = env.step(action.item())

        states.append(state)

        actions.append(action.item())

        rewards.append(reward)

        dones.append(done)

        old_log_probs.append(log_prob.detach())

        values.append(value.detach())

        state = next_state

        if done:
            steps_episode.append(step + 1)
            break

    else:
        steps_episode.append(max_steps)

    #基础advantage计算
    # returns = []

    # G = 0.0

    # for reward in reversed(rewards):

    #     G = reward + gamma * G

    #     returns.append(G)

    # returns.reverse()

    # returns_tensor = torch.tensor(returns,dtype = torch.float32)

    # values_tensor = torch.stack(values)

    # advantages_tensor = returns_tensor - values_tensor

    # =========================
    # Compute GAE
    # =========================

    values_tensor = torch.stack(values)

    advantages = []

    gae = 0.0


    # 1. 先确定 rollout 最后状态的 V(s)
    #
    # 如果已经真正到 terminal：
    #     V(next_state) = 0
    #
    # 如果只是因为 max_steps 截断：
    #     用 Critic 估计最后状态的 value

    with torch.no_grad():

        if done:
            next_value = 0.0
        else:
            final_state_tensor = state_to_tensor(state)
            next_value = critic_network(final_state_tensor)


    # 2. 从最后一步往前计算 GAE

    for t in reversed(range(len(rewards))):

        # 当前这一步是不是 terminal
        done_t = dones[t]


        # 当前状态价值
        value_t = values[t]


        # TD Error:
        #
        # δ_t =
        # r_t
        # + γ(1-done)V(s_{t+1})
        # - V(s_t)

        delta = rewards[t] + gamma * (1 - done_t) * next_value - value_t


        # GAE:
        #
        # A_t =
        # δ_t
        # + γ λ (1-done) A_{t+1}

        gae = delta + gamma * gae_lambda * (1 - done_t) * gae


        advantages.append(gae)


        # 往前移动以后，
        # 当前 V(s_t)
        # 会成为上一时刻的 V(s_{t+1})

        next_value = value_t


    # 3. 因为刚才是倒着保存
    # 恢复正常时间顺序

    advantages.reverse()


    # 4. 转成 Tensor

    advantages_tensor = torch.stack(advantages)


    # 5. Critic 的训练 target
    #
    # A_t = Return_target - V(s_t)
    #
    # 所以：
    #
    # Return_target = A_t + V(s_t)

    returns_tensor = advantages_tensor + values_tensor

    # =========================
    # Advantage Normalization
    # =========================

    advantages_tensor = (
        advantages_tensor
        - advantages_tensor.mean()
    ) / (
        advantages_tensor.std(
            unbiased=False
        )
        + 1e-8
    )

    #这里的处理要注意
    states_tensor = torch.stack(
        [state_to_tensor(state)
         for state in states]
    )

    actions_tensor = torch.tensor(actions,dtype = torch.long)

    old_log_probs_tensor = torch.stack(old_log_probs)

    for _ in range(ppo_epochs):

        logits = actor_network(states_tensor)

        distribution = torch.distributions.Categorical(
            logits = logits
        )

        new_log_probs = distribution.log_prob(actions_tensor)

        ratio = torch.exp(new_log_probs - old_log_probs_tensor)

        surr1 = ratio * advantages_tensor

        clipped_ratio = torch.clamp(

            ratio,
            1 - clip_epsilon,
            1 + clip_epsilon
        )

        surr2 = clipped_ratio * advantages_tensor

        entropy = distribution.entropy().mean()

        actor_loss = -torch.mean(torch.minimum(surr1,surr2)) - entropy_coef * entropy

        actor_optimizer.zero_grad()

        actor_loss.backward()

        actor_optimizer.step()

        predicted_values = critic_network(states_tensor)

        critic_loss = nn.MSELoss()(predicted_values,returns_tensor)

        critic_optimizer.zero_grad()

        critic_loss.backward()

        critic_optimizer.step()

    if (episode + 1) % 100 == 0:

        avg_steps = np.mean(
            steps_episode[-100:]
        )

        print(
            f"Episode {episode + 1:4d} "
            f"| avg steps = {avg_steps:.2f}"
        )

eval_episodes = 100
eval_max_steps = 200

eval_steps = []
success_count = 0


for episode in range(eval_episodes):

    state = env.reset()

    for step in range(eval_max_steps):

        state_tensor = state_to_tensor(state)

        with torch.no_grad():

            logits = actor_network(
                state_tensor
            )

            action = torch.argmax(
                logits
            ).item()

        next_state, reward, done = env.step(
            action
        )

        state = next_state

        if done:

            eval_steps.append(
                step + 1
            )

            success_count += 1

            break

    else:

        eval_steps.append(
            eval_max_steps
        )


average_steps = np.mean(eval_steps)

max_eval_steps = np.max(eval_steps)

min_eval_steps = np.min(eval_steps)

success_rate = (
    success_count / eval_episodes
)


print("\nPPO Evaluation")

print(
    f"平均步长: {average_steps:.2f}"
)

print(
    f"最大步长: {max_eval_steps}"
)

print(
    f"最短步长: {min_eval_steps}"
)

print(
    f"成功率: {success_rate:.2f}"
)

print(
    advantages_tensor.mean().item(),
    advantages_tensor.std(
        unbiased=False
    ).item()
)

