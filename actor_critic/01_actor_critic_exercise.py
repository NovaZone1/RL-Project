import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
from environment import GridWorld

rng = np.random.default_rng(42)
torch.manual_seed(42) #主管网络参数的固定随机

env = GridWorld()

max_episodes = 1000
max_steps = 500
steps_episode = []

state_dim = env.rows * env.cols
action_dim = 4

learning_rate = 0.01
gamma = 1.0

class ActorNetwork(nn.Module):

    def __init__(self,state_dims,action_dims):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dims,64),
            nn.ReLU(),
            nn.Linear(64,action_dims)
        )

    def forward(self,x):

        return self.network(x)

class CriticNetwork(nn.Module):

    def __init__(self,state_dims):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dims,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):

        value = self.network(x)

        return value.squeeze(-1)

def state_to_tensor(state):

    row,col = state

    index = env.cols * row + col

    one_hot = np.zeros(state_dim)

    one_hot[index] = 1

    state_tensor = torch.tensor(one_hot,dtype = torch.float32)

    return state_tensor

actor_network = ActorNetwork(state_dim,action_dim)

actor_optimizer = optim.Adam(

    actor_network.parameters(),
    lr = learning_rate

)

critic_network = CriticNetwork(state_dim)

critic_optimizer = optim.Adam(

    critic_network.parameters(),
    lr = learning_rate

)

for episode in range(max_episodes):

    state = env.reset()

    for step in range(max_steps):

        state_tensor = state_to_tensor(state)

        logits = actor_network(state_tensor)

        distribution = torch.distributions.Categorical(

            logits = logits

        )

        action = distribution.sample()

        prob_log = distribution.log_prob(action)

        value = critic_network(state_tensor)

        next_state,reward,done = env.step(action.item())

        next_state_tensor = state_to_tensor(next_state)

        next_value = critic_network(next_state_tensor)

        with torch.no_grad():
            if done:

                td_target = reward

            else:

                td_target = reward + gamma * next_value

        td_error = td_target - value

        actor_loss = -prob_log * td_error.detach()

        actor_optimizer.zero_grad()

        actor_loss.backward()

        actor_optimizer.step()

        critic_loss = td_error ** 2

        critic_optimizer.zero_grad()

        critic_loss.backward()

        critic_optimizer.step()

        state = next_state

        if done:

            steps_episode.append(step+1)

            break

    else:

        steps_episode.append(max_steps)

    if (episode + 1) % 100 == 0:

        avg_steps = np.mean(
        steps_episode[-100:]
    )

        print(
            f"Episode {episode + 1:4d} "
            f"| avg steps = {avg_steps:.2f}"
        )

max_evaluation_episodes = 100
max_evaluation_steps = 10
steps_evaluation = []

for episode in range(max_evaluation_episodes):

    state = env.reset()

    for step in range(max_evaluation_steps):

        state_tensor = state_to_tensor(state)

        logits = actor_network(state_tensor)

        action = torch.argmax(logits).item()

        next_state,reward,done = env.step(action)

        state = next_state 

        if done:

            steps_evaluation.append(step+1)

            break

    else:

        steps_evaluation.append(max_evaluation_steps)

steps_evaluation = np.array(steps_evaluation)

print("平均步长:",np.mean(steps_evaluation))
print("最大步长:",np.max(steps_evaluation))
print("最短步长:",np.min(steps_evaluation))
print("成功率:",np.mean(steps_evaluation < max_evaluation_episodes))









