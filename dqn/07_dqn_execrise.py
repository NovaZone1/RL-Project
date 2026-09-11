import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from collections import deque
from environment import GridWorld

rng = np.random.default_rng(40)
env = GridWorld()

max_episodes = 2000
decay_episodes = 1500
max_steps = 500

state_dims = env.rows * env.cols
action_dims = 4

gamma = 1.0
learn_rating = 0.01

epsilon_start = 0.8
epsilon_end = 0.01

max_capicity = 5000
min_capicity = 100
batch_size = 64

update_freq = 100
steps_perepisode = []

gradient_steps = 0

#定义Q网络类(需要补一下这个类的写法)
class Q_Network(nn.Module):

    def __init__(self,state_dim,action_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_dim,64),

            nn.ReLU(),

            nn.Linear(64,action_dim)

        )

    def forward(self,x):

        return self.network(x)

def state_to_tensor(state):

    row,col = state

    index = env.cols * row + col

    one_hot = np.zeros(state_dims)

    one_hot[index] = 1

    one_hot = torch.tensor(one_hot,dtype = torch.float32)

    return one_hot

def epsilon_decay(episode):

    process = (epsilon_end - epsilon_start)/decay_episodes

    epsilon = max((epsilon_start + process * episode),epsilon_end)

    return epsilon
#创建一个回放buffer,实现buffer初始化，加入数据，抽取mini_batch的功能
class Replay_Buffer():

    def __init__(self,capacity):

        self.buffer = deque(maxlen = capacity)

    def push(
            self,
            state,
            action,
            reward,
            next_state,
            done
    ):
        #这里的处理要注意，将SARSAD作为一整体作为一条数据传入buffer
        transition = (state,action,reward,next_state,done)
        self.buffer.append(transition)

    def __len__(self):

        return len(self.buffer)

    #抽取mini_batch
    def sample(self,batch_size):

        indices = rng.choice(len(self.buffer),
                             batch_size,
                             replace = False)

        batch = [self.buffer[i]
                 for i in indices]

        return batch

def select_action(state,epsilon):

    if rng.random() < epsilon:

        action = rng.integers(action_dims)

    else:
        #注意这里的处理方式，要先将numpy转化为torch
        state_tensor = state_to_tensor(state)

        with torch.no_grad():

            q_values = train_network(state_tensor)

            # 这里的 item 的作用是取出下标
            action = torch.argmax(q_values).item()

    return action

#初始化

train_network = Q_Network(state_dims,action_dims)
target_network = Q_Network(state_dims,action_dims)
#这里相当于同步两个网络的参数
target_network.load_state_dict(train_network.state_dict())

buffer = Replay_Buffer(max_capicity)

loss_fun = nn.MSELoss()

optimizer = optim.Adam(

    train_network.parameters(),

    lr = learn_rating   
)

for episode in range(max_episodes):

    state = env.reset()

    epsilon = epsilon_decay(episode)

    for step in range (max_steps):

        action = select_action(state,epsilon)

        next_state,reward,done = env.step(action)

        buffer.push(state,action,reward,next_state,done)

        #先收集足够的数据，然后开始训练
        if len(buffer) >= min_capicity:

            batch = buffer.sample(batch_size)

            states,actions,rewards,next_states,dones = zip(
                * batch
            )

            states_tensor = torch.stack(
                [state_to_tensor(state)
                 for state in states]
            )
            
            actions_tensor = torch.tensor(actions,dtype = torch.long)

            rewards_tensor = torch.tensor(rewards,dtype = torch.float32)

            next_states_tensor = torch.stack(
                [state_to_tensor(next_state)
                 for next_state in next_states]
            )

            dones_tensor = torch.tensor(dones,dtype = torch.float32)

            q_values = train_network(states_tensor)

            #这里的处理要再学一下
            predicted_q = q_values.gather(1,actions_tensor.unsqueeze(1)).squeeze(1)

            with torch.no_grad():

                next_q_values = target_network(next_states_tensor)

                best_next_q = next_q_values.max(dim = 1).values

                targets = rewards_tensor + gamma * best_next_q * (1 - dones_tensor)

            loss = loss_fun(predicted_q,targets)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            gradient_steps += 1

            if gradient_steps % update_freq == 0:

                target_network.load_state_dict(train_network.state_dict())

        state = next_state  

        if done:

            steps_perepisode.append(step + 1)

            break

    else:

        steps_perepisode.append(max_steps)

    if (episode + 1) % 100 == 0:

        avg_steps = np.mean(steps_perepisode[-100:])

        print(
            f"Episode {episode + 1:4d} "
            f"| epsilon = {epsilon:.3f} "
            f"| avg steps = {avg_steps:.2f}"
        )

evaluation_episodes = 100
evaluation_steps = 100
per_evaluation_steps = []

for episode in range(evaluation_episodes):

    state = env.reset()

    for step in range(evaluation_steps):

        state_tensor = state_to_tensor(state)

        with torch.no_grad():

            q_values = train_network(state_tensor)

            action = torch.argmax(q_values).item()

            next_state,reward,done = env.step(action)

            state = next_state

        if done:

            per_evaluation_steps.append(step + 1)

            break

    else:

        per_evaluation_steps.append(evaluation_steps)

per_evaluation_steps = np.array(per_evaluation_steps)

print("平均步数:",np.mean(per_evaluation_steps))
print("最大步数:",per_evaluation_steps.max())
print("最小步数:",per_evaluation_steps.min())
print("成功率:",np.mean(per_evaluation_steps < evaluation_steps))


        









        








        






