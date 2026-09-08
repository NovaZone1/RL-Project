import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

from environment import GridWorld


# =========================
# 1. Random Seed
# =========================

torch.manual_seed(42)


# =========================
# 2. Q Network
# =========================

class QNetwork(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )


    def forward(self, x):

        return self.network(x)


# =========================
# 3. Replay Buffer
# =========================

class ReplayBuffer:

    def __init__(
        self,
        capacity
    ):

        self.buffer = deque(
            maxlen=capacity
        )


    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def sample(
        self,
        batch_size,
        rng
    ):

        indices = rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False
        )

        batch = [
            self.buffer[i]
            for i in indices
        ]

        return batch


    def __len__(self):

        return len(
            self.buffer
        )


# =========================
# 4. State -> One-Hot Tensor
# =========================

def state_to_tensor(
    state,
    rows,
    cols
):

    row, col = state

    state_index = (
        row * cols + col
    )

    one_hot = np.zeros(
        rows * cols,
        dtype=np.float32
    )

    one_hot[state_index] = 1.0

    return torch.tensor(
        one_hot,
        dtype=torch.float32
    )


# =========================
# 5. Environment
# =========================

env = GridWorld()

rng = np.random.default_rng(
    42
)

state_dim = (
    env.rows * env.cols
)

action_dim = 4


# =========================
# 6. Online Network
# =========================

q_network = QNetwork(
    state_dim,
    action_dim
)


# =========================
# 7. Target Network
# =========================

target_network = QNetwork(
    state_dim,
    action_dim
)


# 一开始让 Target Network
# 和 Online Network 完全相同

target_network.load_state_dict(
    q_network.state_dict()
)


# =========================
# 8. Optimizer + Loss
# =========================

optimizer = optim.Adam(
    q_network.parameters(),
    lr=0.01
)

loss_fn = nn.MSELoss()


# =========================
# 9. Hyperparameters
# =========================

gamma = 1.0

batch_size = 8


# =========================
# 10. Replay Buffer
# =========================

buffer = ReplayBuffer(
    capacity=500
)


# =========================
# 11. Collect Experience
# =========================

state = env.reset()


for step in range(200):

    action = int(
        rng.integers(4)
    )

    next_state, reward, done = env.step(
        action
    )


    buffer.push(
        state,
        action,
        reward,
        next_state,
        done
    )


    state = next_state


    if done:

        state = env.reset()


print(
    "Buffer size:",
    len(buffer)
)


# =========================
# 12. Sample Batch
# =========================

batch = buffer.sample(
    batch_size,
    rng
)


states = []
actions = []
rewards = []
next_states = []
dones = []


for transition in batch:

    state, action, reward, next_state, done = transition

    states.append(state)

    actions.append(action)

    rewards.append(reward)

    next_states.append(
        next_state
    )

    dones.append(done)


# =========================
# 13. Convert To Tensors
# =========================

states_tensor = torch.stack(
    [
        state_to_tensor(
            state,
            env.rows,
            env.cols
        )
        for state in states
    ]
)


next_states_tensor = torch.stack(
    [
        state_to_tensor(
            state,
            env.rows,
            env.cols
        )
        for state in next_states
    ]
)


actions_tensor = torch.tensor(
    actions,
    dtype=torch.long
).unsqueeze(1)


rewards_tensor = torch.tensor(
    rewards,
    dtype=torch.float32
)


dones_tensor = torch.tensor(
    dones,
    dtype=torch.float32
)


# =========================
# 14. Before Update
# =========================

with torch.no_grad():

    online_before = q_network(
        states_tensor
    )

    target_before = target_network(
        states_tensor
    )


print(
    "\nBefore Online Update:"
)

print(
    "Online Q:"
)

print(
    online_before
)

print(
    "\nTarget Q:"
)

print(
    target_before
)


print(
    "\nNetworks equal:",
    torch.allclose(
        online_before,
        target_before
    )
)


# =========================
# 15. Predicted Q
# Online Network
# =========================

q_values = q_network(
    states_tensor
)


predicted_q = q_values.gather(
    1,
    actions_tensor
).squeeze(1)


# =========================
# 16. TD Targets
# Target Network
# =========================

with torch.no_grad():

    next_q_values = target_network(
        next_states_tensor
    )

    best_next_q = torch.max(
        next_q_values,
        dim=1
    ).values


    targets = (
        rewards_tensor
        +
        gamma
        *
        best_next_q
        *
        (1.0 - dones_tensor)
    )


# =========================
# 17. Loss
# =========================

loss = loss_fn(
    predicted_q,
    targets
)


print(
    "\nPredicted Q:"
)

print(
    predicted_q.detach()
)


print(
    "\nTargets:"
)

print(
    targets
)


print(
    "\nLoss:",
    loss.item()
)


# =========================
# 18. Update Online Network
# =========================

optimizer.zero_grad()

loss.backward()

optimizer.step()


# =========================
# 19. Compare Again
# =========================

with torch.no_grad():

    online_after = q_network(
        states_tensor
    )

    target_after = target_network(
        states_tensor
    )


print(
    "\nAfter Online Update:"
)

print(
    "Online Q:"
)

print(
    online_after
)

print(
    "\nTarget Q:"
)

print(
    target_after
)


print(
    "\nNetworks equal:",
    torch.allclose(
        online_after,
        target_after
    )
)


# =========================
# 20. Synchronize Target Network
# =========================

target_network.load_state_dict(
    q_network.state_dict()
)


# =========================
# 21. Compare After Sync
# =========================

with torch.no_grad():

    online_synced = q_network(
        states_tensor
    )

    target_synced = target_network(
        states_tensor
    )


print(
    "\nAfter Target Sync:"
)

print(
    "Networks equal:",
    torch.allclose(
        online_synced,
        target_synced
    )
)