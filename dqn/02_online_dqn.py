import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
# 3. State -> One-Hot Tensor
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
# 4. Environment
# =========================

env = GridWorld()

state_dim = (
    env.rows * env.cols
)

action_dim = 4


# =========================
# 5. Q Network
# =========================

q_network = QNetwork(
    state_dim,
    action_dim
)


# =========================
# 6. Optimizer
# =========================

optimizer = optim.Adam(
    q_network.parameters(),
    lr=0.01
)


# =========================
# 7. Loss Function
# =========================

loss_fn = nn.MSELoss()


# =========================
# 8. Hyperparameters
# =========================

gamma = 1.0


# =========================
# 9. Create One Transition
# =========================

state = (2, 2)

action = GridWorld.RIGHT

env.state = state

next_state, reward, done = env.step(
    action
)


print("Transition:")

print(
    "state:",
    state
)

print(
    "action:",
    action
)

print(
    "reward:",
    reward
)

print(
    "next_state:",
    next_state
)

print(
    "done:",
    done
)


# =========================
# 10. Current State Tensor
# =========================

state_tensor = state_to_tensor(
    state,
    env.rows,
    env.cols
)


# =========================
# 11. Predict Q Values
# =========================

q_values = q_network(
    state_tensor
)


# 当前真正执行的是 action
# 所以只取对应动作的 Q 值

predicted_q = q_values[
    action
]


# =========================
# 12. TD Target
# =========================

if done:

    target = torch.tensor(
        float(reward),
        dtype=torch.float32
    )

else:

    next_state_tensor = state_to_tensor(
        next_state,
        env.rows,
        env.cols
    )


    # Target 不参与反向传播
    with torch.no_grad():

        next_q_values = q_network(
            next_state_tensor
        )

        best_next_q = torch.max(
            next_q_values
        )

        target = (
            reward
            +
            gamma * best_next_q
        )


# =========================
# 13. Loss
# =========================

loss = loss_fn(
    predicted_q,
    target
)


# =========================
# 14. Print Before Update
# =========================

print("\nBefore Update:")

print(
    "Q values:",
    q_values.detach()
)

print(
    "Predicted Q:",
    predicted_q.item()
)

print(
    "Target:",
    target.item()
)

print(
    "Loss:",
    loss.item()
)


# =========================
# 15. Gradient Update
# =========================

optimizer.zero_grad()

loss.backward()

optimizer.step()


# =========================
# 16. Predict Again
# =========================

new_q_values = q_network(
    state_tensor
)


# =========================
# 17. Print After Update
# =========================

print("\nAfter Update:")

print(
    "Q values:",
    new_q_values.detach()
)

print(
    "New predicted Q:",
    new_q_values[action].item()
)