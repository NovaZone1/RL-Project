import numpy as np
import torch
import torch.nn as nn

from environment import GridWorld


# =========================
# 1. Random Seed
# =========================

torch.manual_seed(42)


# =========================
# 2. Policy Network
# =========================

class PolicyNetwork(nn.Module):

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

        logits = self.network(x)

        return logits


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
# 5. Policy Network
# =========================

policy_network = PolicyNetwork(
    state_dim,
    action_dim
)


# =========================
# 6. Test State
# =========================

state = (2, 2)

state_tensor = state_to_tensor(
    state,
    env.rows,
    env.cols
)


# =========================
# 7. Network Output
# =========================

logits = policy_network(
    state_tensor
)


# =========================
# 8. Softmax -> Probabilities
# =========================

action_probs = torch.softmax(
    logits,
    dim=0
)


# =========================
# 9. Print
# =========================

print(
    "State:",
    state
)

print(
    "\nLogits:"
)

print(
    logits
)

print(
    "\nAction probabilities:"
)

print(
    action_probs
)

print(
    "\nProbability sum:"
)

print(
    action_probs.sum()
)