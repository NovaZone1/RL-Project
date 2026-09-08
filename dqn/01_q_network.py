import numpy as np
import torch
import torch.nn as nn

from environment import GridWorld

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

env = GridWorld()

state_dim = (
    env.rows * env.cols
)

action_dim = 4


q_network = QNetwork(
    state_dim,
    action_dim
)

state = (2, 2)

state_tensor = state_to_tensor(
    state,
    env.rows,
    env.cols
)

print(
    "State:",
    state
)

print(
    "State tensor:",
    state_tensor
)

print(
    "Shape:",
    state_tensor.shape
)

q_values = q_network(
    state_tensor
)

print(
    "\nQ values:"
)

print(
    q_values
)

print(
    "Q shape:",
    q_values.shape
)

    