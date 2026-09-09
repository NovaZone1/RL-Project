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
# 5. Policy Network
# =========================

policy_network = PolicyNetwork(
    state_dim,
    action_dim
)


# =========================
# 6. Episode
# =========================

max_steps = 500

state = env.reset()

trajectory = [
    state
]

rewards = []

log_probs = []

actions = []


for step in range(max_steps):

    # =========================
    # State Tensor
    # =========================

    state_tensor = state_to_tensor(
        state,
        env.rows,
        env.cols
    )


    # =========================
    # Policy Network
    # =========================

    logits = policy_network(
        state_tensor
    )


    # =========================
    # Action Distribution
    # =========================

    distribution = (
        torch.distributions.Categorical(
            logits=logits
        )
    )


    # =========================
    # Sample Action
    # =========================

    action = distribution.sample()


    # =========================
    # Log Probability
    # =========================

    log_prob = distribution.log_prob(
        action
    )


    # =========================
    # Environment Step
    # =========================

    next_state, reward, done = env.step(
        action.item()
    )


    # =========================
    # Save Experience
    # =========================

    actions.append(
        action.item()
    )

    rewards.append(
        reward
    )

    log_probs.append(
        log_prob
    )

    trajectory.append(
        next_state
    )


    state = next_state


    if done:
        break


# =========================
# 7. Results
# =========================

print(
    "Episode finished."
)

print(
    "Steps:",
    len(actions)
)

print(
    "Total reward:",
    sum(rewards)
)


print(
    "\nTrajectory:"
)

print(
    trajectory
)


print(
    "\nActions:"
)

print(
    actions
)


print(
    "\nFirst 10 log probabilities:"
)

for log_prob in log_probs[:10]:

    print(
        log_prob.item()
    )