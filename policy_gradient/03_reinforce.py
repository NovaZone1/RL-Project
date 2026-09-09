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
# 6. Optimizer
# =========================

optimizer = optim.Adam(
    policy_network.parameters(),
    lr=0.01
)


# =========================
# 7. Hyperparameters
# =========================

gamma = 1.0

max_steps = 500


# =========================
# 8. Check Policy Before Update
# =========================

test_state = (2, 2)

test_tensor = state_to_tensor(
    test_state,
    env.rows,
    env.cols
)


with torch.no_grad():

    logits_before = policy_network(
        test_tensor
    )

    probs_before = torch.softmax(
        logits_before,
        dim=0
    )


print(
    "Policy at state (2,2) before update:"
)

print(
    probs_before
)


# =========================
# 9. Run One Episode
# =========================

state = env.reset()

log_probs = []

rewards = []

trajectory = [
    state
]


for step in range(max_steps):

    state_tensor = state_to_tensor(
        state,
        env.rows,
        env.cols
    )


    logits = policy_network(
        state_tensor
    )


    distribution = (
        torch.distributions.Categorical(
            logits=logits
        )
    )


    action = distribution.sample()


    log_prob = distribution.log_prob(
        action
    )


    next_state, reward, done = env.step(
        action.item()
    )


    log_probs.append(
        log_prob
    )

    rewards.append(
        reward
    )

    trajectory.append(
        next_state
    )


    state = next_state


    if done:
        break


# =========================
# 10. Compute Returns
# =========================

returns = []

G = 0.0


for reward in reversed(
    rewards
):

    G = (
        reward
        +
        gamma * G
    )

    returns.append(
        G
    )


returns.reverse()


returns_tensor = torch.tensor(
    returns,
    dtype=torch.float32
)


# =========================
# 11. Policy Loss
# =========================

policy_loss = 0.0


for log_prob, G in zip(
    log_probs,
    returns_tensor
):

    policy_loss = (
        policy_loss
        -
        log_prob * G
    )


# =========================
# 12. Print Episode Info
# =========================

print(
    "\nEpisode:"
)

print(
    "Steps:",
    len(rewards)
)

print(
    "Total reward:",
    sum(rewards)
)


print(
    "\nFirst 10 Returns:"
)

print(
    returns[:10]
)


print(
    "\nPolicy Loss:",
    policy_loss.item()
)


# =========================
# 13. Update Policy
# =========================

optimizer.zero_grad()

policy_loss.backward()

optimizer.step()


# =========================
# 14. Check Policy After Update
# =========================

with torch.no_grad():

    logits_after = policy_network(
        test_tensor
    )

    probs_after = torch.softmax(
        logits_after,
        dim=0
    )


print(
    "\nPolicy at state (2,2) after update:"
)

print(
    probs_after
)