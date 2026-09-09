import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import GridWorld


# =========================
# 1. Random Seed
# =========================

seed = 42

torch.manual_seed(seed)


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
# 3. Value Network
# =========================

class ValueNetwork(nn.Module):

    def __init__(
        self,
        state_dim
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, x):

        value = self.network(x)

        return value.squeeze(-1)


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

state_dim = (
    env.rows * env.cols
)

action_dim = 4


# =========================
# 6. Networks
# =========================

policy_network = PolicyNetwork(
    state_dim,
    action_dim
)

value_network = ValueNetwork(
    state_dim
)


# =========================
# 7. Optimizers
# =========================

policy_optimizer = optim.Adam(
    policy_network.parameters(),
    lr=0.001
)

value_optimizer = optim.Adam(
    value_network.parameters(),
    lr=0.001
)


# =========================
# 8. Value Loss
# =========================

value_loss_fn = nn.MSELoss()


# =========================
# 9. Hyperparameters
# =========================

n_episodes = 3000

max_steps = 200

gamma = 1.0


# =========================
# 10. Statistics
# =========================

steps_per_episode = []

returns_per_episode = []

policy_loss_history = []

value_loss_history = []


# =========================
# 11. Training
# =========================

for episode in range(
    n_episodes
):

    state = env.reset()

    log_probs = []

    values = []

    rewards = []


    # =========================
    # One Episode
    # =========================

    for step in range(
        max_steps
    ):

        state_tensor = state_to_tensor(
            state,
            env.rows,
            env.cols
        )


        # -------------------------
        # Policy
        # -------------------------

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


        # -------------------------
        # Value
        # -------------------------

        value = value_network(
            state_tensor
        )


        # -------------------------
        # Environment
        # -------------------------

        next_state, reward, done = env.step(
            action.item()
        )


        # -------------------------
        # Save
        # -------------------------

        log_probs.append(
            log_prob
        )

        values.append(
            value
        )

        rewards.append(
            reward
        )


        state = next_state


        if done:
            break


    # =========================
    # Episode Statistics
    # =========================

    steps_per_episode.append(
        len(rewards)
    )

    returns_per_episode.append(
        sum(rewards)
    )


    # =========================
    # Compute Monte Carlo Returns
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


    log_probs_tensor = torch.stack(
        log_probs
    )


    values_tensor = torch.stack(
        values
    )


    # =========================
    # Advantage
    # =========================

    advantages = (
        returns_tensor
        -
        values_tensor.detach()
    )


    # =========================
    # Policy Loss
    # =========================

    policy_loss = -(
        log_probs_tensor
        *
        advantages
    ).sum()


    # =========================
    # Value Loss
    # =========================

    value_loss = value_loss_fn(
        values_tensor,
        returns_tensor
    )


    # =========================
    # Update Policy Network
    # =========================

    policy_optimizer.zero_grad()

    policy_loss.backward()

    policy_optimizer.step()


    # =========================
    # Update Value Network
    # =========================

    value_optimizer.zero_grad()

    value_loss.backward()

    value_optimizer.step()


    # =========================
    # Statistics
    # =========================

    policy_loss_history.append(
        policy_loss.item()
    )

    value_loss_history.append(
        value_loss.item()
    )


    # =========================
    # Progress
    # =========================

    if (
        episode + 1
    ) % 100 == 0:

        avg_steps = np.mean(
            steps_per_episode[-100:]
        )

        avg_return = np.mean(
            returns_per_episode[-100:]
        )


        print(
            f"Episode {episode + 1:4d} "
            f"| avg steps = {avg_steps:.2f} "
            f"| avg return = {avg_return:.2f}"
        )


# =========================
# 12. Training Finished
# =========================

print(
    "\nTraining finished."
)


print(
    "Average steps in first 100 episodes:",
    np.mean(
        steps_per_episode[:100]
    )
)


print(
    "Average steps in last 100 episodes:",
    np.mean(
        steps_per_episode[-100:]
    )
)


# =========================
# 13. Evaluation
# =========================

n_eval_episodes = 100

eval_steps = []


for episode in range(
    n_eval_episodes
):

    state = env.reset()


    for step in range(
        max_steps
    ):

        state_tensor = state_to_tensor(
            state,
            env.rows,
            env.cols
        )


        with torch.no_grad():

            logits = policy_network(
                state_tensor
            )


        action = int(
            torch.argmax(
                logits
            ).item()
        )


        next_state, reward, done = env.step(
            action
        )


        state = next_state


        if done:

            eval_steps.append(
                step + 1
            )

            break


    else:

        eval_steps.append(
            max_steps
        )


eval_steps = np.array(
    eval_steps
)


# =========================
# 14. Evaluation Results
# =========================

print(
    "\nEvaluation Results:"
)


print(
    "Average steps:",
    np.mean(
        eval_steps
    )
)


print(
    "Minimum steps:",
    np.min(
        eval_steps
    )
)


print(
    "Maximum steps:",
    np.max(
        eval_steps
    )
)


print(
    "Success rate:",
    np.mean(
        eval_steps < max_steps
    )
)


# =========================
# 15. Learned Policy
# =========================

action_names = {
    GridWorld.UP: "↑",
    GridWorld.DOWN: "↓",
    GridWorld.LEFT: "←",
    GridWorld.RIGHT: "→"
}


print(
    "\nLearned Policy:"
)


for row in range(
    env.rows
):

    line = []


    for col in range(
        env.cols
    ):

        state = (
            row,
            col
        )


        if state == env.goal_state:

            line.append(
                "G"
            )

            continue


        state_tensor = state_to_tensor(
            state,
            env.rows,
            env.cols
        )


        with torch.no_grad():

            logits = policy_network(
                state_tensor
            )


        best_action = int(
            torch.argmax(
                logits
            ).item()
        )


        line.append(
            action_names[
                best_action
            ]
        )


    print(
        "  ".join(line)
    )


# =========================
# 16. Learned State Values
# =========================

print(
    "\nLearned State Values:"
)


for row in range(
    env.rows
):

    line = []


    for col in range(
        env.cols
    ):

        state = (
            row,
            col
        )


        if state == env.goal_state:

            line.append(
                "   G   "
            )

            continue


        state_tensor = state_to_tensor(
            state,
            env.rows,
            env.cols
        )


        with torch.no_grad():

            value = value_network(
                state_tensor
            ).item()


        line.append(
            f"{value:6.2f}"
        )


    print(
        " ".join(line)
    )