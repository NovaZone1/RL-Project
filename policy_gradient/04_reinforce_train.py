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
    lr=0.001
)


# =========================
# 7. Hyperparameters
# =========================

n_episodes = 3000

max_steps = 200

gamma = 1.0


# =========================
# 8. Statistics
# =========================

steps_per_episode = []

returns_per_episode = []

loss_history = []


# =========================
# 9. Training
# =========================

for episode in range(n_episodes):

    state = env.reset()

    log_probs = []

    rewards = []


    # =========================
    # Run One Episode
    # =========================

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
    # Compute Returns
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
    # Policy Loss
    # =========================

    policy_loss = torch.tensor(
        0.0
    )


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
    # Update Policy
    # =========================

    optimizer.zero_grad()

    policy_loss.backward()

    optimizer.step()


    loss_history.append(
        policy_loss.item()
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
# 10. Training Finished
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
# 11. Greedy Evaluation
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


        # Evaluation:
        # 不再采样
        # 直接选择概率最大的动作

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
# 12. Evaluation Results
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
# 13. Learned Policy
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