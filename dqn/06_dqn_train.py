import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque

from environment import GridWorld


# =========================
# 1. Random Seed
# =========================

seed = 42

torch.manual_seed(seed)

rng = np.random.default_rng(seed)


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

target_network.load_state_dict(
    q_network.state_dict()
)


# =========================
# 8. Optimizer + Loss
# =========================

optimizer = optim.Adam(
    q_network.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()


# =========================
# 9. Hyperparameters
# =========================

n_episodes = 1200

max_steps = 200

gamma = 1.0


epsilon_start = 1.0

epsilon_end = 0.01

epsilon_decay_episodes = 800


buffer_capacity = 5000

batch_size = 32

min_buffer_size = 64


target_update_frequency = 100


# =========================
# 10. Replay Buffer
# =========================

buffer = ReplayBuffer(
    capacity=buffer_capacity
)


# =========================
# 11. Training Statistics
# =========================

steps_per_episode = []

loss_history = []

gradient_steps = 0


# =========================
# 12. Training
# =========================

for episode in range(n_episodes):

    # -------------------------
    # Epsilon Decay
    # -------------------------

    ratio = min(
        episode
        /
        epsilon_decay_episodes,
        1.0
    )

    epsilon = (
        epsilon_start
        +
        (
            epsilon_end
            -
            epsilon_start
        )
        * ratio
    )


    # -------------------------
    # Reset Environment
    # -------------------------

    state = env.reset()


    # -------------------------
    # One Episode
    # -------------------------

    for step in range(max_steps):

        row, col = state


        # =========================
        # Epsilon-Greedy
        # =========================

        if rng.random() < epsilon:

            action = int(
                rng.integers(
                    action_dim
                )
            )

        else:

            state_tensor = state_to_tensor(
                state,
                env.rows,
                env.cols
            )


            with torch.no_grad():

                q_values = q_network(
                    state_tensor
                )


            q_values_np = (
                q_values.numpy()
            )


            best_actions = np.flatnonzero(
                np.isclose(
                    q_values_np,
                    q_values_np.max()
                )
            )


            action = int(
                rng.choice(
                    best_actions
                )
            )


        # =========================
        # Environment Step
        # =========================

        next_state, reward, done = env.step(
            action
        )


        # =========================
        # Store Transition
        # =========================

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )


        state = next_state


        # =========================
        # Mini-Batch Training
        # =========================

        if len(buffer) >= min_buffer_size:

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

                (
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done
                ) = transition


                states.append(
                    batch_state
                )

                actions.append(
                    batch_action
                )

                rewards.append(
                    batch_reward
                )

                next_states.append(
                    batch_next_state
                )

                dones.append(
                    batch_done
                )


            # =========================
            # Batch -> Tensors
            # =========================

            states_tensor = torch.stack(
                [
                    state_to_tensor(
                        s,
                        env.rows,
                        env.cols
                    )
                    for s in states
                ]
            )


            next_states_tensor = torch.stack(
                [
                    state_to_tensor(
                        s,
                        env.rows,
                        env.cols
                    )
                    for s in next_states
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
            # Online Network
            # Q(s, a)
            # =========================

            q_values = q_network(
                states_tensor
            )


            predicted_q = q_values.gather(
                1,
                actions_tensor
            ).squeeze(1)


            # =========================
            # Target Network
            # TD Target
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
                    (
                        1.0
                        -
                        dones_tensor
                    )
                )


            # =========================
            # Loss
            # =========================

            loss = loss_fn(
                predicted_q,
                targets
            )


            # =========================
            # Backpropagation
            # =========================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            loss_history.append(
                loss.item()
            )


            gradient_steps += 1


            # =========================
            # Update Target Network
            # =========================

            if (
                gradient_steps
                %
                target_update_frequency
                ==
                0
            ):

                target_network.load_state_dict(
                    q_network.state_dict()
                )


        # =========================
        # Episode Finished
        # =========================

        if done:

            steps_per_episode.append(
                step + 1
            )

            break


    else:

        steps_per_episode.append(
            max_steps
        )


    # =========================
    # Training Progress
    # =========================

    if (
        episode + 1
    ) % 100 == 0:

        recent_average = np.mean(
            steps_per_episode[-100:]
        )

        print(
            f"Episode {episode + 1:4d} "
            f"| epsilon = {epsilon:.3f} "
            f"| avg steps = {recent_average:.2f}"
        )


# =========================
# 13. Training Finished
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
# 14. Evaluation
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


        # Evaluation:
        # epsilon = 0
        # pure greedy

        with torch.no_grad():

            q_values = q_network(
                state_tensor
            )


        action = int(
            torch.argmax(
                q_values
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


# =========================
# 15. Evaluation Results
# =========================

eval_steps = np.array(
    eval_steps
)


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
        eval_steps
        <
        max_steps
    )
)


# =========================
# 16. Greedy Trajectory
# =========================

state = env.reset()

trajectory = [
    state
]


for step in range(
    max_steps
):

    state_tensor = state_to_tensor(
        state,
        env.rows,
        env.cols
    )


    with torch.no_grad():

        q_values = q_network(
            state_tensor
        )


    action = int(
        torch.argmax(
            q_values
        ).item()
    )


    next_state, reward, done = env.step(
        action
    )


    trajectory.append(
        next_state
    )


    state = next_state


    if done:

        break


print(
    "\nGreedy Trajectory:"
)

print(
    trajectory
)


print(
    "Steps:",
    len(trajectory) - 1
)

# =========================
# 17. Moving Average
# =========================

def moving_average(
    data,
    window
):

    data = np.array(
        data,
        dtype=np.float32
    )

    return np.convolve(
        data,
        np.ones(window) / window,
        mode="valid"
    )


# =========================
# 18. Training Steps Curve
# =========================

steps_window = 50

smooth_steps = moving_average(
    steps_per_episode,
    steps_window
)

steps_x = np.arange(
    steps_window,
    len(steps_per_episode) + 1
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    steps_x,
    smooth_steps
)

plt.axhline(
    y=6,
    linestyle="--",
    label="Optimal = 6 steps"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Steps to Goal"
)

plt.title(
    "DQN Training Steps"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/training_steps.png",
    dpi=150
)

plt.close()


# =========================
# 19. Loss Curve
# =========================

loss_window = 200

smooth_loss = moving_average(
    loss_history,
    loss_window
)

loss_x = np.arange(
    loss_window,
    len(loss_history) + 1
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    loss_x,
    smooth_loss
)

plt.xlabel(
    "Gradient Step"
)

plt.ylabel(
    "MSE Loss"
)

plt.title(
    "DQN Training Loss"
)

plt.tight_layout()

plt.savefig(
    "plots/training_loss.png",
    dpi=150
)

plt.close()


# =========================
# 20. Learned Policy
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

            q_values = q_network(
                state_tensor
            )


        best_action = int(
            torch.argmax(
                q_values
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
# 21. Plot Info
# =========================

print(
    "\nSaved plots:"
)

print(
    "plots/training_steps.png"
)

print(
    "plots/training_loss.png"
)