import os

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from models import ActorNetwork, CriticNetwork


# ============================================================
# Hyperparameters
# ============================================================

rollout_steps = 2048
num_updates = 200

gamma = 0.99
gae_lambda = 0.95

ppo_epochs = 10
batch_size = 64
clip_epsilon = 0.2

actor_lr = 1e-4
critic_lr = 3e-4

target_kl = 0.05


# ============================================================
# Random Seed
# ============================================================

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)


# ============================================================
# Environment
# ============================================================

env = gym.make("HalfCheetah-v5")
eval_env = gym.make(
    "HalfCheetah-v5"
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

state, info = env.reset(seed=seed)


# ============================================================
# Actor / Critic
# ============================================================

actor_network = ActorNetwork(
    state_dim,
    action_dim
)

critic_network = CriticNetwork(
    state_dim
)


# ============================================================
# Optimizers
# ============================================================

actor_optimizer = optim.Adam(
    actor_network.parameters(),
    lr=actor_lr
)

critic_optimizer = optim.Adam(
    critic_network.parameters(),
    lr=critic_lr
)


# ============================================================
# Action Range
# ============================================================

action_high = torch.tensor(
    env.action_space.high,
    dtype=torch.float32
)

action_low = torch.tensor(
    env.action_space.low,
    dtype=torch.float32
)

action_scale = (
    action_high - action_low
) / 2.0

action_bias = (
    action_high + action_low
) / 2.0


# ============================================================
# Episode Statistics
# ============================================================

episode_return = 0.0

completed_returns = []

# ============================================================
# Training
# ============================================================

def evaluate_policy(
    actor_network,
    eval_env,
    action_scale,
    action_bias,
    num_episodes=5
):

    eval_returns = []

    actor_network.eval()

    for episode in range(num_episodes):

        state, info = eval_env.reset(
            seed=1000 + episode
        )

        episode_return = 0.0

        while True:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            with torch.no_grad():

                mean, std = actor_network(
                    state_tensor
                )

                action = (
                    action_scale
                    * torch.tanh(mean)
                    + action_bias
                )

            next_state, reward, terminated, truncated, info = eval_env.step(
                action.cpu().numpy()
            )

            episode_return += reward

            state = next_state

            if terminated or truncated:
                break

        eval_returns.append(
            episode_return
        )

    actor_network.train()

    return np.mean(eval_returns)

best_eval_return = -float("inf")

for update in range(num_updates):

    # --------------------------------------------------------
    # Rollout Buffer
    # --------------------------------------------------------

    states = []
    raw_actions = []
    rewards = []

    terminateds = []
    episode_ends = []

    log_probs = []

    values = []
    next_values = []


    # ========================================================
    # Collect Rollout
    # ========================================================

    for step in range(rollout_steps):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )


        # ----------------------------------------------------
        # Actor + Critic
        # ----------------------------------------------------

        with torch.no_grad():

            mean, std = actor_network(
                state_tensor
            )

            distribution = torch.distributions.Normal(
                mean,
                std
            )


            # Gaussian raw action
            raw_action = distribution.sample()


            # tanh squashing
            squashed_action = torch.tanh(
                raw_action
            )


            # map [-1, 1] -> environment action range
            final_action = (
                action_scale
                * squashed_action
                + action_bias
            )


            # ------------------------------------------------
            # Corrected Log Probability
            # ------------------------------------------------

            base_log_prob = distribution.log_prob(
                raw_action
            )

            log_prob = (
                base_log_prob
                - torch.log(
                    action_scale
                    * (
                        1
                        - squashed_action.pow(2)
                    )
                    + 1e-6
                )
            )

            log_prob = log_prob.sum(
                dim=-1
            )


            # ------------------------------------------------
            # State Value
            # ------------------------------------------------

            value = critic_network(
                state_tensor
            )


        # ----------------------------------------------------
        # Environment Step
        # ----------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            final_action.cpu().numpy()
        )

        episode_end = (
            terminated or truncated
        )


        # ----------------------------------------------------
        # Next State Value
        # ----------------------------------------------------

        next_state_tensor = torch.tensor(
            next_state,
            dtype=torch.float32
        )

        with torch.no_grad():

            # true terminal state:
            # no future value
            if terminated:

                next_value = torch.tensor(
                    0.0,
                    dtype=torch.float32
                )

            # truncation:
            # still bootstrap V(next_state)
            else:

                next_value = critic_network(
                    next_state_tensor
                )


        # ----------------------------------------------------
        # Store Rollout
        # ----------------------------------------------------

        states.append(
            state
        )

        raw_actions.append(
            raw_action
        )

        rewards.append(
            reward
        )

        terminateds.append(
            terminated
        )

        episode_ends.append(
            episode_end
        )

        log_probs.append(
            log_prob
        )

        values.append(
            value
        )

        next_values.append(
            next_value
        )


        # ----------------------------------------------------
        # Episode Statistics
        # ----------------------------------------------------

        episode_return += reward


        # ----------------------------------------------------
        # Move Environment
        # ----------------------------------------------------

        if episode_end:

            completed_returns.append(
                episode_return
            )

            episode_return = 0.0

            state, info = env.reset()

        else:

            state = next_state


    # ========================================================
    # GAE
    # ========================================================

    advantages = []

    gae = 0.0


    for t in reversed(
        range(len(rewards))
    ):

        # ----------------------------------------------------
        # TD Error
        # ----------------------------------------------------

        delta = (
            rewards[t]
            + gamma
            * (
                1.0
                - float(terminateds[t])
            )
            * next_values[t]
            - values[t]
        )


        # ----------------------------------------------------
        # Generalized Advantage Estimation
        # ----------------------------------------------------

        gae = (
            delta
            + gamma
            * gae_lambda
            * (
                1.0
                - float(episode_ends[t])
            )
            * gae
        )

        advantages.append(
            gae
        )


    advantages.reverse()


    # ========================================================
    # Convert Rollout to Tensor
    # ========================================================

    advantages_tensor = torch.stack(
        advantages
    )

    values_tensor = torch.stack(
        values
    )


    # Critic target
    #
    # return_target
    # =
    # old_value + GAE advantage

    returns_tensor = (
        advantages_tensor
        + values_tensor
    )


    # --------------------------------------------------------
    # Advantage Normalization
    # --------------------------------------------------------

    advantages_tensor = (
        advantages_tensor
        - advantages_tensor.mean()
    ) / (
        advantages_tensor.std(
            unbiased=False
        )
        + 1e-8
    )


    states_tensor = torch.tensor(
        np.array(states),
        dtype=torch.float32
    )

    raw_actions_tensor = torch.stack(
        raw_actions
    )

    old_log_probs_tensor = torch.stack(
        log_probs
    )

    # ========================================================
    # PPO Update
    # ========================================================

    num_samples = len(
        states_tensor
    )

    kl_values = []
    clip_fractions = []

    epochs_used = 0

    for epoch in range(ppo_epochs):

        epoch_kl_values = []

        indices = torch.randperm(
            num_samples
        )


        for start in range(
            0,
            num_samples,
            batch_size
        ):

            batch_indices = indices[
                start:start + batch_size
            ]


            # ------------------------------------------------
            # Mini-batch
            # ------------------------------------------------

            batch_states = states_tensor[
                batch_indices
            ]

            batch_raw_actions = raw_actions_tensor[
                batch_indices
            ]

            batch_old_log_probs = old_log_probs_tensor[
                batch_indices
            ]

            batch_advantages = advantages_tensor[
                batch_indices
            ]

            batch_returns = returns_tensor[
                batch_indices
            ]


            # =================================================
            # Current Actor
            # =================================================

            mean, std = actor_network(
                batch_states
            )

            distribution = torch.distributions.Normal(
                mean,
                std
            )


            # -------------------------------------------------
            # New Log Probability
            # -------------------------------------------------

            base_new_log_probs = distribution.log_prob(
                batch_raw_actions
            )

            batch_squashed_actions = torch.tanh(
                batch_raw_actions
            )

            new_log_probs = (
                base_new_log_probs
                - torch.log(
                    action_scale
                    * (
                        1
                        - batch_squashed_actions.pow(2)
                    )
                    + 1e-6
                )
            )

            new_log_probs = new_log_probs.sum(
                dim=-1
            )


            # =================================================
            # PPO Ratio
            # =================================================

            ratio = torch.exp(
                new_log_probs
                - batch_old_log_probs
            )

            # =================================================
            # PPO Diagnostics
            # =================================================

            approx_kl = (
                batch_old_log_probs
                - new_log_probs
            ).mean()

            kl_values.append(
                approx_kl.item()
            )

            epoch_kl_values.append(
                approx_kl.item()
            )

            clip_fraction = (
                (
                    torch.abs(
                        ratio - 1.0
                    )
                    > clip_epsilon
                )
                .float()
                .mean()
            )

            clip_fractions.append(
                clip_fraction.item()
            )


            # =================================================
            # Actor Loss
            # =================================================

            surr1 = (
                ratio
                * batch_advantages
            )

            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon
                )
                * batch_advantages
            )

            actor_loss = -torch.min(
                surr1,
                surr2
            ).mean()


            # =================================================
            # Critic Loss
            # =================================================

            new_values = critic_network(
                batch_states
            )

            critic_loss = torch.nn.functional.mse_loss(
                new_values,
                batch_returns
            )


            # =================================================
            # Update Actor
            # =================================================

            actor_optimizer.zero_grad()

            actor_loss.backward()

            actor_optimizer.step()


            # =================================================
            # Update Critic
            # =================================================

            critic_optimizer.zero_grad()

            critic_loss.backward()

            critic_optimizer.step()

        epochs_used = epoch + 1

        mean_epoch_kl = np.mean(
            epoch_kl_values
        )

        # if mean_epoch_kl > target_kl:
        #     break

        


    # ========================================================
    # Training Log
    # ========================================================

    if (update + 1) % 10 == 0:

        recent_returns = completed_returns[-10:]

        policy_std = torch.exp(
            actor_network.log_std
        )

        std_mean = policy_std.mean().item()
        std_min = policy_std.min().item()
        std_max = policy_std.max().item()

        eval_return = evaluate_policy(
            actor_network,
            eval_env,
            action_scale,
            action_bias,
            num_episodes=5
        )

        print(
            f"Update {update + 1}, "
            f"TrainReturn: {np.mean(recent_returns):.2f}, "
            f"EvalReturn: {eval_return:.2f}, "
            f"StdMean: {std_mean:.3f}, "
            f"StdMin: {std_min:.3f}, "
            f"StdMax: {std_max:.3f}, "
            f"KL: {np.mean(kl_values):.4f}, "
            f"ClipFrac: {np.mean(clip_fractions):.3f},"
            f"EpochsUsed: {epochs_used}"
        )

        if eval_return > best_eval_return:

            best_eval_return = eval_return

            os.makedirs(
                "checkpoints",
                exist_ok=True
            )

            torch.save(
                actor_network.state_dict(),
                "checkpoints/best_actor.pth"
            )

            torch.save(
                critic_network.state_dict(),
                "checkpoints/best_critic.pth"
            )

            print(
                f"New best model saved: "
                f"{best_eval_return:.2f}"
            )


# ============================================================
# Save Model
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

torch.save(
    actor_network.state_dict(),
    "checkpoints/actor.pth"
)

torch.save(
    critic_network.state_dict(),
    "checkpoints/critic.pth"
)

print("Models saved.")


# ============================================================
# Close Environment
# ============================================================

env.close()