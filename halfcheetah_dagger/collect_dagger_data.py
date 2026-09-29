import gymnasium as gym
import torch
import numpy as np

from models import PPOActorNetwork, BCActorNetwork


# ==========================
# 1. 创建环境
# ==========================

env = gym.make(
    "HalfCheetah-v5"
)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


action_high=torch.tensor(
    env.action_space.high,
    dtype=torch.float32
)

action_low=torch.tensor(
    env.action_space.low,
    dtype=torch.float32
)


action_scale=(
    action_high-action_low
)/2.0


action_bias=(
    action_high+action_low
)/2.0

# ==========================
# 2. 创建 Expert PPO Actor
# ==========================

expert_actor = PPOActorNetwork(
    state_dim,
    action_dim
)


expert_actor.load_state_dict(
    torch.load(
        "checkpoints/expert_actor.pth",
        weights_only=True
    )
)


expert_actor.eval()


# ==========================
# 3. 创建 Learner BC Actor
# ==========================

learner_actor = BCActorNetwork(
    state_dim,
    action_dim
)


learner_actor.load_state_dict(
    torch.load(
        "checkpoints/bc_actor.pth",
        weights_only=True
    )
)


learner_actor.eval()



# ==========================
# 4. 保存 DAgger 数据
# ==========================

dagger_states = []

dagger_actions = []



# ==========================
# 5. Learner rollout
# ==========================

num_episodes = 50


for episode in range(num_episodes):


    state, info = env.reset()


    while True:


        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )


        with torch.no_grad():


            # --------------------------
            # Learner执行动作
            # --------------------------

            learner_action = (
                action_scale
                * learner_actor(state_tensor)
                + action_bias
            )



            # --------------------------
            # Expert提供标签
            # --------------------------

            expert_mean, expert_std = expert_actor(
                state_tensor
            )


            expert_action = (
                action_scale
                * torch.tanh(expert_mean)
                + action_bias
            )



        # ==========================
        # 环境执行 learner action
        # ==========================

        next_state, reward, terminated, truncated, info = env.step(
            learner_action.numpy()
        )



        # ==========================
        # 保存:
        #
        # learner访问的state
        #
        # expert给出的action
        # ==========================

        dagger_states.append(
            state
        )


        dagger_actions.append(
            expert_action.numpy()
        )



        state = next_state



        if terminated or truncated:

            break



# ==========================
# 6. numpy转换
# ==========================

dagger_states = np.array(
    dagger_states,
    dtype=np.float32
)


dagger_actions = np.array(
    dagger_actions,
    dtype=np.float32
)



print(
    "dagger states:",
    dagger_states.shape
)


print(
    "dagger actions:",
    dagger_actions.shape
)



# ==========================
# 7. 保存
# ==========================

np.savez(
    "dagger_dataset.npz",
    states=dagger_states,
    actions=dagger_actions
)


print(
    "DAgger dataset saved"
)