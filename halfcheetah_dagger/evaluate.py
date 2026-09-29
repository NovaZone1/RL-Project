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


# 动作缩放

action_high = torch.tensor(
    env.action_space.high,
    dtype=torch.float32
)

action_low = torch.tensor(
    env.action_space.low,
    dtype=torch.float32
)


action_scale = (
    action_high-action_low
) / 2.0


action_bias = (
    action_high+action_low
) / 2.0



# ==========================
# 2. 加载三个策略
# ==========================


# Expert PPO

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



# BC

bc_actor = BCActorNetwork(
    state_dim,
    action_dim
)


bc_actor.load_state_dict(
    torch.load(
        "checkpoints/bc_actor.pth",
        weights_only=True
    )
)

bc_actor.eval()



# DAgger

dagger_actor = BCActorNetwork(
    state_dim,
    action_dim
)


dagger_actor.load_state_dict(
    torch.load(
        "checkpoints/dagger_actor.pth",
        weights_only=True
    )
)

dagger_actor.eval()



# ==========================
# 3. evaluation函数
# ==========================


def evaluate_policy(policy, policy_type):


    returns=[]


    for episode in range(10):

        state,info=env.reset()


        episode_return=0.0


        while True:


            state_tensor=torch.tensor(
                state,
                dtype=torch.float32
            )


            with torch.no_grad():


                if policy_type=="ppo":


                    mean,std=policy(
                        state_tensor
                    )

                    action=(
                        action_scale
                        *
                        torch.tanh(mean)
                        +
                        action_bias
                    )


                else:


                    action=policy(
                        state_tensor
                    )


                    action=(
                        action_scale
                        *
                        action
                        +
                        action_bias
                    )



            next_state,reward,terminated,truncated,info=env.step(
                action.numpy()
            )


            episode_return += reward


            state=next_state


            if terminated or truncated:

                break



        returns.append(
            episode_return
        )


    print(
        policy_type,
        "Average:",
        np.mean(returns)
    )

    print(
        policy_type,
        "Std:",
        np.std(returns)
    )

    print(
        policy_type,
        "Max:",
        np.max(returns)
    )

    print(
        policy_type,
        "Min:",
        np.min(returns)
    )

    print()



# ==========================
# 4. 测试
# ==========================


evaluate_policy(
    expert_actor,
    "ppo"
)


evaluate_policy(
    bc_actor,
    "bc"
)


evaluate_policy(
    dagger_actor,
    "dagger"
)