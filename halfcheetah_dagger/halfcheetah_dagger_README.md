# PPO + Behavior Cloning + DAgger Imitation Learning Practice

本项目完成了一个完整的强化学习与模仿学习流程：

-   PPO 连续控制 Expert Policy
-   Expert Demonstration 数据采集
-   Behavior Cloning (BC)
-   DAgger Dataset Aggregation

环境：

-   Gymnasium
-   HalfCheetah-v5
-   PyTorch

------------------------------------------------------------------------

## 1. Overall Pipeline

    PPO Expert
        |
        v
    Expert Dataset
        |
        v
    Behavior Cloning
        |
        v
    BC Policy
        |
        v
    DAgger Data Collection
        |
        v
    DAgger Policy

------------------------------------------------------------------------

## 2. Environment

Environment:

    HalfCheetah-v5

Observation:

    17 dimensions

Action:

    6 dimensions

Action range:

    [-1,1]

------------------------------------------------------------------------

## 3. PPO Expert

PPO 在 HalfCheetah 环境训练连续控制策略。

Expert Actor:

    state
     |
    MLP
     |
    mean, std
     |
    Gaussian Policy
     |
    tanh
     |
    action

保存：

    checkpoints/expert_actor.pth

------------------------------------------------------------------------

## 4. Expert Dataset

使用 PPO Expert rollout 收集示范：

数据：

    (state, expert_action)

保存：

    expert_dataset.npz

尺寸：

    states:
    (50000,17)

    actions:
    (50000,6)

------------------------------------------------------------------------

## 5. Behavior Cloning

BC 将模仿学习转化为监督学习：

目标：

    min ||policy(state)-expert_action||^2

网络：

    17
     |
    64
     |
    ReLU
     |
    64
     |
    ReLU
     |
    6
     |
    tanh

训练结果：

    Epoch 100 Loss:
    0.0186

Evaluation:

    Average Return:
    2848.23

    Std:
    91.10

------------------------------------------------------------------------

## 6. DAgger

Behavior Cloning 存在 Distribution Shift：

训练：

    state ~ Expert

部署：

    state ~ Learner

DAgger：

    Learner rollout

          |

    collect learner states

          |

    Expert labeling

          |

    Dataset aggregation

          |

    Retrain policy

训练数据：

    expert_dataset.npz

    +

    dagger_dataset.npz

保存：

    checkpoints/dagger_actor.pth

------------------------------------------------------------------------

## 7. Final Evaluation

  Policy         Average Return      Std
  ------------ ---------------- --------
  PPO Expert            2834.92   109.24
  BC                    2848.23    91.10
  DAgger                2801.01    82.56

------------------------------------------------------------------------

## 8. Analysis

### PPO Expert

作为教师策略：

    Return:
    2834.92

提供专家示范。

### BC

BC 达到：

    2848.23

说明在 HalfCheetah 连续控制任务中，监督学习可以有效逼近专家策略。

### DAgger

DAgger:

    2801.01

相比 BC 没有明显提升。

原因：

-   HalfCheetah 状态转移较平滑
-   BC 已经接近 Expert
-   当前只完成一次 DAgger iteration

同时 DAgger 降低了策略方差。

------------------------------------------------------------------------

## 9. Project Structure

    halfcheetah_dagger/

    ├── models.py
    ├── collect_dagger_data.py
    ├── train_dagger.py
    ├── evaluate.py
    ├── expert_dataset.npz
    ├── dagger_dataset.npz

    └── checkpoints/

        ├── expert_actor.pth
        ├── bc_actor.pth
        └── dagger_actor.pth

------------------------------------------------------------------------

## 10. Learned Concepts

Reinforcement Learning:

-   PPO
-   Continuous Control
-   Actor-Critic
-   Gaussian Policy

Imitation Learning:

-   Expert Demonstration
-   Behavior Cloning
-   Distribution Shift
-   DAgger
-   Dataset Aggregation

------------------------------------------------------------------------

## 11. Future Extensions

-   Multi-iteration DAgger
-   Offline Reinforcement Learning
-   TD3+BC
-   CQL
-   IQL
