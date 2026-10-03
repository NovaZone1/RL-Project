# Reinforcement Learning Practice

本仓库记录从强化学习基础、模仿学习和 Offline RL，逐步进入机器人学习的实践过程。

## MuJoCo 机器人仿真

当前仿真阶段使用 Franka Emika Panda 机械臂，入口见 [`robot_sim/README.md`](robot_sim/README.md)。

```bash
conda activate rl-practice
python robot_sim/01_inspect_panda.py
python robot_sim/02_control_joint.py
```

