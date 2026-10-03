# MuJoCo 机器人仿真

本目录是从强化学习过渡到机器人操作的第一阶段。当前目标不是立即训练策略，而是先打通最小闭环：

```text
读取机器人状态 -> 生成关节目标 -> MuJoCo 执行动作 -> 读取新状态
```

## 当前环境

- Conda 环境：`rl-practice`
- Python：3.10
- MuJoCo：已安装
- Gymnasium：已安装
- 机器人模型：`models/franka_emika_panda` 中的官方 MuJoCo Menagerie 模型

进入项目目录后激活环境：

```bash
conda activate rl-practice
```

## 第一课：读取 Panda 状态

```bash
python robot_sim/01_inspect_panda.py
```

这个脚本会加载 Panda、恢复到 `home` 位姿，并输出：

- `qpos`：关节位置 `q`
- `qvel`：关节速度 `q_dot`
- `ctrl`：位置控制器的目标值
- `hand.xpos`：末端执行器在世界坐标系中的位置
- `hand.xquat`：末端执行器在世界坐标系中的四元数，顺序为 `wxyz`

## 第二课：控制一个关节

打开交互式窗口：

```bash
python robot_sim/02_control_joint.py
```

脚本让 `joint1` 围绕 home 位置做平滑的正弦运动，同时保持其他关节目标不变。

没有图形显示时，可以只运行物理仿真：

```bash
python robot_sim/02_control_joint.py --headless
```

缩短或延长运行时间：

```bash
python robot_sim/02_control_joint.py --duration 12
```

## 第三课：键盘控制多个关节

```bash
python robot_sim/03_keyboard_control.py
```

启动后先点击 MuJoCo 窗口，让它获得键盘焦点：

- `↑` / `↓`：选择上一个 / 下一个机械臂关节。
- `←` / `→`：将目标角度减小 / 增大 `0.05 rad`。
- `Insert` / `Delete`：打开 / 关闭夹爪。
- `Home`：所有关节回到 home 目标。
- `Space`：暂停 / 恢复物理仿真。
- 关闭 MuJoCo 窗口：退出程序。

终端中的 `joint=actual/target` 分别表示实际关节角和控制器目标角。机器人具有质量和惯性，所以实际角度会稍微滞后于目标角度。

切换关节只会改变当前选中的控制对象，不会清除其他关节已经设置的目标。因此可以依次修改多个关节，程序每一步都会把完整的目标数组写入 `data.ctrl[:]`。

本脚本不使用字母和数字键，因为它们大多是 MuJoCo Viewer 的调试快捷键。例如：`0`～`5` 控制几何显示组，`I` 显示惯性框，`J` 显示关节，`C` 显示接触点。使用方向键也可以避免中文输入法截获控制按键。

## 第四课：Jacobian 末端 reaching

```bash
python robot_sim/04_cartesian_reaching.py
```

程序启动后，在终端输入 TCP 目标坐标：

```text
输入下一个 TCP 目标 x y z（单位 m，输入 q 退出）: 0.45 0.15 0.55
```

到达后可以继续输入第二个、第三个目标；输入 `q` 结束实验。红色小球表示目标位置，绿色小球表示真正被控制的 TCP，也就是两指之间的夹爪中心。

程序不会预先指定 7 个关节各自转多少，而是重复执行：

```text
末端位置误差 -> Jacobian -> 关节速度 -> 更新关节目标
```

为了自动提供第一个目标，也可以使用 `--target`；到达后程序仍会继续询问后续目标：

```bash
python robot_sim/04_cartesian_reaching.py --target 0.50 -0.15 0.60
```

无窗口运行：

```bash
python robot_sim/04_cartesian_reaching.py --headless
```

TCP 使用 Panda 官方模型采用的定义：相对 `hand` 坐标系沿局部 z 轴前移 `0.1 m`。这一课只控制 TCP 位置 `(x, y, z)`，不控制夹爪朝向。终端中的 `error` 是当前 TCP 到目标点的直线距离。

### 可达范围

Franka 官方资料给出的最大法兰臂展为 `855 mm`，但机械臂工作空间并不是以底座为中心的完整球体。关节限位、夹爪朝向、自碰撞和地面都会影响一个目标是否可达。

对当前 MuJoCo Panda 模型进行 `100,000` 组关节限位内随机采样后，TCP 的运动学包络约为：

- `x: -0.94 .. 0.94 m`
- `y: -0.94 .. 0.94 m`
- `z: -0.41 .. 1.27 m`

这些极值不能组合成一个全部可达的长方体，而且包含不安全姿态。第四课采用经过网格测试的保守教学范围：

- `x: 0.25 .. 0.65 m`
- `y: -0.35 .. 0.35 m`
- `z: 0.25 .. 0.85 m`

Viewer 中的蓝色透明框表示这个建议范围。输入框外目标时程序会警告，但确认后仍可继续实验，以便观察不可达目标、关节限位和局部 IK 的表现。

## 第五课：桌面 pushing

```bash
python robot_sim/05_push_cube.py
```

这一课在场景中加入桌子和一个带自由关节的方块。机械臂会自动执行四个阶段：

```text
移动到方块后方 -> 下降到推送高度 -> 沿 +y 方向推方块 -> 抬起夹爪
```

Viewer 中的红色小球表示当前阶段的 TCP 目标。脚本同时控制 TCP 的位置和夹爪朝向，因此推送过程中手腕不会随意翻转。这里使用的是完整的 `6 x 7` Jacobian：前三行描述线速度，后三行描述角速度。

方块不是通过代码直接修改坐标移动的。它具有质量、摩擦和一个 `freejoint`，夹爪先接触方块，再由 MuJoCo 的接触力推动它。终端最后会输出方块的初始位置、最终位置和位移；沿 `+y` 移动至少 `0.15 m` 即判定成功。

无窗口验证：

```bash
python robot_sim/05_push_cube.py --headless
```

## 接下来的路线

1. 读取关节状态和末端位姿。
2. 控制单关节和多关节。
3. 用 Jacobian 做末端 reaching。
4. 加入桌面和方块，完成 pushing。
5. 记录 `(observation, action)` demonstration。
6. 用 BC 和 Diffusion Policy 学习动作。
