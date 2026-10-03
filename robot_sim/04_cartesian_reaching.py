import argparse
import threading
import time

import mujoco
import numpy as np

from panda_common import load_panda


TCP_OFFSET_IN_HAND = np.array([0.0, 0.0, 0.1])
DAMPING = 0.05
POSITION_GAIN = 2.0
MAX_JOINT_SPEED = 0.8
POSITION_TOLERANCE = 0.002
RECOMMENDED_WORKSPACE_MIN = np.array([0.25, -0.35, 0.25])
RECOMMENDED_WORKSPACE_MAX = np.array([0.65, 0.35, 0.85])
SAMPLED_KINEMATIC_MIN = np.array([-0.94, -0.94, -0.41])
SAMPLED_KINEMATIC_MAX = np.array([0.94, 0.94, 1.27])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the Panda gripper centre to repeatedly entered targets."
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional first target; later targets are entered interactively.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Maximum simulation seconds allowed for each target (default: 8).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def get_tcp_position(data: mujoco.MjData) -> np.ndarray:
    hand = data.body("hand")
    hand_rotation = hand.xmat.reshape(3, 3)
    return hand.xpos + hand_rotation @ TCP_OFFSET_IN_HAND


def is_in_recommended_workspace(target_position: np.ndarray) -> bool:
    return bool(
        np.all(target_position >= RECOMMENDED_WORKSPACE_MIN)
        and np.all(target_position <= RECOMMENDED_WORKSPACE_MAX)
    )


def print_workspace_summary() -> None:
    print(
        "建议 TCP 工作区（蓝色透明框，单位 m）:\n"
        f"  x: {RECOMMENDED_WORKSPACE_MIN[0]:.2f} .. "
        f"{RECOMMENDED_WORKSPACE_MAX[0]:.2f}\n"
        f"  y: {RECOMMENDED_WORKSPACE_MIN[1]:.2f} .. "
        f"{RECOMMENDED_WORKSPACE_MAX[1]:.2f}\n"
        f"  z: {RECOMMENDED_WORKSPACE_MIN[2]:.2f} .. "
        f"{RECOMMENDED_WORKSPACE_MAX[2]:.2f}\n"
        "该区域是从 home 位姿出发验证过的保守教学范围，不代表完整工作空间。\n"
        "100,000 次关节采样得到的单轴运动学包络：\n"
        f"  x: {SAMPLED_KINEMATIC_MIN[0]:.2f} .. "
        f"{SAMPLED_KINEMATIC_MAX[0]:.2f}\n"
        f"  y: {SAMPLED_KINEMATIC_MIN[1]:.2f} .. "
        f"{SAMPLED_KINEMATIC_MAX[1]:.2f}\n"
        f"  z: {SAMPLED_KINEMATIC_MIN[2]:.2f} .. "
        f"{SAMPLED_KINEMATIC_MAX[2]:.2f}"
    )


def print_outside_workspace_warning(target_position: np.ndarray) -> None:
    print(
        f"警告：目标 {target_position} 位于建议工作区之外。\n"
        "Panda 的真实工作空间不是长方体；该目标可能仍然可达，也可能因关节限位、"
        "自碰撞或地面碰撞而无法到达。"
    )


def prompt_for_target() -> np.ndarray | None:
    while True:
        try:
            command = input(
                "\n输入下一个 TCP 目标 x y z（单位 m，输入 q 退出）: "
            ).strip()
        except EOFError:
            return None

        if command.lower() in {"q", "quit", "exit"}:
            return None

        values = command.split()
        if len(values) != 3:
            print("请输入三个数字，例如：0.45 0.15 0.55")
            continue

        try:
            target_position = np.array([float(value) for value in values])
        except ValueError:
            print("目标包含无法解析的内容，请重新输入三个数字。")
            continue

        if not np.all(np.isfinite(target_position)):
            print("目标必须是三个有限数值。")
            continue

        if not is_in_recommended_workspace(target_position):
            print_outside_workspace_warning(target_position)
            try:
                confirmation = input("仍然尝试这个目标？[y/N]: ").strip().lower()
            except EOFError:
                return None
            if confirmation not in {"y", "yes"}:
                continue

        return target_position


def calculate_joint_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_body_id: int,
    target_position: np.ndarray,
    position_jacobian: np.ndarray,
    rotation_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tcp_position = get_tcp_position(data)
    mujoco.mj_jac(
        model,
        data,
        position_jacobian,
        rotation_jacobian,
        tcp_position,
        hand_body_id,
    )

    position_error = target_position - tcp_position
    arm_jacobian = position_jacobian[:, :7]
    desired_velocity = POSITION_GAIN * position_error

    regularized_matrix = (
        arm_jacobian @ arm_jacobian.T + DAMPING**2 * np.eye(3)
    )
    joint_velocity = arm_jacobian.T @ np.linalg.solve(
        regularized_matrix,
        desired_velocity,
    )

    speed = np.linalg.norm(joint_velocity)
    if speed > MAX_JOINT_SPEED:
        joint_velocity *= MAX_JOINT_SPEED / speed

    return joint_velocity, position_error


def step_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_body_id: int,
    target_position: np.ndarray,
    arm_target: np.ndarray,
    position_jacobian: np.ndarray,
    rotation_jacobian: np.ndarray,
) -> np.ndarray:
    joint_velocity, position_error = calculate_joint_velocity(
        model,
        data,
        hand_body_id,
        target_position,
        position_jacobian,
        rotation_jacobian,
    )

    arm_target += joint_velocity * model.opt.timestep
    arm_target[:] = np.clip(
        arm_target,
        model.actuator_ctrlrange[:7, 0],
        model.actuator_ctrlrange[:7, 1],
    )
    data.ctrl[:7] = arm_target
    mujoco.mj_step(model, data)
    return position_error


def print_progress(
    data: mujoco.MjData,
    target_position: np.ndarray,
) -> None:
    tcp_position = get_tcp_position(data)
    position_error = target_position - tcp_position
    print(
        f"t={data.time:5.2f} s  "
        f"error={np.linalg.norm(position_error):.4f} m  "
        f"tcp=({tcp_position[0]:.3f}, {tcp_position[1]:.3f}, "
        f"{tcp_position[2]:.3f})  "
        f"target=({target_position[0]:.3f}, {target_position[1]:.3f}, "
        f"{target_position[2]:.3f})"
    )


def initialize_markers(
    viewer: object,
    target_position: np.ndarray,
    tcp_position: np.ndarray,
) -> None:
    with viewer.lock():
        viewer.user_scn.ngeom = 3
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.025, 0.025, 0.025]),
            target_position,
            np.eye(3).reshape(-1),
            np.array([1.0, 0.1, 0.1, 0.8], dtype=np.float32),
        )
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.015, 0.015, 0.015]),
            tcp_position,
            np.eye(3).reshape(-1),
            np.array([0.1, 1.0, 0.2, 0.9], dtype=np.float32),
        )
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[2],
            mujoco.mjtGeom.mjGEOM_BOX,
            (RECOMMENDED_WORKSPACE_MAX - RECOMMENDED_WORKSPACE_MIN) / 2.0,
            (RECOMMENDED_WORKSPACE_MAX + RECOMMENDED_WORKSPACE_MIN) / 2.0,
            np.eye(3).reshape(-1),
            np.array([0.1, 0.45, 1.0, 0.08], dtype=np.float32),
        )


def update_markers(
    viewer: object,
    target_position: np.ndarray,
    tcp_position: np.ndarray,
) -> None:
    with viewer.lock():
        viewer.user_scn.geoms[0].pos[:] = target_position
        viewer.user_scn.geoms[1].pos[:] = tcp_position


def wait_for_viewer_threads(viewer_threads: set[threading.Thread]) -> None:
    for thread in viewer_threads:
        thread.join(timeout=5.0)

    running_threads = [thread.name for thread in viewer_threads if thread.is_alive()]
    if running_threads:
        raise RuntimeError(
            f"MuJoCo viewer threads did not stop cleanly: {running_threads}"
        )


def move_to_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_body_id: int,
    target_position: np.ndarray,
    arm_target: np.ndarray,
    position_jacobian: np.ndarray,
    rotation_jacobian: np.ndarray,
    maximum_duration: float,
    viewer: object | None,
) -> bool:
    started_at_simulation_time = data.time
    next_report_time = data.time

    while data.time - started_at_simulation_time < maximum_duration:
        if viewer is not None and not viewer.is_running():
            return False

        step_started_at = time.monotonic()
        position_error = step_controller(
            model,
            data,
            hand_body_id,
            target_position,
            arm_target,
            position_jacobian,
            rotation_jacobian,
        )

        tcp_position = get_tcp_position(data)
        if viewer is not None:
            update_markers(viewer, target_position, tcp_position)
            viewer.sync()

        if data.time >= next_report_time:
            print_progress(data, target_position)
            next_report_time += 1.0

        if np.linalg.norm(position_error) <= POSITION_TOLERANCE:
            print_progress(data, target_position)
            return True

        if viewer is not None:
            remaining = model.opt.timestep - (
                time.monotonic() - step_started_at
            )
            if remaining > 0:
                time.sleep(remaining)

    print_progress(data, target_position)
    return False


def run_target_sequence(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    maximum_duration: float,
    first_target: np.ndarray | None,
    viewer: object | None,
) -> None:
    hand_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand",
    )
    arm_target = data.ctrl[:7].copy()
    position_jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros((3, model.nv))
    target_position = first_target
    markers_initialized = False
    target_number = 1

    print(f"初始 TCP 位置: {get_tcp_position(data)} m")
    print_workspace_summary()

    while viewer is None or viewer.is_running():
        if target_position is None:
            target_position = prompt_for_target()
            if target_position is None:
                break

        print(f"\n目标 {target_number}: {target_position} m")
        if viewer is not None:
            if not markers_initialized:
                initialize_markers(
                    viewer,
                    target_position,
                    get_tcp_position(data),
                )
                markers_initialized = True
            else:
                update_markers(
                    viewer,
                    target_position,
                    get_tcp_position(data),
                )
            viewer.sync()

        reached = move_to_target(
            model,
            data,
            hand_body_id,
            target_position,
            arm_target,
            position_jacobian,
            rotation_jacobian,
            maximum_duration,
            viewer,
        )

        if viewer is not None and not viewer.is_running():
            break

        if reached:
            print(f"目标 {target_number} 已到达。")
        else:
            print(
                f"目标 {target_number} 在 {maximum_duration:.1f} s 内未收敛，"
                "可能超出工作空间。"
            )

        target_number += 1
        target_position = None


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    maximum_duration: float,
    first_target: np.ndarray | None,
) -> None:
    import mujoco.viewer

    threads_before_launch = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - threads_before_launch

    try:
        run_target_sequence(
            model,
            data,
            maximum_duration,
            first_target,
            viewer,
        )
    finally:
        viewer.close()
        wait_for_viewer_threads(viewer_threads)


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    first_target = None
    if args.target is not None:
        first_target = np.asarray(args.target, dtype=float)
        if not np.all(np.isfinite(first_target)):
            raise ValueError("--target must contain three finite numbers.")
        if not is_in_recommended_workspace(first_target):
            print_outside_workspace_warning(first_target)

    model, data = load_panda()

    if args.headless:
        run_target_sequence(model, data, args.duration, first_target, None)
    else:
        run_viewer(model, data, args.duration, first_target)

    print("Reaching experiment finished.")


if __name__ == "__main__":
    main()
