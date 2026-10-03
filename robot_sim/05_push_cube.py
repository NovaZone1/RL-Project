import argparse
import threading
import time
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "franka_emika_panda"
    / "pushing_scene.xml"
)
TCP_OFFSET_IN_HAND = np.array([0.0, 0.0, 0.1])
DAMPING = 0.06
POSITION_GAIN = 2.0
ORIENTATION_GAIN = 1.5
MAX_JOINT_SPEED = 0.8
POSITION_TOLERANCE = 0.004
ORIENTATION_TOLERANCE = 0.03
WAYPOINT_TIMEOUT = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the Panda gripper to push a cube across a table."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def load_pushing_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "pushing_home",
    )
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    return model, data


def get_tcp_position(data: mujoco.MjData) -> np.ndarray:
    hand = data.body("hand")
    hand_rotation = hand.xmat.reshape(3, 3)
    return hand.xpos + hand_rotation @ TCP_OFFSET_IN_HAND


def get_orientation_error(
    current_rotation: np.ndarray,
    desired_rotation: np.ndarray,
) -> np.ndarray:
    rotation_error = desired_rotation @ current_rotation.T
    return 0.5 * np.array(
        [
            rotation_error[2, 1] - rotation_error[1, 2],
            rotation_error[0, 2] - rotation_error[2, 0],
            rotation_error[1, 0] - rotation_error[0, 1],
        ]
    )


def calculate_joint_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_body_id: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    position_jacobian: np.ndarray,
    rotation_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    hand_rotation = data.body("hand").xmat.reshape(3, 3)
    orientation_error = get_orientation_error(hand_rotation, target_rotation)
    task_error = np.concatenate(
        [
            POSITION_GAIN * position_error,
            ORIENTATION_GAIN * orientation_error,
        ]
    )
    task_jacobian = np.vstack(
        [position_jacobian[:, :7], rotation_jacobian[:, :7]]
    )
    regularized_matrix = (
        task_jacobian @ task_jacobian.T + DAMPING**2 * np.eye(6)
    )
    joint_velocity = task_jacobian.T @ np.linalg.solve(
        regularized_matrix,
        task_error,
    )

    speed = np.linalg.norm(joint_velocity)
    if speed > MAX_JOINT_SPEED:
        joint_velocity *= MAX_JOINT_SPEED / speed

    return joint_velocity, position_error, orientation_error


def initialize_target_marker(viewer: object, target_position: np.ndarray) -> None:
    with viewer.lock():
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.018, 0.018, 0.018]),
            target_position,
            np.eye(3).reshape(-1),
            np.array([1.0, 0.1, 0.1, 0.85], dtype=np.float32),
        )


def update_target_marker(viewer: object, target_position: np.ndarray) -> None:
    with viewer.lock():
        viewer.user_scn.geoms[0].pos[:] = target_position


def step_and_sync(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: object | None,
) -> None:
    step_started_at = time.monotonic()
    mujoco.mj_step(model, data)

    if viewer is not None:
        viewer.sync()
        remaining = model.opt.timestep - (time.monotonic() - step_started_at)
        if remaining > 0:
            time.sleep(remaining)


def settle_and_close_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: object | None,
) -> None:
    data.ctrl[7] = model.actuator_ctrlrange[7, 0]
    finish_time = data.time + 0.6
    while data.time < finish_time:
        if viewer is not None and not viewer.is_running():
            return
        step_and_sync(model, data, viewer)


def print_waypoint_progress(
    phase_name: str,
    data: mujoco.MjData,
    target_position: np.ndarray,
    position_error: np.ndarray,
    orientation_error: np.ndarray,
) -> None:
    tcp_position = get_tcp_position(data)
    print(
        f"{phase_name:<8}  t={data.time:5.2f} s  "
        f"position_error={np.linalg.norm(position_error):.4f} m  "
        f"orientation_error={np.linalg.norm(orientation_error):.4f}  "
        f"tcp=({tcp_position[0]:.3f}, {tcp_position[1]:.3f}, "
        f"{tcp_position[2]:.3f})  "
        f"target=({target_position[0]:.3f}, {target_position[1]:.3f}, "
        f"{target_position[2]:.3f})"
    )


def move_to_waypoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_body_id: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    arm_target: np.ndarray,
    position_jacobian: np.ndarray,
    rotation_jacobian: np.ndarray,
    phase_name: str,
    viewer: object | None,
) -> bool:
    started_at = data.time
    next_report_time = data.time

    if viewer is not None:
        update_target_marker(viewer, target_position)

    while data.time - started_at < WAYPOINT_TIMEOUT:
        if viewer is not None and not viewer.is_running():
            return False

        joint_velocity, position_error, orientation_error = (
            calculate_joint_velocity(
                model,
                data,
                hand_body_id,
                target_position,
                target_rotation,
                position_jacobian,
                rotation_jacobian,
            )
        )
        arm_target += joint_velocity * model.opt.timestep
        arm_target[:] = np.clip(
            arm_target,
            model.actuator_ctrlrange[:7, 0],
            model.actuator_ctrlrange[:7, 1],
        )
        data.ctrl[:7] = arm_target
        step_and_sync(model, data, viewer)

        if data.time >= next_report_time:
            print_waypoint_progress(
                phase_name,
                data,
                target_position,
                position_error,
                orientation_error,
            )
            next_report_time += 0.5

        if (
            np.linalg.norm(position_error) <= POSITION_TOLERANCE
            and np.linalg.norm(orientation_error) <= ORIENTATION_TOLERANCE
        ):
            print_waypoint_progress(
                phase_name,
                data,
                target_position,
                position_error,
                orientation_error,
            )
            return True

    print_waypoint_progress(
        phase_name,
        data,
        target_position,
        position_error,
        orientation_error,
    )
    return False


def run_task(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: object | None,
) -> bool:
    hand_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand",
    )
    cube_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "cube",
    )
    initial_cube_position = data.xpos[cube_body_id].copy()
    target_rotation = data.body("hand").xmat.reshape(3, 3).copy()
    arm_target = data.ctrl[:7].copy()
    position_jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros((3, model.nv))

    push_height = initial_cube_position[2] + 0.01
    push_start_y = initial_cube_position[1] - 0.13
    push_end_y = initial_cube_position[1] + 0.25
    waypoints = [
        (
            "approach",
            np.array([initial_cube_position[0], push_start_y, 0.60]),
        ),
        (
            "descend",
            np.array(
                [initial_cube_position[0], push_start_y, push_height]
            ),
        ),
        (
            "push",
            np.array([initial_cube_position[0], push_end_y, push_height]),
        ),
        (
            "lift",
            np.array([initial_cube_position[0], push_end_y, 0.60]),
        ),
    ]

    print(f"Initial cube position: {initial_cube_position} m")
    print("Closing the gripper before pushing.")
    settle_and_close_gripper(model, data, viewer)
    if viewer is not None and not viewer.is_running():
        return False

    if viewer is not None:
        initialize_target_marker(viewer, waypoints[0][1])

    for phase_name, target_position in waypoints:
        print(f"\nPhase: {phase_name}")
        reached = move_to_waypoint(
            model,
            data,
            hand_body_id,
            target_position,
            target_rotation,
            arm_target,
            position_jacobian,
            rotation_jacobian,
            phase_name,
            viewer,
        )
        if not reached:
            print(f"Phase '{phase_name}' did not reach its waypoint.")
            return False

    final_cube_position = data.xpos[cube_body_id].copy()
    cube_displacement = final_cube_position - initial_cube_position
    print(f"\nFinal cube position: {final_cube_position} m")
    print(f"Cube displacement: {cube_displacement} m")

    succeeded = cube_displacement[1] >= 0.15
    if succeeded:
        print("Pushing task succeeded.")
    else:
        print("Pushing task failed: the cube moved less than 0.15 m in +y.")
    return succeeded


def wait_for_viewer_threads(viewer_threads: set[threading.Thread]) -> None:
    for thread in viewer_threads:
        thread.join(timeout=5.0)

    running_threads = [thread.name for thread in viewer_threads if thread.is_alive()]
    if running_threads:
        raise RuntimeError(
            f"MuJoCo viewer threads did not stop cleanly: {running_threads}"
        )


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    import mujoco.viewer

    threads_before_launch = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - threads_before_launch

    try:
        return run_task(model, data, viewer)
    finally:
        viewer.close()
        wait_for_viewer_threads(viewer_threads)


def main() -> None:
    args = parse_args()
    model, data = load_pushing_scene()

    if args.headless:
        succeeded = run_task(model, data, None)
    else:
        succeeded = run_viewer(model, data)

    if not succeeded:
        raise RuntimeError("Pushing experiment did not complete successfully.")


if __name__ == "__main__":
    main()
