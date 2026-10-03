import argparse
import queue
import threading
import time

import glfw
import mujoco
import numpy as np

from panda_common import load_panda


JOINT_STEP_RAD = 0.05
ARM_JOINT_COUNT = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control Panda arm joints from the MuJoCo viewer keyboard."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Wall-clock duration in seconds; 0 runs until window close.",
    )
    return parser.parse_args()


def print_help() -> None:
    print(
        "\nKeyboard controls (focus the MuJoCo window):\n"
        "  Up / Down       select the previous / next arm joint\n"
        "  Left / Right    decrease / increase its target by 0.05 rad\n"
        "  Insert / Delete open / close the gripper\n"
        "  Home            return all joints to the home target\n"
        "  Space    pause / resume physics\n"
        "  Close the viewer window to quit\n"
    )


def clamp_joint_target(
    model: mujoco.MjModel, joint_index: int, target: float
) -> float:
    lower, upper = model.actuator_ctrlrange[joint_index]
    return float(np.clip(target, lower, upper))


def wait_for_viewer_threads(viewer_threads: set[threading.Thread]) -> None:
    for thread in viewer_threads:
        thread.join(timeout=5.0)

    running_threads = [thread.name for thread in viewer_threads if thread.is_alive()]
    if running_threads:
        raise RuntimeError(
            f"MuJoCo viewer threads did not stop cleanly: {running_threads}"
        )


def main() -> None:
    args = parse_args()
    if args.duration < 0:
        raise ValueError("--duration cannot be negative.")

    model, data = load_panda()
    home_target = data.ctrl.copy()
    target = home_target.copy()
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()

    def key_callback(keycode: int) -> None:
        key_events.put(keycode)

    import mujoco.viewer

    threads_before_launch = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    )
    viewer_threads = set(threading.enumerate()) - threads_before_launch

    selected_joint = 0
    paused = False
    keep_running = True
    started_at = time.monotonic()
    next_report_time = 0.0

    print_help()
    print("Selected joint1.")

    try:
        while viewer.is_running() and keep_running:
            step_started_at = time.monotonic()

            while True:
                try:
                    keycode = key_events.get_nowait()
                except queue.Empty:
                    break

                if keycode in {glfw.KEY_UP, glfw.KEY_DOWN}:
                    direction = -1 if keycode == glfw.KEY_UP else 1
                    selected_joint = (
                        selected_joint + direction
                    ) % ARM_JOINT_COUNT
                    print(
                        f"Selected joint{selected_joint + 1}; "
                        f"target={target[selected_joint]: .3f} rad."
                    )
                elif keycode in {glfw.KEY_LEFT, glfw.KEY_RIGHT}:
                    direction = -1.0 if keycode == glfw.KEY_LEFT else 1.0
                    requested_target = (
                        target[selected_joint] + direction * JOINT_STEP_RAD
                    )
                    target[selected_joint] = clamp_joint_target(
                        model, selected_joint, requested_target
                    )
                    print(
                        f"joint{selected_joint + 1}: "
                        f"actual={data.qpos[selected_joint]: .3f} rad, "
                        f"target={target[selected_joint]: .3f} rad"
                    )
                elif keycode == glfw.KEY_INSERT:
                    target[7] = model.actuator_ctrlrange[7, 1]
                    print("Gripper target: open.")
                elif keycode == glfw.KEY_DELETE:
                    target[7] = model.actuator_ctrlrange[7, 0]
                    print("Gripper target: closed.")
                elif keycode == glfw.KEY_HOME:
                    target[:] = home_target
                    print("All targets reset to the home pose.")
                elif keycode == glfw.KEY_SPACE:
                    paused = not paused
                    print("Physics paused." if paused else "Physics resumed.")

            if not paused:
                data.ctrl[:] = target
                mujoco.mj_step(model, data)

                if data.time >= next_report_time:
                    hand_position = data.body("hand").xpos
                    print(
                        f"t={data.time:5.2f} s  "
                        f"joint{selected_joint + 1}="
                        f"{data.qpos[selected_joint]: .3f}/"
                        f"{target[selected_joint]: .3f} rad  "
                        f"hand=({hand_position[0]: .3f}, "
                        f"{hand_position[1]: .3f}, "
                        f"{hand_position[2]: .3f}) m"
                    )
                    next_report_time += 1.0

            viewer.sync()

            if args.duration and time.monotonic() - started_at >= args.duration:
                keep_running = False

            remaining = model.opt.timestep - (
                time.monotonic() - step_started_at
            )
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()
        wait_for_viewer_threads(viewer_threads)

    print("Keyboard control finished.")


if __name__ == "__main__":
    main()
