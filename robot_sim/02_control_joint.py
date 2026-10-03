import argparse
import math
import threading
import time

import mujoco

from panda_common import load_panda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move Panda joint1 with a position-control target."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Simulation duration in seconds (default: 8).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def update_control(data: mujoco.MjData) -> None:
    amplitude = 0.45
    frequency_hz = 0.25
    data.ctrl[0] = amplitude * math.sin(2.0 * math.pi * frequency_hz * data.time)


def print_progress(data: mujoco.MjData) -> None:
    print(
        f"t={data.time:5.2f} s  "
        f"joint1={data.qpos[0]: .3f} rad  "
        f"target={data.ctrl[0]: .3f} rad  "
        f"hand=({data.body('hand').xpos[0]: .3f}, "
        f"{data.body('hand').xpos[1]: .3f}, "
        f"{data.body('hand').xpos[2]: .3f}) m"
    )


def run_headless(
    model: mujoco.MjModel, data: mujoco.MjData, duration: float
) -> None:
    next_report_time = 0.0
    while data.time < duration:
        update_control(data)
        mujoco.mj_step(model, data)
        if data.time >= next_report_time:
            print_progress(data)
            next_report_time += 1.0


def run_viewer(
    model: mujoco.MjModel, data: mujoco.MjData, duration: float
) -> None:
    import mujoco.viewer

    threads_before_launch = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - threads_before_launch

    try:
        next_report_time = 0.0
        while viewer.is_running() and data.time < duration:
            step_start = time.time()
            update_control(data)
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_report_time:
                print_progress(data)
                next_report_time += 1.0

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join(timeout=5.0)

        running_threads = [thread.name for thread in viewer_threads if thread.is_alive()]
        if running_threads:
            raise RuntimeError(
                f"MuJoCo viewer threads did not stop cleanly: {running_threads}"
            )


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    model, data = load_panda()
    print("Controlling joint1 with a smooth position target.")

    if args.headless:
        run_headless(model, data, args.duration)
    else:
        run_viewer(model, data, args.duration)

    print("Simulation finished.")


if __name__ == "__main__":
    main()
