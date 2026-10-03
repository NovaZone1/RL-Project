import mujoco
import numpy as np

from panda_common import MODEL_PATH, load_panda


def format_vector(values: np.ndarray) -> str:
    return np.array2string(values, precision=4, suppress_small=True)


def main() -> None:
    model, data = load_panda()

    print(f"Model: {MODEL_PATH}")
    print(f"Dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("\nJoint map:")

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_address = model.jnt_qposadr[joint_id]
        dof_address = model.jnt_dofadr[joint_id]
        limited = bool(model.jnt_limited[joint_id])
        joint_range = model.jnt_range[joint_id] if limited else np.array([np.nan, np.nan])
        print(
            f"  {joint_name:>13}  qpos[{qpos_address}]  qvel[{dof_address}]  "
            f"range={format_vector(joint_range)}"
        )

    print("\nState at the home keyframe:")
    print(f"  qpos       = {format_vector(data.qpos)}")
    print(f"  qvel       = {format_vector(data.qvel)}")
    print(f"  ctrl       = {format_vector(data.ctrl)}")
    print(f"  hand.xpos  = {format_vector(data.body('hand').xpos)}")
    print(f"  hand.xquat = {format_vector(data.body('hand').xquat)}  (wxyz)")

    for _ in range(500):
        mujoco.mj_step(model, data)

    print("\nAfter one simulated second while holding the home target:")
    print(f"  time       = {data.time:.3f} s")
    print(f"  qpos       = {format_vector(data.qpos)}")
    print(f"  qvel       = {format_vector(data.qvel)}")
    print(f"  hand.xpos  = {format_vector(data.body('hand').xpos)}")


if __name__ == "__main__":
    main()

