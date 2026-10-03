from pathlib import Path

import mujoco


ROBOT_SIM_ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROBOT_SIM_ROOT / "models" / "franka_emika_panda" / "scene.xml"


def load_panda() -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Panda model not found at {MODEL_PATH}. "
            "Make sure robot_sim/models/franka_emika_panda is present."
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home_keyframe_id == -1:
        raise RuntimeError("The Panda model does not define a 'home' keyframe.")

    mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
    mujoco.mj_forward(model, data)
    return model, data
