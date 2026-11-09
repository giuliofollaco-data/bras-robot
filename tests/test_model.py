"""Le modèle officiel du SO-101 se charge et a la structure attendue."""

from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).parent.parent / "assets" / "so101" / "so101_new_calib.xml"

EXPECTED_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def test_model_loads():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    assert joint_names == EXPECTED_JOINTS


def test_one_position_actuator_per_joint():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert actuator_names == EXPECTED_JOINTS
