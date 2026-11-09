"""La scène (bras + table + cible) se charge et reste stable au repos."""

from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = Path(__file__).parent.parent / "assets" / "scene.xml"


def load_home():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def test_scene_has_table_target_and_gripper_site():
    model, _ = load_home()
    assert model.geom("table") is not None
    assert model.body("target").mocapid[0] >= 0  # la cible est bien un corps mocap
    assert model.site("gripperframe") is not None


def test_target_cannot_collide():
    model, _ = load_home()
    target = model.geom("target")
    assert target.contype[0] == 0 and target.conaffinity[0] == 0


def test_home_pose_is_stable():
    model, data = load_home()
    q_home = data.qpos.copy()
    for _ in range(1000):  # 2 secondes simulées
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.abs(data.qpos - q_home).max() < 0.01
    assert data.ncon == 0  # le bras ne touche pas la table


def test_gripper_is_above_table_at_home():
    _, data = load_home()
    assert data.site("gripperframe").xpos[2] > 0.0
