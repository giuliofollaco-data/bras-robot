"""Accès bas niveau au bras SO-101 simulé.

Cette couche ne connaît que le bras et la cible : pas de récompense, pas
d'épisode. C'est elle que la cinématique inverse (étape 2) utilisera, et sur
laquelle l'environnement d'apprentissage (reach_env.py) est construit.

Toutes les positions sont en mètres, dans le repère du monde :
x vers l'avant du bras, y vers sa gauche, z vers le haut (table à z = 0).
Tous les angles sont en radians. Attention : shoulder_pan > 0 fait tourner
le bras vers sa droite (y négatif), c'est la convention du modèle officiel.
"""

from pathlib import Path

import mujoco
import numpy as np

DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "assets" / "scene.xml"

# Les 5 articulations du bras, de la base vers la pince.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"


class SO101:
    """Le bras SO-101 dans sa scène, piloté en position.

    Chaque articulation a un moteur « position » : on lui donne un angle
    consigne et il s'en rapproche, comme le ferait un vrai servomoteur STS3215.
    """

    def __init__(self, scene_path=DEFAULT_SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Indices utiles, calculés une fois pour toutes.
        self._arm_qpos = np.array([self.model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self._arm_qvel = np.array([self.model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self._arm_ctrl = np.array([self.model.actuator(j).id for j in ARM_JOINTS])
        self._gripper_ctrl = self.model.actuator(GRIPPER_JOINT).id
        self._gripper_qpos = self.model.joint(GRIPPER_JOINT).qposadr[0]
        self._tip_site = self.model.site("gripperframe").id
        self._target_mocap = self.model.body("target").mocapid[0]
        self._home_key = self.model.key("home").id

        self.joint_names = list(ARM_JOINTS)
        # Limites [min, max] de chaque articulation du bras, forme (5, 2).
        self.joint_limits = np.array([self.model.joint(j).range for j in ARM_JOINTS])
        self.gripper_limits = self.model.joint(GRIPPER_JOINT).range.copy()

        self.reset()

    # --- Cycle de simulation -------------------------------------------------

    def reset(self):
        """Remet le bras en position de repos (keyframe « home »)."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        """Avance la simulation de n pas physiques (2 ms chacun)."""
        mujoco.mj_step(self.model, self.data, nstep=n)

    @property
    def timestep(self):
        return self.model.opt.timestep

    # --- État du bras ---------------------------------------------------------

    def qpos(self):
        """Angles actuels des 5 articulations du bras (rad)."""
        return self.data.qpos[self._arm_qpos].copy()

    def qvel(self):
        """Vitesses actuelles des 5 articulations du bras (rad/s)."""
        return self.data.qvel[self._arm_qvel].copy()

    def joint_targets(self):
        """Consignes actuellement envoyées aux 5 moteurs du bras (rad)."""
        return self.data.ctrl[self._arm_ctrl].copy()

    def gripper_position(self):
        """Position (x, y, z) du bout de la pince, en mètres.

        C'est le repère « gripperframe » fourni par le modèle officiel.
        """
        return self.data.site_xpos[self._tip_site].copy()

    # --- Commandes ------------------------------------------------------------

    def set_joint_targets(self, q):
        """Envoie des consignes d'angle aux 5 moteurs du bras.

        Les consignes hors limites sont ramenées dans les limites. Le bras
        ne bouge qu'en appelant ensuite step().
        """
        q = np.clip(np.asarray(q, dtype=float), self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.data.ctrl[self._arm_ctrl] = q
        return q

    def set_gripper(self, value):
        """Consigne d'angle de la pince (rad) : petit = fermée, grand = ouverte."""
        lo, hi = self.gripper_limits
        self.data.ctrl[self._gripper_ctrl] = float(np.clip(value, lo, hi))

    def open_gripper(self):
        self.set_gripper(self.gripper_limits[1])

    def close_gripper(self):
        self.set_gripper(self.gripper_limits[0])

    def gripper_angle(self):
        """Angle actuel de la pince (rad)."""
        return float(self.data.qpos[self._gripper_qpos])

    # --- Cible ----------------------------------------------------------------

    def set_target(self, p):
        """Place la cible (sphère rouge) à la position p = (x, y, z)."""
        self.data.mocap_pos[self._target_mocap] = p
        mujoco.mj_forward(self.model, self.data)

    def target_position(self):
        return self.data.mocap_pos[self._target_mocap].copy()

    def distance_to_target(self):
        """Distance (m) entre le bout de la pince et la cible."""
        return float(np.linalg.norm(self.gripper_position() - self.target_position()))
