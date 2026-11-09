"""Tâche « atteindre la cible », au format standard Gymnasium.

À chaque épisode, le bras part du repos et une cible apparaît au hasard devant
lui. L'agent doit amener le bout de la pince à moins de 2 cm de la cible.

Boucle typique :

    env = ReachEnv()
    obs, info = env.reset(seed=0)
    while True:
        action = env.action_space.sample()      # ici : votre algorithme
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

import time

import gymnasium as gym
import numpy as np

from so101_sim.robot import SO101
from so101_sim.viewer import launch_viewer

# Zone où la cible peut apparaître (m) : devant le bras, au-dessus de la table.
# Vérifiée atteignable en échantillonnant des configurations aléatoires du bras.
TARGET_LOW = np.array([0.15, -0.15, 0.05])
TARGET_HIGH = np.array([0.35, 0.15, 0.25])

SUCCESS_DISTANCE = 0.02  # m : la cible est atteinte sous cette distance
MAX_DELTA = 0.05  # rad : variation maximale d'une consigne par pas d'action
FRAME_SKIP = 20  # pas physiques (2 ms) par action -> l'agent agit à 25 Hz
MAX_EPISODE_STEPS = 200  # 200 actions = 8 secondes simulées


def compute_reward(gripper_pos, target_pos):
    """Récompense d'un pas.

    VERSION DE BASE, VOLONTAIREMENT NAÏVE — à repenser à l'étape 3.
    Moins la distance pince-cible (plus on est près, moins c'est négatif),
    plus un bonus quand la cible est atteinte.
    Pistes à explorer : pénaliser les mouvements brusques, récompenser le
    progrès plutôt que la distance, changer l'échelle, retirer le bonus...
    """
    distance = np.linalg.norm(gripper_pos - target_pos)
    reward = -distance
    if distance < SUCCESS_DISTANCE:
        reward += 10.0
    return float(reward)


class ReachEnv(gym.Env):
    """Amener le bout de la pince du SO-101 sur une cible aléatoire.

    Action (5 nombres dans [-1, 1]) : variation des consignes d'angle des
        5 articulations du bras, multipliée par MAX_DELTA. La pince n'est
        pas pilotée dans cette tâche.
    Observation (16 nombres) : angles (5), vitesses (5), position de la
        pince (3), position de la cible (3).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1 / (FRAME_SKIP * 0.002)}

    def __init__(self, render_mode=None):
        self.robot = SO101()
        self.render_mode = render_mode
        self._viewer = None
        self._steps = 0

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

        if render_mode == "human":
            self._viewer = launch_viewer(self.robot.model, self.robot.data)

    def _observation(self):
        r = self.robot
        return np.concatenate(
            [r.qpos(), r.qvel(), r.gripper_position(), r.target_position()]
        ).astype(np.float32)

    def _info(self):
        distance = self.robot.distance_to_target()
        return {"distance": distance, "is_success": distance < SUCCESS_DISTANCE}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # initialise self.np_random (reproductibilité)
        self.robot.reset()
        self.robot.set_target(self.np_random.uniform(TARGET_LOW, TARGET_HIGH))
        self._steps = 0
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        # Nouvelle consigne = consigne actuelle + petite variation.
        self.robot.set_joint_targets(self.robot.joint_targets() + MAX_DELTA * action)
        self.robot.step(FRAME_SKIP)
        self._steps += 1

        reward = compute_reward(self.robot.gripper_position(), self.robot.target_position())
        info = self._info()
        terminated = info["is_success"]
        truncated = self._steps >= MAX_EPISODE_STEPS

        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self._viewer is None:
            return
        self._viewer.sync()
        time.sleep(1 / self.metadata["render_fps"])  # vitesse temps réel

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
