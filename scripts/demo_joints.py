"""Étape 1 — cinématique directe observée.

Lancement :  uv run python -m scripts.demo_joints

Pour une série de configurations du bras (angles des 5 articulations), on
envoie les consignes, on laisse le bras se stabiliser, puis on mesure où se
trouve le bout de la pince. Ce tableau « angles -> position » servira de
référence pour vérifier votre cinématique inverse à l'étape 2 : partir de la
position, retrouver des angles qui y mènent.
"""

import numpy as np

from so101_sim.robot import SO101

# (description, [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll])
CONFIGURATIONS = [
    ("repos", [0.0, -1.57, 1.57, 0.8, 0.0]),
    ("tout à zéro (milieu des plages)", [0.0, 0.0, 0.0, 0.0, 0.0]),
    ("base tournée +0.8", [0.8, 0.0, 0.0, 0.0, 0.0]),
    ("base tournée -0.8", [-0.8, 0.0, 0.0, 0.0, 0.0]),
    ("épaule levée -1.0", [0.0, -1.0, 0.0, 0.0, 0.0]),
    ("coude plié +1.0", [0.0, 0.0, 1.0, 0.0, 0.0]),
    ("poignet plié +1.0", [0.0, 0.0, 0.0, 1.0, 0.0]),
    ("rotation poignet +1.5", [0.0, 0.0, 0.0, 0.0, 1.5]),
    ("vers le haut", [0.0, -1.0, 0.5, 0.5, 0.0]),
    ("devant, bas", [0.3, -0.5, 1.0, 1.0, 0.0]),
]


def settle(robot, q, seconds=2.0):
    """Envoie la consigne q et laisse le bras s'y stabiliser."""
    robot.set_joint_targets(q)
    robot.step(int(seconds / robot.timestep))


def main():
    robot = SO101()
    np.set_printoptions(precision=2, suppress=True)
    print(f"{'configuration':<34}{'angles mesurés (rad)':<40}pince x, y, z (m)")
    print("-" * 100)
    for description, q in CONFIGURATIONS:
        robot.reset()
        settle(robot, q)
        angles = np.array2string(robot.qpos(), precision=2, suppress_small=True)
        x, y, z = robot.gripper_position()
        print(f"{description:<34}{angles:<40}{x:+.3f}, {y:+.3f}, {z:+.3f}")


if __name__ == "__main__":
    main()
