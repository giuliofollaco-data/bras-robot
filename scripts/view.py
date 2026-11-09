"""Étape 1 — piloter le bras à la main.

Lancement (macOS) :  uv run mjpython scripts/view.py

Dans la fenêtre : panneau de droite, section « Control », un curseur par
moteur. Bougez-les et regardez la position du bout de la pince, affichée en
haut à gauche (et dans le terminal).
Double-clic sur une pièce puis Ctrl + glisser : pousser le bras à la main.
"""

import time

import mujoco

from so101_sim.robot import SO101
from so101_sim.viewer import launch_viewer


def main():
    robot = SO101()
    viewer = launch_viewer(robot.model, robot.data)
    last_print = 0.0

    with viewer:
        while viewer.is_running():
            start = time.time()
            with viewer.lock():
                robot.step()

            x, y, z = robot.gripper_position()
            angles = "  ".join(
                f"{name}={angle:+.2f}" for name, angle in zip(robot.joint_names, robot.qpos())
            )
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Pince (m)\nDistance cible",
                    f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}\n{robot.distance_to_target():.3f} m",
                )
            )
            viewer.sync()

            if start - last_print > 0.5:
                print(f"\rpince=({x:+.3f}, {y:+.3f}, {z:+.3f})  {angles}", end="", flush=True)
                last_print = start

            # Garder le temps simulé aligné sur le temps réel.
            time.sleep(max(0.0, robot.timestep - (time.time() - start)))
    print()


if __name__ == "__main__":
    main()
