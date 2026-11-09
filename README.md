# virtual-robot-arm

Simulateur d'un bras robot SO-101 (Hugging Face / The Robot Studio) avec MuJoCo,
pour apprendre la cinématique inverse puis l'apprentissage par renforcement
avant de passer sur un vrai bras.

## Feuille de route

1. Cinématique directe : piloter les articulations à la main
2. Cinématique inverse : amener la pince sur une cible
3. Apprentissage par renforcement : même tâche, apprise
4. Attraper et déplacer un cube

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync
```

`uv` installe Python 3.12 et les dépendances (`mujoco`, `gymnasium`, `numpy`)
dans `.venv/`, sans toucher au Python du système.

Toutes les commandes se lancent **depuis la racine du projet**.

## Étape 1 — découvrir le bras

Fenêtre 3D interactive : un curseur par moteur (panneau de droite, section
« Control »), position du bout de la pince affichée en direct.

```bash
uv run mjpython -m scripts.view
```

> Sur macOS, la fenêtre 3D exige `mjpython` (fourni avec MuJoCo) au lieu de
> `python`. Sans fenêtre, `python` suffit.

Tableau « angles des articulations → position de la pince » (sans fenêtre) :

```bash
uv run python -m scripts.demo_joints
```

## Utiliser le bras depuis votre code

```python
from so101_sim.robot import SO101

robot = SO101()
robot.set_joint_targets([0.0, -1.0, 0.5, 0.5, 0.0])  # 5 angles (rad)
robot.step(1000)                                      # 2 s simulées
print(robot.qpos(), robot.gripper_position())         # angles, position (m)
```

Repère : x vers l'avant du bras, y vers sa gauche, z vers le haut, dessus de la
table à z = 0. Attention : `shoulder_pan > 0` tourne le bras vers sa **droite**.

## L'environnement d'apprentissage

```python
import gymnasium
import so101_sim  # enregistre l'environnement

env = gymnasium.make("SO101Reach-v0")  # render_mode="human" pour voir (mjpython)
obs, info = env.reset(seed=0)
for _ in range(200):
    action = env.action_space.sample()  # remplacez par votre algorithme
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

- **Action** : 5 nombres dans [-1, 1], variation des consignes d'angle.
- **Observation** : angles (5), vitesses (5), position pince (3), position cible (3).
- **Récompense** : `compute_reward` dans [so101_sim/reach_env.py](so101_sim/reach_env.py),
  version naïve à repenser à l'étape 3.
- **Fin d'épisode** : pince à moins de 2 cm de la cible, ou 200 pas (8 s).

## Organisation

```
assets/scene.xml        la scène : table, cible, position de repos
assets/so101/           modèle officiel du SO-101 (Apache 2.0, voir son README)
so101_sim/robot.py      le bras : état, consignes, pince, cible
so101_sim/reach_env.py  la tâche « atteindre la cible » (Gymnasium)
scripts/                view.py, demo_joints.py
kinematics/             étape 2 — votre cinématique inverse
rl/                     étape 3 — votre apprentissage par renforcement
tests/                  uv run pytest
```

Pour vos propres scripts dans `kinematics/` ou `rl/`, lancez-les aussi en
module depuis la racine : `uv run python -m kinematics.mon_script`.

## Tests

```bash
uv run pytest
```
