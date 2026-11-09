# Plan de construction — environnement SO-101

Design de référence : `docs/superpowers/specs/2026-11-09-environnement-so101-design.md`

Principes : une tâche = un commit ; tests écrits avec le code (pytest) ;
code court et commenté en français, noms en anglais ; rien dans
`src/kinematics/` ni `src/rl/` (réservés à l'utilisateur).

## Tâche 1 — Outillage Python

- `brew install uv`.
- `uv init` → `pyproject.toml` (Python 3.12, paquet `so101_sim` sous `src/`).
- Dépendances : `mujoco`, `gymnasium`, `numpy` ; dev : `pytest`.
- Créer `src/kinematics/` et `src/rl/` avec un `README.md` d'une ligne chacun
  (« réservé à l'étape 2/3 »).
- Vérification : `uv run python -c "import mujoco, gymnasium; print(mujoco.__version__)"`.
- Commit : « Outillage : uv, dépendances, arborescence ».

## Tâche 2 — Import du modèle officiel SO-101

- Clone superficiel de `TheRobotStudio/SO-ARM100` dans un dossier temporaire.
- Copier dans `assets/so101/` : `so101_new_calib.xml`, les fichiers XML qu'il
  inclut (ex. `joints_properties.xml`) et uniquement les maillages référencés.
- `assets/so101/README.md` : source, hash du commit, licence (copier le fichier
  LICENCE si présent), liste des modifications locales.
- Test `tests/test_model.py` : le modèle se charge ; 6 articulations attendues
  (5 bras + pince) avec leurs noms ; actionneurs présents.
- Commit.

## Tâche 3 — Scène

- `assets/scene.xml` : inclut le modèle ; sol, table, éclairage, caméra par
  défaut lisible.
- Cible : corps `mocap` avec une sphère visuelle (contype/conaffinity = 0).
- Site `gripper_tip` entre les mors de la pince (seule modification locale du
  fichier officiel, notée dans le README du modèle).
- Keyframe `home` : bras replié au repos.
- Tests : la scène se charge ; la cible n'entre pas en collision ; quelques
  centaines de pas au repos restent stables (pas de NaN, bras immobile).
- Commit.

## Tâche 4 — `robot.py` (couche bras)

Classe `SO101` :
- `__init__(scene_path)`, `reset()` (keyframe `home`).
- `joint_names`, `joint_limits`.
- `qpos()` / `qvel()` des 5 articulations du bras.
- `set_joint_targets(q)` (consignes de position, bornées aux limites).
- `open_gripper()` / `close_gripper()` / `set_gripper(valeur)`.
- `step(n=1)`.
- `gripper_position()` (site `gripper_tip`), `set_target(p)`, `target_position()`.
- Tests : bornage des consignes ; après consigne + pas, les angles convergent
  vers la consigne ; `set_target` déplace la cible ; ouvrir/fermer change
  l'angle de la pince.
- Commit.

## Tâche 5 — `scripts/view.py` (étape 1, interactif)

- Visualiseur passif MuJoCo, curseurs des actionneurs, affichage en continu de
  la position de la pince (texte en overlay ou terminal).
- Si lancé sans `mjpython` sur macOS : message clair et sortie.
- Vérification manuelle : `uv run mjpython scripts/view.py`.
- Commit.

## Tâche 6 — `scripts/demo_joints.py` (étape 1, scripté)

- Sans fenêtre : parcourt quelques configurations articulaires (repos, chaque
  articulation seule, combinaisons), attend la stabilisation, affiche un tableau
  angles → position de pince.
- Commit.

## Tâche 7 — `reach_env.py` (environnement Gymnasium)

- `ReachEnv(gymnasium.Env)` construit sur `SO101`.
- Action : `Box(-1, 1, (5,))` → variation de consigne (amplitude max par pas
  à fixer, ex. 0,05 rad), plusieurs sous-pas physiques par action.
- Observation : angles (5), vitesses (5), position pince (3), position cible (3).
- `compute_reward(gripper_pos, target_pos)` isolée, naïve, commentée
  « à repenser à l'étape 3 ».
- Fin : `terminated` si distance < seuil (≈ 2 cm) ; `truncated` après N pas.
- `reset(seed)` : repos + cible tirée dans une zone atteignable devant le bras.
  Valider la zone en vérifiant que des configurations aléatoires du bras y
  amènent bien la pince.
- `render_mode="human"` : visualiseur passif, exige `mjpython` (message clair).
- `info` : distance, succès.
- Tests : `gymnasium.utils.env_checker.check_env` ; reproductibilité avec
  seed ; récompense cohérente avec la distance ; épisode tronqué au bon pas.
- Enregistrement Gymnasium (`gymnasium.register("SO101Reach-v0", ...)`).
- Commit.

## Tâche 8 — Documentation

- README : installation (`uv sync`), lancer le visualiseur, la démo, les
  tests, utiliser l'environnement en 10 lignes (boucle avec actions
  aléatoires), où coder la cinématique inverse et l'apprentissage par
  renforcement, rappel `mjpython` sur Mac.
- Vérification finale : `uv run pytest` vert, les deux scripts fonctionnent.
- Commit.
