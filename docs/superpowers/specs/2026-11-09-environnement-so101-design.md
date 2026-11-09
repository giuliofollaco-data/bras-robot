# Environnement de simulation SO-101 — design

Date : 2026-11-09

## Intention

Apprendre la robotique (cinématique inverse puis apprentissage par renforcement)
sur un simulateur avant d'acheter un vrai bras SO-101. Le projet est un support
d'apprentissage : la lisibilité et la compréhension priment sur la performance
ou l'exhaustivité. L'utilisateur code lui-même la cinématique inverse et les
algorithmes d'apprentissage par renforcement ; ce design couvre la **première
couche** : l'environnement simulé qui servira de base à tout le reste.

## Feuille de route (contexte)

1. Cinématique directe : piloter les articulations à la main, observer la pince.
2. Cinématique inverse : amener la pince sur une cible (codé par l'utilisateur).
3. Apprentissage par renforcement : même tâche, apprise (codé par l'utilisateur).
4. Attraper et déplacer un cube.

Ce design couvre l'environnement nécessaire aux étapes 1 à 3, en laissant la
porte ouverte à l'étape 4.

## Objectifs

- Un bras SO-101 simulé physiquement (MuJoCo), posé sur une table.
- Une cible (petite sphère visuelle, sans collision) placée aléatoirement dans
  l'espace atteignable du bras à chaque réinitialisation.
- Une API Python simple pour : lire l'état (angles, vitesses articulaires,
  position de la pince), envoyer des consignes de position articulaire,
  avancer la simulation, mesurer la distance pince–cible.
- Une interface Gymnasium (`reset`, `step`, espaces d'observation et d'action,
  récompense) pour la tâche « atteindre la cible ».
- Un visualiseur 3D (fenêtre MuJoCo) pour regarder et manipuler le bras.
- Un script de démonstration de l'étape 1 : bouger les articulations et
  afficher la position de la pince.

## Non-objectifs

- Écrire un moteur physique : MuJoCo s'en charge.
- Fournir la cinématique inverse ou les algorithmes d'apprentissage : ce sont
  les exercices de l'utilisateur. Le projet prévoit seulement leur emplacement.
- Le cube et la préhension (étape 4) : plus tard, sans refonte.
- Caméras, rendu d'images comme observation, transfert vers le vrai robot.
- Performance d'entraînement (simulation parallèle, GPU).

## Anti-objectifs

- Une boîte noire : si l'utilisateur ne peut pas lire et comprendre chaque
  fichier de l'environnement, c'est un échec même si ça marche.
- Dépendre de LeRobot ou d'un environnement tout fait.

## Contraintes

- Mac Apple Silicon (arm64). Python système 3.9 trop ancien → Python récent
  géré par `uv` (installé via Homebrew), environnement isolé dans le projet.
- Sur macOS, le visualiseur interactif MuJoCo se lance avec `mjpython`.
- Le SO-101 a 5 articulations + une pince : il ne peut pas atteindre une pose
  6D arbitraire. La tâche porte donc sur la **position** de la pince, pas son
  orientation, et les cibles sont tirées dans l'espace atteignable.
- Dépôt git, destiné à être publié sur GitHub.

## Approches considérées

- **A. Environnement maison sur MuJoCo, au format Gymnasium** — choisie.
  Tout le code est lisible, c'est l'objet même de l'apprentissage, et le format
  standard permet plus tard de comparer l'algorithme de l'utilisateur à une
  implémentation de référence.
- B. Réutiliser les simulations LeRobot — rejetée : boîte noire pensée pour
  d'autres usages, on apprend l'outil plutôt que la robotique.
- C. MuJoCo seul, sans Gymnasium — rejetée : perd la compatibilité avec les
  tutoriels et bibliothèques d'apprentissage par renforcement.

## Décisions prises avec l'utilisateur

- Moteur physique MuJoCo, en Python.
- Approche A.
- Progression en 4 étapes ci-dessus ; première tâche = atteindre une cible.
- Suivi avec git, publication éventuelle sur GitHub.

## Design

### Modèle du robot

Le modèle MJCF officiel vient du dépôt de The Robot Studio
(`TheRobotStudio/SO-ARM100`, dossier `Simulation/SO101`) : fichier
`so101_new_calib.xml` et ses maillages (`assets/`). Convention « new calib » :
le zéro de chaque articulation est au milieu de sa plage, cohérent avec LeRobot.
Les fichiers sont copiés dans le dépôt (pas de sous-module), avec mention de la
source, du commit d'origine et de la licence dans un README du dossier.

Notre propre `scene.xml` inclut ce modèle et ajoute : sol, table, éclairage,
la cible (site/géom visuel sans collision, déplacé via `mocap`), et un site à
l'extrémité de la pince pour mesurer sa position.

### Organisation du code (proposition)

```
assets/so101/          modèle officiel copié (+ README source/licence)
assets/scene.xml       scène : table, cible, éclairage
src/so101_sim/
  robot.py             accès bas niveau : état, consignes, pas de simulation
  reach_env.py         environnement Gymnasium « atteindre la cible »
src/kinematics/        étape 2 — vide, réservé à l'utilisateur
src/rl/                étape 3 — vide, réservé à l'utilisateur
scripts/
  view.py              ouvrir la scène dans le visualiseur
  demo_joints.py       étape 1 : bouger les articulations, afficher la pince
tests/                 tests de base de l'environnement
```

Deux couches séparées : `robot.py` (le bras, réutilisable par la cinématique
inverse sans notion de récompense) et `reach_env.py` (la tâche pour
l'apprentissage par renforcement).

### Environnement « atteindre la cible »

- **Action** : variation des consignes de position des 5 articulations du bras,
  bornée, normalisée dans [-1, 1]. La pince est hors de l'espace d'action pour
  cette tâche seulement (inutile pour atteindre un point, et chaque dimension
  d'action en plus ralentit l'apprentissage) ; elle reste entièrement modélisée
  et pilotable via `robot.py` (ouvrir/fermer), et la tâche de l'étape 4 l'ajoutera
  à l'action.
- **Observation** : angles et vitesses des 5 articulations, position de la
  pince, position de la cible.
- **Récompense** : convention Gymnasium, `step` renvoie une récompense, donc
  l'environnement en a une dès le départ. Elle est isolée dans une seule petite
  fonction `compute_reward(...)`, version de base volontairement naïve (moins la
  distance pince–cible, plus un bonus sous un seuil, ex. 2 cm), commentée comme
  « à repenser à l'étape 3 » : concevoir la récompense fait partie des
  exercices d'apprentissage par renforcement de l'utilisateur.
- **Épisode** : se termine en succès sous le seuil, ou après un nombre maximal
  de pas.
- **Réinitialisation** : bras en position de repos, cible tirée aléatoirement
  dans une zone atteignable devant le bras (au-dessus de la table).
- Rendu optionnel (`render_mode="human"`).

### Scripts de l'étape 1 (choix de Claude)

- `view.py` : interactif. Ouvre la scène dans le visualiseur ; l'utilisateur
  bouge les articulations avec les curseurs des actionneurs, et la position de
  la pince s'affiche en continu (overlay ou terminal). Se lance avec `mjpython`.
- `demo_joints.py` : scripté, sans fenêtre, lancé avec `python`. Envoie une
  série de consignes articulaires et affiche la position de pince obtenue :
  c'est la cinématique directe « observée », qui servira de référence pour
  vérifier la cinématique inverse de l'étape 2.

### Rendu et macOS (choix de Claude)

`render_mode="human"` utilise le visualiseur passif MuJoCo et exige donc
`mjpython` sur macOS. Lancé avec `python`, l'environnement échoue avec un
message clair indiquant la commande à utiliser. Pas de rendu de secours, pour
garder `reach_env.py` court. Sans rendu, tout fonctionne avec `python`.

### Environnement logiciel

`uv` + `pyproject.toml`. Dépendances : `mujoco`, `gymnasium`, `numpy` ;
`pytest` en développement. Python 3.12.

## Laissé au builder

- Valeurs numériques : pas de temps, nombre de sous-pas physiques par action,
  amplitude maximale d'une action, durée d'un épisode, seuil exact de succès,
  zone exacte de tirage de la cible.
- Dimensions et position de la table, emplacement exact du site de pince.
- Détail des tests (au minimum : la scène se charge, `reset`/`step` respectent
  l'API Gymnasium, la distance calculée est correcte).
- Langue du code : noms en anglais, commentaires et docs en français.
