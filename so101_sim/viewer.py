"""Ouverture de la fenêtre 3D MuJoCo, avec un message clair sur macOS."""

import sys

import mujoco
import mujoco.viewer

MJPYTHON_HELP = (
    "Sur macOS, la fenêtre 3D de MuJoCo exige de lancer le script avec "
    "`mjpython` au lieu de `python`, par exemple :\n"
    "    uv run mjpython -m scripts.view"
)


def launch_viewer(model, data):
    """Ouvre la fenêtre 3D « passive » : c'est notre code qui fait avancer la
    simulation, la fenêtre se contente d'afficher (appeler viewer.sync())."""
    try:
        return mujoco.viewer.launch_passive(model, data)
    except RuntimeError as error:
        if sys.platform == "darwin" and "mjpython" in str(error):
            raise RuntimeError(MJPYTHON_HELP) from None
        raise
