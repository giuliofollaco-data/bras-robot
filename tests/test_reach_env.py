"""Tests de l'environnement « atteindre la cible »."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import so101_sim  # noqa: F401  (enregistre SO101Reach-v0)
from so101_sim.reach_env import (
    MAX_EPISODE_STEPS,
    SUCCESS_DISTANCE,
    TARGET_HIGH,
    TARGET_LOW,
    ReachEnv,
    compute_reward,
)


@pytest.fixture
def env():
    e = ReachEnv()
    yield e
    e.close()


def test_passes_gymnasium_checker(env):
    check_env(env, skip_render_check=True)


def test_make_by_name():
    e = gymnasium.make("SO101Reach-v0")
    obs, _ = e.reset(seed=0)
    assert obs.shape == (16,)
    e.close()


def test_target_is_in_zone_and_reproducible(env):
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    target = obs_a[13:16]
    assert np.all(target >= TARGET_LOW - 1e-6) and np.all(target <= TARGET_HIGH + 1e-6)


def test_reward_follows_distance():
    target = np.zeros(3)
    near = compute_reward(np.array([0.05, 0, 0]), target)
    far = compute_reward(np.array([0.20, 0, 0]), target)
    assert near > far
    assert compute_reward(np.array([SUCCESS_DISTANCE / 2, 0, 0]), target) > 0


def test_reward_matches_info_distance(env):
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(5))
    assert reward == pytest.approx(-info["distance"])


def test_episode_truncates_after_max_steps(env):
    env.reset(seed=0)
    env.robot.set_target([0.35, 0.15, 0.25])  # loin du repos : pas de succès par hasard
    for i in range(MAX_EPISODE_STEPS):
        _, _, terminated, truncated, _ = env.step(np.zeros(5))
        assert not terminated
        assert truncated == (i == MAX_EPISODE_STEPS - 1)


def test_episode_terminates_on_success(env):
    env.reset(seed=0)
    # On place la cible exactement sur la pince : succès dès le premier pas.
    env.robot.set_target(env.robot.gripper_position())
    _, reward, terminated, _, info = env.step(np.zeros(5))
    assert terminated and info["is_success"]
    assert reward > 0
