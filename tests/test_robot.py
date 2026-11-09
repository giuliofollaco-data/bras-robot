"""Tests de la couche bras (robot.py)."""

import numpy as np
import pytest

from so101_sim.robot import SO101


@pytest.fixture
def robot():
    return SO101()


def test_state_shapes(robot):
    assert robot.qpos().shape == (5,)
    assert robot.qvel().shape == (5,)
    assert robot.gripper_position().shape == (3,)
    assert robot.joint_limits.shape == (5, 2)


def test_targets_are_clipped_to_limits(robot):
    applied = robot.set_joint_targets([10, -10, 10, -10, 10])
    assert np.all(applied <= robot.joint_limits[:, 1])
    assert np.all(applied >= robot.joint_limits[:, 0])
    np.testing.assert_allclose(robot.joint_targets(), applied)


def test_joints_converge_to_targets(robot):
    target = np.array([0.5, -1.0, 1.0, 0.5, 0.3])
    robot.set_joint_targets(target)
    robot.step(1000)  # 2 secondes
    np.testing.assert_allclose(robot.qpos(), target, atol=0.05)


def test_positive_shoulder_pan_turns_arm_right(robot):
    # Convention du modèle officiel : shoulder_pan > 0 fait tourner le bras
    # vers sa droite (y négatif).
    y_before = robot.gripper_position()[1]
    q = robot.qpos()
    q[0] = 0.8
    robot.set_joint_targets(q)
    robot.step(1000)
    assert robot.gripper_position()[1] < y_before - 0.05


def test_set_target_and_distance(robot):
    robot.set_target([0.2, 0.1, 0.15])
    np.testing.assert_allclose(robot.target_position(), [0.2, 0.1, 0.15])
    expected = np.linalg.norm(robot.gripper_position() - np.array([0.2, 0.1, 0.15]))
    assert robot.distance_to_target() == pytest.approx(expected)


def test_gripper_opens_and_closes(robot):
    robot.open_gripper()
    robot.step(500)
    opened = robot.gripper_angle()
    robot.close_gripper()
    robot.step(500)
    closed = robot.gripper_angle()
    assert opened > closed + 1.0


def test_reset_returns_home(robot):
    robot.set_joint_targets([0.5, 0, 0, 0, 0])
    robot.step(500)
    robot.reset()
    np.testing.assert_allclose(robot.qpos(), [0, -1.57, 1.57, 0.8, 0], atol=1e-9)
