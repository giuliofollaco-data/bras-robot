"""Simulation du bras SO-101 avec MuJoCo."""

import gymnasium

# Permet : gymnasium.make("SO101Reach-v0")
gymnasium.register(id="SO101Reach-v0", entry_point="so101_sim.reach_env:ReachEnv")
