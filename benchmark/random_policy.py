"""
Random policy for AI agent benchmarking.

This policy generates random actions, useful for measuring the overhead
of running multiple AI agents with policy execution.
"""

import numpy as np


class RandomPolicy:
    """Policy that generates random actions."""

    def __init__(self, action_space=None, seed=None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, **kwargs):
        """Return a random action."""
        # For discrete actions, return random integer 0-3
        # For continuous, return random float
        return int(self.rng.integers(0, 4))

    def update(self, *args, **kwargs):
        """No learning for random policy."""
        pass


# SHARPIE expects a `policy` variable
policy = RandomPolicy()