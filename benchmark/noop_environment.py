"""
No-op environment for measuring SHARPIE infrastructure overhead.

This minimal environment generates random noise images and discards all actions,
allowing us to measure the overhead of the WebSocket communication,
message serialization, and framework infrastructure.
"""

import numpy as np


class NoOpEnvironment:
    """Minimal environment for benchmarking infrastructure overhead."""

    def __init__(self, render_size=(64, 64, 3), max_steps=100):
        self.render_size = render_size
        self.step_count = 0
        self.max_steps = max_steps

    def reset(self):
        self.step_count = 0
        return {}, {}

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        return {}, 0, terminated, truncated, {"step": self.step_count}

    def render(self):
        return np.random.randint(0, 255, self.render_size, dtype=np.uint8)


# SHARPIE expects an `environment` variable
# This is the class - the runner will instantiate it with metadata params
environment = NoOpEnvironment