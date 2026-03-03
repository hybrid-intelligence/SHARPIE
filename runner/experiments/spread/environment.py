from pettingzoo.mpe import simple_spread_v3
from collections.abc import Iterable



class EnvironmentWrapper:
    """Wrapper for the simple_spread_v3 PettingZoo environment."""

    def __init__(self):
        """Initialize the environment."""
        self.env = simple_spread_v3.parallel_env(max_cycles=200, render_mode="rgb_array")

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation (dict with agent names as keys)
            info: Additional information
        """
        observation, info = self.env.reset()
        return observation, info

    def step(self, actions):
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary with agent id as keys and their corresponding actions as values

        Returns:
            observation: New observation (dict with agent names as keys)
            reward: Reward for the action (dict with agent names as keys)
            terminated: Whether each agent's episode has ended (dict)
            truncated: Whether the episode was truncated for each agent (dict)
            info: Additional information (dict)
        """
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, all(a == 0 for a in terminated), all(a == 0 for a in truncated), info

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (numpy array)
        """
        image = self.env.render()
        return image


environment = EnvironmentWrapper()