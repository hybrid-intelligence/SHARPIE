"""
Template for an Environment implementation.

This is a basic template showing the expected structure for an environment file.
Your environment should define a variable called `environment` with reset, step, and render methods.
"""


class EnvironmentWrapper:
    """A basic environment wrapper template."""

    def __init__(self, env):
        """Initialize the environment."""
        self.env = env

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation
            info: Additional information (optional)
        """
        observation, info = self.env.reset()
        return observation, info

    def step(self, action):
        """
        Execute one step in the environment. Because this is a multi-agent environment, the action will be a dictionary with agent id as keys and their corresponding actions as values.

        Args:
            action: Action to take

        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated
            info: Additional information
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (or None for text-based rendering)
        """
        image = self.env.render()
        return image
    

environment = EnvironmentWrapper(env=None)  # Replace with actual environment initialization