import gymnasium as gym


class EnvironmentWrapper:
    """Wrapper for the FrozenLake-v1 Gymnasium environment."""

    def __init__(self):
        """Initialize the environment."""
        self.env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation (numpy array)
            info: Additional information
        """
        observation, info = self.env.reset()
        return observation, info

    def step(self, action_dict):
        """
        Execute one step in the environment.

        Args:
            action_dict: Dictionary with agent id as keys and action as value

        Returns:
            observation: New observation (numpy array)
            reward: Reward for the action (float)
            terminated: Whether the episode has ended (bool)
            truncated: Whether the episode was truncated (bool)
            info: Additional information (dict)
        """
        # Convert dict action to discrete action (int) - FrozenLake is single-agent
        action = list(action_dict.values())[0]
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (numpy array)
        """
        image = self.env.render()
        return image


environment = EnvironmentWrapper()