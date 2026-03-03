import gymnasium as gym

class EnvironmentWrapper:
    """Wrapper for the MountainCar-v0 Gymnasium environment."""

    def __init__(self):
        """Initialize the environment."""
        # Porting the environment choice from your old code
        self.env = gym.make('MountainCar-v0', render_mode="rgb_array")

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
        Execute one step in the environment with custom input mapping.

        Args:
            action_dict: Dictionary with agent id as keys and action strings/lists as values
        """
        # Convert dict action to discrete action (int) - mountaincar is single-agent
        action = list(action_dict.values())[0]
        observation, reward, terminated, truncated, info = self.env.step(action)
        # erase environment reward and use human feedback instead
        return observation, None, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (numpy array)
        """
        return self.env.render()

    def is_done(self, terminated, truncated):
        """Logic from your old 'termination_condition'."""
        return terminated or truncated

# Instantiate the new format
environment = EnvironmentWrapper()