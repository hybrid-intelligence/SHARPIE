import numpy as np
from tamer import Tamer
from environment import environment


class Policy:
    """
    A TAMER-based policy for the AMaze environment.
    TAMER (Training an Agent Manually via Evaluative Reinforcement)
    uses human feedback to train the agent.
    """

    def __init__(self, id=""):
        """Initialize the TAMER policy."""
        self.name = "TAMER_Policy"
        self.id = id
        self.env = environment
        # Load or create Tamer model
        model_file = f"{id}_model" if id else None
        self.tamer = Tamer(environment=self.env, model_file_to_load=model_file)

    def predict(self, observation, participant_input=None):
        """
        Predict an action based on the observation.

        Args:
            observation: Current observation from the environment
            participant_input: Optional human feedback (reward signal)

        Returns:
            action: The action to take
        """
        # Use the TAMER model to select action
        return self.tamer.act(observation)

    def update(self, state, action, reward, done, next_state):
        """
        Update the policy based on human feedback.

        Args:
            state: Previous observation
            action: Action taken
            reward: Reward (human feedback)
            done: Whether the episode ended
            next_state: Next observation
        """
        # Only update for meaningful feedback
        if reward == 0 and not done:
            return
        if not done:
            td_target = reward
            self.tamer.train(state, action, td_target)
            self.tamer.save_model(f"{self.id}_model")


# Create an instance of the policy for use by the runner
policy = Policy('save')