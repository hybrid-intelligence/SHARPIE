class Policy:
    """A simple heuristic policy for the simple_spread environment."""

    def __init__(self, name):
        """Initialize the agent with its name."""
        self.name = name

    def predict(self, obs):
        """
        Predict an action based on the observation.

        The agent moves towards the nearest landmark (entity) when velocity is low.

        Args:
            obs: Observation for this agent (numpy array)
                - obs[0:2]: velocity
                - obs[4:6]: first entity position
                - obs[6:8]: second entity position
                - obs[8:10]: third entity position

        Returns:
            action: The action to take (0=NOOP, 1=LEFT, 2=RIGHT, 3=DOWN, 4=UP)
        """
        velocity = obs[0:2]
        first_entity = obs[4:6]
        second_entity = obs[6:8]
        third_entity = obs[8:10]

        # Find the nearest landmark
        if abs(first_entity[0]) + abs(first_entity[1]) < abs(second_entity[0]) + abs(second_entity[1]):
            target_entity = first_entity
        else:
            target_entity = second_entity
        if abs(third_entity[0]) + abs(third_entity[1]) < abs(target_entity[0]) + abs(target_entity[1]):
            target_entity = third_entity

        # Don't move if already moving
        if abs(velocity[0]) > 0.1 or abs(velocity[1]) > 0.1:
            return 0

        # Move towards the nearest landmark
        if abs(target_entity[0]) < abs(target_entity[1]):
            # Move vertically
            if target_entity[1] > 0 and abs(target_entity[1]) > 0.1:
                return 4  # UP
            elif abs(target_entity[1]) > 0.1:
                return 3  # DOWN
        else:
            # Move horizontally
            if target_entity[0] > 0 and abs(target_entity[0]) > 0.1:
                return 2  # RIGHT
            elif abs(target_entity[0]) > 0.1:
                return 1  # LEFT
        return 0  # NOOP


# Create an instance of the agent for use by the runner
policy = Policy(name="agent")