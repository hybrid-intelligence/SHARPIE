Hello World Example
===================

In this minimal example, we will create a random mountain car environment with a random policy.

In your project root create a new file ``environment.py`` with the following content:

.. code-block:: python

   import gymnasium as gym

   class EnvironmentWrapper:
      """SHARPIE wrapper for the MountainCar-v0 Gymnasium environment."""

      def __init__(self):
         """Initialize the environment."""
         self.env = gym.make("MountainCar-v0", render_mode="rgb_array")

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
         # Convert dict action to discrete action (int) - MountainCar is single-agent
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

Now a second new file called ``policy.py`` with the following content:

.. code-block:: python

   import gymnasium as gym

   class Policy:
      """Random policy for the MountainCar-v0 environment."""

      def __init__(self):
         """Initialize the policy."""
         self.action_space = gym.make("MountainCar-v0").action_space

      def predict(self, observation, participant_input=None):
         """
         Select an action based on the observation.

         Args:
               observation: Current observation (numpy array)

         Returns:
               action: Selected action (int)
         """
        if participant_input is None: # no participant input, sample random action
            action = self.action_space.sample()
        else:
            action = participant_input
        return action
   
   policy = Policy()

Now browse to `localhost:8000/admin <http://localhost:8000/admin>`_ and create several database entries.

| First create an Environment entry.
| Experiment > Environments > Add Environment > Name: MountainCar, add the following list of environment files.

.. code-block:: json
   
   {
      "environment": "/full/path/to/environment.py",
   }

.. note::

    Get ``/full/path/to`` by running ``pwd`` in the terminal:

| Second, create Policy entry.
| Experiment > Policies > Add Policy > Name: Random Policy, add the following in list of policy files:

.. code-block:: json

   {
      "policy": "/full/path/to/policy.py",
   }

| Finally, create an Agent entry.
| Experiment > Agents > Add Agent

* Role: random_agent
* Name: Random Agent
* Policy: Random Policy
* Can the participant act?: yes
* Inputs captured from participant:
  ``{"ArrowLeft": 0, "ArrowRight": 2, "default": 1}``
* Display config:
  ``{"ArrowLeft": {"symbol": "←", "label": "Left"}, "ArrowRight": {"symbol": "→", "label": "Right"}}``

Leave all other fields as default and select `SAVE`. 

| Now create a new Experiment entry.
| Experiment > Experiments > Add Experiment

* Name: Hello world Experiment
* Link: hello-world
* Home page text: home test
* Experiment page text: experiment test
* Environment: mountain_test
* Agents: Random Agent
* Number of episodes: 30

Keep all other fields in their defaults and select `SAVE`.

| Install gymnasium with the classic_control in your virtual environment:

.. code-block:: console
   
   pip install "gymnasium[classic_control]"
   sharpie-runner runserver --connection-key my_secret


| Now browse to `localhost:8000 <http://localhost:8000>`_, select your experiment and start it.
| You should see the MountainCar environment rendered and a random policy controlling the agent.
| You can take over control by an arrow key press.

For more extensive examples, including on how to capture human demonstrations, conduct studies in a multi-agent multi-participant setting, and provide textual input, have a look at the `SHARPIE Gallery <https://github.com/hybrid-intelligence/SHARPIE_Gallery/>`_.