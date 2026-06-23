Glossary
========

This page contains definitions of terms commonly used in the SHARPIE documentation.

.. glossary::

   SHARPIE
      Shared Human-AI Reinforcement Learning Platform for Interactive Experiments, a Python-based modular framework for Reinforcement Learning and Human-AI interaction experiments.

   Reinforcement Learning (RL)
      A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.

   Human-AI Interaction
      The study and design of systems that facilitate effective collaboration between humans and artificial intelligence agents.

   Django
      A high-level Python web framework that encourages rapid development and clean, pragmatic design. SHARPIE uses Django to serve its web interface to users.

   Web server
      A software system that delivers web pages to users' browsers upon request. In SHARPIE, Django acts as the web server.

   Runner
      A component in SHARPIE responsible for managing the execution of experiments, including initializing/running environments and AI agents.

   Experiment
      A configured study combining zero, one or more RL agent(s), a single environment, one or more participants.

   Session
      The run of a single participant or group of participants through an experiment. Consists of one or more episodestrials.

   Episode
      A complete sequence of interactions between a set of agents, and their environment—from an initial state to a terminal or truncated state—representing one coherent task or decision-making cycle.

   User
      A human that interacts with the web interface. Can be either a Participant or Researcher.

   Participant
      A human subject selected to participate in an experiment, this extends the base user.

   Policy
      The instance of an RL policy, i.e. a particular stochastic or deterministic mapping of states to actions.

   Agent
      The union of policies and participants, i.e. an entity that perceives information from its environment, makes decisions based on those perceptions and internal goals, and acts to influence the environment's future state.

   Environment
      A simulated Markov decision process (MDP) adhering to the Gymnasium API with a tensor actions and rewards to support multi-agent and multi-objective settings.

   EventLog
      a chronological record of all events within an episodea trial, including environment states, agent observations and actions, human inputs, rewards, and other contextual data. It should provide a comprehensive trace of the hybrid Human-AI system's behavior and dynamics over time.

   Event
      a single entry within an event log that captures the state of the environment, an agent's (human or artificial) actions or other outputs, the resulting outcomes or rewards, and any relevant contextual information at a specific point during a trial.

