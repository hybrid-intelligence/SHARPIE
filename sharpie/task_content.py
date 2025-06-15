"""
This module contains all the task descriptions and related content for different use cases.
Each use case has its own dictionary containing task description, learning objectives, and instructions.
"""

TASK_CONTENT = {
    # Mountain Car
    'mountain': {
        'task_description': """
        <p>The Mountain Car problem is a classic reinforcement learning challenge where you control a car that needs to climb a steep hill. 
        The car's engine is not powerful enough to climb the hill directly, so you need to learn to build up momentum by moving back and forth.</p>
        """,
        'learning_objectives': """
        <ul>
            <li>Understand the concept of building momentum to overcome physical limitations.</li>
            <li>Learn how to balance exploration and exploitation in reinforcement learning.</li>
            <li>Experience the challenges of solving problems with continuous state and action spaces.</li>
        </ul>
        """,
        'instructions': """
        <ol>
            <li>Read through the task description and learning objectives carefully.</li>
            <li>Proceed to the configuration page to set up your experiment parameters.</li>
            <li>Once configured, you'll be able to run the experiment and observe the car's behavior.</li>
            <li>Try different configurations to understand how they affect the learning process.</li>
        </ol>
        """
    },
    # AMaze
    'AMaze': {
        'task_description': """
        <p>AMaze is primarily a maze generator: its goal is to provide an easy way to generate arbitrarily complex (or simple) mazes for agents to navigate in.
        Clues point agents towards the correct direction, which is required for them to solve intersections. 
        Traps serve the opposite purpose and are meant to provide more challenge to the maze navigation tasks. 
        Finally, Lures are low-danger traps that can be detected using local information only (i.e. go into a wall).</p>
        """,
        'learning_objectives': """
        <ul>
            <li>Spatial Reasoning: Interpret local observations to infer global maze structure.</li>
            <li>Temporal Context Integration: Leverage past movement direction to avoid loops.</li>
            <li>Symbolic Understanding: Decode clues/traps from discrete inputs or visual patterns (continuous mode).</li>
            <li>Input Adaptation: Generalize across discrete, hybrid (continuous input + discrete actions), and fully continuous control schemes.</li>
        </ul>
        """,
        'instructions': """
        <ol>
            <li>Read through the task description and learning objectives carefully.</li>
            <li>Proceed to the configuration page to set up your experiment parameters.</li>
            <li>Run the experiment to observe lures through immediate surroundings (e.g., adjacent walls).</li>
        </ol>
        """
    },
    # Spread
    'spread': {
        'task_description': """
        <p>The Spread experiment simulates a multi-agent system where agents must spread out evenly across a space while maintaining 
        connectivity. This challenge tests coordination and spatial reasoning abilities in a distributed system.</p>
        """,
        'learning_objectives': """
        <ul>
            <li>Understand distributed spatial coordination strategies</li>
            <li>Learn how to maintain connectivity while achieving optimal spacing</li>
            <li>Experience the challenges of balancing local and global objectives</li>
        </ul>
        """,
        'instructions': """
        <ol>
            <li>Read through the task description and learning objectives carefully.</li>
            <li>Proceed to the configuration page to set up your experiment parameters.</li>
            <li>Configure the number of agents and their movement constraints.</li>
            <li>Run the experiment to observe how agents spread out while maintaining connectivity.</li>
        </ol>
        """
    },
    # Tag
    'tag': {
        'task_description': """
        <p>The Tag experiment is a predator-prey environment. 
        Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). 
        Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). 
        Obstacles (large black circles) block the way. 
        By default, there is 1 good agent, 3 adversaries and 2 obstacles.</p>
        """,
        'learning_objectives': """
        <ul>
            <li>Understand pursuit-evasion strategies in multi-agent systems.</li>
            <li>Develop strategies for predicting and intercepting moving targets.</li>
            <li>Experience the challenges of coordinating multiple chasers to catch evasive runners.</li>
        </ul>
        """,
        'instructions': """
        <ol>
            <li>Read through the task description and learning objectives carefully.</li>
            <li>Proceed to the configuration page to set up your experiment setup.</li>
            <li>Set the number and capabilities of both adversaries and good agents.</li>
            <li>Launch the simulation and observe how the agents interact in the environment.</li>
        </ol>
        """
    },
    # Minecraft
    'minecraft': {
        'task_description': """
        <p>The Minecraft experiment is a 3D environment where agents must learn to navigate, build, and interact with the world. 
        This challenge combines spatial reasoning, resource management, and creative problem-solving in a rich, interactive environment.</p>
        """,
        'learning_objectives': """
        <ul>
            <li>Understand 3D navigation and spatial reasoning in complex environments.</li>
            <li>Learn resource gathering and crafting mechanicm.</li>
            <li>Experience the challenges of building and modifying structures in a 3D world.</li>
        </ul>
        """,
        'instructions': """
        <ol>
            <li>Read through the task description and learning objectives carefully.</li>
            <li>Proceed to the configuration page to set up your experiment parameters.</li>
            <li>Configure the world settings and agent capabilities.</li>
            <li>Run the experiment to observe how agents interact with the Minecraft environment.</li>
        </ol>
        """
    }
} 