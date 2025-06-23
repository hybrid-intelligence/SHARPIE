# AMaze - Task Description

## Overview

AMaze is primarily a maze generator: its goal is to provide an easy way to generate arbitrarily complex (or simple) mazes for agents to navigate in. Clues point agents towards the correct direction, which is required for them to solve intersections. Traps serve the opposite purpose and are meant to provide more challenge to the maze navigation tasks. Finally, Lures are low-danger traps that can be detected using local information only (i.e. go into a wall).

## Learning Objectives

- **Spatial Reasoning**: Interpret local observations to infer global maze structure
- **Temporal Context Integration**: Leverage past movement direction to avoid loops
- **Symbolic Understanding**: Decode clues/traps from discrete inputs or visual patterns (continuous mode)
- **Input Adaptation**: Generalize across discrete, hybrid (continuous input + discrete actions), and fully continuous control schemes

## Instructions

1. Read through the task description and learning objectives carefully
2. Proceed to the configuration page to set up your experiment parameters
3. Run the experiment to observe lures through immediate surroundings (e.g., adjacent walls) 