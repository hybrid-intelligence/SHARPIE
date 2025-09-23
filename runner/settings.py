from environment import *
from agent import *

from collections.abc import Iterable

def input_mapping(inputs):
    if len(inputs) == 0:
        return 1
    
    for agent, actions in inputs.items():
        if 'ArrowLeft' in actions:
            inputs[agent] = 0
        elif 'ArrowRight' in actions:
            inputs[agent] = 2
        else:
            inputs[agent] = 1

    return inputs['agent_0']

def termination_condition(terminated, truncated):
    return terminated or truncated

user_per_experiment = 1