import gymnasium as gym

def input_mapping(inputs):
    # No annotation given, immediate reward = 0
    if len(inputs) == 0:
        return 0
    
    for agent, actions in inputs.items():
        # positive reward
        if 'ArrowRight' in actions:
            inputs[agent] = 1
        # negative reward
        elif 'ArrowLeft' in actions:
            inputs[agent] = -1
        else:
            # ignore input
            inputs[agent] = 0

    return inputs['agent_0']

def termination_condition(terminated, truncated):
    return terminated or truncated

environment = gym.make('MountainCar-v0', render_mode="rgb_array")