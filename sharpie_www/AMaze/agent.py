import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")  # a very simple application
done = False
env.reset()  # 2 observables - position and velocity

DISCRETE_OBSERVATION_SPACE_SIZE = [20] * len(env.observation_space.high)  # will give out 20*20 list

# see how big is the range for each of the 20 different buckets
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE

LEARNING_RATE = 0.9
DISCOUNT = 0.95  # how important we find the new future actions are ; future reward over current reward
EPISODES = 20000
render = False

# even though the solution might have been found, we still wish to look for other solutions
epsilon = 0.01  # 0-1 ; higher it is, more likely for it to perform something random action
START_EPSILON_DECAYING = 1
# python2 style division - gives only int values
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))  # return as tuple

def update_q_table(discrete_obs, action, reward, agent, terminated, truncated, observation):
    if not (terminated or truncated or observation[0] > 0.5):
        # max q value for the next state calculated above
        max_future_q = np.max(agent[discrete_obs])
        # q value for the current action and state
        current_q = agent[discrete_obs + (action,)]
        new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        # based on the new q, we update the current Q value
        agent[discrete_obs + (action,)] = new_q
        # goal reached; reward = 0 and no more negative
    elif observation[0] > 0.5:
        agent[discrete_obs + (action,)] = 100