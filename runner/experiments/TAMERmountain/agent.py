import gymnasium as gym
from tamer import TAMERAgent
import numpy as np
import os

class Agent:
    def __init__(self, name, room_name):
        self.name = name
        self.room_name = room_name
        self.tamer = TAMERAgent(state_dim=2, action_dim=3, learning_rate=0.1, discount_factor=0.9)
        if os.path.exists(f"collected/{self.room_name}_tamer_model.npy"):
            self.tamer.load_model(f"experiments/TAMERmountain/collected/{self.room_name}_tamer_model.npy")

    def sample(self, observation):
        return self.tamer.select_action(observation)
    
    def predict(self, observation):
        return self.tamer.select_action(observation)
    
    def train(self, state, action, reward, done, next_state):
        # if reward == 0 and not done: 
        #     return
        print("reward received:", reward)
        td_target = reward
        if reward is not None and reward != 0:
            self.tamer.update_reward_model(state, action, td_target)
            self.tamer.save_model(f"experiments/TAMERmountain/collected/{self.room_name}_tamer_model.npy")
    
def create_agents(room_name):
    return [Agent('agent_0', room_name)]