from pathlib import Path

from overcooked_ai_py.agents.agent import RandomAgent

import environment

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')


class Agent:
    def __init__(self):
        self.env = environment.environment

    def sample(self):
        return self.env.action_space.sample()
    
    def predict(self, observation):
        return self.sample()


policy = Agent()
