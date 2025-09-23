import gymnasium as gym

class Environment:
    def reset(self):
        pass
    
    def step(self, action):
        pass

    def render(self):
        pass

environment = gym.make("MountainCar-v0", render_mode="rgb_array")