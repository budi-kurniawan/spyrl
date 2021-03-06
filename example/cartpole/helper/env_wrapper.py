import numpy as np
class GymEnvWrapper():
    
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        self.env.reset()
        self.env.state = np.array([0, 0, 0, 0])
        return self.env.state
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        self.env.render()
        
    def seed(self, seed):
        self.env.seed(seed)