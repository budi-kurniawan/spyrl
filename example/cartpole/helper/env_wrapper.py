class GymEnvWrapper():
    
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        self.env.reset()
        self.env.state = [0, 0, 0, 0]
        return [0,0,0,0]
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        self.env.render()
        
    def seed(self, seed):
        self.env.seed(seed)