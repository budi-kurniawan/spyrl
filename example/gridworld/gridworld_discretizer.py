from spyrl.discretizer.discretizer import Discretizer

class GridworldDiscretizer(Discretizer):
    def __init__(self, num_states):
        self.num_states = num_states
        
    def get_num_discrete_states(self):
        return self.num_states
    
    def discretize(self, state):
        return int(state)