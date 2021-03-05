from spyrl.discretiser.discretiser import Discretiser
from spyrl.util.util import override

class GridworldDiscretiser(Discretiser):
    def __init__(self, num_states):
        self.num_states = num_states
        
    @override(Discretiser)
    def get_num_discrete_states(self):
        return self.num_states
    
    @override(Discretiser)
    def discretise(self, state):
        return int(state)