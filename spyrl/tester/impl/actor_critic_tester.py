import pickle
import numpy as np
from spyrl.tester.tester import Tester
from spyrl.util.util import override

class ActorCriticTester(Tester):

    @override(Tester)
    def select_action(self, state)->int:
        discrete_state = self.discretizer.discretize(state);
        return np.argmax(self.theta[discrete_state])
    
    @override(Tester)
    def load_policy(self): # called by the constructor
        file = open(self.policy_path, 'rb')
        self.theta = pickle.load(file)
        file.close()
