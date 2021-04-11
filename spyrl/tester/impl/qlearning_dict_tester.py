import pickle
import numpy as np
from spyrl.tester.tester import Tester
from spyrl.util.util import override

class QLearningDictTester(Tester):

    @override(Tester)
    def select_action(self, state)->int:
        discrete_state = self.discretiser.discretise(state);
        if discrete_state in self.q:
            return np.argmax(self.q[discrete_state])
        else:
            return 0
    
    @override(Tester)
    def load_policy(self): # called by the constructor
        file = open(self.policy_path, 'rb')
        self.q = pickle.load(file)
        print('len(q):', len(self.q))
        file.close()
