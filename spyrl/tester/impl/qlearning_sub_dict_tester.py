import pickle
import os
from spyrl.util.util import override
from spyrl.tester.impl.qlearning_dict_tester import QLearningDictTester

class QLearningSubDictTester(QLearningDictTester):

    @override(QLearningDictTester)
    def load_policy(self): # called by the constructor
        file_index = 1
        self.q = {}
        while True:
            p = self.policy_path + '-' + str(file_index).zfill(4)
            if not os.path.exists(p):
                break
            file = open(p, 'rb')
            temp = pickle.load(file)
            self.q = {**self.q, **temp}
            file.close()
            file_index += 1
        print("loaded len(q):", len(self.q))