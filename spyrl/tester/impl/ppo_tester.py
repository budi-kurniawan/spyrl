import pickle
import torch
from spyrl.tester.tester import Tester
from spyrl.util.util import override

class PPOTester(Tester):
    @override(Tester)
    def select_action(self, state)->int:
        if self.normaliser is not None:
            state = self.normaliser.normalise(state)
        a, _, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
        return a

    def load_policy(self):
        file = open(self.policy_path, 'rb')
        self.ac = pickle.load(file)
        file.close()