import pickle
from spyrl.tester.tester import Tester
from spyrl.util.util import override

class D2DSPLActorCriticTracesTester(Tester):
    @override(Tester)
    def __init__(self, policy_path, **kwargs):
        self.input_dim = kwargs.get('input_dim', None)
        super().__init__(policy_path, **kwargs)
        
    @override(Tester)
    def select_action(self, state)->int:
        if self.normaliser is not None:
            state = self.normaliser.normalise(state)
        p = self.classifier.predict([state])
        return p[0]

    def load_policy(self):
        file = open(self.policy_path, 'rb')
        self.classifier = pickle.load(file)
        file.close()