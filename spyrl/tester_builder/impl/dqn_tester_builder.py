from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override
from spyrl.tester.impl.dqn_tester import DQNTester
from spyrl.normaliser.normaliser import Normaliser

class DQNTesterBuilder(TesterBuilder):
    
    @override(TesterBuilder)
    def __init__(self, policy_parent_path: str, num_learning_episodes: int, normaliser: Normaliser, input_dim: int) -> None:
        self.policy_parent_path = policy_parent_path
        self.num_learning_episodes = num_learning_episodes
        self.normaliser = normaliser
        self.input_dim=input_dim

    @override(TesterBuilder)
    def create_tester(self, trial):
        policy_path = self.policy_parent_path + 'policy' + str(trial).zfill(2) + '-' + str(self.num_learning_episodes) + '.p'
        return DQNTester(policy_path, normaliser=self.normaliser, input_dim=self.input_dim)