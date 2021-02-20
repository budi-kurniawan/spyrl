from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override
from spyrl.tester.dqn_tester import DQNTester
from spyrl.normalizer.normalizer import Normalizer

class DQNTesterBuilder(TesterBuilder):
    
    @override(TesterBuilder)
    def __init__(self, policy_parent_path: str, num_learning_episodes: int, normalizer: Normalizer, input_dim: int) -> None:
        self.policy_parent_path = policy_parent_path
        self.num_learning_episodes = num_learning_episodes
        self.normalizer=normalizer
        self.input_dim=input_dim

    @override(TesterBuilder)
    def create_tester(self, trial):
        #format of policy filename: policy[XX]-[NumLearningEpisodes].p', where XX is the trial, e.g. policy00-100000.p
        policy_path = self.policy_parent_path + 'policy' + str(trial).zfill(2) + '-' + str(self.num_learning_episodes) + '.p'
        print('dqn policy path for activity:', policy_path)
        return DQNTester(policy_path, normalizer=self.normalizer, input_dim=self.input_dim)