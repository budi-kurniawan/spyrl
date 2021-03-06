from spyrl.tester.qlearning_tester import QLearningTester
from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override
from spyrl.discretizer.discretizer import Discretizer

class QLearningTesterBuilder(TesterBuilder):
    def __init__(self, policy_parent_path: str, num_learning_episodes: int, discretizer: Discretizer) -> None:
        self.policy_parent_path = policy_parent_path
        self.num_learning_episodes = num_learning_episodes
        self.discretizer = discretizer
    
    @override(TesterBuilder)
    def create_tester(self, trial):
        #format of policy filename: policy[XX]-[NumLearningEpisodes].p', where XX is the trial, e.g. policy00-100000.p
        policy_path = self.policy_parent_path + 'policy' + str(trial).zfill(2) + '-' + str(self.num_learning_episodes) + '.p'
        print('policy path for activity:', policy_path)
        return QLearningTester(policy_path, discretizer=self.discretizer)    