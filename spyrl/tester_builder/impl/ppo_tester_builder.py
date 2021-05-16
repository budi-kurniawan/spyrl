from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override
from spyrl.tester.impl.ppo_tester import PPOTester
from spyrl.normaliser.normaliser import Normaliser

class PPOTesterBuilder(TesterBuilder):
    
    @override(TesterBuilder)
    def __init__(self, policy_parent_path: str, num_learning_episodes: int, normaliser: Normaliser) -> None:
        self.policy_parent_path = policy_parent_path
        self.num_learning_episodes = num_learning_episodes
        self.normaliser = normaliser

    @override(TesterBuilder)
    def create_tester(self, trial):
        policy_path = self.policy_parent_path + 'policy' + str(trial).zfill(2) + '-' + str(self.num_learning_episodes) + '.p'
        return PPOTester(policy_path, normaliser=self.normaliser)