from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override
from spyrl.tester.impl.actor_critic_tester import ActorCriticTester
from spyrl.discretiser.discretiser import Discretiser

class ActorCriticTesterBuilder(TesterBuilder):

    def __init__(self, policy_parent_path: str, num_learning_episodes: int, discretiser: Discretiser) -> None:
        self.policy_parent_path = policy_parent_path
        self.num_learning_episodes = num_learning_episodes
        self.discretiser = discretiser
    
    @override(TesterBuilder)
    def create_tester(self, trial):
        #format of policy filename: policy[XX]-[NumLearningEpisodes].p', where XX is the trial, e.g. policy00-100000.p
        policy_path = self.policy_parent_path + 'policy' + str(trial).zfill(2) + '-' + str(self.num_learning_episodes) + '.p'
        return ActorCriticTester(policy_path, discretiser=self.discretiser)