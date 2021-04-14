from spyrl.tester.impl.qlearning_sub_dict_tester import QLearningSubDictTester
from spyrl.util.util import override
from spyrl.tester_builder.impl.qlearning_dict_tester_builder import QLearningDictTesterBuilder

class QLearningSubDictTesterBuilder(QLearningDictTesterBuilder):
    @override(QLearningDictTesterBuilder)
    def create_tester(self, trial):
        #format of policy filename: policy[XX]-[NumLearningEpisodes].p', where XX is the trial, e.g. policy00-100000.p
        policy_path = self.policy_parent_path + 'policy' + str(trial).zfill(2) + '-' + str(self.num_learning_episodes) + '.p'
        print('policy path for activity:', policy_path)
        return QLearningSubDictTester(policy_path, discretiser=self.discretiser)    