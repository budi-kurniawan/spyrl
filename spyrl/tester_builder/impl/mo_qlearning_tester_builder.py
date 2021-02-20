from spyrl.util.util import override
from spyrl.tester_builder.mo_tester_builder import MultiObjectiveTesterBuilder
from spyrl.tester.mo_qlearning_tester import MultiObjectiveQLearningTester

class MultiObjectiveQLearningTesterBuilder(MultiObjectiveTesterBuilder):

    @override(MultiObjectiveTesterBuilder)
    def create_tester(self):
        return MultiObjectiveQLearningTester(self.policy_path, discretizer=self.discretizer, num_rewards=self.num_rewards)