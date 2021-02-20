from spyrl.util.util import override
from spyrl.tester_builder.mo_tester_builder import MultiObjectiveTesterBuilder
from spyrl.tester.mo_actor_critic_tester import MultiObjectiveActorCriticTester

class MultiObjectiveActorCriticTesterBuilder(MultiObjectiveTesterBuilder):
    @override(MultiObjectiveTesterBuilder)
    def create_tester(self, trial):
        return MultiObjectiveActorCriticTester(self.policy_path, discretizer=self.discretizer, num_rewards=self.num_rewards)