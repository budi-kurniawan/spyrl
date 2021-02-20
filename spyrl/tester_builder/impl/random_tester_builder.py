from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override
from spyrl.tester.random_tester import RandomTester

class RandomTesterBuilder(TesterBuilder):

    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
    
    @override(TesterBuilder)
    def create_tester(self, trial):
        return RandomTester(self.num_actions, trial)