from spyrl.tester_builder.tester_builder import TesterBuilder
from spyrl.util.util import override

class MultiObjectiveTesterBuilder(TesterBuilder):
    @override(TesterBuilder)
    def __init__(self, policy_path: int, **kwargs) -> None:
        super().__init__(policy_path, **kwargs)
        self.num_rewards = kwargs.get('num_rewards', None)