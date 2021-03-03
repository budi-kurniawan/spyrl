from numpy.random import Generator, PCG64
from spyrl.tester.tester import Tester
from spyrl.util.util import override

class RandomTester(Tester):

    def __init__(self, num_actions: int, seed=None):
        if seed is None:
            self.np_random = Generator(PCG64())
        else:
            self.np_random = Generator(PCG64(seed))
        self.num_actions = num_actions

    @override(Tester)
    def select_action(self, state)->int:
        return self.np_random.choice(self.num_actions)