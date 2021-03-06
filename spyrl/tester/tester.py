__author__ = 'bkurniawan'

""" A Tester is used to test a policy """
from spyrl.activity.activity_context import ActivityContext
class Tester:

    def __init__(self, policy_path, **kwargs):
        self.policy_path = policy_path
        self.discretiser = kwargs.get('discretiser', None)
        self.normaliser = kwargs.get('normaliser', None)
        self.load_policy()

    def trial_start(self, activity_context: ActivityContext) -> None:
        pass

    # called at the end of a trial
    def trial_end(self, activity_context: ActivityContext) -> None:
        pass

    # called at the beginning of an episode
    def episode_start(self, activity_context: ActivityContext) -> None:
        pass

    # called at the end of an episode
    def episode_end(self, activity_context: ActivityContext) -> None:
        pass

    def select_action(self, state) -> int:
        pass
    
    def load_policy(self):
        pass