""" A class representing D2D-SQL agents with fixed epsilon """
from spyrl.util.util import override
from spyrl.activity.activity_context import ActivityContext
from spyrl.agent.dqn_agent import DQNAgent
from spyrl.agent.d2dsql_agent import D2DSQLAgent

__author__ = 'bkurniawan'

class D2DSQLAgent2(D2DSQLAgent):    
    @override(DQNAgent)
    def episode_start(self, activity_context: ActivityContext):
        self.current_epsilon = 0.05