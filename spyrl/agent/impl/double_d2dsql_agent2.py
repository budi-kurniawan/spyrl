""" A class representing D2D-SQL agents """
from spyrl.util.util import override
from spyrl.activity.activity_context import ActivityContext
from spyrl.agent.impl.double_d2dsql_agent import DoubleD2DSQLAgent

__author__ = 'bkurniawan'

class DoubleD2DSQLAgent2(DoubleD2DSQLAgent):    
    @override(DoubleD2DSQLAgent)
    def episode_start(self, activity_context: ActivityContext):
        self.current_epsilon = 0.05