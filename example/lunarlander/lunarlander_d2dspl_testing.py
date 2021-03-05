#!/usr/bin/env python3
"""
    LunarLander learning with D2DSPL agent that uses actor critic with eligibility traces (ACET) in reinforcement phase 
"""
import gym
import sys
import os
sys.path.insert(0, "../spyrl")
from spyrl.listener.impl.file_log_listener import RewardType
from spyrl.tester_builder.impl.d2dspl_actor_critic_traces_tester_builder import D2DSPLActorCriticTracesTesterBuilder
from spyrl.activity.testing import Testing
from spyrl.listener.impl.console_log_listener import ConsoleLogListener
from spyrl.listener.impl.test_result_logger import TestResultLogger
from spyrl.activity.activity_config import ActivityConfig
from spyrl.util.util import get_project_dir

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    num_learning_episodes = 5000
    policy_parent_path = 'result/lunarlander/d2dspl-5000-06/'
    out_path = policy_parent_path + 'performance/'
    config = ActivityConfig(num_trials=10, num_episodes=100, out_path=out_path)    
    policy_parent_path = os.path.join(get_project_dir(), policy_parent_path)    
    
    tester_builder = D2DSPLActorCriticTracesTesterBuilder(policy_parent_path, num_learning_episodes, None)    
    testing = Testing(listeners=[ConsoleLogListener(), TestResultLogger(RewardType.TOTAL)])
    testing.test(env, tester_builder, config)