#!/usr/bin/env python3
"""
    LunarLander learning with D2DSPL agent that uses actor critic with eligibility traces (ACET) in reinforcement phase 
"""
import gym
import sys
import os
sys.path.insert(0, "../spyrl")
from spyrl.tester_builder.impl.actor_critic_tester_builder import ActorCriticTesterBuilder
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser
from spyrl.listener.impl.file_log_listener import RewardType
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
    num_learning_episodes = 10000
    policy_parent_path = 'result/lunarlander/acet-01/'
    out_path = policy_parent_path + 'performance-'+ str(num_learning_episodes) + '/'
    config = ActivityConfig(num_trials=10, num_episodes=100, out_path=out_path)    
    policy_parent_path = os.path.join(get_project_dir(), policy_parent_path)
    tester_builder = ActorCriticTesterBuilder(policy_parent_path, num_learning_episodes, LunarLanderDiscretiser())
    testing = Testing(listeners=[ConsoleLogListener(), TestResultLogger(RewardType.TOTAL)])
    testing.test(env, tester_builder, config)