#!/usr/bin/env python3
"""
    LunarLander learning with DQN agent
"""
import gym
import sys
import os
sys.path.insert(0, "../spyrl")
from spyrl.activity.testing import Testing
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.test_result_logger import TestResultLogger
from spyrl.listener.impl.console_log_listener import ConsoleLogListener
from spyrl.listener.impl.file_log_listener import RewardType
from spyrl.tester_builder.impl.dqn_tester_builder import DQNTesterBuilder
from spyrl.util.util import get_project_dir

if __name__ == '__main__':    
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    num_learning_episodes = 20000
    policy_parent_path = 'result/lunarlander/dqn-01/'
    out_path = policy_parent_path + 'performance-20000/'
    config = ActivityConfig(num_trials=10, num_episodes=100, out_path=out_path)    
    policy_parent_path = os.path.join(get_project_dir(), policy_parent_path)    
    
    tester_builder = DQNTesterBuilder(policy_parent_path, num_learning_episodes, None, input_dim=8)
    testing = Testing(listeners=[ConsoleLogListener(), TestResultLogger(RewardType.TOTAL)])
    testing.test(env, tester_builder, config)