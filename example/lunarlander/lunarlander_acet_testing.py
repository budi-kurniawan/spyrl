#!/usr/bin/env python3
"""
    LunarLander learning with D2DSPL agent that uses actor critic with eligibility traces (ACET) in reinforcement phase 
"""
import gym
import os
from spyrl.tester_builder.impl.actor_critic_tester_builder import ActorCriticTesterBuilder
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser,\
    LunarLanderDiscretiser12288, LunarLanderDiscretiser24576
from spyrl.listener.impl.file_log_listener import RewardType
from spyrl.activity.testing import Testing
from spyrl.listener.impl.console_log_listener import ConsoleLogListener
from spyrl.listener.impl.test_result_logger import TestResultLogger
from spyrl.activity.activity_config import ActivityConfig
from spyrl.util.util import get_project_dir

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    num_learning_episodes = 5000
    policy_parent_path = 'result/lunarlander/d2dspl-acet-10000-24/'
    out_path = policy_parent_path + 'performance-acet-'+ str(num_learning_episodes) + '/'
    config = ActivityConfig(start_trial=1, num_trials=10, num_episodes=100, out_path=out_path)    
    policy_parent_path = os.path.join(get_project_dir(), policy_parent_path)
    tester_builder = ActorCriticTesterBuilder(policy_parent_path, num_learning_episodes, LunarLanderDiscretiser24576())
    testing = Testing(listeners=[ConsoleLogListener(), TestResultLogger(RewardType.TOTAL)])
    testing.test(env, tester_builder, config)
