#!/usr/bin/env python3
"""
    LunarLander learning with DQN agent
"""
import gym
import sys
import os
from spyrl.tester_builder.impl.ppo_tester_builder import PPOTesterBuilder
from spyrl.listener.impl.renderer import Renderer
from spyrl.activity.testing import Testing
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.test_result_logger import TestResultLogger
from spyrl.listener.impl.console_log_listener import ConsoleLogListener
from spyrl.listener.impl.file_log_listener import RewardType
from spyrl.tester_builder.impl.dqn_tester_builder import DQNTesterBuilder
from spyrl.util.util import get_project_dir

if __name__ == '__main__':
    id = 'CartPole-v2'
    gym.envs.register(
        id=id,
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100_000
    )
    env = gym.make(id)
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    nn_dims = (num_states, 64, 64, num_actions)
    num_learning_episodes = 626
    policy_parent_path = os.path.join(get_project_dir(), 'result/cartpole/ppo-01/')
    out_path = policy_parent_path + 'performance-' + str(num_learning_episodes) + '/'
    config = ActivityConfig(num_trials=1, num_episodes=5, out_path=out_path)    
    policy_parent_path = os.path.join(get_project_dir(), policy_parent_path)
    
    tester_builder = PPOTesterBuilder(policy_parent_path, num_learning_episodes, None)
    testing = Testing(listeners=[ConsoleLogListener(), TestResultLogger(RewardType.TOTAL)])
    testing.test(env, tester_builder, config)