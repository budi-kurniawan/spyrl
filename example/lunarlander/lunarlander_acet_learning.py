#!/usr/bin/env python3
"""
    LunarLander learning with actor critic with eligibility traces (ACET) agent
"""
import gym
import sys
from spyrl.listener.impl.file_log_listener import RewardType
sys.path.insert(0, "../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.agent_builder.impl.actor_critic_traces_agent_builder import ActorCriticTracesAgentBuilder
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    config = ActivityConfig(num_episodes=10_000, out_path='result/lunarlander/acet-01/')
    agent_builder = ActorCriticTracesAgentBuilder(num_actions, discretiser=LunarLanderDiscretiser())
    learning = Learning(listener=BasicFunctions(render=False, draw=False, reward_type=RewardType.TOTAL))
    learning.learn(env, agent_builder, config)