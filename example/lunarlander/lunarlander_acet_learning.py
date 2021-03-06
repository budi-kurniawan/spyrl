#!/usr/bin/env python3
"""
    LunarLander learning with actor critic with eligibility traces (ACET) agent
"""
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.agent_builder.impl.actor_critic_traces_agent_builder import ActorCriticTracesAgentBuilder
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.listener.impl.file_log_listener import RewardType
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser,\
    LunarLanderDiscretiser12288, LunarLanderDiscretiser24576

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    config = ActivityConfig(start_trial=4, num_trials=7, num_episodes=10000, out_path='result/lunarlander/acet-21/')
    agent_builder = ActorCriticTracesAgentBuilder(num_actions, discretiser=LunarLanderDiscretiser24576())
    milestone_episodes = [1000, 2000, 4000, 5000, 8000]
    learning = Learning(listener=BasicFunctions(render=False, draw=False, reward_type=RewardType.TOTAL, 
                        milestone_episodes=milestone_episodes))
    learning.learn(env, agent_builder, config)