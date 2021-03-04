#!/usr/bin/env python3
"""
    LunarLander learning with D2DSPL agent that uses actor critic with eligibility traces (ACET) in reinforcement phase 
"""
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.agent_builder.impl.d2dspl_actor_critic_traces_agent_builder import D2DSPLActorCriticTracesAgentBuilder
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.listener.impl.file_log_listener import RewardType
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    max_num_samples_for_classifier = 1000 #4000 #3000 #200 #300 #500 #1000 #1500 #2000
    config = ActivityConfig(num_trials = 10, num_episodes=5_000, out_path='result/lunarlander/d2dspl-01/')
    agent_builder = D2DSPLActorCriticTracesAgentBuilder(num_actions, LunarLanderDiscretiser(), 
                        max_num_samples_for_classifier, None, [128, 128])
    learning = Learning(listener=BasicFunctions(render=False, draw=False, reward_type=RewardType.TOTAL))
    learning.learn(env, agent_builder, config)