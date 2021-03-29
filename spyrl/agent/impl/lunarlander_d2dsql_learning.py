#!/usr/bin/env python3
"""
    LunarLander learning with D2DSPL agent that uses actor critic with eligibility traces (ACET) in reinforcement phase 
"""
import os
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.util.util import get_project_dir
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.d2dsql_agent2 import D2DSQLAgent2
from spyrl.listener.impl.session_logger import SessionLogger
from spyrl.listener.impl.gmailer import Gmailer
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.listener.impl.file_log_listener import RewardType

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

class D2DSQLAgentBuilder(AgentBuilder):
    def create_agent(self, seed, initial_policy_path=None):
        num_inputs = 8
        memory_size = 50_000; batch_size = 64; dqn_dims = [num_inputs, 128, self.num_actions]
        trial = seed
        normalised_training_set_path = os.path.join(normalised_training_set_parent_path,
                'd2dspl-normalised_training_set-' + str(trial).zfill(2) + '-00005000.txt')
        return D2DSQLAgent2(normalised_training_set_path, target_loss, memory_size, batch_size, dqn_dims, self.normaliser, seed)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    max_num_samples_for_classifier = 500
    num_episodes = 10000
    session_id = '22'
    target_loss = 0.01 #0.001

    normalised_training_set_parent_path = 'result/lunarlander/d2dspl-acet-' + str(num_episodes) + '-' + session_id
    description = 'd2dsql-' + session_id + '. mem_size=50,000, batch_size=64, hidden dims=128, D2DSQLAgent2 (fixed epsilon)' + \
            'training set from ' + normalised_training_set_parent_path + \
            '/d2dspl-normalised_training_set-0x-00050000.txt\n' + 'target_loss: ' + str(target_loss)
    
    milestone_episodes = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    config = ActivityConfig(start_trial=1, num_trials = 10, num_episodes=num_episodes, 
                            out_path='result/lunarlander/d2dsql-' + session_id + '/')
    agent_builder = D2DSQLAgentBuilder(num_actions)
    listeners = [BasicFunctions(render=False, draw=False, reward_type=RewardType.TOTAL, milestone_episodes=milestone_episodes),
                Gmailer("D2DSQL-22"), SessionLogger(description)]
    learning = Learning(listeners=listeners)
    learning.learn(env, agent_builder, config)