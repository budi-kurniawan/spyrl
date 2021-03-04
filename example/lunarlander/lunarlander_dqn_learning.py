#!/usr/bin/env python3
"""
    LunarLander learning with DQN agent
"""
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.dqn_agent import DQNAgent
from spyrl.listener.impl.file_log_listener import RewardType
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

class AceZeroDQNAgentBuilder(AgentBuilder):
    def create_agent(self, seed, initial_policy_path=None):
        num_inputs = 8
        memory_size = 1_000_000; batch_size = 64; dqn_dims = [num_inputs, 300, self.num_actions]
        return DQNAgent(memory_size, batch_size, dqn_dims, self.normaliser, seed)

if __name__ == '__main__':
    
    start_trial = 1
    if len(sys.argv) > 1:
        start_trial = int(sys.argv[1])
    
    env = gym.make('LunarLander-v2')
    config = ActivityConfig(start_trial=start_trial, num_trials=10, num_episodes=20000, out_path='result/lunarlander/dqn-00/')
    agent_builder = AceZeroDQNAgentBuilder(env.action_space.n)
    milestone_episodes = [10000, 12000, 14000, 16000, 18000]
    learning = Learning(listener=BasicFunctions(render=False, draw=False, milestone_episodes=milestone_episodes, 
            reward_type=RewardType.TOTAL))
    learning.learn(env, agent_builder, config)