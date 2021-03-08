#!/usr/bin/env python3
"""
    LunarLander learning with DQN agent
"""
import gym
import sys
from spyrl.agent.impl.double_dqn_agent import DoubleDQNAgent
sys.path.insert(0, "../spyrl")
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.dqn_agent import DQNAgent
from spyrl.listener.impl.file_log_listener import RewardType
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.listener.impl.gmailer import Gmailer

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

class DDQNAgentBuilder(AgentBuilder):
    def create_agent(self, seed, initial_policy_path=None):
        num_inputs = 8
        memory_size = 50_000; batch_size = 64; dqn_dims = [num_inputs, 128, self.num_actions]
        return DoubleDQNAgent(memory_size, batch_size, dqn_dims, self.normaliser, seed)

if __name__ == '__main__':
    
    start_trial = 1
    if len(sys.argv) > 1:
        start_trial = int(sys.argv[1])
    
    env = gym.make('LunarLander-v2')
    config = ActivityConfig(start_trial=start_trial, num_trials=10, num_episodes=10000, out_path='result/lunarlander/ddqn-03/')
    agent_builder = DDQNAgentBuilder(env.action_space.n)
    milestone_episodes = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
    learning = Learning(listener=BasicFunctions(render=False, draw=False, milestone_episodes=milestone_episodes, 
            reward_type=RewardType.TOTAL))
    learning.add_listener(Gmailer("DDQN-03"))
    learning.learn(env, agent_builder, config)