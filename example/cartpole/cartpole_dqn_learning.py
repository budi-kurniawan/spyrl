#!/usr/bin/env python3
"""
    LunarLander learning with DQN agent
"""
import gym
import sys
from example.cartpole.helper.env_wrapper import GymEnvWrapper
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

# class DQNAgentBuilder(AgentBuilder):
#     def create_agent(self, seed, initial_policy_path=None):
#         num_inputs = 8
#         memory_size = 1_000_000; batch_size = 64; dqn_dims = [num_inputs, 300, self.num_actions]
#         return DQNAgent(memory_size, batch_size, dqn_dims, self.normaliser, seed)
class DQNAgentBuilder(AgentBuilder):
    def create_agent(self, seed, initial_policy_path=None):
        num_inputs = 4
        memory_size = 50_000; batch_size = 64; dqn_dims = [num_inputs, 128, self.num_actions]
        return DQNAgent(memory_size, batch_size, dqn_dims, self.normaliser, seed)

if __name__ == '__main__':
    
    start_trial = 1
    if len(sys.argv) > 1:
        start_trial = int(sys.argv[1])
    
    id = 'CartPole-v2'
    gym.envs.register(
        id=id,
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100_000
    )
    env = gym.make(id)
    num_actions = env.action_space.n
    env = GymEnvWrapper(env)

    config = ActivityConfig(start_trial=start_trial, num_trials=1, num_episodes=1000, out_path='result/cartpole/dqn-01-with-wrapper/')
    agent_builder = DQNAgentBuilder(num_actions)
    milestone_episodes = []
    learning = Learning(listener=BasicFunctions(render=True, draw=True, milestone_episodes=milestone_episodes, 
            reward_type=RewardType.TOTAL))
    learning.learn(env, agent_builder, config)