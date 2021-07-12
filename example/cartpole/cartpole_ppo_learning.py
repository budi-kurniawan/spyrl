#!/usr/bin/env python3
"""
    Cartpole learning with PPO
"""
import os
import gym
import sys
from spyrl.util.util import get_project_dir
sys.path.insert(0, "../spyrl")
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.agent.impl.ppo_agent import PPOAgent
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

class PPOAgentBuilder(AgentBuilder):
    def create_agent(self, seed, initial_policy_path=None):
        normaliser = None
        seed = 1
        return PPOAgent(nn_dims, normaliser, seed, local_steps_per_epoch=7)

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
    
    out_path = os.path.join(get_project_dir(), 'result/cartpole/ppo-test/')
    config = ActivityConfig(num_episodes=2, out_path=out_path)
    
    agent_builder = PPOAgentBuilder(num_actions)
    learning = Learning(listener=BasicFunctions(render=False))
    learning.learn(env, agent_builder, config)